/*
 * Implementation of a distributed memory  kd-tree
 * The idea is to have a top level domain decomposition with a shallow shared
 * top level tree between computational nodes.
 *
 * Then each domain has a different set of points to work on separately
 * the top tree serves as a map to know later on in which processor ask for
 * neighbors
 */
#include "tree.h"
#include "heap.h"
#include "kdtree.h"
#include "mpi.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/sysinfo.h>
#include <float.h>

#define NBINS 20
/* 
 * Maximum bytes to send with a single mpi send/recv, used 
 * while communicating results of ngbh search
 */

/* Maximum allowed is 4GB */
//#define MAX_MSG_SIZE 4294967296

/* Used slices of 10 mb ? Really good? Maybe at the cause of TID thing */
// #define MAX_MSG_SIZE (10000 * k * sizeof(heap_node_t))
#define MAX_MSG_SIZE (100000 * k * sizeof(heap_node_t))


#define TOP_TREE_RCH 1
#define TOP_TREE_LCH 0
#define NO_CHILD -1

unsigned int data_dims;


int cmp_float_t(const void* a, const void* b)
{
    float_t aa = *((float_t*)a);
    float_t bb = *((float_t*)b);
    return  (aa > bb) - (aa < bb);
}




/* quickselect for an element along a dimension */
void swap_data_element(float_t *a, float_t *b, size_t vec_len) {
    float_t tmp;
    for (size_t i = 0; i < vec_len; ++i) 
    {
        tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
}

int compare_data_element(float_t *a, float_t *b, int compare_dim) {
    return -((a[compare_dim] - b[compare_dim]) > 0.) + ((a[compare_dim] - b[compare_dim]) < 0.);
}

int partition_data_element(float_t *array, int vec_len, int compare_dim,
                           int left, int right, int pivot_index) 
{
    int store_index = left;
    int i;

    /* Move pivot to end */
    swap_data_element(array + pivot_index * vec_len, array + right * vec_len, vec_len);
    for (i = left; i < right; ++i) 
    {
        // if(compare_data_element(array + i*vec_len, array + pivot_index*vec_len,
        // compare_dim ) >= 0){
        if (array[i * vec_len + compare_dim] < array[right * vec_len + compare_dim]) 
        {
            swap_data_element(array + store_index * vec_len, array + i * vec_len, vec_len);
            store_index += 1;
        }
    }
    /* Move pivot to its final place */
    swap_data_element(array + (store_index)*vec_len, array + right * vec_len, vec_len);

    return store_index;
}

int qselect_data_element(float_t *array, int vec_len, int compare_dim, int left, int right, int n) 
{
    int pivot_index;
    if (left == right) 
    {
        return left;
    }
    pivot_index = left; // + (rand() % (right-left + 1)); /* random int left <= x <= right */
    pivot_index = partition_data_element(array, vec_len, compare_dim, left, right, pivot_index);
    /* The pivot is in its final sorted position */
    if (n == pivot_index) 
    {
        return pivot_index;
    } 
    else if (n < pivot_index) 
    {
        return qselect_data_element(array, vec_len, compare_dim, left, pivot_index - 1, n);
    } 
    else 
    {
        return qselect_data_element(array, vec_len, compare_dim, pivot_index + 1, right, n);
    }
}

int quickselect_data_element(float_t *array, int vec_len, int array_size, int compare_dim, int k) 
{
    return qselect_data_element(array, vec_len, compare_dim, 0, array_size - 1, k - 1);
}

int CMP_DIM;
int compare_data_element_sort(const void *a, const void *b) {
    float_t aa = *((float_t *)a + CMP_DIM);
    float_t bb = *((float_t *)b + CMP_DIM);
    return ((aa - bb) > 0.) - ((aa - bb) < 0.);
}

void compute_bounding_box(global_context_t *ctx) {
    ctx->lb_box = (float_t *)MY_MALLOC(ctx->dims * sizeof(float_t));
    ctx->ub_box = (float_t *)MY_MALLOC(ctx->dims * sizeof(float_t));

    for (size_t d = 0; d < ctx->dims; ++d) {
    ctx->lb_box[d] = 99999999.;
    ctx->ub_box[d] = -99999999.;
    }

    #define local_data ctx->local_data
    #define lb ctx->lb_box
    #define ub ctx->ub_box

    /* compute minimum and maximum for each dimensions, store them in local bb */
    /* each processor on its own */
    for (size_t i = 0; i < ctx->local_n_points; ++i) {
    for (size_t d = 0; d < ctx->dims; ++d) {
      lb[d] = MIN(local_data[i * ctx->dims + d], lb[d]);
      ub[d] = MAX(local_data[i * ctx->dims + d], ub[d]);
    }
    }

    /* Reduce to obtain bb */
    /*
    MPI_Allreduce(  const void *sendbuf,
                                  void *recvbuf,
                                  int count,
                                  MPI_Datatype datatype,
                                  MPI_Op op,
                                  MPI_Comm comm)
    */

    /*get the bounding box */

    MPI_Allreduce(MPI_IN_PLACE, lb, ctx->dims, MPI_MY_FLOAT, MPI_MIN, ctx->mpi_communicator);
    MPI_Allreduce(MPI_IN_PLACE, ub, ctx->dims, MPI_MY_FLOAT, MPI_MAX, ctx->mpi_communicator);

    #undef local_data
    #undef lb
    #undef ub
}

/* i want a queue to enqueue the partitions to deal with */
void enqueue_partition(partition_queue_t *queue, partition_t p) 
{
    if (queue->count == queue->_capacity) 
    {
        queue->_capacity = queue->_capacity * 1.10;
        queue->data = realloc(queue->data, queue->_capacity);
    }
    /* insert point */
    memmove(queue->data + 1, queue->data, queue->count * sizeof(partition_t));
    queue->data[0] = p;
    queue->count++;
}

partition_t dequeue_partition(partition_queue_t *queue) 
{
  return queue->data[--(queue->count)];
}

void compute_medians_and_check(global_context_t *ctx, float_t *data) {
    float_t prop = 0.5;
    int k = (int)(ctx->local_n_points * prop);
    int d = 1;

    /*quick select on a particular dimension */
    CMP_DIM = d;
    int kk = (k - 1) * ctx->dims;

    int count = 0;
    // idx = idx - 1;
    //
    int aaa = quickselect_data_element(ctx->local_data, (int)(ctx->dims), (int)(ctx->local_n_points), d, k);
    /*
    * sanity check
    * check if the median found in each node is
    * a median
    */

    float_t *medians_rcv = (float_t *)MY_MALLOC(ctx->dims * ctx->world_size * sizeof(float_t));

    /*
    * MPI_Allgather(     const void *sendbuf,
    *                     int sendcount,
    *                     MPI_Datatype sendtype,
    *                     void *recvbuf,
    *                     int recvcount,
    *                     MPI_Datatype recvtype,
    *                     MPI_Comm comm)
    */

    /* Exchange medians */

    MPI_Allgather(ctx->local_data + kk, ctx->dims, MPI_MY_FLOAT, medians_rcv, ctx->dims, MPI_MY_FLOAT, ctx->mpi_communicator);

    /* sort medians on each node */

    CMP_DIM = d;
    qsort(medians_rcv, ctx->world_size, ctx->dims * sizeof(float_t), compare_data_element_sort);

    /*
    * Evaluate goodness of the median on master which has whole dataset
    */

    if (ctx->mpi_rank == 0) {
    int count = 0;
    int idx = (int)(prop * (ctx->world_size));
    // idx = idx - 1;
    for (int i = 0; i < ctx->n_points; ++i) 
    {
        count += data[i * ctx->dims + d] <= medians_rcv[idx * ctx->dims + d];
    }
    mpi_printf(ctx, "Choosing %lf percentile on dimension %d: empirical prop %lf\n", prop, d, (float_t)count / (float_t)(ctx->n_points));
    }
    free(medians_rcv);
}

float_t check_pc_pointset_parallel(global_context_t *ctx, pointset_t *ps, guess_t g, int d, float_t prop) {
    /*
     * ONLY FOR TEST PURPOSES
     * gather on master all data
     * perform the count on master
     */

    size_t pvt_count = 0;
    #pragma omp parallel for reduction(+:pvt_count)
    for (size_t i = 0; i < ps->n_points; ++i) 
    {
        pvt_count += ps->data[i * ps->dims + d] <= g.x_guess;
    }

    size_t pvt_n_and_tot[2] = {pvt_count, ps->n_points};
    size_t tot_count[2];
    MPI_Allreduce(pvt_n_and_tot, tot_count, 2, MPI_UINT64_T, MPI_SUM, ctx->mpi_communicator);

    float_t ep = (float_t)tot_count[0] / (float_t)(tot_count[1]);
    /*
    mpi_printf(ctx,"[PS TEST PARALLEL]: ");
    mpi_printf(ctx,"Condsidering %d points, searching for %lf percentile on
    dimension %d: empirical measure %lf\n",tot_count[1],prop, d, ep);
    */
    return ep;
}

void compute_bounding_box_pointset(global_context_t *ctx, pointset_t *ps) {
    #define local_data ps->data
    #define lb ps->lb_box
    #define ub ps->ub_box

    for (size_t d = 0; d < ps->dims; ++d)
    {
        ps->lb_box[d] =  FLT_MAX;
        ps->ub_box[d] = -FLT_MAX;
    }


    /* compute minimum and maximum for each dimensions, store them in local bb */
    /* each processor on its own */

    // this moves memory maybe there is a better way
    #pragma omp parallel 
    {
        float_t* pvt_lb = (float_t*)MY_MALLOC(ctx -> dims * sizeof(float_t));
        float_t* pvt_ub = (float_t*)MY_MALLOC(ctx -> dims * sizeof(float_t));
        for (size_t d = 0; d < ps->dims; ++d)
        {
            pvt_lb[d] = FLT_MAX;
            pvt_ub[d] = -FLT_MAX;
        }

        #pragma omp for
        for (size_t i = 0; i < ps->n_points; ++i) 
        {
            for (size_t d = 0; d < ps->dims; ++d) 
            {
                pvt_lb[d] = MIN(local_data[i * ps->dims + d], pvt_lb[d]);
                pvt_ub[d] = MAX(local_data[i * ps->dims + d], pvt_ub[d]);
            }
        }

        #pragma omp critical (bounding_box_reduction)
        {
            for (size_t d = 0; d < ps->dims; ++d) 
            {
                lb[d] = MIN(pvt_lb[d], lb[d]);
                ub[d] = MAX(pvt_ub[d], ub[d]);
            }
        }


        free(pvt_lb);
        free(pvt_ub);

    }

    /*get the bounding box */

    MPI_Allreduce(MPI_IN_PLACE, lb, ps->dims, MPI_MY_FLOAT, MPI_MIN, ctx->mpi_communicator);
    MPI_Allreduce(MPI_IN_PLACE, ub, ps->dims, MPI_MY_FLOAT, MPI_MAX, ctx->mpi_communicator);

    /*
    MPI_DB_PRINT("[PS BOUNDING BOX]: ");
    for(size_t d = 0; d < ps -> dims; ++d) MPI_DB_PRINT("d%d:[%lf, %lf] ",(int)d,
    lb[d], ub[d]); MPI_DB_PRINT("\n");
    */



    #undef local_data
    #undef lb
    #undef ub
}


guess_t retrieve_guess_pure(global_context_t *ctx, pointset_t *ps,
                            uint64_t *global_bin_counts, 
                            int k_global, int d, float_t pc)
{

    /*
    * retrieving the best median guess from pure binning
    */

    float_t total_count = 0.;
    for (int i = 0; i < k_global; ++i) total_count += (float_t)global_bin_counts[i];

    /*
    MPI_DB_PRINT("[ ");
    for(int i = 0; i < k_global; ++i)
    {
          MPI_DB_PRINT( "%lu %lf --- ", global_bin_counts[i],
                        (float_t)global_bin_counts[i]/(float_t)total_count);
    }
    MPI_DB_PRINT("\n");
    */

    float_t cumulative_count = 0;
    int idx = 0;
    while ((cumulative_count + (float_t)global_bin_counts[idx]) / total_count < pc) 
    {
        cumulative_count += (float_t)global_bin_counts[idx];
        idx++;
    }
    /* find best spot in the bin */
    float_t box_lb = ps->lb_box[d];
    float_t box_ub = ps->ub_box[d];
    float_t box_width = box_ub - box_lb;
    float_t global_bin_width = box_width / (float_t)k_global;

    float_t x0 = box_lb + (global_bin_width * (idx));
    float_t x1 = box_lb + (global_bin_width * (idx + 1));

    float_t y0 = (cumulative_count) / total_count;
    float_t y1 = (cumulative_count + global_bin_counts[idx]) / total_count;

    float_t x_guess = (pc - y0) / (y1 - y0) * (x1 - x0) + x0;

        
    /*
    MPI_DB_PRINT("[MASTER] best guess @ %lf is %lf on bin %d on dimension %d --- x0 %lf x1 %lf y0 %lf y1 %lf\n",pc, x_guess,idx, d, x0, x1, y0, y1);
    */

    guess_t g = {.bin_idx = idx, .x_guess = x_guess};
    return g;
}


void global_binning_check(global_context_t *ctx, float_t *data, int d, int k) 
{
    /*
    * sanity check
    * find if global bins are somehow similar to acutal binning on master
    */

    if (I_AM_MASTER) 
    {
        int *counts = (int *)MY_MALLOC(k * sizeof(int));
        for (int bin_idx = 0; bin_idx < k; ++bin_idx) counts[bin_idx] = 0;

        float_t box_lb = ctx->lb_box[d];
        float_t box_ub = ctx->ub_box[d];
        float_t box_width = box_ub - box_lb;
        float_t bin_width = box_width / (float_t)k;

        for (size_t i = 0; i < ctx->n_points; ++i) 
        {
            int bin_idx = (int)((data[i * ctx->dims + d] - box_lb) / bin_width);
            if (bin_idx < k) counts[bin_idx]++;
            // counts[bin_idx]++
        }
        int cc = 0;

        free(counts);
    }
}

// TODO: k_global better to define it as a #define 
void compute_pure_global_binning(global_context_t *ctx, pointset_t *ps,
                                 uint64_t *global_bin_counts, int k_global,
                                 int d) 
{
    /* compute binning of data along dimension d */
    uint64_t local_bin_count[NBINS];
    for (size_t k = 0; k < k_global; ++k) 
    {
        local_bin_count[k] = 0;
        global_bin_counts[k] = 0;
    }

    /*
    MPI_DB_PRINT("[PS BOUNDING BOX %d]: ", ctx -> mpi_rank);
    for(size_t d = 0; d < ps -> dims; ++d) MPI_DB_PRINT("d%d:[%lf, %lf] ",(int)d, ps -> lb_box[d], ps -> ub_box[d]); MPI_DB_PRINT("\n");
    MPI_DB_PRINT("\n");
    */

    float_t bin_w = (ps-> ub_box[d] - ps->lb_box[d]) / (float_t)k_global;

    /* THIS IS problematic when ps -> ub_box == ps -> lb_box */

    // TODO: this should be moved to a thread_private reduction
    #pragma omp parallel for
    for (size_t i = 0; i < ps->n_points; ++i) 
    {
        float_t p = ps->data[i * ps->dims + d];
        /* to prevent the border point in the box to have bin_idx == k_global causing invalid memory access */
        int bin_idx = MIN((int)((p - ps->lb_box[d]) / bin_w), k_global - 1);
        
        #pragma omp atomic update
        local_bin_count[bin_idx]++;
    }

    MPI_Allreduce(local_bin_count, global_bin_counts, k_global, MPI_UNSIGNED_LONG, MPI_SUM, ctx->mpi_communicator);
    //free(local_bin_count);
}


size_t partition_data_around_value(float_t *array, size_t vec_len, size_t compare_dim,
                                   size_t left, size_t right, float_t pivot_value) 
{
    /*
    * returns the number of elements less than the pivot
    */
    size_t store_index = left;
    size_t i;
    /* Move pivot to end */

    // TODO: Does it exist a way to get a parallel partition?
    for (i = left; i < right; ++i) 
    {
        // if(compare_data_element(array + i*vec_len, array + pivot_index*vec_len, compare_dim ) >= 0){
        if (array[i * vec_len + compare_dim] < pivot_value) 
        {
            swap_data_element(array + store_index * vec_len, array + i * vec_len, vec_len);
            store_index += 1;
        }
    }
    /* Move pivot to its final place */
    // swap_data_element(array + (store_index)*vec_len , array + right*vec_len,
    // vec_len);

    return store_index; 
}

size_t parallel_partition_data_around_value(partition_utils_t* p_utils, float_t* in, 
                                            float_t* out, size_t vec_len, size_t compare_dim,
                                            size_t left, size_t right, float_t pivot_value) 
{
    /*
    * returns the number of elements less than the pivot
    */
    size_t store_index = left;
    
    int num_threads = omp_get_num_threads();
    size_t total_lt_count = 0;

    // init to 0 the count and displs
    for(int i=0; i<num_threads;++i)
    {
        p_utils->lt_count[i]=0;
        p_utils->gt_count[i]=0;
        p_utils->gt_displ[i]=0;
        p_utils->lt_displ[i]=0;
    }

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for schedule(static, 128)
        for(size_t i=left; i<right; ++i)
        {
            bool is_less_than_pivot = in[i*vec_len + compare_dim] < pivot_value;
            p_utils->lt_count[thread_id] +=  is_less_than_pivot;
            p_utils->gt_count[thread_id] += !is_less_than_pivot;
        }

        // exclusive prefix sum (starting from 0)
        // displacements should include the offest on the left
        #pragma omp single
        total_lt_count += p_utils->lt_count[0];
        for(int th=1; th<num_threads; ++th)
        {
            total_lt_count += p_utils->lt_count[th];
            p_utils->lt_displ[th] = left + p_utils->lt_count[th-1] + p_utils->lt_displ[th-1]; 
            p_utils->gt_displ[th] = left + p_utils->gt_count[th-1] + p_utils->gt_displ[th-1]; 
        }

        #pragma omp barrier

        idx_t thread_lt_count = 0;
        idx_t thread_gt_count = 0;

        #pragma omp for schedule(static, 128)
        for(size_t i=left; i<right; ++i)
        {
            idx_t target_idx = 0;
            bool is_less_than_pivot = in[i*vec_len + compare_dim] < pivot_value;
            if(is_less_than_pivot) 
            {
                target_idx = p_utils->lt_displ[thread_id] + thread_lt_count;
                thread_lt_count++;
            }
            else 
            {
                target_idx = total_lt_count + p_utils->gt_displ[thread_id] + thread_gt_count;
                thread_gt_count++;
            }
            memcpy(in + i*vec_len, out + target_idx*vec_len, vec_len*sizeof(float_t));
        }
    }
    
    return store_index + total_lt_count; 
}



guess_t refine_pure_binning(global_context_t *ctx, pointset_t *ps,
                            guess_t best_guess, uint64_t *global_bin_count,
                            int k_global, int d, float_t f, float_t tolerance)
{
    /* refine process from obtained binning */
    if (fabs(best_guess.ep - f) < tolerance) 
    {
        /*
        MPI_DB_PRINT("[MASTER] No need to refine, finishing\n");
        */
        return best_guess;
    }
    float_t total_count = 0;
    float_t starting_cumulative = 0;

    for (int i = 0; i < best_guess.bin_idx; ++i) starting_cumulative += global_bin_count[i];
    for (int i = 0; i < k_global; ++i) total_count += global_bin_count[i];

    float_t bin_w = (ps->ub_box[d] - ps->lb_box[d]) / k_global;
    float_t bin_lb = ps->lb_box[d] + (bin_w * (best_guess.bin_idx));
    float_t bin_ub = ps->lb_box[d] + (bin_w * (best_guess.bin_idx + 1));

    uint64_t *tmp_global_bins = (uint64_t *)MY_MALLOC(sizeof(uint64_t) * k_global);
    for (int i = 0; i < k_global; ++i) tmp_global_bins[i] = global_bin_count[i];

    /*
    MPI_DB_PRINT("STARTING REFINE global bins: ");
    for(int i = 0; i < k_global; ++i)
    {
          MPI_DB_PRINT("%lf ", global_bin_count[i]);
    }
    MPI_DB_PRINT("\n");
    */

    guess_t g;
    while (fabs(best_guess.ep - f) > tolerance) {
        /* compute the target */
        float_t ff, b0, b1;
        ff = -1;
        b0 = starting_cumulative;
        b1 = tmp_global_bins[best_guess.bin_idx];
        ff = (f * total_count - b0) / ((float_t)tmp_global_bins[best_guess.bin_idx]);

        /*
         * generate a partset of points in the bin considered
         * each one has to partition its dataset according to the
         * fact that points on dimension d has to be in the bin
         *
         * then make into in place alg for now, copy data in another pointer
         * will be done in place
         * */

        
        /*
        MPI_DB_PRINT("---- ---- ----\n");
        MPI_DB_PRINT("[MASTER] Refining on bin %d lb %lf ub %lf starting c %lf %lf\n", 
                best_guess.bin_idx, bin_lb, bin_ub, starting_cumulative/total_count,
                (tmp_global_bins[best_guess.bin_idx] + starting_cumulative)/total_count);
        */
    

        for (int i = 0; i < k_global; ++i)  tmp_global_bins[i] = 0;

        pointset_t tmp_ps;

        int end_idx   = partition_data_around_value(ps->data, (int)ps->dims, d, 0, (int)ps->n_points, bin_ub);
        int start_idx = partition_data_around_value(ps->data, (int)ps->dims, d, 0,end_idx, bin_lb);

        tmp_ps.n_points = end_idx - start_idx;
        tmp_ps.data = ps->data + start_idx * ps->dims;
        tmp_ps.dims = ps->dims;
        tmp_ps.lb_box = (float_t*)MY_MALLOC(ctx -> dims * sizeof(float_t));
        tmp_ps.ub_box = (float_t*)MY_MALLOC(ctx -> dims * sizeof(float_t));

        compute_bounding_box_pointset(ctx, &tmp_ps);

        /*
        MPI_DB_PRINT("[MASTER] searching for %lf of the bin considered\n",ff);
        */

        // DB_PRINT("%lu\n",tmp_ps.n_points );
        MPI_Barrier(ctx->mpi_communicator);
        compute_pure_global_binning(ctx, &tmp_ps, tmp_global_bins, k_global, d);

        /* sum to global bins */
        // for(int i = 0; i < k_global; ++i) tmp_global_bins[i] +=
        // starting_cumulative;

        best_guess = retrieve_guess_pure(ctx, &tmp_ps, tmp_global_bins, k_global, d, ff);

        best_guess.ep = check_pc_pointset_parallel(ctx, ps, best_guess, d, f);
        // ep = check_pc_pointset_parallel(ctx, &tmp_ps, best_guess, d, f);

        bin_w = (tmp_ps.ub_box[d] - tmp_ps.lb_box[d]) / k_global;
        bin_lb = tmp_ps.lb_box[d] + (bin_w * (best_guess.bin_idx));
        bin_ub = tmp_ps.lb_box[d] + (bin_w * (best_guess.bin_idx + 1));

        for (int i = 0; i < best_guess.bin_idx; ++i) starting_cumulative += tmp_global_bins[i];

        // free(tmp_ps.data);
        free(tmp_ps.lb_box);
        free(tmp_ps.ub_box);
    }

    /*
    MPI_DB_PRINT("SUCCESS!!! \n");
    */

    free(tmp_global_bins);

    return best_guess;
}

void init_queue(partition_queue_t *pq) 
{
    pq->count = 0;
    pq->_capacity = 1000;
    pq->data = (partition_t *)MY_MALLOC(pq->_capacity * sizeof(partition_t));
}

void free_queue(partition_queue_t *pq) { free(pq->data); }

void get_pointset_from_partition(pointset_t *ps, partition_t *part) 
{
    ps->n_points  = part->n_points;
    ps->data      = part->base_ptr;
    ps->n_points  = part->n_points;
}

guess_t compute_median_pure_binning(global_context_t *ctx, pointset_t *ps, float_t fraction, int selected_dim, idx_t nbins, float_t tolerance)
{
    int best_bin_idx;
    float_t ep;


    // uint64_t *global_bin_counts_int = (uint64_t *)MY_MALLOC(n_bins * sizeof(uint64_t));

    uint64_t global_bin_counts_int[NBINS]; 

    compute_pure_global_binning(ctx, ps, global_bin_counts_int, NBINS, selected_dim);
    guess_t g = retrieve_guess_pure(ctx, ps, global_bin_counts_int, NBINS, selected_dim, fraction);
    g.ep = check_pc_pointset_parallel(ctx, ps, g, selected_dim, fraction);
    g = refine_pure_binning(ctx, ps, g, global_bin_counts_int, NBINS, selected_dim, fraction, tolerance);

    //free(global_bin_counts_int);
    return g;
}

guess_t parallel_compute_median_pure_binning(global_context_t *ctx, pointset_t *ps, float_t fraction, int selected_dim, idx_t nbins, float_t tolerance)
{
    int best_bin_idx;
    float_t ep;


    // uint64_t *global_bin_counts_int = (uint64_t *)MY_MALLOC(n_bins * sizeof(uint64_t));

    uint64_t global_bin_counts_int[NBINS]; 

    compute_bounding_box_pointset(ctx, ps);
    compute_pure_global_binning(ctx, ps, global_bin_counts_int, NBINS, selected_dim);
    guess_t g = retrieve_guess_pure(ctx, ps, global_bin_counts_int, NBINS, selected_dim, fraction);
    // check_pc_pointset(ctx, ps, best_guess, d, f);
    g.ep = check_pc_pointset_parallel(ctx, ps, g, selected_dim, fraction);
    g = refine_pure_binning(ctx, ps, g, global_bin_counts_int, NBINS, selected_dim, fraction, tolerance);

    //free(global_bin_counts_int);
    return g;
}

int compute_n_nodes(int n)
{
    if(n == 1) return 1;
    int nl = n/2;
    int nr = n - nl;
    return 1 + compute_n_nodes(nl) + compute_n_nodes(nr);
}

void top_tree_init(global_context_t *ctx, top_kdtree_t *tree) 
{
    /* we want procs leaves */
    int l = (int)(ceil(log2((float_t)ctx -> world_size)));    
    int tree_nodes = (1 << (l + 1)) - 1;
    //int tree_nodes = compute_n_nodes(ctx -> world_size);    
    //MPI_DB_PRINT("Tree nodes %d %d %d %d\n", ctx -> world_size,l, tree_nodes, compute_n_nodes(ctx -> world_size));
    tree->_nodes      = (top_kdtree_node_t*)MY_MALLOC(tree_nodes * sizeof(top_kdtree_node_t));
    for(int i = 0; i < tree_nodes; ++i)
    {
        tree -> _nodes[i].lch = NULL;
        tree -> _nodes[i].rch = NULL;
        tree -> _nodes[i].parent = NULL;
        tree -> _nodes[i].owner = -1;
        tree -> _nodes[i].n_points = 0;
        tree -> _nodes[i].split_dim = -1;
        tree -> _nodes[i].split_val = 0.f;
        tree -> _nodes[i].lb_node_box = NULL;
        tree -> _nodes[i].ub_node_box = NULL;

    }
    tree->_capacity = tree_nodes;
    tree->dims         = ctx->dims;
    tree->count     = 0;
    return;
}

void top_tree_free(global_context_t *ctx, top_kdtree_t *tree) 
{
    for(int i = 0; i < tree -> count; ++i)
    {
        if(tree -> _nodes[i].lb_node_box) free(tree -> _nodes[i].lb_node_box);
        if(tree -> _nodes[i].ub_node_box) free(tree -> _nodes[i].ub_node_box);
    }
    free(tree->_nodes);
    return;
}

top_kdtree_node_t* top_tree_generate_node(global_context_t* ctx, top_kdtree_t* tree)
{
    top_kdtree_node_t* ptr = tree -> _nodes + tree -> count;
    ptr -> lch = NULL;
    ptr -> rch = NULL;
    ptr -> parent = NULL;
    ptr -> lb_node_box = (float_t*)MY_MALLOC(ctx -> dims * sizeof(float_t));
    ptr -> ub_node_box = (float_t*)MY_MALLOC(ctx -> dims * sizeof(float_t));
    ptr -> owner       = -1;
    ptr -> split_dim   = 0;
    ++(tree -> count);
    return ptr;
 
}

void tree_print(global_context_t* ctx, top_kdtree_node_t* root)
{
    MPI_DB_PRINT("Node %p: \n\tsplit_dim %d \n\tsplit_val %lf", root, root -> split_dim, root -> split_val);
    MPI_DB_PRINT("\n\tparent %p", root -> parent);
    MPI_DB_PRINT("\n\towner  %d", root -> owner);
    MPI_DB_PRINT("\n\tbox");
    MPI_DB_PRINT("\n\tlch %p", root -> lch);
    MPI_DB_PRINT("\n\trch %p\n", root -> rch);
    for(size_t d = 0; d < ctx -> dims; ++d) MPI_DB_PRINT("\n\t  d%d:[%lf, %lf]",(int)d, root -> lb_node_box[d], root -> ub_node_box[d]); 
    MPI_DB_PRINT("\n");
    if(root -> lch) tree_print(ctx, root -> lch);
    if(root -> rch) tree_print(ctx, root -> rch);
}
void _recursive_nodes_to_file(global_context_t* ctx, FILE* nodes_file, top_kdtree_node_t* root, int level)
{
    fprintf(nodes_file, "%d,", level);
    fprintf(nodes_file, "%d,", root -> owner);
    fprintf(nodes_file, "%d,", root -> split_dim);
    fprintf(nodes_file, "%lf,", root -> split_val);
    for(int i = 0; i < ctx -> dims; ++i)
    {
        fprintf(nodes_file,"%lf,",root -> lb_node_box[i]);
    }
    for(int i = 0; i < ctx -> dims - 1; ++i)
    {
        fprintf(nodes_file,"%lf,",root -> ub_node_box[i]);
    }
    fprintf(nodes_file,"%lf\n",root -> ub_node_box[ctx -> dims - 1]);
    if(root -> lch) _recursive_nodes_to_file(ctx, nodes_file, root -> lch, level + 1);
    if(root -> rch) _recursive_nodes_to_file(ctx, nodes_file, root -> rch, level + 1);
}
void write_nodes_to_file( global_context_t* ctx,top_kdtree_t* tree, 
                        const char* nodes_path) 
{
    FILE* nodes_file  = fopen(nodes_path,"w");

    if(!nodes_file) 
    {
        printf("Cannot open hp file\n");
        return;
    }
    _recursive_nodes_to_file(ctx, nodes_file, tree -> root, 0);
    fclose(nodes_file);

    
}

void tree_print_leaves(global_context_t* ctx, top_kdtree_node_t* root)
{
    if(root -> owner != -1)
    {
        MPI_DB_PRINT("Node %p: \n\tsplit_dim %d \n\tsplit_val %lf", root, root -> split_dim, root -> split_val);
        MPI_DB_PRINT("\n\tparent %p", root -> parent);
        MPI_DB_PRINT("\n\towner  %d", root -> owner);
        MPI_DB_PRINT("\n\tbox");
        MPI_DB_PRINT("\n\tlch %p", root -> lch);
        MPI_DB_PRINT("\n\trch %p\n", root -> rch);
        for(size_t d = 0; d < ctx -> dims; ++d) MPI_DB_PRINT("\n\t  d%d:[%lf, %lf]",(int)d, root -> lb_node_box[d], root -> ub_node_box[d]); 
        MPI_DB_PRINT("\n");
    }
    if(root -> lch) tree_print_leaves(ctx, root -> lch);
    if(root -> rch) tree_print_leaves(ctx, root -> rch);
}

void parallel_build_top_kdtree(global_context_t *ctx, pointset_t *og_pointset, top_kdtree_t *tree, idx_t n_bins, float_t tolerance) 
{

    // NOTE: For the me of the future
    // pointsests are only "helper structs" partitions 
    size_t tot_n_points = 0;
    MPI_Allreduce(&(og_pointset->n_points), &tot_n_points, 1, MPI_UINT64_T, MPI_SUM, ctx->mpi_communicator);

    /*
    MPI_DB_PRINT("[MASTER] Top tree builder invoked\n");
    */
    MPI_DB_PRINT("\n");
    MPI_DB_PRINT("Building top tree on %lu points with %d processors\n", tot_n_points, ctx->world_size);
    MPI_DB_PRINT("\n");

    size_t current_partition_n_points = tot_n_points;
    size_t expected_points_per_node = tot_n_points / ctx->world_size;

    /* enqueue the two partitions */

    compute_bounding_box_pointset(ctx, og_pointset);

    partition_queue_t queue;
    init_queue(&queue);

    int selected_dim = 0;
    partition_t current_partition = {  .d          = selected_dim,
                                       .base_ptr   = og_pointset->data,
                                       .n_points   = og_pointset->n_points,
                                       .n_procs    = ctx->world_size,
                                       .parent     = NULL,
                                       .lr         = NO_CHILD };

    enqueue_partition(&queue, current_partition);

    // this struct holds the current slice of points
    pointset_t current_pointset;
    //current_pointset.lb_box = (float_t*)MY_MALLOC(ctx -> dims * sizeof(float_t));
    //current_pointset.ub_box = (float_t*)MY_MALLOC(ctx -> dims * sizeof(float_t));

    while (queue.count) 
    {
        /*dequeue the partition to process */
        current_partition = dequeue_partition(&queue);

        /* generate e pointset for that partition */

        get_pointset_from_partition(&current_pointset, &current_partition);
        current_pointset.dims = ctx->dims;


        top_kdtree_node_t* current_node  = top_tree_generate_node(ctx, tree);
        /* insert node */
        
        // MPI_DB_PRINT(   "[RANK %d] Handling partition:\n"\
        //                 "    current_node %p,\n"\
        //                 "    dim %d,\n"\
        //                 "    n_points %lu,\n"\
        //                 "    start_proc %d,\n"\
        //                 "    n_procs %d\n"\
        //                 "    parent %p\n"\
        //                 "    base_ptr %p\n"\
        //                 "    lr %d\n", 
        //         ctx -> mpi_rank,
        //         current_node,
        //         current_partition.d,
        //         current_partition.n_points,
        //         current_partition.start_proc,
        //         current_partition.n_procs,
        //         current_partition.parent,
        //         current_partition.base_ptr,
        //         current_partition.lr);

        /*generate a tree node and point the bounding box to the pointset */
        switch (current_partition.lr) {
            case TOP_TREE_LCH:
                if(current_partition.parent)
                {
                    current_node -> parent        = current_partition.parent;
                    current_node -> parent -> lch = current_node;
                    /* compute the box */
                    /*
                     * left child has lb equal to parent
                     * ub equal to parent except for the dim of splitting 
                     */
                    int parent_split_dim = current_node -> parent -> split_dim;
                    float_t parent_hp    = current_node -> parent -> split_val;

                    memcpy(current_node -> lb_node_box, current_node -> parent -> lb_node_box, ctx -> dims * sizeof(float_t));
                    memcpy(current_node -> ub_node_box, current_node -> parent -> ub_node_box, ctx -> dims * sizeof(float_t));


                    current_node -> ub_node_box[parent_split_dim] = parent_hp;
                }
                break;

            case TOP_TREE_RCH:
                if(current_partition.parent)
                {
                    current_node -> parent        = current_partition.parent;
                    current_node -> parent -> rch = current_node;

                    int parent_split_dim = current_node -> parent -> split_dim;
                    float_t parent_hp    = current_node -> parent -> split_val;

                    /*
                     * right child has ub equal to parent
                     * lb equal to parent except for the dim of splitting 
                     */

                    memcpy(current_node -> lb_node_box, current_node -> parent -> lb_node_box, ctx -> dims * sizeof(float_t));
                    memcpy(current_node -> ub_node_box, current_node -> parent -> ub_node_box, ctx -> dims * sizeof(float_t));

                    current_node -> lb_node_box[parent_split_dim] = parent_hp;
                }
                break;
            case NO_CHILD:
                {
                    tree -> root = current_node;
                    memcpy(current_node -> lb_node_box, og_pointset -> lb_box, ctx -> dims * sizeof(float_t));
                    memcpy(current_node -> ub_node_box, og_pointset -> ub_box, ctx -> dims * sizeof(float_t));
                }
                break;
        }

        current_node -> split_dim = current_partition.d;
        current_node -> parent = current_partition.parent;
        current_node -> lch = NULL;
        current_node -> rch = NULL;

        current_pointset.lb_box = current_node->lb_node_box;
        current_pointset.ub_box = current_node->ub_node_box;

        MPI_Barrier(ctx -> mpi_communicator);
        /* handle partition */
        if(current_partition.n_procs > 1)
        {
            float_t fraction = (current_partition.n_procs / 2) / (float_t)current_partition.n_procs;
            guess_t g = compute_median_pure_binning(ctx, &current_pointset, fraction, current_partition.d, n_bins, tolerance);
            size_t pv = partition_data_around_value(current_pointset.data, ctx->dims, current_partition.d, 0, current_pointset.n_points, g.x_guess);

            current_node -> split_val = g.x_guess;

            size_t points_left = (size_t)pv;
            size_t points_right = current_partition.n_points - points_left;

            int procs_left = current_partition.n_procs * fraction;
            int procs_right = current_partition.n_procs - procs_left;


            // MPI_DB_PRINT("Chosing as guess: %lf, seareching for %lf, obtained %lf\n", g.x_guess, fraction, g.ep);
            // MPI_DB_PRINT("-------------------\n\n");
    


            int next_dimension = (++selected_dim) % (ctx->dims);
            partition_t left_partition = {
                .n_points     = points_left, 
                .n_procs      = procs_left,
                .start_proc   = current_partition.start_proc,
                .parent       = current_node,
                .lr           = TOP_TREE_LCH,
                .base_ptr     = current_pointset.data,
                .d            = next_dimension,
            };

            partition_t right_partition = {
                .n_points     = points_right, 
                .n_procs      = procs_right,
                .start_proc   = current_partition.start_proc + procs_left,
                .parent       = current_node,
                .lr           = TOP_TREE_RCH,
                .base_ptr     = current_pointset.data + pv * current_pointset.dims,
                .d            = next_dimension
            };

            enqueue_partition(&queue, left_partition);
            enqueue_partition(&queue, right_partition);
        }
        else
        {
            current_node -> owner = current_partition.start_proc;
        }
    }
    tree -> root = tree -> _nodes;

    #if defined(WRITE_TOP_NODES)
    MPI_DB_PRINT("Root is %p\n", tree -> root);
        if(I_AM_MASTER)
        {
            tree_print(ctx, tree -> root);
            write_nodes_to_file(ctx, tree, "bb/top_nodes.csv");
        }
    #endif

    
    //free(current_pointset.lb_box);
    //free(current_pointset.ub_box);
    free_queue(&queue);

}

void build_top_kdtree(global_context_t *ctx, pointset_t *og_pointset, top_kdtree_t *tree, idx_t n_bins, float_t tolerance) 
{

    // NOTE: For the me of the future
    // pointsests are only "helper structs" partitions 
    size_t tot_n_points = 0;
    MPI_Allreduce(&(og_pointset->n_points), &tot_n_points, 1, MPI_UINT64_T, MPI_SUM, ctx->mpi_communicator);



    /*
    MPI_DB_PRINT("[MASTER] Top tree builder invoked\n");
    */
    MPI_DB_PRINT("\n");
    MPI_DB_PRINT("Building top tree on %lu points with %d processors\n", tot_n_points, ctx->world_size);
    MPI_DB_PRINT("\n");

    size_t current_partition_n_points = tot_n_points;
    size_t expected_points_per_node = tot_n_points / ctx->world_size;

    /* enqueue the two partitions */

    compute_bounding_box_pointset(ctx, og_pointset);

    partition_queue_t queue;
    init_queue(&queue);

    int selected_dim = 0;
    partition_t current_partition = {  .d          = selected_dim,
                                       .base_ptr   = og_pointset->data,
                                       .n_points   = og_pointset->n_points,
                                       .n_procs    = ctx->world_size,
                                       .parent     = NULL,
                                       .lr         = NO_CHILD };

    enqueue_partition(&queue, current_partition);
    pointset_t current_pointset;

    while (queue.count) 
    {
        /*dequeue the partition to process */
        current_partition = dequeue_partition(&queue);

        /* generate e pointset for that partition */

        get_pointset_from_partition(&current_pointset, &current_partition);
        current_pointset.dims = ctx->dims;

        /*generate a tree node */

        top_kdtree_node_t* current_node  = top_tree_generate_node(ctx, tree);
        /* insert node */
        
        // MPI_DB_PRINT(   "[RANK %d] Handling partition:\n"\
        //                 "    current_node %p,\n"\
        //                 "    dim %d,\n"\
        //                 "    n_points %lu,\n"\
        //                 "    start_proc %d,\n"\
        //                 "    n_procs %d\n"\
        //                 "    parent %p\n"\
        //                 "    base_ptr %p\n"\
        //                 "    lr %d\n", 
        //         ctx -> mpi_rank,
        //         current_node,
        //         current_partition.d,
        //         current_partition.n_points,
        //         current_partition.start_proc,
        //         current_partition.n_procs,
        //         current_partition.parent,
        //         current_partition.base_ptr,
        //         current_partition.lr);

        switch (current_partition.lr) {
            case TOP_TREE_LCH:
                if(current_partition.parent)
                {
                    current_node -> parent        = current_partition.parent;
                    current_node -> parent -> lch = current_node;
                    /* compute the box */
                    /*
                     * left child has lb equal to parent
                     * ub equal to parent except for the dim of splitting 
                     */
                    int parent_split_dim = current_node -> parent -> split_dim;
                    float_t parent_hp    = current_node -> parent -> split_val;

                    memcpy(current_node -> lb_node_box, current_node -> parent -> lb_node_box, ctx -> dims * sizeof(float_t));
                    memcpy(current_node -> ub_node_box, current_node -> parent -> ub_node_box, ctx -> dims * sizeof(float_t));

                    current_node -> ub_node_box[parent_split_dim] = parent_hp;
                }
                break;

            case TOP_TREE_RCH:
                if(current_partition.parent)
                {
                    current_node -> parent        = current_partition.parent;
                    current_node -> parent -> rch = current_node;

                    int parent_split_dim = current_node -> parent -> split_dim;
                    float_t parent_hp    = current_node -> parent -> split_val;

                    /*
                     * right child has ub equal to parent
                     * lb equal to parent except for the dim of splitting 
                     */

                    memcpy(current_node -> lb_node_box, current_node -> parent -> lb_node_box, ctx -> dims * sizeof(float_t));
                    memcpy(current_node -> ub_node_box, current_node -> parent -> ub_node_box, ctx -> dims * sizeof(float_t));

                    current_node -> lb_node_box[parent_split_dim] = parent_hp;
                }
                break;
            case NO_CHILD:
                {
                    tree -> root = current_node;
                    memcpy(current_node -> lb_node_box, og_pointset -> lb_box, ctx -> dims * sizeof(float_t));
                    memcpy(current_node -> ub_node_box, og_pointset -> ub_box, ctx -> dims * sizeof(float_t));
                }
                break;
        }

        current_node -> split_dim = current_partition.d;
        current_node -> parent = current_partition.parent;
        current_node -> lch = NULL;
        current_node -> rch = NULL;

        current_pointset.lb_box = current_node->lb_node_box;
        current_pointset.ub_box = current_node->ub_node_box;

        MPI_Barrier(ctx -> mpi_communicator);
        /* handle partition */
        if(current_partition.n_procs > 1)
        {
            float_t fraction = (current_partition.n_procs / 2) / (float_t)current_partition.n_procs;
            guess_t g = compute_median_pure_binning(ctx, &current_pointset, fraction, current_partition.d, n_bins, tolerance);
            size_t pv = partition_data_around_value(current_pointset.data, ctx->dims, current_partition.d, 0, current_pointset.n_points, g.x_guess);

            current_node -> split_val = g.x_guess;

            size_t points_left = (size_t)pv;
            size_t points_right = current_partition.n_points - points_left;

            int procs_left = current_partition.n_procs * fraction;
            int procs_right = current_partition.n_procs - procs_left;


            // MPI_DB_PRINT("Chosing as guess: %lf, seareching for %lf, obtained %lf\n", g.x_guess, fraction, g.ep);
            // MPI_DB_PRINT("-------------------\n\n");
    


            // use max_strech int next_dimension = (++selected_dim) % (ctx->dims);
            int next_dimension = 0;
            float_t max_strech = 0.;

            for(int dim=0; dim<ctx->dims; ++dim)
            {
                float_t stretch = current_node->ub_node_box[dim] - current_node->lb_node_box[dim];
                if(stretch > max_strech)
                {
                    max_strech = stretch;
                    next_dimension = dim;
                }
            }

            partition_t left_partition = {
                .n_points     = points_left, 
                .n_procs      = procs_left,
                .start_proc   = current_partition.start_proc,
                .parent       = current_node,
                .lr           = TOP_TREE_LCH,
                .base_ptr     = current_pointset.data,
                .d            = next_dimension,
            };

            partition_t right_partition = {
                .n_points     = points_right, 
                .n_procs      = procs_right,
                .start_proc   = current_partition.start_proc + procs_left,
                .parent       = current_node,
                .lr           = TOP_TREE_RCH,
                .base_ptr     = current_pointset.data + pv * current_pointset.dims,
                .d            = next_dimension
            };

            enqueue_partition(&queue, left_partition);
            enqueue_partition(&queue, right_partition);
        }
        else
        {
            current_node -> owner = current_partition.start_proc;
        }
    }
    tree -> root = tree -> _nodes;

    #if defined(WRITE_TOP_NODES)
    MPI_DB_PRINT("Root is %p\n", tree -> root);
        if(I_AM_MASTER)
        {
            tree_print(ctx, tree -> root);
            write_nodes_to_file(ctx, tree, "bb/top_nodes.csv");
        }
    #endif

    
    free_queue(&queue);

}

int compute_point_owner(global_context_t* ctx, top_kdtree_t* tree, float_t* data)
{
    top_kdtree_node_t* current_node = tree -> root;
    int owner = current_node -> owner;
    while(owner == -1)
    {
        /* compute side */
        int split_dim = current_node -> split_dim;
        int side = data[split_dim] > current_node -> split_val;
        switch (side) 
        {
            case TOP_TREE_RCH:
                {
                    current_node = current_node -> rch;                    
                }
                break;

            case TOP_TREE_LCH:
                {
                    current_node = current_node -> lch;                    
                }
                break;
            default:
                break;
        }
        owner = current_node -> owner;
    }
    return owner;
}

/* to partition points around owners */
int partition_data_around_key(int* key, float_t *val, int vec_len, int ref_key , int left, int right) 
{
    /*
    * returns the number of elements less than the pivot
    */
    int store_index = left;
    int i;
    /* Move pivot to end */
    for (i = left; i < right; ++i) 
    {
        // if(compare_data_element(array + i*vec_len, array + pivot_index*vec_len, compare_dim ) >= 0){
        if (key[i] < ref_key) 
        {
            swap_data_element(val + store_index * vec_len, val + i * vec_len, vec_len);
            /* swap keys */
            int tmp = key[i];
            key[i] = key[store_index];
            key[store_index] = tmp;
            
            store_index += 1;
        }
    }

    return store_index; 
}



void exchange_points(global_context_t* ctx, top_kdtree_t* tree)
{
    int* points_per_proc  = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));    
    int* points_owners    = (int*)MY_MALLOC(ctx -> dims * ctx -> local_n_points * sizeof(float_t));
    int* partition_offset = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));    

    /* compute owner */
    #pragma omp parallel for
    for(size_t i = 0; i < ctx -> local_n_points; ++i)
    {
        /* tree walk */
        points_owners[i] = compute_point_owner(ctx, tree, ctx -> local_data + (i * ctx -> dims));
    }
        
    
    int last_idx = 0;
    int len      = ctx -> local_n_points;
    float_t* curr_data = ctx -> local_data;

    partition_offset[0] = 0;
    for(int owner = 1; owner < ctx -> world_size; ++owner)
    {
        last_idx = partition_data_around_key(points_owners, ctx -> local_data, ctx -> dims, owner, last_idx, ctx -> local_n_points);    
        partition_offset[owner] = last_idx;
        points_per_proc[owner - 1] = last_idx;
    }

    points_per_proc[ctx -> world_size - 1] = ctx -> local_n_points;
    
    
    for(int i = ctx -> world_size - 1; i > 0; --i)
    {
        points_per_proc[i] = points_per_proc[i] - points_per_proc[i - 1];
    }

    int* rcv_count   = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));
    int* rcv_displs  = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));
    int* send_displs = (int*)MY_MALLOC(ctx -> world_size * sizeof(int)); 
    int* send_count  = points_per_proc;

    float_t* rcvbuffer = NULL;
    int tot_count = 0;

    MPI_Barrier(ctx -> mpi_communicator);

    MPI_Alltoall(send_count, 1, MPI_INT, rcv_count, 1, MPI_INT, ctx -> mpi_communicator);
    rcv_displs[0] = 0;
    send_displs[0] = 0;
    for(int i = 1; i < ctx -> world_size; ++i) 
    {
        rcv_displs[i] = rcv_displs[i - 1] + rcv_count[i - 1];
        send_displs[i] = send_displs[i - 1] + send_count[i - 1];
    }

    /*multiply for number of elements */
    for(int i = 0; i < ctx -> world_size; ++i) 
    {
        send_displs[i]= send_displs[i] * ctx -> dims;
        send_count[i] = send_count[i] * ctx -> dims;

        rcv_displs[i] = rcv_displs[i] * ctx -> dims;
        rcv_count[i]  = rcv_count[i] * ctx -> dims;
        tot_count += rcv_count[i];
    }

    rcvbuffer = (float_t*)MY_MALLOC(tot_count * sizeof(float_t));

    /*exchange points */


    MPI_Alltoallv(  ctx -> local_data, send_count, send_displs, MPI_MY_FLOAT, 
                    rcvbuffer, rcv_count, rcv_displs, MPI_MY_FLOAT, 
                    ctx -> mpi_communicator);

    ctx -> local_n_points = tot_count / ctx -> dims; 
    idx_t* ppp = (idx_t*)MY_MALLOC(ctx -> world_size * sizeof(idx_t));

    MPI_Allgather(&(ctx -> local_n_points), 1, MPI_UINT64_T, ppp, 1, MPI_UINT64_T, ctx -> mpi_communicator);
    ctx -> idx_start = 0;
    for(int i = 0; i < ctx -> mpi_rank; ++i)
    {
        ctx -> idx_start += ppp[i];
    }

    /* find slices of indices */
    for(int i = 0; i < ctx -> world_size; ++i) ctx -> rank_n_points[i] = ppp[i]; 

    ctx -> rank_idx_start[0] = 0;
    for(int i = 1; i < ctx -> world_size; ++i) ctx -> rank_idx_start[i] = ppp[i - 1] + ctx -> rank_idx_start[i - 1]; 

    /* free prv pointer */
    free(ppp);
    free(ctx -> local_data);
    ctx -> local_data = rcvbuffer;

    /* check exchange */
    
    for(size_t i = 0; i < ctx -> local_n_points; ++i)
    {
        int o = compute_point_owner(ctx, tree, ctx -> local_data + (i * ctx -> dims));
        if(o != ctx -> mpi_rank) DB_PRINT("rank %d got an error\n",ctx -> mpi_rank);
    }

    free(points_owners);
    free(points_per_proc);
    free(partition_offset);
    free(rcv_count);
    free(rcv_displs);
    free(send_displs);
}

static inline size_t local_to_global_idx(global_context_t* ctx, size_t local_idx)
{
    return local_idx + ctx -> idx_start; 
}

void translate_tree_idx_to_global(global_context_t* ctx, kdtree_t* local_tree) 
{
    for(size_t i = 0; i < ctx -> local_n_points; ++i)        
    {
        local_tree -> __points[i].array_idx = local_to_global_idx(ctx, local_tree -> __points[i].array_idx); 
    }
}

void tree_walk(
        global_context_t* ctx, 
        top_kdtree_node_t* root, 
        int point_idx,
        float_t max_dist,
        float_t* point,
        float_t** data_to_send_per_proc, 
        int** local_idx_of_the_point, 
        int* point_to_send_count, 
        int* point_to_send_capacity)
{
    if(root -> owner != -1 && root -> owner != ctx -> mpi_rank)
    {
        
        #pragma omp critical
        {
            /* put the leaf on the requests array */
            int owner = root -> owner;
            int idx = point_to_send_count[owner];
            int capacity = point_to_send_capacity[owner];
            //int len = 1 + ctx -> dims;
            int len = ctx -> dims;
            if(idx == capacity)
            {
                //data_to_send_per_proc[owner]  = realloc(data_to_send_per_proc[owner], (capacity * 1.1) * (1 + ctx -> dims) * sizeof(float_t));
                data_to_send_per_proc[owner]  = realloc(data_to_send_per_proc[owner], (capacity * 1.1) * (ctx -> dims) * sizeof(float_t));
                local_idx_of_the_point[owner] = realloc(local_idx_of_the_point[owner], (capacity * 1.1) * sizeof(int));
                point_to_send_capacity[owner] = capacity * 1.1;
            }

            float_t* base = data_to_send_per_proc[owner] + (len * idx); 
            /*
            base[0] = max_dist;
            memcpy(base + 1, point, ctx -> dims * sizeof(float_t));
            */
            memcpy(base, point, ctx -> dims * sizeof(float_t));
            local_idx_of_the_point[owner][idx] = point_idx;

            point_to_send_count[owner]++;
        }

    }
    else
    {
        /* tree walk */
        int split_var = root -> split_dim;
        float_t hp_distance = point[split_var] - root -> split_val;
        __builtin_prefetch(root -> lch, 0, 3);
        __builtin_prefetch(root -> rch, 0, 3);

        int side = hp_distance > 0.f;

        switch (side)
        {
            case TOP_TREE_LCH:
                if(root -> lch)
                {
                    /* walk on the left */
                    tree_walk(ctx, root -> lch, point_idx, max_dist, point, 
                            data_to_send_per_proc, local_idx_of_the_point, 
                            point_to_send_count, point_to_send_capacity);
                }
                break;
            
            case TOP_TREE_RCH:
                if(root -> rch)
                {
                    /* walk on the right */
                    tree_walk(ctx, root -> rch, point_idx, max_dist, point, 
                            data_to_send_per_proc, local_idx_of_the_point, 
                            point_to_send_count, point_to_send_capacity);
                }
                break;

            default:
                break;
        }

        int c   = max_dist > (hp_distance * hp_distance);

        //if(c || (H -> count) < (H -> N))
        if(c)
        {

            switch (side)
            {
                case HP_LEFT_SIDE:
                    if(root -> rch) 
                    {
                        /* walk on the right */
                        tree_walk(ctx, root -> rch, point_idx, max_dist, point, 
                                data_to_send_per_proc, local_idx_of_the_point, 
                                point_to_send_count, point_to_send_capacity);
                    }
                    break;
                
                case HP_RIGHT_SIDE:
                    if(root -> lch) 
                    {
                        /* walk on the left */
                        tree_walk(ctx, root -> lch, point_idx, max_dist, point, 
                                data_to_send_per_proc, local_idx_of_the_point, 
                                point_to_send_count, point_to_send_capacity);
                    }
                    break;

                default:
                    break;
            }
        }
    }

}

void tree_walk_v2_find_n_points(
        global_context_t* ctx, 
        top_kdtree_node_t* root, 
        int point_idx,
        float_t max_dist,
        float_t* point,
        int* point_to_send_capacity) 
{
    if(root -> owner != -1 && root -> owner != ctx -> mpi_rank)
    {
        #pragma omp atomic update 
        point_to_send_capacity[root -> owner]++;
    }
    else
    {
        /* tree walk */
        int split_var = root -> split_dim;
        float_t hp_distance = point[split_var] - root -> split_val;
        __builtin_prefetch(root -> lch, 0, 3);
        __builtin_prefetch(root -> rch, 0, 3);

        int side = hp_distance > 0.f;

        switch (side)
        {
            case TOP_TREE_LCH:
                if(root -> lch)
                {
                    /* walk on the left */
                    tree_walk_v2_find_n_points(ctx, root -> lch, point_idx, max_dist, point, point_to_send_capacity);
                }
                break;
            
            case TOP_TREE_RCH:
                if(root -> rch)
                {
                    /* walk on the right */
                    tree_walk_v2_find_n_points(ctx, root -> rch, point_idx, max_dist, point, point_to_send_capacity);
                }
                break;

            default:
                break;
        }

        // int c   = max_dist > (hp_distance * hp_distance);
        //
        // //if(c || (H -> count) < (H -> N))
        // if(c)
        // {
        //
        //     switch (side)
        //     {
        //         case HP_LEFT_SIDE:
        //             if(root -> rch) 
        //             {
        //                 /* walk on the right */
        //                 tree_walk_v2_find_n_points(ctx, root -> rch, point_idx, max_dist, point, point_to_send_capacity);
        //             }
        //             break;
        //
        //         case HP_RIGHT_SIDE:
        //             if(root -> lch) 
        //             {
        //                 /* walk on the left */
        //                 tree_walk_v2_find_n_points(ctx, root -> lch, point_idx, max_dist, point, point_to_send_capacity);
        //             }
        //             break;
        //
        //         default:
        //             break;
        //     }
        // }

        //if(c || (H -> count) < (H -> N))
        switch (side)
        {
            case HP_LEFT_SIDE:
                if(root -> rch) 
                {
                    /* walk on the right */
                    float_t point_to_box_dist = box_dist(point, root->rch->lb_node_box, root->rch->ub_node_box, ctx->dims);
                    if(point_to_box_dist < max_dist) tree_walk_v2_find_n_points(ctx, root -> rch, point_idx, max_dist, point, point_to_send_capacity);
                }
                break;

            case HP_RIGHT_SIDE:
                if(root -> lch) 
                {
                    /* walk on the left */
                    float_t point_to_box_dist = box_dist(point, root->lch->lb_node_box, root->lch->ub_node_box, ctx->dims);
                    if(point_to_box_dist < max_dist) tree_walk_v2_find_n_points(ctx, root -> lch, point_idx, max_dist, point, point_to_send_capacity);
                }
                break;

            default:
                break;
        }
    }

}

void tree_walk_v2_append_points(
        global_context_t* ctx, 
        top_kdtree_node_t* root, 
        int point_idx,
        float_t max_dist,
        float_t* point,
        float_t** data_to_send_per_proc, 
        int** local_idx_of_the_point, 
        int* point_to_send_count) 
{
    if(root -> owner != -1 && root -> owner != ctx -> mpi_rank)
    {
        /* put the leaf on the requests array */
        int owner = root -> owner;


        int idx;

        #pragma omp atomic capture
        idx = point_to_send_count[owner]++;

        int len = ctx -> dims;

        float_t* base = data_to_send_per_proc[owner] + (len * idx); 

        memcpy(base, point, ctx -> dims * sizeof(float_t));
        local_idx_of_the_point[owner][idx] = point_idx;
    }
    else
    {
        /* tree walk */
        int split_var = root -> split_dim;
        float_t hp_distance = point[split_var] - root -> split_val;
        __builtin_prefetch(root -> lch, 0, 3);
        __builtin_prefetch(root -> rch, 0, 3);

        int side = hp_distance > 0.f;

        switch (side)
        {
            case TOP_TREE_LCH:
                if(root -> lch)
                {
                    /* walk on the left */
                    tree_walk_v2_append_points(ctx, root -> lch, point_idx, max_dist, point, 
                            data_to_send_per_proc, local_idx_of_the_point, point_to_send_count);
                }
                break;
            
            case TOP_TREE_RCH:
                if(root -> rch)
                {
                    /* walk on the right */
                    tree_walk_v2_append_points(ctx, root -> rch, point_idx, max_dist, point, 
                            data_to_send_per_proc, local_idx_of_the_point, point_to_send_count);
                }
                break;

            default:
                break;
        }

        // int c   = max_dist > (hp_distance * hp_distance);
        // if(c)
        // {
        //
        //     switch (side)
        //     {
        //         case HP_LEFT_SIDE:
        //             if(root -> rch) 
        //             {
        //                 /* walk on the right */
        //                 tree_walk_v2_append_points(ctx, root -> rch, point_idx, max_dist, point, 
        //                         data_to_send_per_proc, local_idx_of_the_point, point_to_send_count);
        //             }
        //             break;
        //
        //         case HP_RIGHT_SIDE:
        //             if(root -> lch) 
        //             {
        //                 /* walk on the left */
        //                 tree_walk_v2_append_points(ctx, root -> lch, point_idx, max_dist, point, 
        //                         data_to_send_per_proc, local_idx_of_the_point, point_to_send_count);
        //             }
        //             break;
        //
        //         default:
        //             break;
        //     }
        // }

        switch (side)
        {
            case HP_LEFT_SIDE:
                if(root -> rch) 
                {
                    /* walk on the right */
                    float_t point_to_box_dist = box_dist(point, root->rch->lb_node_box, root->rch->ub_node_box, ctx->dims);
                    if(point_to_box_dist < max_dist) tree_walk_v2_append_points(ctx, root -> rch, point_idx, max_dist, point, 
                                                        data_to_send_per_proc, local_idx_of_the_point, point_to_send_count);
                }
                break;

            case HP_RIGHT_SIDE:
                if(root -> lch) 
                {
                    /* walk on the left */
                    float_t point_to_box_dist = box_dist(point, root->lch->lb_node_box, root->lch->ub_node_box, ctx->dims);
                    if(point_to_box_dist < max_dist) tree_walk_v2_append_points(ctx, root -> lch, point_idx, max_dist, point, 
                                                        data_to_send_per_proc, local_idx_of_the_point, point_to_send_count);
                }
                break;

            default:
                break;
        }
    }

}


void convert_heap_idx_to_global(global_context_t* ctx, heap_t* H)
{
    for(uint64_t i = 0; i < H -> count; ++i)
    {
        if(H -> data[i].array_idx != MY_SIZE_MAX) H -> data[i].array_idx = local_to_global_idx(ctx, H -> data[i].array_idx);
    }
}

void print_diagnositcs(global_context_t* ctx, int k)
{
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shmcomm);
    int shm_world_size;
    MPI_Comm_size(shmcomm, &shm_world_size);
    MPI_DB_PRINT("\n");
    MPI_DB_PRINT("[INFO] Got %d ranks per node \n",shm_world_size); 
    /* data */
    float_t memory_use = (float_t)ctx -> local_n_points * ctx -> dims * sizeof(float_t);
    memory_use += (float_t)sizeof(datapoint_info_t)* (float_t)(ctx -> local_n_points); 
    /* ngbh */
    memory_use += (float_t)sizeof(heap_node_t)*(float_t)k * (float_t)(ctx -> local_n_points); 
    memory_use = memory_use / 1e9 * shm_world_size;

    MPI_DB_PRINT("       Got ~%d points per node and %d ngbh per points\n", ctx -> local_n_points * shm_world_size, k); 
    MPI_DB_PRINT("       Expected to use ~%.2lfGB of memory for each node, plus memory required to communicate ngbh\n", memory_use); 
    struct sysinfo info;
    sysinfo(&info);
    
    if(memory_use > 0.5 * (float_t)info.freeram / 1e9)
    MPI_DB_PRINT("/!\\    Projected memory usage is more than half of the node memory, may go into troubles while communicating ngbh\n"); 
    MPI_DB_PRINT("\n");

    MPI_Barrier(ctx -> mpi_communicator);
}


void mpi_ngbh_search(global_context_t* ctx, datapoint_info_t* dp_info, top_kdtree_t* top_tree, kdtree_t* local_tree, float_t* data, int k)
{
    /* local search */
    /* print diagnostics */
    print_diagnositcs(ctx, k);
    ctx -> k = (idx_t)k;
    
    TIME_DEF;
    double elapsed_time;

    TIME_START;
    MPI_Barrier(ctx -> mpi_communicator);
    //ctx -> __local_heap_buffers = (heap_node_t*)MY_MALLOC(ctx -> local_n_points * k * sizeof(heap_node));
    MPI_Alloc_mem(ctx -> local_n_points * k * sizeof(heap_node_t), MPI_INFO_NULL, &(ctx -> __local_heap_buffers));

    #pragma omp parallel for schedule(dynamic, 32)
    for(int p = 0; p < ctx -> local_n_points; ++p)
    {
        idx_t idx = local_tree -> __points[p].array_idx;
        /* actually we want to preserve the heap_t to then insert guesses from other nodes */
        heap_t tmp_heap;
        tmp_heap.data  = ctx -> __local_heap_buffers + k * idx;
        tmp_heap.count = 0;
        tmp_heap.size  = k;

        //dp_info[idx].ngbh = knn_kdtree_t_no_heapsort(local_tree -> _nodes[p].data, local_tree -> root, k);

        idx_t visited_nodes = 0;
        knn_sub_tree_search(local_tree->__points[p].data, local_tree-> __points, local_tree->__pivots,
                            local_tree->root, &tmp_heap, local_tree->dims, &visited_nodes);

        convert_heap_idx_to_global(ctx, &tmp_heap);
        dp_info[idx].cluster_idx = -1;
        dp_info[idx].is_center = 0;
        dp_info[idx].array_idx = idx + ctx -> idx_start;
        dp_info[idx].ngbh = tmp_heap.data;
    }
    elapsed_time = TIME_STOP;
    LOG_WRITE("Local neighborhood search", elapsed_time);
    printf("rank %d elapsed_time %lf\n", ctx->mpi_rank, elapsed_time);


    TIME_START;
    /* find if a points needs a refine on the global tree */
    float_t** data_to_send_per_proc    = (float_t**)MY_MALLOC(ctx -> world_size * sizeof(float_t*));
    int**     local_idx_of_the_point   = (int**)MY_MALLOC(ctx -> world_size * sizeof(int*));
    int*      point_to_snd_count       = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));
    int*      point_to_snd_capacity    = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));

    for(int i = 0; i < ctx -> world_size; ++i)
    {
        /* allocate it afterwards */
        point_to_snd_capacity[i] = 0;
        point_to_snd_count[i]    = 0;
    }

    /* NEW VERSION double tree walk */
    #pragma omp parallel for schedule(dynamic, 128)
    for(int i = 0; i < ctx -> local_n_points; ++i)
    {
        float_t max_dist = dp_info[i].ngbh[0].value;
        float_t* point   = ctx -> local_data + (i * ctx -> dims);
        
        tree_walk_v2_find_n_points(ctx, top_tree -> root, i, max_dist, point, point_to_snd_capacity);

    }

    /* allocate needed space */
    for(int i = 0; i < ctx -> world_size; ++i)
    {
        int np = point_to_snd_capacity[i];
        data_to_send_per_proc[i]  = (float_t*)MY_MALLOC(np * (ctx -> dims) * sizeof(float_t));    
        local_idx_of_the_point[i] = (int*)MY_MALLOC(np * sizeof(int));    

    }

    #pragma omp parallel for schedule(dynamic, 128)
    for(int i = 0; i < ctx -> local_n_points; ++i)
    {
        float_t max_dist = dp_info[i].ngbh[0].value;
        float_t* point   = ctx -> local_data + (i * ctx -> dims);

        tree_walk_v2_append_points(ctx, top_tree -> root, i, max_dist, point, data_to_send_per_proc, local_idx_of_the_point, point_to_snd_count);
    }


    elapsed_time = TIME_STOP;
    LOG_WRITE("Finding points to refine", elapsed_time);

    TIME_START;
    int* point_to_rcv_count = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));

    /* exchange points to work on*/
    MPI_Alltoall(point_to_snd_count, 1, MPI_INT, point_to_rcv_count, 1, MPI_INT, ctx -> mpi_communicator);

    int* rcv_count = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));
    int* snd_count = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));
    int* rcv_displ = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));
    int* snd_displ = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));

    /*compute counts and displs*/
    rcv_displ[0] = 0;
    snd_displ[0] = 0;


    rcv_count[0] = point_to_rcv_count[0] * (ctx -> dims);
    snd_count[0] = point_to_snd_count[0] * (ctx -> dims);

    int tot_points_rcv = point_to_rcv_count[0];
    int tot_points_snd = point_to_snd_count[0];
    int tot_count = rcv_count[0];

    for(int i = 1; i < ctx -> world_size; ++i)
    {
        rcv_count[i] = point_to_rcv_count[i] * (ctx -> dims);        
        snd_count[i] = point_to_snd_count[i] * (ctx -> dims);        

        tot_count += rcv_count[i];
        tot_points_rcv += point_to_rcv_count[i];
        tot_points_snd += point_to_snd_count[i];

        rcv_displ[i] = rcv_displ[i - 1] + rcv_count[i - 1];
        snd_displ[i] = snd_displ[i - 1] + snd_count[i - 1];
    }

    float_t* __rcv_points = (float_t*)MY_MALLOC(tot_points_rcv * (ctx -> dims) * sizeof(float_t));
    float_t* __snd_points = (float_t*)MY_MALLOC(tot_points_snd * (ctx -> dims) * sizeof(float_t)); 

    float_t* __max_dist_snd = (float_t*)MY_MALLOC(tot_points_snd * sizeof(float_t));
    float_t* __max_dist_rcv = (float_t*)MY_MALLOC(tot_points_rcv * sizeof(float_t));



    /* copy data to send in contiguous memory */
    for(int i = 0; i < ctx -> world_size; ++i)
    {
        memcpy(__snd_points + snd_displ[i], data_to_send_per_proc[i], snd_count[i] * sizeof(float_t));
    }

    size_t dist_idx = 0;
    for(idx_t i = 0; i < ctx -> world_size; ++i)
    {
        for(idx_t j = 0; j < point_to_snd_count[i]; ++j)
        {
            idx_t point_idx = local_idx_of_the_point[i][j];
            // for each point take the distance of its furthest nearest neighbor
            // then communicate it in order to prune the search on foreign nodes
            __max_dist_snd[dist_idx] = ctx -> local_datapoints[point_idx].ngbh[0].value; 
            ++dist_idx;
        }
    }



    MPI_Alltoallv(__snd_points, snd_count, snd_displ, MPI_MY_FLOAT, 
                  __rcv_points, rcv_count, rcv_displ, MPI_MY_FLOAT, ctx -> mpi_communicator); 

    float_t** rcv_work_batches = (float_t**)MY_MALLOC(ctx -> world_size * sizeof(float_t*));
    for(int i = 0; i < ctx -> world_size; ++i) 
    {
        rcv_work_batches[i]       = __rcv_points + rcv_displ[i];
    }

    MPI_Status status;
    int flag;

    for(int i = 0; i < ctx -> world_size; ++i)
    {
        rcv_count[i] = rcv_count[i] / ctx -> dims;
        rcv_displ[i] = rcv_displ[i] / ctx -> dims;
        snd_count[i] = snd_count[i] / ctx -> dims;
        snd_displ[i] = snd_displ[i] / ctx -> dims;

    }

    MPI_Alltoallv(__max_dist_snd, snd_count, snd_displ, MPI_MY_FLOAT, 
                  __max_dist_rcv, rcv_count, rcv_displ, MPI_MY_FLOAT, ctx -> mpi_communicator); 


    // THIS CAN BE SUBSTITUTED BY DIVIDING BY ctx -> dims
    // rcv_displ[0] = 0;
    // snd_displ[0] = 0;
    // rcv_count[0] = point_to_rcv_count[0];
    // snd_count[0] = point_to_snd_count[0]; 
    //
    //
    // for(int i = 1; i < ctx -> world_size; ++i)
    // {
    //
    //     rcv_count[i] = point_to_rcv_count[i]; 
    //     snd_count[i] = point_to_snd_count[i]; 
    //
    //     rcv_displ[i] = rcv_displ[i - 1] + rcv_count[i - 1];
    //     snd_displ[i] = snd_displ[i - 1] + snd_count[i - 1];
    // }
    //

    /* prepare heap_t batches */

    //int work_batch_stride = 1 + ctx -> dims;
    int work_batch_stride = ctx -> dims;

    /* Note that I then have to recieve an equal number of heaps as the one I sent out before */
    heap_node_t* __heap_batches_to_snd = (heap_node_t*)MY_MALLOC((uint64_t)k * (uint64_t)tot_points_rcv * sizeof(heap_node_t));

    if( __heap_batches_to_snd == NULL)
    {
        DB_PRINT("Rank %d failed to allocate snd_heaps %luB required\n",ctx -> mpi_rank, (uint64_t)k * (uint64_t)tot_points_snd * sizeof(heap_node_t));
        exit(1);
    }

    MPI_Barrier(ctx -> mpi_communicator);


    // max dists will contain the maximum distance to have to search
    heap_node_t** heap_batches_per_node = (heap_node_t**)MY_MALLOC(ctx -> world_size * sizeof(heap_node_t*));
    float_t**   max_dists_per_node    = (float_t**)MY_MALLOC(ctx -> world_size * sizeof(float_t*));

    for(int p = 0; p < ctx -> world_size; ++p) 
    {
        heap_batches_per_node[p] = __heap_batches_to_snd + (uint64_t)rcv_displ[p] * (uint64_t)k;
        max_dists_per_node[p]    = __max_dist_rcv        + (uint64_t)rcv_displ[p];
    }

    /* compute everything */
    elapsed_time = TIME_STOP;
    LOG_WRITE("Exchanging points", elapsed_time);
    MPI_Barrier(ctx -> mpi_communicator);


    TIME_START;

    /* ngbh search on recieved points */
    for(int p = 0; p < ctx -> world_size; ++p)
    {
        if(point_to_rcv_count[p] > 0 && p != ctx -> mpi_rank)
        {
            #pragma omp parallel for schedule(dynamic, 128)
            for(int batch = 0; batch < point_to_rcv_count[p]; ++batch)
            {
                heap_t H;
                H.count = k;
                H.size = k;
                H.data = heap_batches_per_node[p] + (uint64_t)k * (uint64_t)batch; 
                float_t max_dist = max_dists_per_node[p][batch];
                for(idx_t i = 0; i < H.size; ++i)
                {
                    H.data[i].value = max_dist;
                    H.data[i].array_idx = MY_SIZE_MAX; 
                }

                //float_t* point = rcv_work_batches[p] + batch * work_batch_stride + 1; 
                idx_t visited_nodes = 0;
                float_t* point = rcv_work_batches[p] + (uint64_t)batch * (uint64_t)work_batch_stride; 
                knn_sub_tree_search(local_tree->__points[p].data, local_tree-> __points, local_tree->__pivots,
                                    local_tree->root, &H, local_tree->dims, &visited_nodes);
                convert_heap_idx_to_global(ctx, &H);
            }
        }
    }
    
    /* sendout results */

    /* 
     * dummy pointers to clarify counts in this part
     * act like an alias for rcv and snd counts
     */ 

    int* ngbh_to_send = point_to_rcv_count;
    int* ngbh_to_recv = point_to_snd_count;

    /*
     * counts are inverted since I have to recieve as many batches as points I
     * Have originally sended
     */

    elapsed_time = TIME_STOP;
    LOG_WRITE("Ngbh search for foreing points", elapsed_time);

    TIME_START;
    
    MPI_Datatype MPI_my_heap;
    MPI_Type_contiguous(k * sizeof(heap_node_t), MPI_BYTE, &MPI_my_heap);
    MPI_Barrier(ctx -> mpi_communicator);
    MPI_Type_commit(&MPI_my_heap);


    /* -------------------------------------
     * ALTERNATIVE TO ALL TO ALL FOR BIG MSG
     * ------------------------------------- */
    
    MPI_Barrier(ctx -> mpi_communicator);
    int default_msg_len = MAX_MSG_SIZE / (k * sizeof(heap_node_t));

    int* already_sent_points = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));
    int* already_rcvd_points = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));

    /* allocate a request array to keep track of all requests going out*/
    MPI_Request* req_array;
    int req_num = 0;
    for(int i = 0; i < ctx -> world_size; ++i)
    {
        req_num += ngbh_to_send[i] > 0 ? ngbh_to_send[i]/default_msg_len + 1 : 0;
    }

    req_array = (MPI_Request*)MY_MALLOC(req_num * sizeof(MPI_Request));


    for(int i = 0; i < ctx -> world_size; ++i)
    {
        already_sent_points[i] = 0;
        already_rcvd_points[i] = 0;
    }

    int req_idx = 0;

    // find the maximum number of points to send */
    
    idx_t max_n_recv = 0;
    for(int i = 0; i < ctx -> world_size; ++i)
    {
        max_n_recv = MAX(max_n_recv, (idx_t)ngbh_to_recv[i]);
    }

    MPI_DB_PRINT("Using default message length %lu\n", default_msg_len);

    heap_node_t* __heap_batches_to_rcv = (heap_node_t*)MY_MALLOC((uint64_t)k * (uint64_t)max_n_recv * sizeof(heap_node_t));
    if( __heap_batches_to_rcv == NULL)
    {
        DB_PRINT("Rank %d failed to allocate rcv_heaps %luB required\n",ctx -> mpi_rank, (uint64_t)k * (uint64_t)max_n_recv* sizeof(heap_node_t));
        exit(1);
    }

    /* make a ring */

    MPI_Barrier(ctx -> mpi_communicator);
    for(int i = 1; i < ctx -> world_size; ++i)
    {
        int rank_to_send = (ctx -> mpi_rank + i) % (ctx -> world_size);
        int rank_to_recv = (ctx -> world_size + ctx -> mpi_rank - i) % (ctx -> world_size);

        /* do things */

        /* send out one batch */

        #ifdef PRINT_NGBH_EXCHANGE_SCHEME
            MPI_DB_PRINT("[--- ROUND %d ----]\n", i);
            MPI_Barrier(ctx -> mpi_communicator);
            DB_PRINT("[RANK %d] sending to %d tot: %d [%luB]---- recieving from %d %d\n", ctx -> mpi_rank, 
                    rank_to_send, ngbh_to_send[rank_to_send], ngbh_to_send[rank_to_send]*sizeof(heap_node_t), rank_to_recv, ngbh_to_recv[rank_to_recv]);
        #endif
        if(ngbh_to_send[rank_to_send] > 0)
        {
            int count_send = 0;
            while(already_sent_points[rank_to_send] < ngbh_to_send[rank_to_send])
            {
                MPI_Request request;
                count_send = MIN((int)default_msg_len, (int)(ngbh_to_send[rank_to_send] - already_sent_points[rank_to_send] ));

                MPI_Isend(  heap_batches_per_node[rank_to_send] + k * already_sent_points[rank_to_send], count_send,  
                        MPI_my_heap, rank_to_send, 0, ctx -> mpi_communicator, &request);

                already_sent_points[rank_to_send] += count_send;
                req_array[req_idx] = request;
                ++req_idx;
            }
        }

        if(     ngbh_to_send[rank_to_send] != already_sent_points[rank_to_send] || 
                point_to_rcv_count[rank_to_send] != already_sent_points[rank_to_send])
        {
            DB_PRINT("ERROR OCCURRED in sending points [rank %d] got %d expected %d\n", 
                    ctx -> mpi_rank, already_rcvd_points[rank_to_send], point_to_rcv_count[rank_to_send]);
        }
        
        MPI_Barrier(ctx -> mpi_communicator);

        if(ngbh_to_recv[rank_to_recv] > 0)
        {
            flag = 0;
            while(already_rcvd_points[rank_to_recv] < ngbh_to_recv[rank_to_recv])
            {
                MPI_Probe(rank_to_recv, MPI_ANY_TAG, ctx -> mpi_communicator, &status);
                MPI_Request request;
                int count_recv; 
                int source = status.MPI_SOURCE;
                MPI_Get_count(&status, MPI_my_heap, &count_recv);
                /* recieve each slice */

                MPI_Recv(__heap_batches_to_rcv + k * already_rcvd_points[rank_to_recv], 
                        count_recv, MPI_my_heap, source, MPI_ANY_TAG, ctx -> mpi_communicator, &status);

                already_rcvd_points[rank_to_recv] += count_recv;
            }
        }

        if(     ngbh_to_recv[rank_to_recv] != already_rcvd_points[rank_to_recv] || 
                point_to_snd_count[rank_to_recv] != already_rcvd_points[rank_to_recv])
        {
            DB_PRINT("ERROR OCCURRED in recieving points [rank %d] got %d expected %d\n", 
                    ctx -> mpi_rank, already_rcvd_points[rank_to_recv], point_to_snd_count[rank_to_recv]);
        }

        /* merge lists */
        #pragma omp parallel for schedule(dynamic, 32)
        for(int b = 0; b < ngbh_to_recv[rank_to_recv]; ++b)
        {
            int idx = local_idx_of_the_point[rank_to_recv][b];
            /* retrieve the heap_t */
            heap_t H;
            H.count = k;
            H.size  = k;
            H.data  = __heap_batches_to_rcv + k*b;
            /* insert the points into the heap_t */
            for(int j = 0; j < k; ++j)
            {
                heap_t tmp_heap;
                tmp_heap.size  = k;
                tmp_heap.count = k;
                tmp_heap.data  = dp_info[idx].ngbh;
                // insert it only of != from max
                if(H.data[j].array_idx != MY_SIZE_MAX) max_heap_insert(&(tmp_heap), H.data[j].value, H.data[j].array_idx);
            }
        }



        MPI_Barrier(ctx -> mpi_communicator);
    }

    elapsed_time = TIME_STOP;
    LOG_WRITE("Merging results", elapsed_time);

    
    MPI_Testall(req_idx, req_array, &flag, MPI_STATUSES_IGNORE);

    if(flag == 0)
    {
        DB_PRINT("ERROR OCCURRED Rank %d has unfinished communications\n", ctx -> mpi_rank);
        exit(1);
    }
    free(req_array);
    free(already_sent_points);
    free(already_rcvd_points);

    /* -------------------------------------------------------- */
    /* heapsort them */

    TIME_START;

    #pragma omp parallel for schedule(dynamic, 128)
    for(int i = 0; i < ctx -> local_n_points; ++i)
    {
        heap_t tmp_heap;
        tmp_heap.size   = k;
        tmp_heap.count = k;
        tmp_heap.data  = dp_info[i].ngbh;

        heap_sort(&(tmp_heap));
    }

    elapsed_time = TIME_STOP;
    LOG_WRITE("Sorting negihborhoods", elapsed_time);


    #if defined(WRITE_NGBH)
    MPI_DB_PRINT("Writing ngbh to files\n");
        char ngbh_out[80];
        sprintf(ngbh_out, "./bb/rank_%d.ngbh",ctx -> mpi_rank);
        FILE* file = fopen(ngbh_out,"w");
        if(!file) 
        {
            printf("Cannot open file %s\n",ngbh_out);
        }
        else
        {
            for(int i = 0; i < ctx -> local_n_points; ++i)
            {
                fwrite(dp_info[i].ngbh, sizeof(heap_node_t), k, file);
            }
            fclose(file);
        }
    #endif

    MPI_Barrier(ctx -> mpi_communicator);

    for(int i = 0; i < ctx -> world_size; ++i)
    {
        if(data_to_send_per_proc[i])  free(data_to_send_per_proc[i]);
        if(local_idx_of_the_point[i]) free(local_idx_of_the_point[i]);
    }


    free(data_to_send_per_proc);
    free(local_idx_of_the_point);
    free(heap_batches_per_node);
    free(max_dists_per_node);
    //free(rcv_heap_batches);
    free(rcv_work_batches);
    free(point_to_rcv_count);
    free(point_to_snd_count);
    free(point_to_snd_capacity);

    free(rcv_count);
    free(snd_count);
    free(rcv_displ);
    free(snd_displ);
    free(__heap_batches_to_rcv);
    free(__heap_batches_to_snd);
    free(__rcv_points);
    free(__snd_points);
    free(__max_dist_snd);
    free(__max_dist_rcv);

}

void build_local_tree(global_context_t* ctx, kdtree_t* local_tree)
{
    build_tree_kdtree(local_tree);
}



int foreign_owner(global_context_t* ctx, idx_t idx)
{
    int owner = ctx -> mpi_rank;
    if( idx >= ctx -> idx_start && idx < ctx -> idx_start + ctx -> local_n_points) 
    {
        return ctx -> mpi_rank;
    }

    for(int i = 0; i < ctx -> world_size; ++i)
    {
        owner = i;    
        if( idx >= ctx -> rank_idx_start[i] && idx < ctx -> rank_idx_start[i] + ctx -> rank_n_points[i]) break;
    }
    return owner;
}











