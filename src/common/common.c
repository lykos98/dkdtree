#include "common.h"
#include "mpi.h"
#include <stdlib.h>
#include <time.h>

#define ARRAY_INCREMENT 100

#define FREE_NOT_NULL(x) if(x){free(x); x = NULL;}
#define PATH_LEN 500

void get_context(global_context_t* ctx)
{
	MPI_Comm_size(ctx -> mpi_communicator, &(ctx -> world_size));
	MPI_Get_processor_name(ctx -> processor_mame, &(ctx -> __processor_name_len));
	MPI_Comm_rank(ctx -> mpi_communicator, &(ctx -> mpi_rank));
	ctx -> local_data = NULL;
	ctx -> lb_box     = NULL;
	ctx -> ub_box     = NULL;
    ctx -> og_idxs    = NULL;
    ctx -> rank_n_points  = (idx_t*)malloc(ctx -> world_size * sizeof(idx_t));
    ctx -> rank_idx_start = (idx_t*)malloc(ctx -> world_size * sizeof(idx_t));
    ctx -> local_datapoints = NULL;
    ctx -> __local_heap_buffers = NULL;
    ctx -> input_data_in_float32 = -1;
    ctx -> dims = 0;
    ctx -> k = 300;
    ctx -> z = 3;
}

void get_dataset_diagnostics(global_context_t* ctx, float_t* data)
{
    //print mean and variance per column of the dataset
    float_t* mean = (float_t*)MY_MALLOC(ctx -> dims * sizeof(float_t));
    float_t* var  = (float_t*)MY_MALLOC(ctx -> dims * sizeof(float_t));

    for(int i = 0; i < ctx -> dims; ++i)
    {
        mean[i] = 0.;
        var[i]  = 0.;
    }

    int jmax = ctx -> dims - (ctx -> dims % 4);
    #pragma omp parallel
    {
        float_t* pvt_mean = (float_t*)MY_MALLOC(ctx -> dims * sizeof(float_t));
        for(int j = 0; j < ctx -> dims; ++j) pvt_mean[j] = 0.;

        #pragma omp for
        for(idx_t i = 0; i < ctx -> n_points; ++i)
        {
            int j = 0;
            for(j = 0; j < jmax; j+=4)
            {
                pvt_mean[j    ] += data[i * ctx -> dims + j    ];
                pvt_mean[j + 1] += data[i * ctx -> dims + j + 1];
                pvt_mean[j + 2] += data[i * ctx -> dims + j + 2];
                pvt_mean[j + 3] += data[i * ctx -> dims + j + 3];
            }

            for(j = jmax; j < ctx -> dims; ++j)
            {
                pvt_mean[j] += data[i * ctx -> dims + j];
            }
        }

        for(int j = 0; j < ctx -> dims; ++j)
        {
            #pragma omp atomic update
            mean[j] += pvt_mean[j];
        }

        free(pvt_mean);
    }

    for(int i = 0; i < ctx -> dims; ++i)
    {
        mean[i] = mean[i] / (float_t) ctx -> n_points;
    }

    #pragma omp parallel
    {
        float_t* pvt_var = (float_t*)MY_MALLOC(ctx -> dims * sizeof(float_t));
        for(int j = 0; j < ctx -> dims; ++j) pvt_var[j] = 0.;

        #pragma omp for
        for(idx_t i = 0; i < ctx -> n_points; ++i)
        {
            int j = 0;
            for(j = 0; j < jmax; j+=4)
            {
                float_t v0 = mean[j    ] - data[i * ctx -> dims + j    ];
                float_t v1 = mean[j + 1] - data[i * ctx -> dims + j + 1];
                float_t v2 = mean[j + 2] - data[i * ctx -> dims + j + 2];
                float_t v3 = mean[j + 3] - data[i * ctx -> dims + j + 3];

                pvt_var[j    ] += v0 * v0;
                pvt_var[j + 1] += v1 * v1;
                pvt_var[j + 2] += v2 * v2;
                pvt_var[j + 3] += v3 * v3;
            }

            for(j = jmax; j < ctx -> dims; ++j)
            {
                float_t v = mean[j] - data[i * ctx -> dims + j];
                pvt_var[j] += v * v;
            }
        }

        for(int j = 0; j < ctx -> dims; ++j)
        {
            #pragma omp atomic update
            var[j] += pvt_var[j];
        }

        free(pvt_var);
    }

    for(int i = 0; i < ctx -> dims; ++i)
    {
        var[i]  = var[i]  / ((float_t) ctx -> n_points - 1);
    }

    for(int j = 0; j < ctx -> dims; ++j)
    {
        printf("dim%2d   mean %.2e std %.2e\n", j, mean[j], sqrt(var[j]));
    }
    free(mean);
    free(var);
}

void print_error_code(int err)
{
    switch (err) 
    {
        case MPI_SUCCESS:
            DB_PRINT("MPI_SUCCESS\n");
            break;
        case MPI_ERR_ARG:
            DB_PRINT("MPI_ERR_ARG\n");
            break;
        case MPI_ERR_COMM:
            DB_PRINT("MPI_ERR_COMM\n");
            break;
        case MPI_ERR_DISP:
            DB_PRINT("MPI_ERR_DISP\n");
            break;
        case MPI_ERR_INFO:
            DB_PRINT("MPI_ERR_INFO\n");
            break;
        case MPI_ERR_SIZE:
            DB_PRINT("MPI_ERR_SIZE\n");
            break;
        case MPI_ERR_OTHER:
            DB_PRINT("MPI_ERR_OTHER\n");
            break;
        default:
            break;
    
    }
}

void free_context(global_context_t* ctx)
{

    FREE_NOT_NULL(ctx -> local_data);
    FREE_NOT_NULL(ctx -> ub_box);
    FREE_NOT_NULL(ctx -> lb_box);
    FREE_NOT_NULL(ctx -> og_idxs);
    //FREE_NOT_NULL(ctx -> __local_heap_buffers);
    if(ctx -> __local_heap_buffers) MPI_Free_mem(ctx -> __local_heap_buffers);


    //if(ctx -> local_datapoints)
    //{
    //    for(int i = 0; i < ctx -> local_n_points; ++i) FREE_NOT_NULL(ctx -> local_datapoints[i].ngbh.data);
    //}

    FREE_NOT_NULL(ctx -> local_datapoints);
    FREE_NOT_NULL(ctx -> rank_n_points);
    FREE_NOT_NULL(ctx -> rank_idx_start);
}

void free_pointset(pointset_t* ps)
{
	if(ps -> datapoints) 
	{
		free(ps -> datapoints);
		ps -> datapoints = NULL;
	}

	if(ps -> ub_box)
	{
		free(ps -> ub_box);
		ps -> ub_box = NULL;	
	}

	if(ps -> lb_box)
	{
		free(ps -> lb_box);
		ps -> lb_box = NULL;	
	}
}


void mpi_printf(global_context_t* ctx, const char *fmt, ...)
{
	if(ctx -> mpi_rank == 0)
	{
		va_list l;
		va_start(l, fmt);
//		printf("[MASTER]: ");
		vprintf(fmt, l);
		//        myflush(stdout);
		va_end(l);
	}
    fflush(stdout);
}

void generate_random_matrix(
		float_t** data,
		int dimensions,
		size_t n,
		global_context_t* ctx)
{
	/* seed the random number generator */
	srand((unsigned)time(NULL) + ctx -> mpi_rank * ctx -> world_size + ctx -> __processor_name_len);

	//size_t n = rand() % (nmax - nmin) + nmin;
	float_t* local_data = (float_t*)malloc(dimensions*n*sizeof(float_t));
	for(size_t i = 0; i < dimensions*n; ++i) local_data[i] = (float_t)rand()/(float_t)RAND_MAX;
	*data = local_data;
	
	ctx -> dims = dimensions;
	ctx -> local_n_points = n;

	return;
}


void lu_dynamic_array_allocate(lu_dynamic_array_t * a)
{
    a -> data = (idx_t*)malloc(ARRAY_INCREMENT*sizeof(idx_t));
    a -> count = 0;
    a -> size = ARRAY_INCREMENT;
}

void lu_dynamic_array_pushBack(lu_dynamic_array_t * a, idx_t p)
{
    if(a -> count < a -> size)
    {
        a -> data[a -> count] =  p;
        a -> count += 1;
    }
    else{
        a -> size += ARRAY_INCREMENT;
        a -> data = realloc(a -> data, a -> size * sizeof(idx_t));
        a -> data[a -> count] =  p;
        a -> count += 1;
    }
}

void lu_dynamic_array_Reset(lu_dynamic_array_t * a)
{
    a -> count = 0;
}

void lu_dynamic_array_reserve(lu_dynamic_array_t * a, idx_t n)
{
    a -> data = realloc(a -> data, n*sizeof(idx_t));
    a -> size = n;
}

void lu_dynamic_array_init(lu_dynamic_array_t * a)
{
    a -> data = NULL;
    a -> count = 0;
    a -> size = 0;
}

const char*  __units[3] = {"MB", "GB", "TB"};
const double __multiplier[3] = {1e6, 1e9, 1e12}; 

static inline int get_unit_measure(size_t bytes)
{
    if((double)bytes < (1e9))
    {
        return 0;
    }
    else if ((double)bytes < (1e12)) {
        return 1; 
    }
    else
    {
        return 2;
    }


}

float_t* read_data_file(global_context_t *ctx, const char *fname, const idx_t ndims,
                        const int file_in_float32) 
{
    printf("Reading %s\n",fname);
    FILE *f = fopen(fname, "r");
    if (!f) 
    {
        fprintf(stderr,"Cannot open file %s\n", fname);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    size_t n = ftell(f);
    rewind(f);

    int InputFloatSize = file_in_float32 ? 4 : 8;

    n = n / (InputFloatSize);

    float_t *data = (float_t *)MY_MALLOC(n * sizeof(float_t));

    if (file_in_float32) 
    {
        float *df = (float *)MY_MALLOC(n * sizeof(float));
        size_t fff = fread(df, sizeof(float), n, f);

        int measure = get_unit_measure(fff * sizeof(float));
        double file_len_converted = (double)(fff * sizeof(float))/__multiplier[measure];

        mpi_printf(ctx, "Read %.2lf%s\n", file_len_converted, __units[measure]);

        ctx -> dims = ndims;
        ctx -> n_points = n / ctx -> dims;

        mpi_printf(ctx, "Got ndims %lu npoints %lu\n", ctx -> dims, ctx -> n_points);
        fclose(f);

        for (uint64_t i = 0; i < n; ++i) data[i] = (float_t)(df[i]);

        free(df);
    } 
    else 
    {
        double *df = (double *)MY_MALLOC(n * sizeof(double));
        size_t fff = fread(df, sizeof(double), n, f);

        int measure = get_unit_measure(fff * sizeof(double));
        double file_len_converted = (double)(fff * sizeof(double))/__multiplier[measure];
        mpi_printf(ctx, "Read %.2lf%s\n", file_len_converted, __units[measure]);

        ctx -> dims = ndims;
        ctx -> n_points = n / ctx -> dims;

        mpi_printf(ctx, "Got ndims %lu npoints %lu\n", ctx -> dims, ctx -> n_points);
        fclose(f);

        for (uint64_t i = 0; i < n; ++i) data[i] = (float_t)(df[i]);

        free(df);
    }
    return data;
}

void ordered_buffer_to_file(global_context_t* ctx, void* buffer, size_t el_size, uint64_t n, const char* fname)
{
    //MPI_Barrier(ctx -> mpi_communicator);
    MPI_DB_PRINT("[MASTER] writing to file %s\n", fname);
    void* tmp_data; 
    int* ppp; 
    int* displs;

    MPI_Barrier(ctx -> mpi_communicator);
    
    uint64_t tot_n = 0;
    MPI_Reduce(&n, &tot_n, 1, MPI_UINT64_T , MPI_SUM, 0, ctx -> mpi_communicator);

    if(I_AM_MASTER) 
    {
        tmp_data = (void*)MY_MALLOC(el_size * tot_n );
        ppp      = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));
        displs   = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));

    }
    
    int nn = (int)n;
    MPI_Gather(&nn, 1, MPI_INT, ppp, 1, MPI_INT, 0, ctx -> mpi_communicator);

    if(I_AM_MASTER)
    {
        displs[0] = 0;
        for(int i = 0; i < ctx -> world_size; ++i) ppp[i]    = el_size  * ppp[i];
        for(int i = 1; i < ctx -> world_size; ++i) displs[i] = displs[i - 1] + ppp[i - 1];
            
    }

    MPI_Gatherv(buffer, (int)(el_size * n), 
            MPI_BYTE, tmp_data, ppp, displs, MPI_BYTE, 0, ctx -> mpi_communicator);

    if(I_AM_MASTER)
    {
        FILE* file = fopen(fname,"w");
        if(!file)
        {
            printf("Cannot open file %s ! Aborting \n", fname);
        }
        fwrite(tmp_data, 1, el_size * tot_n, file);
        fclose(file);
        free(tmp_data);
        free(ppp);
        free(displs);

    }
    MPI_Barrier(ctx -> mpi_communicator);
}

void distributed_buffer_to_file(global_context_t* ctx, void* buffer, size_t el_size, uint64_t n, const char* fname)
{
    char out_path_w_proc_name[PATH_LEN]; 
    MPI_DB_PRINT("[MASTER] writing to file %s.%s\n", fname, "[proc]");
    snprintf(out_path_w_proc_name, PATH_LEN, "%s.%d", fname, ctx -> mpi_rank);
    FILE* file = fopen(out_path_w_proc_name,"w");
    if(!file)
    {
        printf("Cannot open file %s ! Aborting \n", fname);
    }
    else
    {
        fwrite(buffer, 1, el_size * n, file);
    }
    fclose(file);

}

void big_ordered_buffer_to_file(global_context_t* ctx, void* buffer, size_t el_size, uint64_t n, const char* fname)
{
    //MPI_Barrier(ctx -> mpi_communicator);
    MPI_DB_PRINT("[MASTER] writing to file %s\n", fname);
    void*  tmp_data; 
    idx_t  already_sent = 0;
    idx_t* ppp; 
    idx_t* displs;
    idx_t* already_recv;

    MPI_Barrier(ctx -> mpi_communicator);
    
    uint64_t tot_n = 0;
    MPI_Reduce(&n, &tot_n, 1, MPI_UINT64_T , MPI_SUM, 0, ctx -> mpi_communicator);

    if(I_AM_MASTER) 
    {
        tmp_data = (void*)MY_MALLOC(el_size * tot_n );
        ppp      = (idx_t*)MY_MALLOC(ctx -> world_size * sizeof(idx_t));
        displs   = (idx_t*)MY_MALLOC(ctx -> world_size * sizeof(idx_t));
        already_recv   = (idx_t*)MY_MALLOC(ctx -> world_size * sizeof(idx_t));

    }
    
    MPI_Gather(&n, 1, MPI_UINT64_T, ppp, 1, MPI_UINT64_T, 0, ctx -> mpi_communicator);

    if(I_AM_MASTER)
    {
        displs[0] = 0;
        for(int i = 0; i < ctx -> world_size; ++i) ppp[i]    = el_size  * ppp[i];
        for(int i = 1; i < ctx -> world_size; ++i) displs[i] = displs[i - 1] + ppp[i - 1];
        for(int i = 0; i < ctx -> world_size; ++i) already_recv[i] = 0;
            
    }


    //Gather on master
    //
    
    uint64_t default_msg_len = 100000000; //bytes
    
    if(I_AM_MASTER)
    {
        //recieve from itself
        memcpy(tmp_data, buffer, n * el_size);   
        for(int r = 1; r < ctx -> world_size; ++r)
        {
            while(already_recv[r] < ppp[r])
            {
                MPI_Status status;
                MPI_Probe(r, MPI_ANY_TAG, ctx -> mpi_communicator, &status);

                MPI_Request request;
                int count_recv; 
                int source = status.MPI_SOURCE;
                MPI_Get_count(&status, MPI_BYTE, &count_recv);

                MPI_Recv(tmp_data + displs[r] + already_recv[r], ppp[r], MPI_BYTE, r, r, ctx -> mpi_communicator, MPI_STATUS_IGNORE);
                already_recv[r] += count_recv;

                #ifdef PRINT_ORDERED_BUFFER
                printf("[MASTER] recieved from %d %lu elements out of %lu\n", r, already_recv[r], ppp[r]);
                #endif
            }
            #ifdef PRINT_ORDERED_BUFFER
            printf("-----------\n");
            #endif
        }
    }
    else
    {
        while(already_sent < n * el_size) 
        {
            int count_send = MIN(default_msg_len, n * el_size - already_sent); 
            MPI_Send(buffer + already_sent, count_send, MPI_BYTE, 0, ctx -> mpi_rank, ctx -> mpi_communicator);
            already_sent += count_send;
        }
    }

    if(I_AM_MASTER)
    {
        FILE* file = fopen(fname,"w");
        if(!file)
        {
            printf("Cannot open file %s ! Aborting \n", fname);
        }
        fwrite(tmp_data, 1, el_size * tot_n, file);
        fclose(file);
        free(tmp_data);
        free(ppp);
        free(displs);

    }
    MPI_Barrier(ctx -> mpi_communicator);
}

void ordered_data_to_file(global_context_t* ctx, const char* fname)
{
    //MPI_Barrier(ctx -> mpi_communicator);
    MPI_DB_PRINT("[MASTER] writing DATA to file\n");
    float_t* tmp_data; 
    int* ppp; 
    int* displs;

    MPI_Barrier(ctx -> mpi_communicator);
    if(I_AM_MASTER) 
    {
        tmp_data = (float_t*)MY_MALLOC(ctx -> dims * ctx -> n_points * sizeof(float_t));
        ppp      = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));
        displs   = (int*)MY_MALLOC(ctx -> world_size * sizeof(int));

    }
    
    MPI_Gather(&(ctx -> local_n_points), 1, MPI_INT, ppp, 1, MPI_INT, 0, ctx -> mpi_communicator);

    if(I_AM_MASTER)
    {
        displs[0] = 0;
        for(int i = 0; i < ctx -> world_size; ++i) ppp[i]    = ctx -> dims * ppp[i];
        for(int i = 1; i < ctx -> world_size; ++i) displs[i] = displs[i - 1] + ppp[i - 1];
            
    }
    MPI_Gatherv(ctx -> local_data, ctx -> dims * ctx -> local_n_points, 
            MPI_MY_FLOAT, tmp_data, ppp, displs, MPI_MY_FLOAT, 0, ctx -> mpi_communicator);

    if(I_AM_MASTER)
    {
        FILE* file = fopen(fname,"w");
        if(file)
        {
            fwrite(tmp_data, sizeof(float_t), ctx -> dims * ctx -> n_points, file);
            fclose(file);
        }
        else
        {
            printf("Cannot open file %s\n", fname);
        }
        free(tmp_data);
        free(ppp);
        free(displs);
    }
    MPI_Barrier(ctx -> mpi_communicator);
}

void test_file_path(const char* fname)
{
    FILE* file = fopen(fname,"w");
    if(file)
    {
        fprintf(file, "This is only to test if I can open a file in the desidered path\n");
        fprintf(file, "Here will be written the output of dadp\n");
        fclose(file);
    }
    else
    {
        printf("Cannot open file %s\n", fname);
        exit(1);
    }
    
}

void test_distributed_file_path(global_context_t* ctx, const char* fname)
{
    char out_path_w_proc_name[PATH_LEN]; 
    snprintf(out_path_w_proc_name, PATH_LEN, "%s.%d", fname, ctx -> mpi_rank);
    FILE* file = fopen(out_path_w_proc_name,"w");
    if(!file)
    {
        printf("Cannot open file %s ! Aborting \n", out_path_w_proc_name);
    }
    else
    {
        fprintf(file, "This is only to test if I can open a file in the desidered path\n");
        fprintf(file, "Here will be written the output of dadp\n");
    }
    fclose(file);

}
