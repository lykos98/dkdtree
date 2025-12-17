#pragma once
#include "heap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <time.h>
#include <float.h>

#ifndef MAX
#define MAX(x,y) ((x > y) ? (x) : (y))
#endif

#ifndef MIN
#define MIN(x,y) ((x < y) ? (x) : (y))
#endif

#define DEFAULT_LEAF_SIZE 32

#define ALIGNMENT 64
#define CHECK_ALLOCATION_NO_CTX(x) if(!x){printf("[!!!] Failed allocation: %s at line %d \n", __FILE__, __LINE__ ); exit(1);}
#define MY_MALLOC(n) ({void* p = aligned_alloc(ALIGNMENT,n); CHECK_ALLOCATION_NO_CTX(p); memset(p, 0, n); p; })

#define FREE(x) if(x); free(x);

#define HP_LEFT_SIDE  0
#define HP_RIGHT_SIDE 1

#define T double
#define DATA_DIMS 0 

#ifdef USE_FLOAT32
    #define float_t float
#else
    #define float_t double 
#endif

#ifdef USE_INT32
    #define MY_SIZE_MAX UINT32_MAX
    #define idx_t uint32_t
#else
    #define MY_SIZE_MAX UINT64_MAX
    #define idx_t uint64_t
#endif

typedef struct 
{
    float_t* lb;
    float_t* ub;
} bounding_box_t;

#ifndef POINT
    #define POINT
    typedef struct point_t
    {
        idx_t    array_idx;
        float_t* data;
    } point_t;
#endif

struct pivot_t;

union pivot_data {

    struct {
        float_t split_value;    // The value used for the split
        uint8_t split_variable; // The dimension used for the split
        uint64_t lch_idx;       // 4-byte index to the left child in the pivot array
        uint64_t rch_idx;       // 4-byte index to the right child in the pivot array
    } internal;

    // Data for a leaf node
    struct {
        uint32_t point_list_idx;   // Index to the first point in the global point array
        uint32_t leaf_point_count; // Number of points in this leaf
    } leaf;
};

// The new, slimmer pivot_t struct
struct pivot_t {
    uint8_t is_leaf; 
    bounding_box_t bounding_box;
    union pivot_data as;
};

struct kdtree_t
{
    float_t* data;
    uint32_t  dims;
    idx_t  n_points;
    idx_t  root;

    struct point_t* __points;
    float_t* __boxes_data;
    struct pivot_t* __pivots;
};

typedef struct pivot_t pivot_t;
//typedef struct point_t point_t;
typedef struct kdtree_t kdtree_t;


static inline float_t euclidean_distance(const float_t* p1, const float_t* p2, const idx_t dims) 
{
    register float_t d = 0;
    for (idx_t i = 0; i < dims; ++i) 
    {
        d += (p1[i] - p2[i])*(p1[i] - p2[i]);
    }
    return d;
    // return sqrt(d);
}

// #include <immintrin.h>
// static inline float_t euclidean_distance_simd(float_t* p1, float_t* p2, idx_t dims) 
// {
//     // Use AVX to process 4 doubles at a time
//     __m256d diff, sum = _mm256_setzero_pd();
//
//     idx_t i = 0;
//     for (; i <= (idx_t)dims - 4; i += 4) {
//         // Load 4 doubles from p1 and p2
//         __m256d v1 = _mm256_loadu_pd(p1 + i);
//         __m256d v2 = _mm256_loadu_pd(p2 + i);
//
//         // Subtract
//         diff = _mm256_sub_pd(v1, v2);
//
//         // Square and add to sum (fused multiply-add)
//         sum = _mm256_fmadd_pd(diff, diff, sum);
//     }
//
//     // Horizontal sum of the 4 partial sums in the vector
//     __m256d temp = _mm256_hadd_pd(sum, sum);
//     __m128d sum_high = _mm256_extractf128_pd(temp, 1);
//     __m128d result = _mm_add_pd(_mm256_castpd256_pd128(temp), sum_high);
//
//     float_t d = _mm_cvtsd_f64(result);
//
//     // Handle any remaining dimensions that aren't a multiple of 4
//     for (; i < dims; ++i) {
//         float_t term = p1[i] - p2[i];
//         d += term * term;
//     }
//
//     return d;
// }

#ifdef USE_FLOAT32
typedef float v4f __attribute__((vector_size(16)));
#else
typedef double v4f __attribute__((vector_size(32)));
#endif

static inline void point_swap(point_t *x, point_t *y, const idx_t dims) {
    // exchange idxs
    
    idx_t tmp_idx;
    tmp_idx = x->array_idx;
    x->array_idx = y->array_idx;
    y->array_idx = tmp_idx;

    //swap datapoint

    for(idx_t i = 0; i < dims; ++i)
    {
        float_t tmp_data = x -> data[i];
        x -> data[i] = y -> data[i];
        y -> data[i] = tmp_data;
    }

}

static void pivot_initialize_all(pivot_t* node_array, idx_t n) {
    for (idx_t i = 0; i < n; ++i) 
    {
        node_array[i].as.internal.lch_idx = -1;
        node_array[i].as.internal.rch_idx = -1;
        node_array[i].is_leaf = false;
        node_array[i].as.internal.split_variable = -1;
        node_array[i].as.internal.split_value = FLT_MAX;
    }
}


static void point_initialize_all(point_t* point_array, float_t* data, idx_t n, idx_t dims)
{
    for (idx_t i = 0; i < n; ++i) 
    {
        point_array[i].array_idx = i;
        point_array[i].data = data + i*dims;
    }

}

static inline int point_compare(const point_t *a, const point_t *b, int var) {
    float_t res = a->data[var] - b->data[var];
    return (res > 0);
}

static void pivot_print(pivot_t node, idx_t node_idx, idx_t dims) {
    printf("Node %lu:\n", node_idx);
    printf(" data: ");

    printf("\n");

    if(node.is_leaf)
    {
        printf("  LEAF node\n");
        printf("  list len: %d\n", node.as.leaf.leaf_point_count);
    }
    else 
    {
        printf("  INTERNAL node\n");
        printf("  split_value: %.4lf split_variable: %d\n", node.as.internal.split_value, node.as.internal.split_variable);
        printf("  lch: %lu\n", node.as.internal.lch_idx);
        printf("  rch: %lu\n", node.as.internal.rch_idx);
    
    }
    printf("  lb_box: [ ");
    for(int i=0; i<dims; ++i)
    {
        printf("%.4lf ", node.bounding_box.lb[i]);
    }
    printf(" ]\n");

    printf("  ub_box: [ ");
    for(int i=0; i<dims; ++i)
    {
        printf("%.4lf ", node.bounding_box.ub[i]);
    }
    printf(" ]\n");


    printf("\n");
}

// Standard Lomuto partition function

static inline int point_partition(point_t *arr, idx_t low, idx_t high, int split_var, idx_t dims) {
    point_t pivot = arr[high];

    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (!point_compare(arr + j, &pivot, split_var)) 
        {
            i++;
            point_swap(arr + i, arr + j, dims);
        }
    }
    point_swap(arr + i + 1, arr + high, dims);
    return (i + 1);
}


static inline int faster_point_partition_random(point_t *arr, idx_t low, idx_t high, int split_var, idx_t dims) {
    // Seed the random number generator only once for the application's lifetime.
    static bool seeded = false;
    if (!seeded) {
        srand(42); // Using a fixed seed for reproducible behavior. Use time(NULL) for non-deterministic pivots.
        seeded = true;
    }
    
    // In a subarray of size > 1, choose a random pivot index.
    if (low < high) {
        idx_t pivot_idx = low + rand() % (high - low + 1);
        // Swap the chosen pivot with the last element to use the standard Lomuto scheme.
        point_swap(arr + pivot_idx, arr + high, dims);
    }
    
    // The rest of this function is the standard Lomuto partition scheme.
    point_t pivot = arr[high];
    idx_t i = (low - 1);
    for (idx_t j = low; j < high; j++) {
        // If current element is smaller than or equal to the pivot
        if (arr[j].data[split_var] <= pivot.data[split_var]) {
            i++;
            point_swap(arr + i, arr + j, dims);
        }
    }
    point_swap(arr + i + 1, arr + high, dims);
    return (i + 1);
}

static int faster_point_compute_median(point_t *a, idx_t left, idx_t right, int split_var, idx_t dims) 
{
    // `k` is the absolute index of the median in the original full array.
    int k = left + ((right - left + 1) / 2);

    while (left <= right) {
        // Base case: if the subarray has only one element, it's the median.
        if (left == right) {
            return left;
        }

        // Partition the array using a random pivot and get the pivot's final index.
        int pivotIndex = faster_point_partition_random(a, left, right, split_var, dims);

        // If the pivot is the median, we are done.
        if (pivotIndex == k) {
            return pivotIndex;
        }
        // If the median is in the left subarray, adjust the `right` boundary.
        else if (pivotIndex > k) {
            right = pivotIndex - 1;
        }
        // If the median is in the right subarray, adjust the `left` boundary.
        else {
            left = pivotIndex + 1;
        }
    }
    return -1; // Should not be reached in a valid scenario.
}

// Function to find the n-th smallest element in a sub-array of points using Quickselect.
static int point_compute_nth_element(point_t *a, idx_t left, idx_t right, idx_t rank, int split_var, idx_t dims) 
{
    // `k` is the absolute index of the rank-th smallest element in the full array.
    // `rank` is the 0-indexed rank in the subarray `a[left...right]`.
    int k = left + rank;

    while (left <= right) {
        // Base case: if the subarray has only one element, it's the desired element.
        if (left == right) {
            return left;
        }

        // Partition the array using a random pivot and get the pivot's final index.
        int pivotIndex = faster_point_partition_random(a, left, right, split_var, dims);

        // If the pivot is the rank-th element, we are done.
        if (pivotIndex == k) {
            return pivotIndex;
        }
        // If the rank-th element is in the left subarray, adjust the `right` boundary.
        else if (pivotIndex > k) {
            right = pivotIndex - 1;
        }
        // If the rank-th element is in the right subarray, adjust the `left` boundary.
        else {
            left = pivotIndex + 1;
        }
    }
    return -1; // Should not be reached in a valid scenario.
}

// Implementation of QuickSelect
static int point_compute_median(point_t *a, idx_t left, idx_t right, int split_var, idx_t dims) 
{
    // printf("----------\n");
    int k = left + ((right - left + 1) / 2);

    if (left == right) return left;
    if (left == (right - 1)) 
    {
        if (point_compare(a + left, a + right, split_var)) 
        {
            point_swap(a + left, a + right, dims);
        }
        return right;
    }
    while (left <= right) {

        // Partition a[left..right] around a pivot
        // and find the position of the pivot
        int pivotIndex = point_partition(a, left, right, split_var, dims);
        // printf("%d %d %d %d\n",left, right, k, pivotIndex);

        // If pivot itself is the k-th smallest element
        if (pivotIndex == k)
        return pivotIndex;

        // If there are more than k-1 elements on
        // left of pivot, then k-th smallest must be
        // on left side.

        else if (pivotIndex > k) right = pivotIndex - 1;

        // Else k-th smallest is on right side.
        else left = pivotIndex + 1;
    }
    return -1;
}

static idx_t recursive_make_tree(point_t* points, pivot_t* pivots, 
                             idx_t child_pivot_idx, int left_or_right, 
                             idx_t start, idx_t end, int level, idx_t dims) 
{
    struct timespec start_node, stop_node;

    clock_gettime(CLOCK_MONOTONIC, &start_node);

    // printf("Entering Node: [%d %d] @level %d\n", 
    //        start, end, level );

    idx_t parent_pivot_idx = (child_pivot_idx > 0) ? ((child_pivot_idx - 1) / 2) : -1;
    pivot_t *pivot = pivots + child_pivot_idx;
    pivot_t *parent = pivots + parent_pivot_idx;

    switch (left_or_right) 
    {
        case HP_LEFT_SIDE:
        {
            memcpy(pivot->bounding_box.lb, parent->bounding_box.lb, dims*sizeof(float_t));
            memcpy(pivot->bounding_box.ub, parent->bounding_box.ub, dims*sizeof(float_t));
            idx_t parent_splitting_dim = pivots[parent_pivot_idx].as.internal.split_variable;
            float_t parent_split_val = pivots[parent_pivot_idx].as.internal.split_value;

            pivot->bounding_box.ub[parent_splitting_dim] = parent_split_val; 
        }
        break;

        case HP_RIGHT_SIDE:
        {
            memcpy(pivot->bounding_box.lb, parent->bounding_box.lb, dims*sizeof(float_t));
            memcpy(pivot->bounding_box.ub, parent->bounding_box.ub, dims*sizeof(float_t));
            idx_t parent_splitting_dim = pivots[parent_pivot_idx].as.internal.split_variable;
            float_t parent_split_val = pivots[parent_pivot_idx].as.internal.split_value;

            pivot->bounding_box.lb[parent_splitting_dim] = parent_split_val; 
        }
        break;

        default:
        {
            for(int i=0; i<dims; ++i)
            {
                pivot->bounding_box.lb[i] = FLT_MAX;
                pivot->bounding_box.ub[i] = -FLT_MAX;
            }

            // compute master bounding_box
            //

            for(idx_t i=start; i<=end; ++i)
            {
                for(idx_t j=0; j<dims; ++j)
                {
                    pivot->bounding_box.lb[j] = MIN(pivot->bounding_box.lb[j], points[i].data[j]);
                    pivot->bounding_box.ub[j] = MAX(pivot->bounding_box.ub[j], points[i].data[j]);
                }
            }
        }
        break;
         
    }
    
    //int split_var = level % dims;

    int split_var = 0;
    float_t max_extension = 0;
    for(idx_t d=0; d < dims; ++d)
    {
        float_t box_extension = pivot->bounding_box.ub[d] - pivot->bounding_box.lb[d];
        box_extension = box_extension*box_extension;

        split_var = (box_extension > max_extension) ? d : split_var;
        max_extension = MAX(box_extension, max_extension);
    }

    if (end - start < DEFAULT_LEAF_SIZE)
    {
        pivots[child_pivot_idx].is_leaf = 1;
        pivots[child_pivot_idx].as.leaf.leaf_point_count = (size_t)(end - start + 1);
        pivots[child_pivot_idx].as.leaf.point_list_idx = start;
        return child_pivot_idx;
    }

    int median_idx = -1;

    median_idx = faster_point_compute_median(points, start, end, split_var, dims);
    // printf("%d median idx\n", median_idx);
    if (median_idx > -1) 
    {
        // generate a pivot
        pivots[child_pivot_idx].as.internal.split_variable = split_var;
        pivots[child_pivot_idx].as.internal.split_value = points[median_idx].data[split_var];
        pivots[child_pivot_idx].as.internal.lch_idx = 
            recursive_make_tree(points, pivots, child_pivot_idx*2 + 1, 
                                HP_LEFT_SIDE, start, median_idx, level + 1, dims);
        pivots[child_pivot_idx].as.internal.rch_idx = 
            recursive_make_tree(points, pivots, child_pivot_idx*2 + 2, 
                                HP_RIGHT_SIDE, median_idx + 1, end, level + 1, dims);

    }

    clock_gettime(CLOCK_MONOTONIC, &stop_node);

    // printf("Exiting Node: [%d %d] @level %d -> elapsed: %f\n", 
    //        start, end, level, ((float)stop_node.tv_sec - (float)start_node.tv_sec) + 
    //                           ((float)stop_node.tv_nsec - (float)start_node.tv_nsec)/1e9  );

    return child_pivot_idx;
}

static inline float_t hyper_plane_dist(const float_t *p1, const float_t p2, const int var) 
{
    return p1[var] - p2;
}

static inline float_t box_dist(const float_t *p, const float_t *lb, const float_t* ub, const int dims) 
{
    float_t r = 0;
    for (idx_t i=0; i<dims; i++) {
        if(p[i] < lb[i]) {
            r += (lb[i] - p[i]) * (lb[i] - p[i]);
        } else if (p[i] > ub[i]) {
            r += (p[i] - ub[i]) * (p[i] - ub[i]);
        }
    }
    return r;
}

static inline int hyper_plane_side(const float_t *p1, const float_t *p2, const int var)
{
    return p1[var] > p2[var];
}

static void knn_sub_tree_search(const float_t *point, const point_t* data_points, const pivot_t* pivots, 
                         const idx_t root_idx, heap_t *H, const idx_t dims, idx_t* vis_nodes) 
{
    pivot_t root = pivots[root_idx];
    if (root.is_leaf) 
    {
        // take the base pointer
        idx_t base_point = root.as.leaf.point_list_idx;
        const point_t* leaf_points = data_points + (size_t)base_point;
        float_t* base_data   = leaf_points[0].data;
        idx_t point_list_count = root.as.leaf.leaf_point_count;


        float_t dists[DEFAULT_LEAF_SIZE];
        #pragma unroll 4
        for (size_t i = 0; i < point_list_count; ++i) 
        {
            *vis_nodes = *vis_nodes + 1;
            float_t distance = euclidean_distance(point, base_data + (size_t)i*dims, dims);
            dists[i] = distance;
        }

        for (size_t i = 0; i < point_list_count; ++i) 
        {
            point_t data_point = leaf_points[i];
            max_heap_insert(H, dists[i], data_point.array_idx);
        }

        // CORRECTED an efficient single loop:
        // for (size_t i = 0; i < point_list_count; ++i)
        // {
        //     *vis_nodes = *vis_nodes + 1;
        //
        //     // Calculate distance using the correct data pointer
        //     float_t distance = euclidean_distance(point, base_data + i*dims, dims);
        //
        //     // Immediately use the distance. No temporary array, less memory traffic.
        //     max_heap_insert(H, distance, leaf_points[i].array_idx);
        // }
        return;
    }



    //float_t hp_distance = hyper_plane_dist(point, root->split_value, root->split_variable);
    float_t hp_distance = hyper_plane_dist(point, root.as.internal.split_value, root.as.internal.split_variable);

    int side = hp_distance > 0.f;

    switch (side) {
    case HP_LEFT_SIDE:
        knn_sub_tree_search(point, data_points, pivots, root.as.internal.lch_idx, H, dims, vis_nodes);
        break;

    case HP_RIGHT_SIDE:
        knn_sub_tree_search(point, data_points, pivots, root.as.internal.rch_idx, H, dims, vis_nodes);
        break;

    default:
        break;
    }
    float_t max_d = H->data[0].value;


    // int c = max_d > (hp_distance * hp_distance);
    // if (c || (H->count) < (H->size))
    // {
    //     switch (side) 
    //     {
    //         case HP_LEFT_SIDE:
    //             if (root->rch) knn_sub_tree_search(point, root->rch, H, dims, vis_nodes);
    //             break;
    //
    //         case HP_RIGHT_SIDE:
    //             if (root->lch) knn_sub_tree_search(point, root->lch, H, dims, vis_nodes);
    //             break;
    //
    //         default:
    //             break;
    //     }
    // }

    int heap_is_not_full = (H->count) < (H->size);
    switch (side) 
    {
        case HP_LEFT_SIDE:
            {
                bounding_box_t box = pivots[root.as.internal.rch_idx].bounding_box; 
                float_t dist = box_dist(point, box.lb, box.ub, dims);
                if((max_d > dist) || heap_is_not_full) 
                    knn_sub_tree_search(point, data_points, pivots, root.as.internal.rch_idx, H, dims, vis_nodes);
            }
            break;

        case HP_RIGHT_SIDE:
            {
                bounding_box_t box = pivots[root.as.internal.lch_idx].bounding_box; 
                float_t dist = box_dist(point, box.lb, box.ub, dims);
                if((max_d > dist) || heap_is_not_full) 
                    knn_sub_tree_search(point, data_points, pivots, root.as.internal.lch_idx, H, dims, vis_nodes);
            }
            break;

        default:
            break;
    }
    return;
}



static void kdtree_initialize(kdtree_t *tree, float_t *data, size_t n_points, unsigned int dims) 
{
    tree->dims = dims;
    tree->data = data;
    tree->n_points = n_points;
    tree->__points = (point_t *)MY_MALLOC(n_points * sizeof(point_t));
    tree->__pivots = (pivot_t *)MY_MALLOC(n_points * sizeof(pivot_t));
    tree->__boxes_data = (float_t *)MY_MALLOC(2 * n_points * sizeof(float_t)*dims);

    for(idx_t i=0; i<n_points; ++i)
    {
        idx_t ii = 2*i;
        tree->__pivots[i].bounding_box.lb = tree->__boxes_data + (size_t)ii*dims; 
        tree->__pivots[i].bounding_box.ub = tree->__boxes_data + (size_t)(ii+1)*dims; 
    }
    pivot_initialize_all(tree->__pivots, n_points);
    point_initialize_all(tree->__points, data, n_points, dims);
    tree->root = -1;
}

static void recursive_print(pivot_t* pivots, idx_t node_idx, idx_t dims, idx_t level)
{
    pivot_t node =  pivots[node_idx];
    if(level < 3)
    {
        printf("---LEVEL %lu ---\n", level);
        pivot_print(node, node_idx, dims);

    }
    if(!node.is_leaf) 
    {
        recursive_print(pivots, node.as.internal.lch_idx, dims, level+1);
        recursive_print(pivots, node.as.internal.rch_idx, dims, level+1);
    }
}

static void kdtree_print(kdtree_t* tree)
{
    idx_t level = 0;
    recursive_print(tree->__pivots, tree -> root, tree -> dims, level);
}

// void kdtree_compact_data(kdtree_t* tree)
// {
//     float_t* new_data = (float_t*)calloc(tree->dims * tree->n_points * sizeof(float_t), 1); 
//     for(idx_t i = 0; i < tree->n_points; ++i)
//     {
//         kdnode_t* node = tree->_nodes + i;
//                     memcpy(new_data + i*tree->dims, node->data, tree->dims * sizeof(float_t));        node->data = new_data + i*tree->dims; 
//     }
//     free(tree -> data);
//     tree -> data = new_data;
// }




#include <omp.h>

#define SERIAL_BUILD_CUTOFF 4096

static idx_t parallel_recursive_make_tree(point_t* points, pivot_t* pivots, 
                                   idx_t child_pivot_idx, int left_or_right, 
                                   idx_t start, idx_t end, int level, idx_t dims);

static void build_tree_kdtree_parallel(kdtree_t* tree) 
{
    /*************************************************
    * Wrapper for the parallel make_tree function.  *
    *************************************************/
    #pragma omp parallel
    {
        #pragma omp single
        {
            tree->root = parallel_recursive_make_tree(tree->__points, tree->__pivots, 0, -1,
                                                      0, tree->n_points-1, 0, tree->dims);
        }
    }
}

static idx_t parallel_make_tree_w_ranks(point_t* points, pivot_t* pivots, 
                                        idx_t child_pivot_idx, int left_or_right, 
                                        idx_t start, idx_t end, int level, idx_t dims, 
                                        idx_t start_rank, idx_t end_rank) 
{
    // this only build the first log2(ranks) levels
    idx_t parent_pivot_idx = (child_pivot_idx > 0) ? ((child_pivot_idx - 1) / 2) : -1;
    pivot_t *pivot = pivots + child_pivot_idx;
    pivot_t *parent = (parent_pivot_idx != (idx_t)-1) ? (pivots + parent_pivot_idx) : NULL;

    switch (left_or_right) 
    {
        case HP_LEFT_SIDE:
        {
            memcpy(pivot->bounding_box.lb, parent->bounding_box.lb, dims*sizeof(float_t));
            memcpy(pivot->bounding_box.ub, parent->bounding_box.ub, dims*sizeof(float_t));
            idx_t parent_splitting_dim = parent->as.internal.split_variable;
            float_t parent_split_val = parent->as.internal.split_value;
            pivot->bounding_box.ub[parent_splitting_dim] = parent_split_val; 
        }
        break;

        case HP_RIGHT_SIDE:
        {
            memcpy(pivot->bounding_box.lb, parent->bounding_box.lb, dims*sizeof(float_t));
            memcpy(pivot->bounding_box.ub, parent->bounding_box.ub, dims*sizeof(float_t));
            idx_t parent_splitting_dim = parent->as.internal.split_variable;
            float_t parent_split_val = parent->as.internal.split_value;
            pivot->bounding_box.lb[parent_splitting_dim] = parent_split_val;
        }
        break;

        default:
        {
            for(int i=0; i<dims; ++i) {
                pivot->bounding_box.lb[i] = FLT_MAX;
                pivot->bounding_box.ub[i] = -FLT_MAX;
            }
            for(idx_t i=start; i<=end; ++i) {
                for(idx_t j=0; j<dims; ++j) {
                    pivot->bounding_box.lb[j] = MIN(pivot->bounding_box.lb[j], points[i].data[j]);
                    pivot->bounding_box.ub[j] = MAX(pivot->bounding_box.ub[j], points[i].data[j]);
                }
            }
        }
        break;
    }
    
    //handle the case it is a leaf

    if(start_rank == end_rank)
    {
        pivots[child_pivot_idx].is_leaf = true;
        // write here the owner!!!
        pivots[child_pivot_idx].as.leaf.leaf_point_count = start_rank;
        return child_pivot_idx;
    }
    
    int split_var = 0;
    float_t max_extension = 0;
    for(idx_t d=0; d < dims; ++d) {
        float_t box_extension = pivot->bounding_box.ub[d] - pivot->bounding_box.lb[d];
        box_extension = box_extension*box_extension;
        if (box_extension > max_extension) {
            max_extension = box_extension;
            split_var = d;
        }
    }


    idx_t middle_rank = (start_rank + end_rank)/2;

    float_t fraction = (float_t)(middle_rank - start_rank + 1)/(float_t)(end_rank - start_rank + 1);
    float_t nth_place  = (end - start)*fraction;

    int nth_idx = point_compute_nth_element(points, start, end, (idx_t)nth_place, split_var, dims); 

    // printf("child_pivot_idx %lu start_rank %lu end_rank %lu middle %lu nth_place %.2lf fraction %.2lf\n", 
    //         child_pivot_idx, start_rank, end_rank, middle_rank, nth_place, fraction);
    //printf("child_pivot_idx %lu start %lu end %lu nth %lu\n", child_pivot_idx, start, end, (idx_t)nth_idx);

    if (nth_idx > -1) {
        pivots[child_pivot_idx].as.internal.split_variable = split_var;
        pivots[child_pivot_idx].as.internal.split_value = points[nth_idx].data[split_var];
        
        pivots[child_pivot_idx].as.internal.lch_idx = 
            parallel_make_tree_w_ranks(points, pivots, child_pivot_idx*2 + 1, 
                                    HP_LEFT_SIDE, start, nth_idx, level + 1, 
                                    dims, start_rank, middle_rank);
        
        pivots[child_pivot_idx].as.internal.rch_idx = 
            parallel_make_tree_w_ranks(points, pivots, child_pivot_idx*2 + 2, 
                                        HP_RIGHT_SIDE, nth_idx + 1, end, level + 1, 
                                        dims, middle_rank + 1, end_rank);
        
    }

    return child_pivot_idx;
}

static idx_t parallel_recursive_make_tree(point_t* points, pivot_t* pivots, 
                                          idx_t child_pivot_idx, int left_or_right, 
                                          idx_t start, idx_t end, int level, idx_t dims) 
{
    // Fallback to serial version for small subproblems
    if ((end - start) < SERIAL_BUILD_CUTOFF) {
        return recursive_make_tree(points, pivots, child_pivot_idx, left_or_right, start, end, level, dims);
    }

    // This logic is identical to the serial version, as the partitioning step itself remains serial.
    // The parallelism comes from recursively calling this function in parallel tasks.
    idx_t parent_pivot_idx = (child_pivot_idx > 0) ? ((child_pivot_idx - 1) / 2) : -1;
    pivot_t *pivot = pivots + child_pivot_idx;
    pivot_t *parent = (parent_pivot_idx != (idx_t)-1) ? (pivots + parent_pivot_idx) : NULL;

    switch (left_or_right) 
    {
        case HP_LEFT_SIDE:
        {
            memcpy(pivot->bounding_box.lb, parent->bounding_box.lb, dims*sizeof(float_t));
            memcpy(pivot->bounding_box.ub, parent->bounding_box.ub, dims*sizeof(float_t));
            idx_t parent_splitting_dim = parent->as.internal.split_variable;
            float_t parent_split_val = parent->as.internal.split_value;
            pivot->bounding_box.ub[parent_splitting_dim] = parent_split_val; 
        }
        break;

        case HP_RIGHT_SIDE:
        {
            memcpy(pivot->bounding_box.lb, parent->bounding_box.lb, dims*sizeof(float_t));
            memcpy(pivot->bounding_box.ub, parent->bounding_box.ub, dims*sizeof(float_t));
            idx_t parent_splitting_dim = parent->as.internal.split_variable;
            float_t parent_split_val = parent->as.internal.split_value;
            pivot->bounding_box.lb[parent_splitting_dim] = parent_split_val;
        }
        break;

        default:
        {
            for(int i=0; i<dims; ++i) {
                pivot->bounding_box.lb[i] = FLT_MAX;
                pivot->bounding_box.ub[i] = -FLT_MAX;
            }
            for(idx_t i=start; i<=end; ++i) {
                for(idx_t j=0; j<dims; ++j) {
                    pivot->bounding_box.lb[j] = MIN(pivot->bounding_box.lb[j], points[i].data[j]);
                    pivot->bounding_box.ub[j] = MAX(pivot->bounding_box.ub[j], points[i].data[j]);
                }
            }
        }
        break;
    }
    
    int split_var = 0;
    float_t max_extension = 0;
    for(idx_t d=0; d < dims; ++d) {
        float_t box_extension = pivot->bounding_box.ub[d] - pivot->bounding_box.lb[d];
        box_extension = box_extension*box_extension;
        if (box_extension > max_extension) {
            max_extension = box_extension;
            split_var = d;
        }
    }

    if (end - start < DEFAULT_LEAF_SIZE) {
        pivots[child_pivot_idx].is_leaf = 1;
        pivots[child_pivot_idx].as.leaf.leaf_point_count = (size_t)(end - start + 1);
        pivots[child_pivot_idx].as.leaf.point_list_idx = start;
        return child_pivot_idx;
    }

    int median_idx = faster_point_compute_median(points, start, end, split_var, dims);
    
    if (median_idx > -1) {
        pivots[child_pivot_idx].as.internal.split_variable = split_var;
        pivots[child_pivot_idx].as.internal.split_value = points[median_idx].data[split_var];
        
        #pragma omp task
        pivots[child_pivot_idx].as.internal.lch_idx = 
            parallel_recursive_make_tree(points, pivots, child_pivot_idx*2 + 1, 
                                         HP_LEFT_SIDE, start, median_idx, level + 1, dims);
        
        #pragma omp task
        pivots[child_pivot_idx].as.internal.rch_idx = 
            parallel_recursive_make_tree(points, pivots, child_pivot_idx*2 + 2, 
                                         HP_RIGHT_SIDE, median_idx + 1, end, level + 1, dims);
        
        #pragma omp taskwait
    }

    return child_pivot_idx;
}


static void kdtree_free(kdtree_t *tree) 
{

    FREE(tree->__pivots)
    FREE(tree->__points)
    FREE(tree->__boxes_data);
}

static void build_tree_kdtree(kdtree_t* tree) 
{
    /*************************************************
    * Wrapper for make_tree function.               *
    * Simplifies interfaces and takes time measures *
    *************************************************/
    tree->root = recursive_make_tree(tree->__points, tree->__pivots, 0, -1, 0, tree->n_points-1, 0, tree->dims);
    //tree->root = parallel_recursive_make_tree(tree->__points, tree->__pivots, 0, -1, 0, tree->n_points-1, 0, tree->dims);
}

