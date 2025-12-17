#pragma once 
#include "../common/common.h"
#include "heap.h"
#include "kdtree.h"
#include <stdlib.h>

#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>

#define DTHR 23.92812698
#define PI_F 3.1415926f
#define ARRAY_INCREMENT 500
#define DA_DTYPE idx_t
#define NOBORDER MY_SIZE_MAX

#define VERBOSE_TRUE 1
#define VERBOSE_FALSE 0

typedef struct center_t
{
    int cluster_idx;
    idx_t idx;
    float_t density;
} center_t;


typedef struct mpi_double_int{
	float_t val;
	int key;
} mpi_double_int;

typedef struct guess_t
{
	float_t x_guess;
	int bin_idx;
	float_t ep; 

} guess_t;


typedef struct partition_t
{
	int d;
	int n_procs;
	int start_proc;
	size_t n_points;
	int lr;
	struct top_kdtree_node_t* parent;
    //support for parallel partition to be put into a union maybe
    idx_t  offset;
    //support for single thread paritition to be put into a union maybe
	point_t* base_ptr;

} partition_t;

typedef struct partition_queue_t
{
	int count;
	int _capacity;
	struct partition_t* data;
} partition_queue_t;


typedef struct top_kdtree_node_t
{
	float_t split_val;
	float_t* lb_node_box; //Needed? 
	float_t* ub_node_box; //Needed?
	int owner;
	int split_dim;
	size_t n_points;
	struct top_kdtree_node_t* lch;
	struct top_kdtree_node_t* rch;
	struct top_kdtree_node_t* parent;
} top_kdtree_node_t;

typedef struct top_kdtree_t
{
	int dims;
	size_t count;
	size_t _capacity;
	struct top_kdtree_node_t* _nodes;
	struct top_kdtree_node_t* root;
} top_kdtree_t;

typedef struct partition_utils_t
{
   size_t* lt_count; 
   size_t* gt_count; 
   size_t* lt_displ; 
   size_t* gt_displ; 
} partition_utils_t;



void     simulate_master_read_and_scatter(int, size_t, global_context_t* ); 
void     top_tree_init(global_context_t *ctx, top_kdtree_t *tree); 
void     build_top_kdtree(global_context_t *ctx, top_kdtree_t *tree, idx_t n_bins, float_t tolerance);
void     exchange_points(global_context_t* ctx, top_kdtree_t* tree);
void     build_local_tree(global_context_t* ctx, kdtree_t* local_tree);
size_t   partition_data_around_value(point_t *array, size_t vec_len, size_t compare_dim, size_t left, size_t right, float_t pivot_value);
size_t   parallel_partition_data_around_value(partition_utils_t* p_putils, float_t* in, 
                                            float_t* out, size_t vec_len, size_t compare_dim,
                                            size_t left, size_t right, float_t pivot_value);
void     mpi_ngbh_search(global_context_t* ctx, datapoint_info_t* dp_info, top_kdtree_t* top_tree, kdtree_t* local_tree, float_t* data, int k);
float_t  compute_ID_two_NN_ML(global_context_t* ctx, datapoint_info_t* dp_info, idx_t n, int verbose);
void     compute_density_kstarnn_rma_v2(global_context_t* ctx, const float_t d, int verbose);
void     compute_correction(global_context_t* ctx, float_t Z);
int      foreign_owner(global_context_t*, idx_t);
void     top_tree_free(global_context_t *ctx, top_kdtree_t *tree);
void     parallel_build_top_kdtree(global_context_t *ctx, top_kdtree_t *tree, float_t tolerance, idx_t oversampling);





