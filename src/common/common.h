#pragma once
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <stdint.h>
#include <time.h>
#include "../tree/heap.h"

#define DEFAULT_MSG_LEN 10000000
//#include <stdarg.h>

#define PARALLEL_FIX_BORDERS
// #define WRITE_SHUFFLED_DATA
// #define WRITE_NGBH
// #define WRITE_TOP_NODES
// #define WRITE_DENSITY
// #define WRITE_CLUSTER_ASSIGN_H1
// #define WRITE_BORDERS
// #define WRITE_CENTERS_PRE_MERGING
// #define WRITE_MERGES_INFO
// #define WRITE_MERGING_TABLE
// #define WRITE_FINAL_ASSIGNMENT

// #define PRINT_NGBH_EXCHANGE_SCHEME
// #define PRINT_H2_COMM_SCHEME
// #define PRINT_H1_CLUSTER_ASSIGN_COMPLETION
// #define PRINT_ORDERED_BUFFER
#define PRINT_BALANCE_FACTOR
// #define CHECK_CORRECT_EXCHANGE

#define DEFAULT_STR_LEN 200

#define MAX(A,B) ((A) > (B) ? (A) : (B))
#define MIN(A,B) ((A) < (B) ? (A) : (B))

#ifdef USE_FLOAT32
	#define float_t float
#else
	#define float_t double
#endif

#ifdef USE_FLOAT32
#define MPI_MY_FLOAT MPI_FLOAT
#else
#define MPI_MY_FLOAT MPI_DOUBLE
#endif


#define I_AM_MASTER ctx->mpi_rank == 0


#define MY_TRUE  1
#define MY_FALSE 0

#define HERE printf("%d in file %s reached line %d\n", ctx -> mpi_rank, __FILE__, __LINE__); fflush(stdout); MPI_Barrier(ctx -> mpi_communicator);

#define CHECK_ALLOCATION(x) if(!x){printf("[!!!] %d rank encountered failed allocation: %s at line %s \n", ctx -> mpi_rank, __FILE__, __LINE__ ); exit(1);};


#define CHECK_ALLOCATION_NO_CTX(x) if(!x){printf("[!!!] Failed allocation: %s at line %d \n", __FILE__, __LINE__ ); exit(1);}
#ifndef MY_MALLOC
#define MY_MALLOC(n) ({void* p = calloc(n,1); CHECK_ALLOCATION_NO_CTX(p); p; })
#endif

#define DB_PRINT(...) printf(__VA_ARGS__); fflush(stdout)
#ifdef NDEBUG
	#undef DB_PRINT(...)
	#define DB_PRINT(...)
#endif

#define MPI_DB_PRINT(...) mpi_printf(ctx,__VA_ARGS__)
#ifdef NDEBUG
	#undef MPI_DB_PRINT(...)
	#define MPI_DB_PRINT(...)
#endif

#define MPI_PRINT(...) mpi_printf(ctx,__VA_ARGS__)

#ifdef NDEBUG
    #define TIME_DEF 
    #define TIME_START 
    #define TIME_STOP 
    #define LOG_WRITE
#else 
    #define TIME_DEF struct timespec __start, __end;
    #define TIME_START { \
        MPI_Barrier(ctx -> mpi_communicator); \
        clock_gettime(CLOCK_MONOTONIC,&__start); \
    }
    #define TIME_STOP \
        (clock_gettime(CLOCK_MONOTONIC,&__end), \
        (double)(__end.tv_sec - __start.tv_sec) + (__end.tv_nsec - __start.tv_nsec)/1e9)
    #define LOG_WRITE(sec_name,time) { \
        MPI_Barrier(ctx -> mpi_communicator); \
        if(time > 0) \
        { \
            double max, min, avg; \
            MPI_Reduce(&time, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, ctx -> mpi_communicator); \
            MPI_Reduce(&time, &min, 1, MPI_DOUBLE, MPI_MIN, 0, ctx -> mpi_communicator); \
            MPI_Reduce(&time, &max, 1, MPI_DOUBLE, MPI_MAX, 0, ctx -> mpi_communicator); \
            MPI_Barrier(ctx->mpi_communicator); \
            MPI_DB_PRINT("%50.50s -> [avg: %.2lfs, min: %.2lfs, max: %.2lfs]\n", sec_name, avg/((double)ctx -> world_size), min, max); \
            fflush(stdout); \
        } \
        else \
        { \
            MPI_DB_PRINT("%s\n", sec_name);\
            fflush(stdout); \
        }\
    }
    
#endif

/*
 * from Spriengel code Gadget4
 */

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202000L
/* C2x does not require the second parameter for va_start. */
#define va_start(ap, ...) __builtin_va_start(ap, 0)
#else
/* Versions before C2x do require the second parameter. */
#define va_start(ap, param) __builtin_va_start(ap, param)
#endif
#define va_end(ap)          __builtin_va_end(ap)
#define va_arg(ap, type)    __builtin_va_arg(ap, type)

#if defined(NDEBUG)
    FILE* __log_file;
    #define LOG_START __log_file = fopen("","w"); 
    #define LOG
    #define LOG_END
#else 
    #define LOG_START 
    #define LOG
    #define LOG_END
#endif

#ifndef POINT
    #define POINT
    typedef struct point_t
    {
        idx_t    array_idx;
        float_t* data;
    } point_t;
#endif

typedef struct datapoint_info_t {
    /*
     * datapoint object 
     */
    heap_node_t* ngbh;      //heap object stores nearest neighbors of each point
    int is_center;          //flag signaling if a point is a cluster center
    int cluster_idx;        //cluster index
    idx_t array_idx;        //global index of the point in the dataset
    idx_t kstar;            //kstar value required for the density computation
    float_t g;              //density quantities, required by ADP the procedure
    float_t log_rho;        //
    float_t log_rho_c;      //
    float_t log_rho_err;    //
} datapoint_info_t;


typedef struct global_context_t 
{
    /*
     * context object to store info related to each 
     * MPI process
     */
    char processor_mame[MPI_MAX_PROCESSOR_NAME];    //processor name
    int __processor_name_len;                       //processor name len
    int world_size;  
    int mpi_rank;                                   //rank of the processor
    idx_t k;                                        //number of neighbors
    uint32_t dims;                                  //number of dimensions of the dataset
    float_t z;                                      //z parameter
	float_t* local_data;                            //pointer to the dataset stored into the node
	float_t* lb_box;                                //bounding box of the dataset
	float_t* ub_box;                                //bounding box of the dataset
    size_t n_points;                                //total number of points
    size_t idx_start;                               //starting index of the points on the processor
    size_t local_n_points;                          //number of points stored in the current processor
    datapoint_info_t*  local_datapoints;            //pointer to the datapoint properties
    idx_t* rank_idx_start;                          //starting index of datapoints in each processor
    idx_t* rank_n_points;                           //processor name
    idx_t* og_idxs;                                 //original indexes
    heap_node_t* __local_heap_buffers;              //buffer that stores nearest neighbors
	MPI_Comm mpi_communicator;                      //mpi communicator
    int input_data_in_float32;
    char input_data_file[DEFAULT_STR_LEN];
    char output_assignment_file[DEFAULT_STR_LEN];
    char output_data_file[DEFAULT_STR_LEN];
} global_context_t;

typedef struct pointset_t
{
    /*
     * Helper object to handle top kdtree 
     * construction, it represents the dataset
     * inside one node of the tree
     */
	size_t n_points;
	size_t __capacity;
	uint32_t dims;
	point_t* datapoints;
	float_t* lb_box;
	float_t* ub_box;
} pointset_t;


typedef struct lu_dynamic_array_t {
  idx_t *data;
  idx_t size;
  idx_t count;
} lu_dynamic_array_t;


void mpi_printf(global_context_t*, const char *fmt, ...);
void get_context(global_context_t*);
void print_global_context(global_context_t* );
void free_context(global_context_t* );
void free_pointset(pointset_t* );

void generate_random_matrix(float_t** ,int ,size_t ,global_context_t*);
void lu_dynamic_array_allocate(lu_dynamic_array_t * a);
void lu_dynamic_array_pushBack(lu_dynamic_array_t * a, idx_t p);
void lu_dynamic_array_Reset(lu_dynamic_array_t * a);
void lu_dynamic_array_reserve(lu_dynamic_array_t * a, idx_t n);
void lu_dynamic_array_init(lu_dynamic_array_t * a);
void print_error_code(int err);

void ordered_data_to_file(global_context_t* ctx, const char* fname);
void ordered_buffer_to_file(global_context_t* ctx, void* buffer, size_t el_size, uint64_t n, const char* fname);
void test_file_path(const char* fname);
void big_ordered_buffer_to_file(global_context_t* ctx, void* buffer, size_t el_size, uint64_t n, const char* fname);
float_t* read_data_file(global_context_t *ctx, const char *fname, const idx_t ndims, const int file_in_float32);
void get_dataset_diagnostics(global_context_t* ctx, float_t* data);

void test_distributed_file_path(global_context_t* ctx, const char* fname);
void distributed_buffer_to_file(global_context_t* ctx, void* buffer, size_t el_size, uint64_t n, const char* fname);
