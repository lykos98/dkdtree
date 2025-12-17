#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include "../common/common.h"
#include "../tree/tree.h"
#include <unistd.h>
#include <getopt.h>


#ifdef AMONRA
    #pragma message "Hi, you are on amonra"
    #define OUT_CLUSTER_ASSIGN "/beegfs/ftomba/phd/results/final_assignment.npy"
    #define OUT_DATA           "/beegfs/ftomba/phd/results/ordered_data.npy"
#endif

#ifdef LEONARDO
    #define OUT_CLUSTER_ASSIGN "/leonardo_scratch/large/userexternal/ftomba00/out_dadp/final_assignment.npy"
    #define OUT_DATA           "/leonardo_scratch/large/userexternal/ftomba00/out_dadp/ordered_data.npy"
#endif

#ifdef LUMI
    #define OUT_CLUSTER_ASSIGN "~/scratch_dadp/out_dadp/final_assignment.npy"
    #define OUT_DATA           "~/scratch_dadp/out_dadp/ordered_data.npy"
#endif

#ifndef  OUT_CLUSTER_ASSIGN
    #define OUT_CLUSTER_ASSIGN "final_assignment.npy"
    #define OUT_DATA           "ordered_data.npy"
#endif


#ifdef THREAD_FUNNELED
    #define THREAD_LEVEL MPI_THREAD_FUNNELED
#else
    #define THREAD_LEVEL MPI_THREAD_MULTIPLE
#endif


struct option long_options[] =
{
    {"in-data" , required_argument, 0, 'i'},
    {"in-dtype", required_argument, 0, 't'},
    {"in-dims" , required_argument, 0, 'd'},
    {"out-data", optional_argument, 0, 'o'},
    {"out-assignment", optional_argument, 0, 'a'},
    {"kngbh", optional_argument, 0, 'k'},
    {"z", optional_argument, 0, 'z'},
    {"help", optional_argument, 0, 'h'},
    {0, 0, 0, 0}
};

const char* help = "Distributed Advanced Density Peak\n"\
                   "    -h --help show this message\n"\
                   "    -i --in-data        (required) path of the input file\n"\
                   "    -t --in-type        (required) datatype of the input file, allowed choices `f32`, `f64`\n"\
                   "    -d --in-dims        (required) number of dimensions of the data file, dadp expects something\n"\
                   "                                   of the form N x d where N is inferred from the lenght of the\n"\
                   "                                   data file\n"\
                   "    -o --out-data       (optional) output path for the data, the datafile is shuffled between\n"\
                   "                                   mpi tasks and datapoints are ordered default is `out_data` \n"\
                   "    -a --out-assignment (optional) output path for the cluster assignment output ranges [0 ... Nc - 1]\n"\
                   "                                   for core points halo points have indices [-Nc ... -1] conversion\n"\
                   "                                   of idx for an halo point is cluster_idx = -halo_idx - 1, default is `out_assignment`\n"\
                   "    -k --kngbh          (optional) number of nearest neighbors to compute\n"\
                   "    -z --z              (optional) number of nearest neighbors to compute\n";

void parse_args(global_context_t* ctx, int argc, char** argv)
{
    int err = 0;
    int opt;
    int input_file_set = 0;
    int input_type_set = 0;
    snprintf(ctx -> output_assignment_file, DEFAULT_STR_LEN, "%s", OUT_CLUSTER_ASSIGN);
    snprintf(ctx -> output_data_file, DEFAULT_STR_LEN, "%s", OUT_DATA);

    while((opt = getopt_long(argc, argv, "i:t:d:o:a:k:z:h", long_options, NULL)) != -1)
    {
        switch(opt)
        {
            case 'i':
                strncpy(ctx -> input_data_file, optarg, DEFAULT_STR_LEN);
                input_file_set = 1;
                break;
            case 't':
                ctx -> input_data_in_float32 = -1;
                if(strncmp(optarg, "f32", DEFAULT_STR_LEN) == 0) ctx -> input_data_in_float32 = MY_TRUE; 
                if(strncmp(optarg, "f64", DEFAULT_STR_LEN) == 0) ctx -> input_data_in_float32 = MY_FALSE; 
                if(ctx -> input_data_in_float32 == -1)
                {
                    fprintf(stderr, "Invalid option of datatype, allowed are f32, f64\n");
                    exit(1);
                }
                input_type_set = 1;
                break;
            case 'd':
                ctx -> dims = atoi(optarg);
                if(ctx -> dims < 0)
                {
                    fprintf(stderr, "Invaild number of dimensions\n");
                    exit(1);
                }
                break;
            case 'o':
                strncpy(ctx -> output_data_file, optarg, DEFAULT_STR_LEN);
                break;
            case 'a':
                strncpy(ctx -> output_assignment_file, optarg, DEFAULT_STR_LEN);
                break;
            case 'k':
                ctx -> k = atoi(optarg);
                break;
            case 'z':
                ctx -> z = atof(optarg);
                break;
            case 'h':
                mpi_printf(ctx, "%s\n", help);
                MPI_Finalize();
                exit(0);

            default:
                mpi_printf(ctx, "%s\n", help);
                MPI_Finalize();
                exit(0);
                break;
        }
    }

    if(ctx -> dims == 0){mpi_printf(ctx,"Please provide number of dimensions with -d\n"); ++err;}; 
    if(!input_file_set){mpi_printf(ctx,"Please provide input file with -i\n"); ++err;};
    if(!input_type_set){mpi_printf(ctx,"Please provide input type with -t\n"); ++err;};
    if(err){MPI_Finalize(); exit(1);};
}

void print_hello(global_context_t* ctx)
{
    char * hello =  ""
                    "     _           _          \n"
                    "  __| | __ _  __| |_ __     \n"
                    " / _` |/ _` |/ _` | '_ \\   \n"
                    "| (_| | (_| | (_| | |_) |   \n"
                    " \\__,_|\\__,_|\\__,_| .__/ \n"
                    "                  |_|       \n"
                    " Distributed Advanced Density Peak";
    mpi_printf(ctx, "%s\n\n", hello);

    #if defined (_OPENMP)
        mpi_printf(ctx,"Running Hybrid (Openmp + MPI) code\n");
    #else
        mpi_printf(ctx,"Running pure MPI code\n");
    #endif

    #if defined (THREAD_FUNNELED)
        mpi_printf(ctx,"/!\\ Code built with MPI_THREAD_FUNNELED level\n");
    #else
        mpi_printf(ctx,"/!\\ Code built with MPI_THREAD_MULTIPLE level\n");
    #endif

    mpi_printf(ctx, "\nConfigs: \n");
    mpi_printf(ctx, "Data file  .............> %s\n", ctx -> input_data_file);
    mpi_printf(ctx, "Input Type .............> float%d\n", ctx -> input_data_in_float32 ? 32 : 64);
    mpi_printf(ctx, "Dimensions .............> %d\n", ctx -> dims);
    mpi_printf(ctx, "Output data file .......> %s\n", ctx -> output_data_file);
    mpi_printf(ctx, "Output assignment file -> %s\n", ctx -> output_assignment_file);
    mpi_printf(ctx, "k ......................> %lu\n", ctx -> k);
    mpi_printf(ctx, "Z ......................> %.2lf\n", ctx -> z);
    mpi_printf(ctx, "\nRUNNING!\n");

}

int main(int argc, char** argv) {
    #if defined (_OPENMP)
        int mpi_provided_thread_level;
        MPI_Init_thread( &argc, &argv, THREAD_LEVEL, &mpi_provided_thread_level);
        if ( mpi_provided_thread_level < THREAD_LEVEL ) 
        {
            switch(THREAD_LEVEL)
            {
                case MPI_THREAD_FUNNELED:
                    printf("a problem arise when asking for MPI_THREAD_FUNNELED level\n");
                    MPI_Finalize();
                    exit( 1 );
                    break;
                case MPI_THREAD_SERIALIZED:
                    printf("a problem arise when asking for MPI_THREAD_SERIALIZED level\n");
                    MPI_Finalize();
                    exit( 1 );
                    break;
                case MPI_THREAD_MULTIPLE:
                    printf("a problem arise when asking for MPI_THREAD_MULTIPLE level\n");
                    MPI_Finalize();
                    exit( 1 );
                    break;
            }
        }
    #else
        MPI_Init(NULL, NULL);
    #endif


    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
	
	global_context_t ctx;

	ctx.mpi_communicator = MPI_COMM_WORLD;
	get_context(&ctx);


	/*
	 * Mock reading some files, one for each processor
	 */

	int d = 5;
    int k = 300;
	
	float_t* data;
    
    //parse command line
    
    parse_args(&ctx, argc, argv);
    print_hello(&ctx);
	/*
	 * Generate a random matrix of lenght of some kind
	 */


	if(ctx.mpi_rank == 0)
	{
		simulate_master_read_and_scatter(5, 1000000, &ctx);	
	}
	else
	{
		simulate_master_read_and_scatter(0, 0, &ctx);	
	}

	//free(data);
	free_context(&ctx);
    MPI_Finalize();
}



void simulate_master_read_and_scatter(int dims, size_t n, global_context_t *ctx) 
{
    float_t *data;
    TIME_DEF
    double elapsed_time;

    int halo = MY_FALSE;
    float_t tol = 0.002;

    if(I_AM_MASTER && ctx -> world_size <= 6)
    {
        test_file_path(ctx -> output_data_file);
        test_file_path(ctx -> output_assignment_file);
    }
    else
    {
        test_distributed_file_path(ctx, ctx -> output_data_file);
        test_distributed_file_path(ctx, ctx -> output_assignment_file);
    }
    

    TIME_START;
    if (ctx->mpi_rank == 0) 
    {
        data = read_data_file(ctx, ctx -> input_data_file, ctx -> dims, ctx -> input_data_in_float32);
        get_dataset_diagnostics(ctx, data);
    }
    
    /* communicate the total number of points*/
    MPI_Bcast(&(ctx->dims), 1, MPI_UINT32_T, 0, ctx->mpi_communicator);
    MPI_Bcast(&(ctx->n_points), 1, MPI_UINT64_T, 0, ctx->mpi_communicator);

    /* compute the number of elements to recieve for each processor */
    idx_t *send_counts = (idx_t *)MY_MALLOC(ctx->world_size * sizeof(idx_t));
    idx_t *displacements = (idx_t *)MY_MALLOC(ctx->world_size * sizeof(idx_t));

    displacements[0] = 0;
    send_counts[0] = ctx->n_points / ctx->world_size;
    send_counts[0] += (ctx->n_points % ctx->world_size) > 0 ? 1 : 0;
    send_counts[0] = send_counts[0] * ctx->dims;

    for (int p = 1; p < ctx->world_size; ++p) 
    {
        send_counts[p] = (ctx->n_points / ctx->world_size);
        send_counts[p] += (ctx->n_points % ctx->world_size) > p ? 1 : 0;
        send_counts[p] = send_counts[p] * ctx->dims;
        displacements[p] = displacements[p - 1] + send_counts[p - 1];
    }


    ctx->local_n_points = send_counts[ctx->mpi_rank] / ctx->dims;

    float_t *pvt_data = (float_t *)MY_MALLOC(send_counts[ctx->mpi_rank] * sizeof(float_t));


    if(I_AM_MASTER)
    {
        memcpy(pvt_data, data, ctx -> dims * ctx -> local_n_points * sizeof(float_t));
        int already_sent_points = 0;
        for(int i = 1; i < ctx -> world_size; ++i)
        {
            already_sent_points = 0;
            while(already_sent_points < send_counts[i])
            {
                int count_send = MIN(DEFAULT_MSG_LEN, send_counts[i] - already_sent_points); 
                MPI_Send(data + displacements[i] + already_sent_points, count_send, MPI_MY_FLOAT, i, ctx -> mpi_rank, ctx -> mpi_communicator);
                already_sent_points += count_send;
            }
        }
    }
    else
    {
        int already_recvd_points = 0;
        while(already_recvd_points < send_counts[ctx -> mpi_rank])
        {
            MPI_Status status;
            MPI_Probe(0, MPI_ANY_TAG, ctx -> mpi_communicator, &status);

            MPI_Request request;
            int count_recv; 
            int source = status.MPI_SOURCE;
            MPI_Get_count(&status, MPI_MY_FLOAT, &count_recv);

            MPI_Recv(pvt_data + already_recvd_points, count_recv, MPI_MY_FLOAT, source, MPI_ANY_TAG, ctx -> mpi_communicator, MPI_STATUS_IGNORE);
            already_recvd_points += count_recv;
        }
    }

    elapsed_time = TIME_STOP;
    LOG_WRITE("Importing file ad scattering", elapsed_time);

    if (I_AM_MASTER) free(data);

    ctx->local_data = pvt_data;

    int k_local  = 20;
    int k_global = 20;

    uint64_t *global_bin_counts_int = (uint64_t *)MY_MALLOC(k_global * sizeof(uint64_t));


    top_kdtree_t tree;
    TIME_START;
    top_tree_init(ctx, &tree);
    elapsed_time = TIME_STOP;
    LOG_WRITE("Initializing global kdtree", elapsed_time);

    TIME_START;
    //build_top_kdtree(ctx, &original_ps, &tree, tol);
    build_top_kdtree(ctx, &tree, k_global, tol);
    //parallel_build_top_kdtree(ctx, &tree, 0.001, 512);
    elapsed_time = TIME_STOP;
    LOG_WRITE("Top kdtree build", elapsed_time);

    TIME_START;
    exchange_points(ctx, &tree);
    elapsed_time = TIME_STOP;
    LOG_WRITE("Top kdtree build and domain decomposition", elapsed_time);

#ifdef WRITE_SHUFFLED_DATA
    distributed_buffer_to_file(ctx, ctx->local_data, 5*sizeof(float_t), ctx->local_n_points, "bb/ordered_data");
#endif

    TIME_START;
    kdtree_t local_tree;
    kdtree_initialize( &local_tree, ctx -> local_data, ctx -> local_n_points, (unsigned int)ctx -> dims);
    datapoint_info_t* dp_info = (datapoint_info_t*)MY_MALLOC(ctx -> local_n_points * sizeof(datapoint_info_t));            
    for(uint64_t i = 0; i < ctx -> local_n_points; ++i)
    {
        dp_info[i].ngbh = NULL;
        dp_info[i].g = 0.f;
        dp_info[i].log_rho = 0.f;
        dp_info[i].log_rho_c = 0.f;
        dp_info[i].log_rho_err = 0.f;
        dp_info[i].array_idx = -1;
        dp_info[i].kstar = -1;
        dp_info[i].is_center = -1;

        dp_info[i].cluster_idx = -1717171717;
    }
    ctx -> local_datapoints = dp_info;
    build_local_tree(ctx, &local_tree);
    elapsed_time = TIME_STOP;
    LOG_WRITE("Local trees init and build", elapsed_time);

    TIME_START;
    MPI_DB_PRINT("----- Performing ngbh search -----\n");
    MPI_Barrier(ctx -> mpi_communicator);

    mpi_ngbh_search(ctx, dp_info, &tree, &local_tree, ctx -> local_data, ctx -> k);

    MPI_Barrier(ctx -> mpi_communicator);
    elapsed_time = TIME_STOP;
    LOG_WRITE("Total time for all knn search", elapsed_time)

    top_tree_free(ctx, &tree);
    kdtree_free(&local_tree);

    free(send_counts);
    free(displacements);
    //free(dp_info);
    
    free(global_bin_counts_int);
}
