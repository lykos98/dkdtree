# Distributed K-D Tree for Parallel k-NN Search

This repository contains a massively parallel implementation of a k-d tree for performing efficient k-nearest neighbor (k-NN) searches on large-scale, high-dimensional datasets. The implementation is designed for distributed memory systems and leverages a hybrid MPI and OpenMP model to achieve high performance on multi-node, multi-core HPC systems.

## Description

Finding nearest neighbors in large datasets is a fundamental operation in many scientific applications. This project provides a scalable solution by distributing the k-NN problem across multiple compute nodes, allowing it to handle datasets that are too large to fit into a single machine's memory.

The core of this project is a two-level k-d tree structure:

1.  **Top-Level Tree**: A global, conceptual k-d tree is built collaboratively by all MPI processes. This "top tree" is not used for searching directly, but for domain decomposition—it partitions the entire data space and assigns a unique hyper-rectangular region to each process.
2.  **Local K-D Trees**: Once the data is distributed according to this decomposition, each MPI process builds its own highly optimized, in-memory k-d tree on its local subset of points. This local tree is used for fast nearest-neighbor queries within the process's domain.

The distributed k-NN search is a hybrid process that combines fast local searches with a targeted exchange of queries between processes, minimizing communication overhead by only contacting nodes whose domains could contain relevant neighbors.

## Features

-   **Massively Parallel**: Scales across thousands of cores using a hybrid MPI + OpenMP model.
-   **Scalable k-Nearest Neighbor Search**: Implements a distributed two-level k-d tree for efficient k-NN queries on massive datasets.
-   **Optimized Data Partitioning**: Employs a parallel, histogram-based median-finding algorithm to ensure a balanced domain decomposition.
-   **Hybrid Search Algorithm**: Combines fast, local k-d tree searches with an intelligent remote query mechanism to find globally-correct nearest neighbors.
-   **Large Dataset Support**: Ingests data from large binary files and distributes it across all processes at startup.
-   **Configurable Precision**: Supports both 32-bit (`float`) and 64-bit (`double`) floating-point precision via compile-time flags.

## Building the Project

The project uses a standard `Makefile`. You will need a C compiler (e.g., GCC, Clang) and an MPI implementation (e.g., OpenMPI, MPICH).

To compile the project, simply run:

```sh
make
```

## Usage

The program is executed using `mpirun`. All command-line arguments must be provided.

```sh
mpirun -np <num_processes> ./main [OPTIONS]
```

### Command-Line Arguments

| Short | Long               | Argument          | Description                                                                                                |
| :---- | :----------------- | :---------------- | :--------------------------------------------------------------------------------------------------------- |
| `-i`  | `--in-data`        | `<path>`          | **(Required)** Path to the input data file.                                                                |
| `-t`  | `--in-type`        | `f32` or `f64`    | **(Required)** The data type of the input file.                                                            |
| `-d`  | `--in-dims`        | `<int>`           | **(Required)** The number of dimensions of the dataset.                                                    |
| `-k`  | `--kngbh`          | `<int>`           | (Optional) The number of nearest neighbors (`k`) to find for each point. Defaults to `300`.                  |
| `-z`  | `--z`              | `<float>`         | (Optional) A user-defined floating-point parameter. Defaults to `3.0`.                                     |
| `-o`  | `--out-data`       | `<path>`          | (Optional) Path to save the shuffled, re-ordered dataset to. Defaults to `ordered_data.npy`.               |
| `-a`  | `--out-assignment` | `<path>`          | (Optional) Path to save the final output assignments to. Defaults to `final_assignment.npy`.               |
| `-h`  | `--help`           |                   | Shows the help message.                                                                                    |

### Example

```sh
# Run on 128 processes with a 10-dimensional, 64-bit float dataset
mpirun -np 128 ./main \
    --in-data /path/to/my/dataset.bin \
    --in-type f64 \
    --in-dims 10 \
    --kngbh 500 \
    --out-assignment /path/to/results/assignments.npy
```

### Hybrid MPI + OpenMP Execution Example

For optimal performance on multi-core, multi-socket HPC nodes, it's recommended to run in a hybrid model. This typically involves launching one MPI process per socket and using OpenMP to parallelize computations across the cores within that socket. This strategy maximizes memory bandwidth utilization for each process.

The following example is based on a SLURM script running on 2 nodes, with 2 sockets per node and 56 cores per socket.

1.  **Set OpenMP Environment Variables**: Configure OpenMP to use all available cores for each MPI task.

    ```sh
    # Set the number of threads for each MPI process
    export OMP_NUM_THREADS=56
    
    # Pin threads to cores and bind them closely
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    ```

2.  **Run with `mpirun` using socket mapping**: Use the `--map-by` option in `mpirun` to control process placement.

    ```sh
    # Total MPI tasks = 2 nodes * 2 tasks/node = 4
    # ppr:1:socket = Place 1 Process Per socket
    # PE=56        = Assign 56 Processing Elements (cores) to each process
    
    mpirun -n 4 --map-by ppr:1:socket:PE=56 \
        ./main -t f32 -i /path/to/data -d 5
    ```
This command launches four MPI processes in total. On each of the two nodes, it places one process on each of the two sockets. Each of these processes then spawns 56 OpenMP threads, fully utilizing the 56 cores available to it.

## Code Structure
-   `src/main/main.c`: The main driver application. Handles argument parsing and orchestrates the k-NN search.
-   `src/common/`: Contains common utilities, including MPI wrappers, data structures (`global_context_t`), and file I/O.
-   `src/tree/`: Contains the core logic for the distributed k-d tree and k-NN search.
    -   `tree.c`: Implements the high-level logic for building the top tree (`parallel_build_top_kdtree`), shuffling data (`exchange_points`), and coordinating the distributed k-NN search (`mpi_ngbh_search`).
    -   `kdtree.h`: A highly optimized, single-node parallel k-d tree used by each process for its local data.
    -   `heap.h`: A max-heap implementation used for efficient k-NN queries.

## Algorithm Workflow

1.  **Initialization**: The MPI environment is initialized. Process 0 parses command-line arguments and broadcasts the configuration.
2.  **Data Ingestion**: Process 0 reads the dataset from the input file and scatters it across all participating processes.
3.  **Top Tree Construction**: The processes collaboratively build a global "top tree" to create a spatial domain decomposition. This ensures the subsequent data shuffling is balanced.
4.  **Data Shuffling**: Using the top tree as a guide, points are globally exchanged (`MPI_Alltoallv`) so that each process's local data corresponds to its assigned spatial domain.
5.  **Local Tree Indexing**: Each process builds a local, in-memory k-d tree on its data subset. This step is parallelized with OpenMP.
6.  **Distributed k-NN Search**: The `mpi_ngbh_search` function finds the `k` nearest neighbors for every point. It first performs a local search and then queries other processes if a point's neighborhood sphere crosses partition boundaries.
7.  **Output**: The final results, such as the re-ordered dataset or derived assignments, are saved to the specified output files.
