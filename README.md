# dADP

contact: francesco.tomba@phd.units.it

# Usage
Clone the repository and compile with `make`
The compilation process produces an executable called `main`, run it with `mpirun`
The suggestion is to run it with one mpi task per socket.
 - -h --help show help message
 - -i --in-data        *(required)* path of the input file
 - -t --in-type        *(required)* datatype of the input file, allowed choices `f32`, `f64`
 - -d --in-dims        *(required)* number of dimensions of the data file, dadp expects something of the form N x d where N is inferred from the lenght of the data file
 - -o --out-data       *(optional)* output path for the data, the datafile is shuffled between mpi tasks and datapoints are ordered default is `out_data`
 - -a --out-assignment *(optional)* output path for the cluster assignment output ranges [0 ... Nc - 1] for core points halo points have indices [-Nc ... -1] conversion of idx for an halo point is `cluster_idx = -halo_idx - 1`, default is `out_assignment`

# Todo

 - [ ] argument parsing: find an elegant way to pass parameters and file (maybe a config file?)
 - [~] H1: implementation of lock free centers elimination (*work in progress*)
 - [ ] context: open all windows in a single shot, close them all togheter
 - [ ] io: curation of IO using mpi IO or other solutions 
 - [ ] kdtree: optimization an profiling
 - [ ] prettify overall stdout
 - [x] ~~arugment parser~~
 - [x] ~~H2: graph reduction~~
 - [x] ~~kdtree: implement slim heap~~

