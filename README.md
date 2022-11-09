This is my attempt to revive this old implementation of the convective parameterization for the Shallow Water model found in [here](https://github.com/aramirezreyes/RamirezReyes_ShallowWaterInFPlane). But this one was implemented on top of the great project [Dedalus](https://github.com/DedalusProject/dedalus), which is a python-oriented framework for PDE solving.

I moved to the other approach based on Julia and Oceananigans because:
1. The convective parameterization can be embarrasingly parallel, so the ease of access to the GPU from Julia/Oceananigans.jl was ideal.
1. My implementation was not the best so it is unbearably slow.

Nevertheless it should be possible to make it work.

# To instal in Cheyenne

It should be as easy as loading the following modules:

module load fftwmpi 
module load openmpi
module unload netcdf
module load hdf5

And then running the `script scripts/install_conda_cheyenne.sh`. 

Following the instructions of the all-conda installation in the Dedadalus webpage for version two does not work from out of the box in Cheyenne becase the conda-installed openmpi does not play well by default with Cheyenne's MPI.

To run a simulation, it is possible that you will need to load the same packages.


