#!/bin/bash

#SBATCH --job-name="Energy_Cost_Metabolism"
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=32

#SBATCH --account=proj137
#SBATCH --partition=prod
#SBATCH --constraint=cpu
#SBATCH --time=10:00:00

#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output="%x-%j.log"

# Here are the modules loaded when the simulation was initialized
# Feel free to edit this section
# -----------------------------------------------------------------
module load unstable
module load gmsh
spack env activate -d /gpfs/bbp.cscs.ch/data/scratch/proj137/farina/MR_Energy_Cost/spackenv

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# BEFORE SUBMITTING THIS SCRIPT TO THE SLURM QUEUE.    !
# please ensure that both the configuration and the    !
# environment of the simulation are sane by executing: !
#                                                      !
#               multiscale-run check                   !
#                                                      !
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

time srun --overlap dplace multiscale-run compute
