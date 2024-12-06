#!/bin/bash
#SBATCH --ntasks=1000
#SBATCH --time=1:00:00
#SBATCH --mem=0
#SBATCH --partition=prod
#SBATCH --constraint=cpu
#SBATCH --account=proj137
#SBATCH --job-name=current

module load unstable py-mpi4py
#module load unstable neurodamus-neocortex-multiscale
module unload python

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

export DASK_DISTRIBUTED__LOGGING__DISTRIBUTED="info"
export DASK_DISTRIBUTED__WORKER__USE_FILE_LOCKING=False
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False  # don't spill to disk
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False  # don't spill to disk
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.80  # pause execution at 80% memory use
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95  # restart the worker at 95% use
export DASK_DISTRIBUTED__WORKER__MULTIPROCESSING_METHOD=spawn
export DASK_DISTRIBUTED__WORKER__DAEMON=True

# Reduce dask profile memory usage/leak (see https://github.com/dask/distributed/issues/4091)
export DASK_DISTRIBUTED__WORKER__PROFILE__INTERVAL=10000ms  # Time between statistical profiling queries
export DASK_DISTRIBUTED__WORKER__PROFILE__CYCLE=1000000ms  # Time between starting new profile

export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=200000ms  # Time for handshake
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=200000ms  # Time for handshake
export DASK_TEMPORARY_DIRECTORY=$TMPDIR

srun \
    emodel-generalisation -v compute_currents \
    --input-path  /gpfs/bbp.cscs.ch/project/proj137/NGVCircuits/rat_sscx_S1HL/V10/build/sonata/networks/nodes/All/nodes.h5 \
    --output-path nodes_new.h5 \
    --morphology-path /gpfs/bbp.cscs.ch/project/proj83/entities/fixed-ais-L23PC-2020-12-10/ascii \
    --hoc-path  /gpfs/bbp.cscs.ch/project/proj137/home/mandge/optimisation/release/v3/ \
    --only-rin \
    --parallel-lib dask_dataframe


# --parallel-lib multiprocessing
# --parallel-lib multiprocessing \