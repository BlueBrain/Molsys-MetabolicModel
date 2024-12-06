#!/bin/bash
source /gpfs/bbp.cscs.ch/project/proj137/home/mandge/proj137venv/bin/activate

nrnivmodl mod
sbatch run.sbatch

