#!/bin/bash -l

#PBS -l ncpus=8
#PBS -l mem=4gb
#PBS -l walltime=00:40:00

module load python/3.7.4-gcccore-8.3.0
source $HOME/712/bin/activate
python3 $HOME/remote/clusters.py
