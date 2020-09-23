#!/bin/bash

#PBS -l ncpus=8
#PBS -l mem=16gb
#PBS -l walltime=3:00:00

source /etc/profile.d/modules.sh
module load python/3.7.4-gcccore-8.3.0
source $HOME/712/bin/activate
python3 main.py
