#!/bin/bash -l

#PBS -l ncpus=4
#PBS -l mem=2gb
#PBS -l walltime=00:30:00

module load python/3.7.4-gcccore-8.3.0
source $HOME/712/bin/activate
python3 $HOME/remote/freq_words.py