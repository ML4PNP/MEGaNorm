#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p normal
#SBATCH -o /home/meganorm-smkia/temp/log/%j.out

source=$1
target=$2

source activate mne 

srun python /home/meganorm-smkia/Code/MEGaNorm/src/mainParallel.py $source $target