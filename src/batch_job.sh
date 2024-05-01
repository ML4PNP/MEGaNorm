#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p normal
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH -o /home/meganorm-smkia/temp/log/%j.out
#SBATCH -e /home/meganorm-smkia/temp/log/%j.err

source=$1
target=$2
subject=$3

source activate mne 

srun --job-name=$subject python ./src/mainParallel.py $source $target