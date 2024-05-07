#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p normal
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH -o /home/meganorm-smkia/temp/log/%x_%j.out
#SBATCH -e /home/meganorm-smkia/temp/log/%x_%j.err

source=$1
target=$2

source activate mne 

srun python ./src/mainParallel.py $source $target
