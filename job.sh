#!/bin/sh

#SBATCH --gres=gpu:tesla:0



if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi


conda activate mne


python fooofAnalysis.py  