#!/bin/sh

#SBATCH --gres=gpu:tesla:0



if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi


conda activate mne


# parallel coputation
python preprocess.py dir /home/smkia/Data/CamCAN/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/sub-CC410226/mf2pt2_sub-CC410226_ses-rest_task-rest_megtransdef.fif saveDir data/parallelData

