#!/bin/sh

#SBATCH --gres=gpu:tesla:0



if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi


conda activate mne


# preprocess
python preprocess.py dir /home/smkia/Data/CamCAN/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/*/*.fif subIdPosition -1

# fooofAnalysis
# python fooofAnalysis.py  basPath data/icaPreprocessed/*.fif subIdPosition -1 savePath data/fooofResults/fooofModels.pkl