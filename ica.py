import matplotlib.pyplot as plt
import numpy as np
import mne
from glob import glob 
from preprocessUtils import AutoICA
import tqdm
import json

import warnings
warnings.filterwarnings('ignore')


basPath = "/home/smkia/Data/CamCAN/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/*/*.fif"
dataPaths = glob(basPath)
print("Number of participants: ", len(dataPaths))


# # loop over all of data 
for count, subjectPath in enumerate(tqdm.tqdm(dataPaths[:])):
    
    subID = subjectPath.split("/")[-2]

    # read the data
    data = mne.io.read_raw_fif(subjectPath,
                               verbose=False,
                               preload=True)

    # apply automated ICA
    ica = AutoICA.autoICA(data, 
                          n_components=30, # FLUX default
                          max_iter=1000, # FLUX default
                          plot=True)
    ica.apply(data, verbose=False)


    # downsample & band pass filter
    data.resample(500, verbose=False, n_jobs=-1) ; data.filter(1, 100, n_jobs=-1, verbose=False)

    data.save(f'/home/zamanzad/trial1/data/icaPreprocessed/{subID}.fif', overwrite=True)
    
    with open("log.txt", "a") as logFile:
        logFile.write(subID)
    
    
    


