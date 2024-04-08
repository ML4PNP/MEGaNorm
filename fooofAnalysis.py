import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import numpy as np
import fooof as f
import mne
import h5py

from config.config import *
from processUtils import neuralParameterize, isNan
from dataManagementUtils import storeFooofRes, logFunc


import warnings
warnings.filterwarnings('ignore')




def fooof(dataPaths, savePath, freqRange, fs, powerNoise) -> None: 
    """
    This function will do the following steps
    1. 
    """

    

    subjId = subjPath.split("/")[-1][:-4]

    # read the data
    data = mne.io.read_raw_fif(subjPath,
                            preload=True,
                            verbose=False).pick(picks=["meg"])


    data.notch_filter(powerNoise, n_jobs=-1, verbose=False)

    segments = mne.make_fixed_length_epochs(data,
                                            duration=10,
                                            overlap=0,
                                            reject_by_annotation=False,
                                            proj=False,
                                            verbose=False
                                            )
    

    
    # mag ==> grad
    # segments.as_type(ch_type='grad', mode='fast')

    # rejecting noisy epchs
    segments = segments.drop_bad(
                    reject=rejectCriteria, 
                    flat=flatCriteria, 
                    verbose=False,
                    )
    

    # if the data related to a person is entierly nan, 
    # exclude them
    if isNan(np.nanmean(segments)): raise ValueError("NaN input")
    
    segments = segments.load_data().interpolate_bads().average().get_data()

    
        
    
    # parametrizing neural spectrum        
    (periodicsPeaks ,
    aperiodics,
    periodic,
    freqs,
    aperiodicParams, 
    periodicParams) = neuralParameterize.fooofModeling(segments, 
                                            "welch", 
                                            fs,
                                            freqRange)
    

    # save the results
    storeFooofRes(savePath, 
                    subjId, 
                    periodicsPeaks, 
                    aperiodics, 
                    periodic,
                    freqs, 
                    aperiodicParams, 
                    periodicParams)

    

    print(30*"-")
            

        


if __name__ == "__main__":

    basPath = "data/icaPreprocessed/*.fif"
    dataPaths = glob(basPath)
    
    savePath = "data/fooofResults.h5"

    freqRange :list = [3.0, 40.0]
    fs = 500 #sampling rate
    powerNoise = 50

    for count, subjPath in enumerate(tqdm(dataPaths[:])):
        subjId = subjPath.split("/")[-1][:-4]

        try:
        # parametrizing neural spectrum and saving the res
            fooof(subjPath, savePath, freqRange, fs, powerNoise)
            logFunc("logs/fooofLog.txt", subjId)
            logFunc("logs/fooofLog2.txt", f"{count}_")
        except ValueError: 
            
            logFunc("logs/excludedSubj.txt", subjId)
            logFunc("logs/fooofLog2.txt", f"{count}_")
        break
        











