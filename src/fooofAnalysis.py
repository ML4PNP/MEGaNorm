import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import numpy as np
import fooof as f
import mne
import h5py

from config.config import *
from processUtils import neuralParameterize, isNan
from dataManagementUtils import storeFooofModels, logFunc


import warnings
warnings.filterwarnings('ignore')




def fooof(dataPaths, savePath, freqRange, fs, powerNoise, tmin, tmax) -> None: 
    """
    This function will do the following steps
    1. 
    """
    subjId = subjPath.split("/")[-1][:-4]

    # read the data
    data = mne.io.read_raw_fif(subjPath,
                            preload=True,
                            verbose=False).pick(picks=["meg"])
    
    # We exclude 20s from both begining and end of signals 
    # since participants usually open and close their eyes
    # in this time interval
    tmax = int(np.shape(data.get_data())[1]/fs + tmax)
    data.crop(tmin=tmin, tmax=tmax)
    
    segments = mne.make_fixed_length_epochs(data,
                                            duration=10,
                                            overlap=2,
                                            reject_by_annotation=True,
                                            verbose=False
                                            )
    
    # mag ==> grad
    # segments.as_type(ch_type='grad', mode='fast')

    # rejecting noisy epchs (this part needs to be modified)
    # segments = segments.drop_bad(
    #                 reject=rejectCriteria, 
    #                 flat=flatCriteria, 
    #                 verbose=False,
    #                 )
    # segments.plot_drop_log()

    

    # if the data related to a person is entierly nan, 
    # exclude them
    # if isNan(np.nanmean(segments.get_data())): raise ValueError("NaN input")
    
    segments = segments.load_data().interpolate_bads()
    
    # parametrizing neural spectrum        
    fooofModels, psds, freqs = neuralParameterize.fooofModeling(segments, 
                                                        "welch", 
                                                        fs,
                                                        freqRange)

    # # save the results
    storeFooofModels(savePath, 
                    subjId, 
                    fooofModels,
                    psds,
                    freqs)
    print(30*"-")
            

        


if __name__ == "__main__":

    basPath = "data/icaPreprocessed/*.fif"
    dataPaths = glob(basPath)
    
    savePath = "data/fooofResults/fooofModels.pkl"

    freqRange :list = [3.0, 40.0]
    fs = 500 #sampling rate
    powerNoise = 50

    tmin, tmax = 20, -20

    for count, subjPath in enumerate(tqdm(dataPaths[:])):
        subjId = subjPath.split("/")[-1][:-4]


        # parametrizing neural spectrum and saving the res
        fooof(subjPath, savePath, freqRange, fs, powerNoise, tmin, tmax)
        logFunc("logs/fooofLog.txt", subjId)
        logFunc("logs/fooofLog2.txt", f"{count}_")

     
        












