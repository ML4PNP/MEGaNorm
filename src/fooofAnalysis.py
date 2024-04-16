import ast
import mne
import h5py
import argparse
import fooof as f
import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

import config
from processUtils import neuralParameterize, isNan
from dataManagementUtils import storeFooofModels, logFunc


import warnings
warnings.filterwarnings('ignore')




def fooof(dataPaths, savePath, freqRange, fs, tmin, tmax, segmentsLength, overlap) -> None: 
    """
    This function will do the following steps
    1. 
    """

    if freqRange[1] > 45:
        raise Exception("You Need to implement a notch filter for power line noise")


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
                                            duration=segmentsLength,
                                            overlap=overlap,
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

    parser = argparse.ArgumentParser()

    # positional arguments (remember to delete --) TODO
    parser.add_argument("--paths", help="Address to your data")
    parser.add_argument("--subIdPosition",
            help="where subject IDs are positioned in paths")
    parser.add_argument("--savePath",
            help="where to save results")

    # optional arguments
    parser.add_argument("--freqRangeLow", 
                help="Desired frequency range to run FOOOF (lower limit)")
    parser.add_argument("--freqRangeHigh", 
                help="Desired frequency range to run FOOOF (higher limit)")
    parser.add_argument("--fs",
                help="sampling rate")
    parser.add_argument("--tmin",
                help="start time of the raw data to use in seconds")
    parser.add_argument("--tmax",
                help="end time of the raw data to use in seconds")
    parser.add_argument("--segmentsLength",
                help="length of MEG segments in seconds")
    parser.add_argument("--overlap",
                help="amount of overlap between MEG sigals")

    args = parser.parse_args()

    if not args.freqRangeLow : args.freqRangeLow = config.freqRange[0]
    if not args.freqRangeHigh : args.freqRangeHigh = config.freqRange[1]
    if not args.fs: args.fs = config.fs
    if not args.tmin: args.tmin = config.tmin
    if not args.tmax: args.tmax = config.tmax
    if not args.segmentsLength: args.segmentsLength = config.segmentsLength
    if not args.overlap: args.overlap = config.overlap


    # remember to remove the following two lines
    basPath = "data/icaPreprocessed/*.fif"
    savePath = "data/fooofResults/fooofModels.pkl"

    dataPaths = glob(basPath)

    for count, subjPath in enumerate(tqdm(dataPaths[:])):
        subjId = subjPath.split("/")[args.subIdPosition][:-4] # -1

        # parametrizing neural spectrum and saving the res
        fooof(subjPath,
        savePath,
        int(args.freqRange),
        int(args.fs),
        int(args.tmin),
        int(args.tmax),
        int(args.segmentsLength),
        int(args.overlap))


     
        












