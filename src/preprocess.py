import ast
import mne
import json
import tqdm
import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from preprocessUtils import AutoICA

import config

import warnings
warnings.filterwarnings('ignore')



def preprocess(subjectPath:str, subIdPosition:int, targetFS:int, n_component:int,
        maxIter:int, IcaMethod:str, cutoffFreqLow:float, cutoffFreqHigh:float):
    """
    This function perprocess MEG signal.
    """
    
    subID = subjectPath.split("/")[subIdPosition]

    # read the data
    data = mne.io.read_raw_fif(subjectPath,
                               verbose=False,
                               preload=True)

    # apply automated ICA
    ica = AutoICA.autoICA(data, 
                          n_components=n_component, # FLUX default
                          max_iter=maxIter, # FLUX default,
                          IcaMethod = IcaMethod,
                          cutoffFreq=[cutoffFreqLow, cutoffFreqHigh],
                          plot=True)
    ica.apply(data, verbose=False)

    # downsample & band pass filter
    data.resample(targetFS, verbose=False, n_jobs=-1) ; data.filter(1, 100, n_jobs=-1, verbose=False)
    print(data.get_data().shape)

    # data.save(f'/home/zamanzad/trial1/data/icaPreprocessed/{subID}.fif', overwrite=True)

    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # positional Arguments 
    parser.add_argument("--dir", 
            help="Address to your data")
    parser.add_argument("--subIdPosition",
            help="where subject IDs are positioned in paths")

    # optional arguments
    parser.add_argument("--targetFS", type=int,
            help="Specify the desired sampling rate for resampling your data.")
    parser.add_argument("--n_component", type=float,
            help="ICA n_components")
    parser.add_argument("--maxIter", type=int,
            help="maximum number of iteration in the ICA algorithm")
    parser.add_argument("--IcaMethod", type=str,
            choices=["fastica", "infomax", "picard"],
            help="which ICA method to use")
    parser.add_argument("--cutoffFreqHigh", type=int,
            help="Cutoff frequency for filtering data prior to feeding it into ICA.")
    parser.add_argument("--cutoffFreqLow", type=int,
            help="Cutoff frequency for filtering data prior to feeding it into ICA.")
    
    args = parser.parse_args()

    
    if not args.targetFS : args.targetFS = config.targetFS
    if not args.n_component: args.n_component = config.n_component
    if not args.maxIter: args.maxIter = config.maxIter
    if not args.IcaMethod: args.IcaMethod = config.IcaMethod
    if not args.cutoffFreqHigh: args.cutoffFreqHigh = config.cutoffFreqHigh
    if not args.cutoffFreqLow: args.cutoffFreqLow = config.cutoffFreqLow

#     # remove the following two lines
#     args.dir = "/home/smkia/Data/CamCAN/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/*/*.fif"
#     args.subIdPosition = -1

    dataPaths = glob(args.dir)
    # loop over all of data 
    for count, subjectPath in enumerate(tqdm.tqdm(dataPaths[:])):
        preprocess(subjectPath,
            args.subIdPosition,
            args.targetFS,
            args.n_component,
            args.maxIter,
            args.IcaMethod,
            args.cutoffFreqLow,
            args.cutoffFreqHigh)
        

