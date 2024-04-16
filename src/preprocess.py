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



def preprocess(subjectPath:str, subIdPosition:int, targetFS:int, 
               n_component:int, maxIter:int, IcaMethod:str, cutoffFreq:list):
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
                          cutoffFreq=cutoffFreq,
                          plot=True)
    ica.apply(data, verbose=False)

    # downsample & band pass filter
    data.resample(targetFS, verbose=False, n_jobs=-1) ; data.filter(1, 100, n_jobs=-1, verbose=False)

    # data.save(f'/home/zamanzad/trial1/data/icaPreprocessed/{subID}.fif', overwrite=True)

    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # positional Arguments (remember to delete --) TODO
    parser.add_argument("--paths", help="Address to your data")
    parser.add_argument("--subIdPosition",
            help="where subject IDs are positioned in paths")

    # optional arguments
    parser.add_argument("--targetFS", 
            help="Specify the desired sampling rate for resampling your data.")
    parser.add_argument("--n_component", 
            help="ICA n_components")
    parser.add_argument("--maxIter", 
            help="maximum number of iteration in the ICA algorithm")
    parser.add_argument("--IcaMethod",
            choices=["fastica", "infomax", "picard"],
            help="which ICA method to use")
    parser.add_argument("--cutoffFreq",
            help="Cutoff frequency for filtering data prior to feeding it into ICA.")
    
    args = parser.parse_args()

    
    if not args.targetFS : args.targetFS = config.targetFS
    if not args.n_component: args.n_component = config.n_component
    if not args.maxIter: args.maxIter = config.maxIter
    if not args.IcaMethod: args.IcaMethod = config.IcaMethod
    if not args.cutoffFreq: args.cutoffFreq = config.cutoffFreq

    basPath = "/home/smkia/Data/CamCAN/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/*/*.fif"
    dataPaths = glob(basPath)
    # loop over all of data 
    for count, subjectPath in enumerate(tqdm.tqdm(dataPaths[:])):
        preprocess(subjectPath,
            int(args.subIdPosition),
            int(args.targetFS),
            int(args.n_component),
            int(args.maxIter),
            args.IcaMethod,
            ast.literal_eval(args.cutoffFreq))

