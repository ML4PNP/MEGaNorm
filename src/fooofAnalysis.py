
import mne
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob


import config
from processUtils import neuralParameterize
from dataManagementUtils import storeFooofModels


import warnings
warnings.filterwarnings('ignore')



## TODO: Separate functions for epoching and fooof
def fooof(data, freqRangeLow=3, freqRangeHigh=40, min_peak_height=0,
        peak_threshold=2, fs=1000, tmin=20, tmax=-20, segmentsLength=10, overlap=2,
        psdMethod="welch", psd_n_overlap=1, psd_n_fft=2) -> None: 
    """
    following steps are included in this function
    1. data segmentation
    2. croping data (using tmin and tmax)
    3. interpolate bad segments
    4. calculate power spectrum
    5. apply fooof algorithm to parametrize neural spectrum

    parameters
    -----------
    dataPaths: str
    data address

    save_path: str
    where to save data

    freqRangeLow: float
    Desired frequency range to run FOOOF

    freqRangeHigh: float
    Desired frequency range to run FOOOF






    returns
    ------------
    None
    """

    if freqRangeLow > 45:
        raise Exception("You Need to implement a notch filter for power line noise")


    
    # We exclude 20s from both begining and end of signals 
    # since participants usually open and close their eyes
    # in this time interval
    tmax = int(np.shape(data.get_data())[1]/fs + tmax)
    data.crop(tmin=tmin, tmax=tmax)
    
    segments = mne.make_fixed_length_epochs(data,
                                            duration=segmentsLength,
                                            overlap=overlap,
                                            reject_by_annotation=True,
                                            verbose=False)

    
    # interpolate bad segments
    segments = segments.load_data().interpolate_bads()
    
    # parametrizing neural spectrum        
    fooofModels, psds, freqs = neuralParameterize.fooofModeling(segments, 
                                                        freqRangeLow,
                                                        freqRangeHigh,
                                                        min_peak_height,
                                                        peak_threshold,
                                                        fs,
                                                        psdMethod,
                                                        psd_n_overlap,
                                                        psd_n_fft)


    return fooofModels, psds, freqs
            

        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # positional arguments (remember to delete --) TODO
    parser.add_argument("--dir", help="Address to your data")
    


    
    # optional arguments
    # fooof

    parser.add_argument("--freqRangeLow", type=float,
                help="Desired frequency range to run FOOOF (lower limit)")
    parser.add_argument("--freqRangeHigh", type=float,
                help="Desired frequency range to run FOOOF (higher limit)")
    parser.add_argument("--min_peak_height", type=int,
                help="Absolute threshold for detecting peaks")
    parser.add_argument("--peak_threshold", type=float,
                help="Relative threshold for detecting peaks")
    # segmentation
    parser.add_argument("--fs", type=int,
                help="sampling rate")
    parser.add_argument("--tmin", type=int,
                help="start time of the raw data to use in seconds")
    parser.add_argument("--tmax", type=int,
                help="end time of the raw data to use in seconds")
    parser.add_argument("--segmentsLength", type=int,
                help="length of MEG segments in seconds")
    parser.add_argument("--overlap", type=float,
                help="amount of overlap between MEG sigals segmentation")
    # psd
    parser.add_argument("-psdMethod", type=str,
                choices=["welch", "multitaper"],
                help="Spectral estimation method.")
    parser.add_argument("--psd_n_overlap", type=float,
                help="amount of overlap between windows in Welch's method")
    parser.add_argument("--psd_n_fft", type=float)


    args = parser.parse_args()

    # fooof
    if not args.freqRangeLow : args.freqRangeLow = config.freqRangeLow
    if not args.freqRangeHigh : args.freqRangeHigh = config.freqRangeHigh
    if not args.min_peak_height: args.min_peak_height = config.min_peak_height
    if not args.peak_threshold: args.peak_threshold = config.peak_threshold
    # segmentation
    if not args.fs: args.fs = config.fs
    if not args.tmin: args.tmin = config.tmin
    if not args.tmax: args.tmax = config.tmax
    if not args.overlap: args.overlap = config.overlap
    if not args.segmentsLength: args.segmentsLength = config.segmentsLength
    # psd
    if not args.psdMethod: args.psdMethod = config.psdMethod
    if not args.psd_n_fft: args.psd_n_fft = config.psd_n_fft
    if not args.psd_n_overlap: args.psd_n_overlap = config.psd_n_overlap


    # remove
    args.dir = "data/icaPreprocessed/*.fif"
    args.savePath = "data/fooofResults/fooofModels.pkl"
    dataPaths = glob(args.dir)

    for count, subjPath in enumerate(tqdm(dataPaths[:])):

        subjId = subjPath.split("/")[-1][:-4] # -1
        # read the data
        data = mne.io.read_raw_fif(subjPath,
                                preload=True,
                                verbose=False).pick(picks=["mag"])

        # parametrizing neural spectrum and saving the res
        fooofModels, psds, freqs = fooof(data = data,
                                        freqRangeLow = args.freqRangeLow,
                                        freqRangeHigh = args.freqRangeHigh,
                                        min_peak_height = args.min_peak_height,
                                        peak_threshold = args.peak_threshold,
                                        fs = args.fs,
                                        tmin = args.tmin,
                                        tmax = args.tmax,
                                        segmentsLength = args.segmentsLength,
                                        overlap = args.overlap,
                                        psdMethod = args.psdMethod,
                                        psd_n_overlap = args.psd_n_overlap,
                                        psd_n_fft = args.psd_n_fft)
        

            # save the results
        storeFooofModels(args.savePath, 
                        subjId, 
                        fooofModels,
                        psds,
                        freqs)

     













