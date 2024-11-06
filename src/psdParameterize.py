
import os
import sys
import json
import mne
import argparse
from tqdm import tqdm
from glob import glob
import fooof as f

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'utils')
sys.path.append(config_path)
from IO import make_config, storeFooofModels

import warnings
warnings.filterwarnings('ignore')





def computePsd(segments, freqRangeLow=3, freqRangeHigh=40, sampling_rate=1000, psdMethod="welch", psd_n_overlap=1, psd_n_fft=2, n_per_seg=2):
    """
    compute power spectrum
    """

    psds, freqs = segments.compute_psd( 
                    method = psdMethod,
                    fmin = freqRangeLow,
                    fmax = freqRangeHigh,
                    n_jobs = -1,
                    average = "mean",
                    # having at least two cycles of 1Hz activity
                    n_overlap = psd_n_overlap*sampling_rate, 
                    n_fft = psd_n_fft*sampling_rate, 
                    n_per_seg = n_per_seg*sampling_rate, 
                    verbose=False).average().get_data(return_freqs=True)
    
    return psds, freqs
    

def parameterizePsd(psds, freqs, freqRangeLow=3, freqRangeHigh=40, min_peak_height=0,
                    peak_threshold=2, peak_width_limits=[1, 12.0], aperiodic_mode="fixed"):

    """
    fooofModeling fit multiple models (group fooof) and returns 
    periodic and aperiodic signals

    parameters
    --------------
    segments: ndarray
    segmented data

    spectrumMethod: str
    the method to be used to calculate power spectrum

    fs:int
    sampling rate
    
    freqRange: list[int]
    desired frequency range to be modeled by fooof model
    default: [1, 60]

    return
    --------------
    fooofModels: object
    """  

    # fitting seperate models for each channel
    fooofModels = f.FOOOFGroup(peak_width_limits=peak_width_limits, 
                                min_peak_height=min_peak_height, 
                                peak_threshold=peak_threshold, 
                                aperiodic_mode=aperiodic_mode)
    fooofModels.fit(freqs, psds, [freqRangeLow, freqRangeHigh], n_jobs=-1)

    return fooofModels, psds, freqs


def psdParameterize(segments, freqRangeLow=3, freqRangeHigh=40, min_peak_height=0,
        peak_threshold=2, sampling_rate=1000, psdMethod="welch", psd_n_overlap=1, 
        psd_n_fft=2, n_per_seg=2, peak_width_limits=[1, 12.0], aperiodic_mode="fixed") -> None: 
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

    psds, freqs = computePsd(segments=segments, 
                            freqRangeLow=freqRangeLow, 
                            freqRangeHigh=freqRangeHigh, 
                            sampling_rate=sampling_rate, 
                            psdMethod=psdMethod, 
                            psd_n_overlap=psd_n_overlap, 
                            psd_n_fft=psd_n_fft,
                            n_per_seg=n_per_seg)

    # parametrizing neural spectrum        
    fooofModels, psds, freqs = parameterizePsd(psds=psds, 
                                            freqs=freqs, 
                                            freqRangeLow=freqRangeLow, 
                                            freqRangeHigh=freqRangeHigh, 
                                            min_peak_height=min_peak_height,
                                            peak_threshold=peak_threshold,
                                            peak_width_limits=peak_width_limits,
                                            aperiodic_mode=aperiodic_mode)

    return fooofModels, psds, freqs


            

        



        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument("dir", help="Address to your data")
    parser.add_argument("saveDir", help="Address to save your data")
    # optional arguments
    parser.add_argument("--configs", type=str, default=None,
        help="Address of configs json file")
    args = parser.parse_args()


    # Loading configs
    if args.configs is not None:
        with open(args.configs, 'r') as f:
            configs = json.load(f)
    else: configs = make_config()


    dataPaths = glob(args.dir)

    for count, subjPath in enumerate(tqdm(dataPaths[:])):

        subjId = subjPath.split("/")[-1][:-4] 

        # read the data
        data = mne.io.read_raw_fif(subjPath,
                                preload=True,
                                verbose=False)


        # parametrizing neural spectrum and saving the res
        fooofModels, psds, freqs = psdParameterize(data = data,
                                        freqRangeLow = configs["freqRangeLow"],
                                        freqRangeHigh = configs["freqRangeHigh"],
                                        min_peak_height = configs["min_peak_height"],
                                        peak_threshold = configs["peak_threshold"],
                                        fs = configs["fs"],
                                        tmin = configs["tmin"],
                                        tmax = configs["tmax"],
                                        segmentsLength = configs["segmentsLength"],
                                        overlap = configs["overlap"],
                                        psdMethod = configs["psdMethod"],
                                        psd_n_overlap = configs["psd_n_overlap"],
                                        psd_n_fft = configs["psd_n_fft"],
                                        n_per_seg = configs["n_per_seg"],
                                        peak_width_limits = configs["peak_width_limits"])
        

        # save the results
        storeFooofModels(configs["savePath"], 
                        subjId, 
                        fooofModels,
                        psds,
                        freqs)

     













