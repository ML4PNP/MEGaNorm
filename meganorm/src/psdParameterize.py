import os
import sys
import json
import mne
import argparse
from tqdm import tqdm
from glob import glob
import fooof as f
from meganorm.utils.IO import make_config, storeFooofModels

import warnings

warnings.filterwarnings("ignore")


def computePsd(segments, freqRangeLow=3, freqRangeHigh=40, sampling_rate=1000, psdMethod="welch", 
               psd_n_overlap=1, psd_n_fft=2, n_per_seg=2):
    """
    Computes the Power Spectral Density (PSD) for the given segments of EEG/MEG data.

    Args:
        segments (mne.Epochs): Segmented data for which PSD will be computed.
        freqRangeLow (int): Lower frequency bound for PSD calculation (Hz).
        freqRangeHigh (int): Upper frequency bound for PSD calculation (Hz).
        sampling_rate (int): The sampling rate of the data (Hz).
        psdMethod (str): Method for computing the PSD. Default is "welch".
        psd_n_overlap (int): The overlap between segments (in seconds) for PSD calculation.
        psd_n_fft (int): The number of FFT points used for the PSD calculation.
        n_per_seg (int): The number of samples per segment used for computing PSD.

    Returns:
        psds (np.ndarray): Array of power spectral density values.
        freqs (np.ndarray): Array of frequency values corresponding to the PSD.
    """
    
    psds, freqs = segments.compute_psd( 
                    method=psdMethod,
                    fmin=freqRangeLow,
                    fmax=freqRangeHigh,
                    n_jobs=-1,
                    average="mean",
                    n_overlap=psd_n_overlap * sampling_rate, 
                    n_fft=psd_n_fft * sampling_rate, 
                    n_per_seg=n_per_seg * sampling_rate, 
                    verbose=False).average().get_data(return_freqs=True)
    
    return psds, freqs


def parameterizePsd(psds, freqs, freqRangeLow=3, freqRangeHigh=40, min_peak_height=0,
                    peak_threshold=2, peak_width_limits=[1, 12.0], aperiodic_mode="fixed"):
    """
    Fits a FOOOF model to the power spectral density (PSD) data to decompose 
    the signal into periodic and aperiodic components.

    Args:
        psds (np.ndarray): Power spectral density values.
        freqs (np.ndarray): Frequency values corresponding to the PSD.
        freqRangeLow (int): Lower frequency bound for the FOOOF model (Hz).
        freqRangeHigh (int): Upper frequency bound for the FOOOF model (Hz).
        min_peak_height (float): Minimum height of peaks to be considered in the FOOOF model.
        peak_threshold (float): Threshold for peak detection in the FOOOF model.
        peak_width_limits (list): Limits on the width of peaks (in Hz).
        aperiodic_mode (str): Mode to model the aperiodic component, options are "fixed", "knee", or "none".

    Returns:
        fooofModels (FOOOFGroup): Fitted FOOOF group model containing periodic and aperiodic components.
        psds (np.ndarray): Original power spectral density values.
        freqs (np.ndarray): Frequency values corresponding to the PSD.
    """
    
    # Fit separate models for each channel
    fooofModels = f.FOOOFGroup(peak_width_limits=peak_width_limits, 
                                min_peak_height=min_peak_height, 
                                peak_threshold=peak_threshold, 
                                aperiodic_mode=aperiodic_mode)
    fooofModels.fit(freqs, psds, [freqRangeLow, freqRangeHigh], n_jobs=-1)

    return fooofModels, psds, freqs


def psdParameterize(segments, freqRangeLow=3, freqRangeHigh=40, min_peak_height=0,
                    peak_threshold=2, sampling_rate=1000, psd_method="welch", psd_n_overlap=1, 
                    psd_n_fft=2, n_per_seg=2, peak_width_limits=[1, 12.0], aperiodic_mode="knee"):
    
    """
    Runs the complete pipeline for spectral parameterization using FOOOF. 
    This includes computing the PSD and fitting FOOOF models for each segment/channel.

    Args:
        segments (mne.Epochs): Epoched MNE object containing segmented data.
        freqRangeLow (float): Lower bound of frequency range for PSD and FOOOF (Hz).
        freqRangeHigh (float): Upper bound of frequency range for PSD and FOOOF (Hz).
        min_peak_height (float): Minimum height of peaks to be detected by FOOOF.
        peak_threshold (float): Threshold for peak detection relative to aperiodic fit.
        sampling_rate (int): Sampling frequency of the signal (Hz).
        psd_method (str): Method used to compute PSD. Options: "welch", "multitaper".
        psd_n_overlap (int): Overlap (in seconds) between segments in PSD computation.
        psd_n_fft (int): Number of FFT points (in seconds) used in PSD.
        n_per_seg (int): Length (in seconds) of each segment used in PSD.
        peak_width_limits (list): Lower and upper bounds on peak width (Hz).
        aperiodic_mode (str): Mode of aperiodic fit. Options: "fixed" or "knee".

    Returns:
        fooofModels (FOOOFGroup): Fitted FOOOF models for each channel.
        psds (ndarray): Power spectral densities.
        freqs (ndarray): Corresponding frequency values.
    """
    if psd_method not in ["multitaper", "welch"]:
        raise ValueError("psd_method must be either 'welch' or 'multitaper'")
    if aperiodic_mode not in ["fixed", "knee"]:
        raise ValueError("aperiodic_mode must be either 'knee' or 'fixed'")
        
    psds, freqs = computePsd(segments=segments, 
                            freqRangeLow=freqRangeLow, 
                            freqRangeHigh=freqRangeHigh, 
                            sampling_rate=sampling_rate, 
                            psd_method=psd_method, 
                            psd_n_overlap=psd_n_overlap, 
                            psd_n_fft=psd_n_fft,
                            n_per_seg=n_per_seg)

    # parametrizing neural spectrum
    fooofModels, psds, freqs = parameterizePsd(
        psds=psds,
        freqs=freqs,
        freqRangeLow=freqRangeLow,
        freqRangeHigh=freqRangeHigh,
        min_peak_height=min_peak_height,
        peak_threshold=peak_threshold,
        peak_width_limits=peak_width_limits,
        aperiodic_mode=aperiodic_mode,
    )

    return fooofModels, psds, freqs
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument("dir", help="Address to your data")
    parser.add_argument("saveDir", help="Address to save your data")
    # optional arguments
    parser.add_argument(
        "--configs", type=str, default=None, help="Address of configs json file"
    )
    args = parser.parse_args()

    # Loading configs
    if args.configs is not None:
        with open(args.configs, "r") as f:
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
        storeFooofModels(configs["savePath"], subjId, fooofModels, psds, freqs)
