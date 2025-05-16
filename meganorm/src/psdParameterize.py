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


def computePsd(
    segments,
    freq_range_low=3,
    freq_range_high=40,
    sampling_rate=1000,
    psd_method="welch",
    psd_n_overlap=1,
    psd_n_fft=2,
    n_per_seg=2,
):
    """
    Compute the Power Spectral Density (PSD) of EEG/MEG data segments.

    Parameters
    ----------
    segments : mne.Epochs
        Segmented data for which PSD will be computed.
    freq_range_low : int
        Lower frequency bound for PSD calculation (Hz).
    freq_range_high : int
        Upper frequency bound for PSD calculation (Hz).
    sampling_rate : int
        Sampling rate of the data (Hz).
    psd_method : str
        Method for computing the PSD. Default is "welch".
    psd_n_overlap : int
        Overlap between segments (in seconds) for PSD calculation.
    psd_n_fft : int
        Number of FFT points used for the PSD calculation.
    n_per_seg : int
        Number of samples per segment used for computing PSD.

    Returns
    -------
    psds : np.ndarray
        Array of power spectral density values.
    freqs : np.ndarray
        Array of frequency values corresponding to the PSD.
    """

    psds, freqs = (
        segments.compute_psd(
            method=psd_method,
            fmin=freq_range_low,
            fmax=freq_range_high,
            n_jobs=-1,
            average="mean",
            n_overlap=psd_n_overlap * sampling_rate,
            n_fft=psd_n_fft * sampling_rate,
            n_per_seg=n_per_seg * sampling_rate,
            verbose=False,
        )
        .average()
        .get_data(return_freqs=True)
    )

    return psds, freqs


def parameterizePsd(
    psds,
    freqs,
    freq_range_low=3,
    freq_range_high=40,
    min_peak_height=0,
    peak_threshold=2,
    peak_width_limits=[1, 12.0],
    aperiodic_mode="fixed",
):
    """
    Fit a FOOOF model to power spectral density (PSD) data to separate
    periodic (oscillatory) and aperiodic (background) components.

    Parameters
    ----------
    psds : np.ndarray
        Power spectral density values.
    freqs : np.ndarray
        Frequency values corresponding to the PSD.
    freq_range_low : int
        Lower frequency bound for the FOOOF model (Hz).
    freq_range_high : int
        Upper frequency bound for the FOOOF model (Hz).
    min_peak_height : float
        Minimum height of peaks to be considered in the FOOOF model.
    peak_threshold : float
        Threshold for peak detection in the FOOOF model.
    peak_width_limits : list
        Limits on the width of peaks (in Hz).
    aperiodic_mode : str
        Mode for modeling the aperiodic component. Options are "fixed", "knee", or "none".

    Returns
    -------
    fooofModels : FOOOFGroup
        Fitted FOOOF group model containing periodic and aperiodic components.
    psds : np.ndarray
        Original power spectral density values.
    freqs : np.ndarray
        Frequency values corresponding to the PSD.
    """

    # Fit separate models for each channel
    fooofModels = f.FOOOFGroup(
        peak_width_limits=peak_width_limits,
        min_peak_height=min_peak_height,
        peak_threshold=peak_threshold,
        aperiodic_mode=aperiodic_mode,
    )
    fooofModels.fit(freqs, psds, [freq_range_low, freq_range_high], n_jobs=-1)

    return fooofModels, psds, freqs


def psdParameterize(
    segments,
    freq_range_low=3,
    freq_range_high=40,
    min_peak_height=0,
    peak_threshold=2,
    sampling_rate=1000,
    psd_method="welch",
    psd_n_overlap=1,
    psd_n_fft=2,
    n_per_seg=2,
    peak_width_limits=None,
    aperiodic_mode="knee",
):
    """
    Runs the complete pipeline for spectral parameterization using FOOOF.
    This includes computing the PSD and fitting FOOOF models for each channel.

    Parameters
    ----------
    segments : mne.Epochs
        Epoched MNE object containing segmented data.
    freq_range_low : float
        Lower bound of frequency range for PSD and FOOOF (Hz).
    freq_range_high : float
        Upper bound of frequency range for PSD and FOOOF (Hz).
    min_peak_height : float
        Minimum height of peaks to be detected by FOOOF.
    peak_threshold : float
        Threshold for peak detection relative to the aperiodic fit.
    sampling_rate : int
        Sampling frequency of the signal (Hz).
    psd_method : str
        Method used to compute PSD. Options: "welch", "multitaper".
    psd_n_overlap : int
        Overlap (in seconds) between segments in PSD computation.
    psd_n_fft : int
        Number of FFT points (in seconds) used in PSD.
    n_per_seg : int
        Length (in seconds) of each segment used in PSD.
    peak_width_limits : list of float, optional
        Lower and upper bounds on peak width (Hz). Default is [1, 12.0].
    aperiodic_mode : str
        Mode of aperiodic fit. Options: "fixed" or "knee".

    Returns
    -------
    fooofModels : FOOOFGroup
        Fitted FOOOF models for each channel.
    psds : np.ndarray
        Power spectral densities.
    freqs : np.ndarray
        Corresponding frequency values.

    Raises
    ------
    ValueError
        If `psd_method` is not 'welch' or 'multitaper'.
    ValueError
        If `aperiodic_mode` is not 'fixed' or 'knee'.
    """
    if peak_width_limits is None:
        peak_width_limits = [1, 12.0]

    if psd_method not in ["multitaper", "welch"]:
        raise ValueError("psd_method must be either 'welch' or 'multitaper'")

    if aperiodic_mode not in ["fixed", "knee"]:
        raise ValueError("aperiodic_mode must be either 'fixed' or 'knee'")

    psds, freqs = computePsd(
        segments=segments,
        freq_range_low=freq_range_low,
        freq_range_high=freq_range_high,
        sampling_rate=sampling_rate,
        psd_method=psd_method,
        psd_n_overlap=psd_n_overlap,
        psd_n_fft=psd_n_fft,
        n_per_seg=n_per_seg,
    )

    fooofModels, psds, freqs = parameterizePsd(
        psds=psds,
        freqs=freqs,
        freq_range_low=freq_range_low,
        freq_range_high=freq_range_high,
        min_peak_height=min_peak_height,
        peak_threshold=peak_threshold,
        peak_width_limits=peak_width_limits,
        aperiodic_mode=aperiodic_mode,
    )

    return fooofModels, psds, freqs
