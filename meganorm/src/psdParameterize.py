import mne
import numpy as np
import fooof as f
from pyrasa.irasa import irasa
from pyrasa.irasa_mne.mne_objs import (
    AperiodicEpochsSpectrum,
    IrasaEpoched,
    PeriodicEpochsSpectrum,
)

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


def fooof(
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


def parameterize_psds(
    segments,
    parametrization_method,
    freq_range_low=3,
    freq_range_high=40,
    min_peak_height=0,
    peak_threshold=2,
    sampling_rate=1000,
    psd_method="welch",
    psd_n_overlap=1,
    psd_n_fft=2,
    n_per_seg=2,
    peak_width_limits=[1, 12.0],
    aperiodic_mode="knee",
    irasa_hset = (1.05, 2.0, 0.05)
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
    spectral_models : FOOOFGroup | pyrasa.irasa_mne.mne_objs.IrasaEpoched
        Fitted spectral models for each channel.
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
    if psd_method not in ["multitaper", "welch"]:
        raise ValueError("psd_method must be either 'welch' or 'multitaper'")

    if aperiodic_mode not in ["fixed", "knee"]:
        raise ValueError("aperiodic_mode must be either 'fixed' or 'knee'")
    
    if parametrization_method == "fooof":

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

        spectral_models, psds, freqs = fooof(
            psds=psds,
            freqs=freqs,
            freq_range_low=freq_range_low,
            freq_range_high=freq_range_high,
            min_peak_height=min_peak_height,
            peak_threshold=peak_threshold,
            peak_width_limits=peak_width_limits,
            aperiodic_mode=aperiodic_mode,
        )

    elif parametrization_method == "irasa":
        psds, freqs, spectral_models = irasa_epochs(
            segments,
            band=(freq_range_low, freq_range_high),
            hset_info=irasa_hset,
            )

    return spectral_models, psds, freqs



def irasa_epochs(
    data: mne.Epochs,
    band: tuple[float, float] = (1.0, 100.0),
    hset_info: tuple[float, float, float] = (1.05, 2.0, 0.05),
) -> IrasaEpoched:
    """
    Separate aperiodic from periodic power spectra using the IRASA algorithm for Epochs data.

    This function applies the Irregular Resampling Auto-Spectral Analysis (IRASA) algorithm
    as described by Wen & Liu (2016) to decompose the power spectrum of neurophysiological
    signals into aperiodic (fractal) and periodic (oscillatory) components. It is specifically
    designed for time-series data in `mne.Epochs` format, making it suitable for event-related
    EEG/MEG analyses.

    Parameters
    ----------
    data : mne.Epochs
        The time-series data used to extract aperiodic and periodic power spectra.
        This should be an instance of `mne.Epochs`.
    band : tuple of (float, float), optional, default: (1.0, 100.0)
        A tuple specifying the lower and upper bounds of the frequency range (in Hz) used
        for extracting the aperiodic and periodic spectra.
    hset_info : tuple of (float, float, float), optional, default: (1.05, 2.0, 0.05)
        Contains the range of up/downsampling factors used in the IRASA algorithm.
        This should be a tuple specifying the (min, max, step) values for the resampling.

    Returns
    -------
    psd_list_original: np.array
        Original power spectrum.
    aperiodic : AperiodicEpochsSpectrum
        The aperiodic component of the data as an `AperiodicEpochsSpectrum` object.
    periodic : PeriodicEpochsSpectrum
        The periodic component of the data as a `PeriodicEpochsSpectrum` object.

    Note
    ---------
    This code is driven and modified from PYRASA: 
    https://github.com/schmidtfa/pyrasa/blob/afb003444131f97d3221abb6d338384d92c12e29/pyrasa/irasa_mne/irasa_mne.py
    """

    # set parameters & safety checks
    # ensure that input data is in the right format
    assert isinstance(data, mne.BaseEpochs), 'Data should be of type mne.BaseEpochs'
    assert (
        data.info['bads'] == []
    ), 'Data should not contain bad channels as this might mess up the creation of the returned data structure'

    info = data.info.copy()
    fs = data.info['sfreq']

    data_array = data.get_data(copy=True)

    nfft = 2 ** (np.ceil(np.log2(int(data_array.shape[2] * np.max(hset_info)))))

    kwargs_psd = {
        'nperseg': None,
        'nfft': nfft,
        'noverlap': 0,
    }

    psd_list_aperiodic, psd_list_periodic, psd_list_original = [], [], []
    for epoch in data_array:
        irasa_spectrum = irasa(
            epoch,
            fs=fs,
            band=band,
            filter_settings=(data.info['highpass'], data.info['lowpass']),
            hset_info=hset_info,
            **kwargs_psd,
        )

        psd_list_aperiodic.append(irasa_spectrum.aperiodic.copy())
        psd_list_periodic.append(irasa_spectrum.periodic.copy())
        psd_list_original.append(irasa_spectrum.raw_spectrum.copy())

    psds_aperiodic = np.array(psd_list_aperiodic).mean(axis=0)
    psds_periodic = np.array(psd_list_periodic).mean(axis=0)
    psd_list_original = np.array(psd_list_original).mean(axis=0)

    psds_periodic = psds_periodic[np.newaxis, :, :]
    psds_aperiodic = psds_aperiodic[np.newaxis, :, :]

    return psd_list_original, irasa_spectrum.freqs, IrasaEpoched(
        periodic=PeriodicEpochsSpectrum(
            psds_periodic, info, freqs=irasa_spectrum.freqs, events=np.array([[0, 0, 1]]*1), event_id={'1': 1}
        ),
        aperiodic=AperiodicEpochsSpectrum(
            psds_aperiodic, info, freqs=irasa_spectrum.freqs, events=np.array([[0, 0, 1]]*1), event_id={'1': 1}
        ),
    )