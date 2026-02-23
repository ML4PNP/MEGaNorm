import numpy as np
import os
import sys
import tqdm
import json
import pickle
import argparse
import logging
import pyrasa
import fooof as f
import pandas as pd
from typing import Union
from typing import Dict, List
from abc import ABC, abstractmethod
from pyrasa.irasa_mne import irasa_epochs
# from layouts import load_specific_layout
from meganorm.utils.IO import make_config
from meganorm.layouts.layouts import load_specific_layout


logger = logging.getLogger(__name__)


# def offset(fm: f.FOOOF) -> float:
#     """
#     Extract the offset parameter from the aperiodic component of a FOOOF model.

#     Parameters
#     ----------
#     fm : f.FOOOF
#         A FOOOF model object that has been fit to data and contains aperiodic parameters.

#     Returns
#     -------
#     float
#         The offset value, which is the first element of the aperiodic parameters.

#     Raises
#     ------
#     TypeError
#         Expected a FOOOF model instance.
#     """
#     if not isinstance(fm, f.FOOOF):
#         raise TypeError("Expected a FOOOF model instance.")

#     return fm.get_params("aperiodic_params")[0]


# def exponent(fm: f.FOOOF, aperiodic_mode: str) -> float:
#     """
#     Extract the exponent value from the aperiodic component of a FOOOF model.

#     Parameters
#     ----------
#     fm : f.FOOOF
#         A FOOOF model object that has been fit to data and contains aperiodic parameters.
#     aperiodic_mode : str
#         The aperiodic mode that has been used to fit the model. Must be one of ['knee', 'fixed'].

#     Returns
#     -------
#     float
#         The exponent value corresponding to the specified mode ('knee' or 'fixed')

#     Raises
#     ------
#     ValueError
#         Unknown aperiodic_mode; Expected 'knee' or 'fixed'.
#     """
#     if aperiodic_mode == "knee":
#         exponent_index = 2
#     elif aperiodic_mode == "fixed":
#         exponent_index = 1
#     else:
#         raise ValueError(
#             f"Unknown aperiodic_mode: {aperiodic_mode}. Expected 'knee' or 'fixed'."
#         )

#     return fm.get_params("aperiodic_params")[exponent_index]


# def find_peak_in_band(
#     fm: f.FOOOF, fmin: Union[int, float], fmax: Union[int, float]
# ) -> list:
#     """
#     Find peaks in a specified frequency band (determined by fmin and fmax) from the peak parameters of a FOOOF model.

#     Parameters
#     ----------
#     fm : f.FOOOF
#         A FOOOF model object that contains peak parameters.
#     fmin : Union[int, float]
#         The lower frequency of the band.
#     fmax : Union[int, float]
#         The upper frequency of the band.

#     Returns
#     -------
#     list
#         A list of tuples where each tuple represents a peak. In the tuples, the first element is the
#         frequency of the corresponding peak, the second element is the peak value (power), and the third element is the width of the peak.
#     """
#     peaks = fm.get_params("peak_params")

#     # filter peaks: check for NaNs and then within thee frequency band
#     band_peaks = [
#         peak for peak in peaks if not np.any(np.isnan(peak)) and fmin <= peak[0] <= fmax
#     ]

#     return band_peaks


# def peak_center(band_peaks: list):
#     """
#     Returns the frequency of the center of a dominant peak.

#     Parameters
#     ----------
#     band_peaks : list
#         A list of tuples where each tuple represents a peak. This list is the output of 'find_peak_in_band'
#         function. In the tuples, the first element is the frequency, the second element is the peak value,
#         and the third element is the width of the dominant peak.
#     Returns
#     -------
#     float
#         The frequency of the dominant peak, or np.nan if the list is empty.
#     """
#     if not band_peaks:
#         return np.nan

#     # Get the dominant peak by selecting the one with the maximum second element (e.g., power)
#     dominant_peak = max(band_peaks, key=lambda x: x[1])

#     # Return the frequency of the dominant peak (first element of the tuple)
#     return dominant_peak[0]


# def peak_power(band_peaks: list):
#     """
#     Returns the power of the center of a dominant peak from a list of peaks.

#     Parameters
#     ----------
#     band_peaks : list
#         A list of tuples where each tuple represents a peak. This list is the output of 'find_peak_in_band'
#         function. In the tuples, the first element is the frequency, the second element is the peak value,
#         and the third element is the width of the dominant peak.

#     Returns
#     -------
#     float
#         The power of the dominant peak, or np.nan if the list is empty.
#     """
#     if not band_peaks:
#         return np.nan

#     dominant_peak = max(band_peaks, key=lambda x: x[1])
#     return dominant_peak[1]


# def peak_width(band_peaks: list):
#     """
#     Returns the width of the dominant peak from a list of peaks.

#     Parameters
#     ----------
#     band_peaks : A list of tuples where each tuple represents a peak. This list is the output of 'find_peak_in_band'
#                 function. In the tuples, the first element is the frequency, the second
#                 element is the peak value, and the third element is the width of the dominant peak.

#     Returns
#     -------
#     float
#         The width of the dominant peak, or np.nan if the list is empty.
#     """
#     if not band_peaks:
#         return np.nan

#     dominant_peak = max(band_peaks, key=lambda x: x[1])
#     return dominant_peak[2]


# def isolate_periodic(fm: f.FOOOF, psd: np.ndarray) -> np.ndarray:
#     """
#     Isolates the periodic component of the power spectrum by subtracting the aperiodic fit
#     from the original pwer spectrum density.

#     Parameters
#     ----------
#     fm : f.FOOOF
#         An already fitted FOOOF model object.
#     psd : np.ndarray
#         Original power spectrum in linear scale.

#     Returns
#     -------
#     np.ndarray
#         A 1D array of the peridic component of the power spectrum.
#     """
#     return psd - 10**fm._ap_fit


def abs_canonical_power(
    psd: np.ndarray, freqs: np.ndarray, fmin: Union[int, float], fmax: Union[int, float]
) -> float:
    """
    Calculates absolute canonical power of a frequency band from a power spectrum density (PSD).

    Parameters
    ----------
    psd : np.ndarray
        Power spectral density values (in linear scale).
    freqs : np.ndarray
        A 1D array of frequency values that were used to compute the PSD.
    fmin : Union[int, float]
        Lower bound of the frequency band
    fmax : Union[int, float]
        Upper bound of the frequency band.

    Returns
    -------
    float
        Log-transformed absolute power in the specified frequency band.

    Notes
    -------
    'psd' can be both original PSD or periodic PSD.
    """

    band_indices = np.logical_and(freqs >= fmin, freqs <= fmax)
    band_power = np.trapezoid(psd[band_indices], freqs[band_indices])

    return np.log10(band_power)


def rel_canonical_power(
    psd: np.ndarray, freqs: np.ndarray, fmin: Union[int, float], fmax: Union[int, float]
) -> float:
    """
    Calculates relative canonical power of a frequency band from a power spectrum density.

    Parameters
    ----------
    psd : np.ndarray
        Power spectral density values (in linear scale).
    freqs : np.ndarray
        A 1D array of frequency values that were used to compute the PSD.
    fmin : Union[int, float]
        Lower bound of the frequency band.
    fmax : Union[int, float]
        Upper bound of the frequency band.

    Returns
    -------
    float
        Relative power in the specified frequency band. Returns np.nan if total power is zero.

    Notes
    -------
    'psd' can be both original PSD or periodic PSD.
    """

    band_indices = np.logical_and(freqs >= fmin, freqs <= fmax)
    band_power = np.trapezoid(psd[band_indices], freqs[band_indices])
    total_power = np.trapezoid(psd, freqs)

    if total_power == 0:
        return np.nan

    return band_power / total_power


def abs_individual_power(psd, freqs, band_peaks, individualized_band_ranges, band_name):
    """Calculates absolute power in an individualized frequency band centered around the dominant peak.

    Parameters
    ----------
    psd : np.ndarray
        Power spectral density values (in linear scale).
    freqs : np.ndarray
        A 1D array of frequency values that were used to compute the PSD.
    band_peaks : list
        List of peak tuples (frequency, power, width).
    individualized_band_ranges : dict
        Dictionary mapping band names to (lower_offset, upper_offset) in Hz.
    band_name : str
         Name of the frequency band to compute power for.

    Returns
    -------
    float
        Log-transformed absolute power in the individualized frequency band. Returns np.nan if no peaks are found.

    Notes
    -------
    'psd' can be both original PSD or periodic PSD.
    """

    if not band_peaks or band_name not in individualized_band_ranges:
        return np.nan

    # Find the dominant peak
    dominant_peak = max(band_peaks, key=lambda x: x[1])
    peak_freq = dominant_peak[0]
    lower_offset, upper_offset = individualized_band_ranges[band_name]

    # Define the frequency range around the peak and find matching indices
    peak_range_indices = np.logical_and(
        freqs >= peak_freq + lower_offset, freqs <= peak_freq + upper_offset
    )

    band_power = np.trapezoid(psd[peak_range_indices], freqs[peak_range_indices])
    return np.log10(band_power)


def rel_individual_power(psd, freqs, band_peaks, individualized_band_ranges, band_name):
    """
    Calculates relative power in an individualized frequency band centered around the dominant peak.

    Parameters
    ----------
    psd : np.ndarray
        Power spectral density values (in linear scale).
    freqs : list
        List of peak tuples (frequency, power, width)
    band_peaks : list
        List of peak tuples (frequency, power, width)
    individualized_band_ranges : dict
        Dictionary mapping band names to (lower_offset, upper_offset) in Hz.
    band_name : str
        Name of the frequency band to compute power for.

    Returns
    -------
    float
        Relative power in the individualized frequency band. Returns np.nan if total power is zero or input is invalid.

    Notes
    -------
    'psd' can be both original PSD or periodic PSD.
    """

    if not band_peaks or band_name not in individualized_band_ranges:
        return np.nan

    # Find the dominant peak
    dominant_peak = max(band_peaks, key=lambda x: x[1])
    peak_freq = dominant_peak[0]
    lower_offset, upper_offset = individualized_band_ranges[band_name]

    # Define the range around the peak frequency
    peak_range_indices = np.logical_and(
        freqs >= peak_freq + lower_offset, freqs <= peak_freq + upper_offset
    )

    band_power = np.trapezoid(psd[peak_range_indices], freqs[peak_range_indices])
    total_power = np.trapezoid(psd, freqs)

    if total_power == 0:
        return np.nan

    return band_power / total_power


def summarizeFeatures(df, device, which_layout, which_sensor):
    """
    Summarizes a feature DataFrame by averaging channels based on a specified sensor layout.

    Since sensor positions may differ across datasets recorded with different MEG hardware systems,
    this function enables consistent feature extraction by averaging signals across the whole brain
    or predefined brain regions (e.g., lobes).

    The function computes the mean of selected channels (e.g., MEG, EEG) according to a layout
    specified in a JSON file. The layout file is selected based on the recording device
    (e.g., 'FIF', 'DS') and contains channel groupings for either whole-brain or regional (lobe-level)
    parcellation.

    Example layout for regional parcellation:
        "FIF_MEG_LOBE": {
            "MAG_frontal_left": ["MEG0121", "MEG0341", "MEG0311", "MEG0321", ...],
            "MAG_frontal_right": ["MEG1411", "MEG1221", "MEG1211", "MEG1231", ...]
        }

    Example layout for whole-brain averaging:
        "FIF_MAG_ALL": {
            "MAG_ALL": ["MEG0121", "MEG0341", "MEG0311", ...]
        }

    Layout files must be stored in a dedicated layout directory and named based on the recording
    device (e.g., 'FIF.json'). The appropriate key in the JSON (e.g., 'FIF_MEG_LOBE') is constructed
    using `device`, `which_layout`, and `which_sensor`.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame where each column represents a channel and each row a sample (subject or epoch).
    device : str
        The recording file type (e.g., 'FIF', 'DS'). Used to locate the correct layout file.
    which_layout : str
        Layout type to use: 'all' for global averaging or 'lobe' for region-based averaging.
    which_sensor : dict
        Dictionary indicating which sensor modalities to include (e.g., {'meg': True, 'eeg': False}).

    Returns
    -------
    pd.DataFrame
        A new DataFrame where columns represent averaged parcels and rows represent samples.
    """
    df.dropna(axis=0, how="all", inplace=True)
    summrized_df = pd.DataFrame(index=df.index)

    # TODO: If both meg and eeg is True, this won't work!
    if which_layout == "all":
        summrized_df[which_layout] = df.mean(axis=1)

    else:
        modality = [
            s_type for s_type, if_alculate in which_sensor.items() if if_alculate
        ][0]

        layout_name = (
            device.upper() + "_" + modality.upper() + "_" + which_layout.upper()
        )
        layout = load_specific_layout(device.upper(), layout_name)

        for parcel_name, channels_list in layout.items():
            summrized_df[parcel_name] = df[list(channels_list)].mean(axis=1)

    return summrized_df


def psd_ratio(
    psd,
    freqs,
    freqRangeNumerator: float,
    freqRangeDenominator: float,
    channelNames: str,
    name: str,
    psdType: str,
):
    """
    Calculates the ratio of power in two frequency bands (numerator/denominator) in the power spectral density (PSD).

    Parameters
    ----------
    psd : np.ndarray
        A 1D array of power spectral density values (in linear scale).
    freqs : np.ndarray
        A 1D array of frequency values corresponding to the PSD.
    freqRangeNumerator : tuple
        A tuple (min_freq, max_freq) representing the numerator frequency range.
    freqRangeDenominator : tuple
        A tuple (min_freq, max_freq) representing the denominator frequency range.
    channelNames : str
        Name of the channel for which the ratio is calculated.
    name : str
        Descriptive name for the output feature.
    psdType : str
        Type of the PSD (e.g., 'power', 'spectrum').

    Returns
    -------
    list
        A list with the computed PSD ratio (log of numerator/denominator)
    list
        A list with the generated feature name

    """
    # TODO check this function, seems incorrect
    # Numerator
    bandIndices = np.logical_and(
        freqs >= freqRangeNumerator[0], freqs <= freqRangeNumerator[1]
    )
    powerNumerator = np.trapezoid(psd[bandIndices], freqs[bandIndices])

    # Denominator
    bandIndices = np.logical_and(
        freqs >= freqRangeDenominator[0], freqs <= freqRangeDenominator[1]
    )
    powerDenominator = np.trapezoid(psd[bandIndices], freqs[bandIndices])

    # ratio
    featRow = np.log10(powerNumerator) / np.log10(powerDenominator)
    featName = f"{psdType}_Canonical_Absolute_Power_{name}_{channelNames}"

    return [featRow], [featName]


def create_feature_container(feature_categories, freq_bands, channel_names):
    """
    Creates a DataFrame to store features for each channel, with feature names corresponding to
    the specified categories and frequency bands.

    Parameters
    ----------
    feature_categories : dict
        Dictionary with feature names as keys and booleans indicating
        whether the feature should be calculated.
    freq_bands : list
        List of frequency bands (e.g., ['theta', 'alpha', 'beta']).
    channel_names : list
        List of channel names (e.g., ['ch1', 'ch2', 'ch3']).

    Returns
    -------
    pd.DataFrame
        A DataFrame with feature names as rows and channels as columns.
    """
    # TODO if band_name != "broadband" although not necessary because we fill the
    # data frame with name (df.at())

    # Features that do not need frequency band appended
    no_freq = ["Offset", "Exponent", "Peak_Center", "Peak_Power", "Peak_Width"]

    feature_names = []

    for feature, if_calculate in feature_categories.items():
        if if_calculate:
            if feature not in no_freq:
                # Append frequency bands to the feature name
                for freq_band in freq_bands:
                    feature_names.append(f"{feature}__{freq_band}")
            else:
                # For features that don't need frequency bands
                feature += "__" # TODO: this '_' should be removed in future
                feature_names.append(feature)

    # Return an empty DataFrame with features as index and channels as columns
    return pd.DataFrame(columns=channel_names, index=feature_names)


def add_feature(feature_container, feature_arr, feature_name, channel_name, band_name):
    """
    Add a feature value to the feature container for a specific channel and frequency band.

    This function appends a feature to a DataFrame by assigning a value (e.g., from an array)
    to a row labeled with the combined feature and band name, and a column labeled with the
    channel name.

    Parameters
    ----------
    feature_container : pd.DataFrame
        DataFrame used to store features, where rows represent feature names and columns represent channels.
    feature_arr : np.ndarray
        Array containing the feature value(s) to add.
    feature_name : str
        Name of the feature (e.g., 'RelativePower_').
    channel_name : str
        Name of the channel (e.g., 'MEG0121') to which the feature value should be assigned.
    band_name : str
        Frequency band to append to the feature name (e.g., 'Alpha').

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with the new feature added.
    """
    feature_name = feature_name + "__" + band_name
    feature_container.at[feature_name, channel_name] = feature_arr

    return feature_container


def feature_extract(
    subject_id: str,
    spectral_models,
    psds: np.ndarray,
    feature_categories: Dict[str, bool],
    freqs: np.ndarray,
    freq_bands: Dict[str, tuple],
    channel_names: List[str],
    individualized_band_ranges: Dict[str, tuple],
    device: str,
    which_layout: str,
    which_sensor: Dict[str, bool],
    aperiodic_mode: str,
    min_r_squared: float,
) -> pd.DataFrame:
    """
    Extract features from FOOOF models for each channel and frequency band.

    This function computes various features from FOOOF models for each channel,
    based on specified frequency bands. Features such as offset, exponent, peak
    characteristics, and canonical power are calculated and stored in a DataFrame.

    Parameters
    ----------
    subject_id : str
        The unique identifier for the subject whose data is being processed.
    spectral_models : 
        Group of FOOOF models or PYRASA models, where each model corresponds to a channel and
        its power spectral data.
    psds : np.ndarray
        Original power spectral density values, with shape (n_channels, n_freqs).
    feature_categories : Dict[str, bool]
        A dictionary where keys are feature names (e.g., 'Offset', 'Exponent') and values are
        booleans indicating whether to compute the feature.
    freqs : np.ndarray
        Frequency values corresponding to the power values in the `psds` array.
    freq_bands : Dict[str, tuple]
        Dictionary mapping frequency band names (e.g., 'Alpha', 'Beta') to their
        corresponding frequency ranges (min_freq, max_freq).
    channel_names : List[str]
        List of channel names corresponding to the rows of the `psds` array.
    individualized_band_ranges : Dict[str, tuple]
        A dictionary mapping band names to individualized frequency ranges, which may differ
        across subjects or datasets.
    device : str
        The device of the subject's recording (e.g., 'FIF', 'DS'). Used to read the
        appropriate layout file from the layout directory.
    which_layout : str
        Specifies the sensor layout for feature averaging, either 'all' for global averaging
        or 'lobe' for averaging within lobes.
    which_sensor : Dict[str, bool]
        A dictionary indicating which modalities (e.g., 'meg', 'eeg') should be included
        in the feature extraction.
    aperiodic_mode : str
        Defines the aperiodic component fitting mode for FOOOF. Options are 'knee' or 'fixed'.
    min_r_squared : float
        Minimum acceptable R-squared value for FOOOF model fitting. Channels with
        R-squared values below this threshold are excluded.

    Returns
    -------
    pd.DataFrame
        A DataFrame with features extracted for each channel and frequency band. The
        DataFrame has features as rows and channels (and frequency bands) as columns.

    Raises
    ------
    ValueError
        If `aperiodic_mode` is not 'knee' or 'fixed'.
    TypeError
        If `spectral_models` is not an instance of f.FOOOF.
    ValueError
        If `min_r_squared` is not between 0 and 1.
    """

    if aperiodic_mode not in ["knee", "fixed"]:
        raise ValueError(
            f"Unknown aperiodic_mode: {aperiodic_mode}. Expected 'knee' or 'fixed'."
        )
    if not isinstance(spectral_models, f.FOOOF) and not isinstance(spectral_models, pyrasa.irasa_mne.mne_objs.IrasaEpoched):
        raise TypeError("Expected a f.FOOOF or pyrasa.irasa_mne.mne_objs.IrasaEpoched object instance.")

    # Store features in a pandas DataFrame with channel names as columns
    # and feature names as the index,
    feature_container = create_feature_container(
        feature_categories, freq_bands, channel_names
    )

    if isinstance(spectral_models, pyrasa.irasa_mne.mne_objs.IrasaEpoched):
        ap = spectral_models.aperiodic.fit_aperiodic_model(fit_func=aperiodic_mode, scale=True)

    for channel_num, channel_name in enumerate(channel_names):

        if isinstance(spectral_models, f.FOOOF):
            spectral_model = FOOOFDecomposer(spectral_models, 
                                            mode=aperiodic_mode, 
                                            ch_num=channel_num)

        elif isinstance(spectral_models, pyrasa.irasa_mne.mne_objs.IrasaEpoched):
            spectral_model = PYRASADecomposer(spectral_models, 
                                            mode=aperiodic_mode, 
                                            ch_name=channel_name, 
                                            ch_num=channel_num, 
                                            aperiodic=ap)
            
        else:
            raise TypeError(f"Unknown spectral model type: {type(spectral_models)}")

        # fitness SQC
        logger.info(f"The R**2 in PSD parametrization of the channel {channel_name} was {spectral_model.get_r_squared()}")
        if spectral_model.get_r_squared() < min_r_squared:
            logger.info(f"The {channel_num}th channel, {channel_name}, was removed" \
                        " since it's corresponding R2 score in PSD parametrization " \
                        f"was less than the threshold: {spectral_model.get_r_squared()} > min_r_squared")
            continue

        # # offset ==================================
        if feature_categories["Offset"]:
            feature_arr = spectral_model.get_aperiodic_params()[0]
            feature_container = add_feature(
                feature_container, feature_arr, "Offset", channel_name, ""
            )
        # # Exponent ==================================
        if feature_categories["Exponent"]:
            feature_arr = spectral_model.get_aperiodic_params()[1]
            feature_container = add_feature(
                feature_container, feature_arr, "Exponent", channel_name, ""
            )

        # isolate periodic parts of signals
        flattened_psd = spectral_model.get_periodic_spectrum(original_psds=psds)
        original_psd = psds[channel_num, :]

        # # whenever aperidic activity is higher than periodic activity
        # # => set the preiodic acitivity to zero
        flattened_psd = np.array(list(map(lambda x: max(0, x), flattened_psd)))

        # Loop through each frequency band
        for band_name, (fmin, fmax) in freq_bands.items():

            # Peak Features ==================================
            peak_params, band_peaks = spectral_model.get_peak_params(fmin=fmin, fmax=fmax)
            if feature_categories["Peak_Center"] and peak_params:
                feature_container = add_feature(
                    feature_container,
                    peak_params[0],
                    "Peak_Center",
                    channel_name,
                    band_name,
                )

            if feature_categories["Peak_Power"] and peak_params:
                feature_container = add_feature(
                    feature_container,
                    peak_params[1],
                    "Peak_Power",
                    channel_name,
                    band_name,
                )

            if feature_categories["Peak_Width"] and peak_params:
                feature_container = add_feature(
                    feature_container,
                    peak_params[2],
                    "Peak_Width",
                    channel_name,
                    band_name,
                )

            # Adjusted absolute canonical power ==================================
            if feature_categories["Adjusted_Canonical_Absolute_Power"]:
                feature_arr = abs_canonical_power(
                    psd=flattened_psd, freqs=freqs, fmin=fmin, fmax=fmax
                )
                feature_container = add_feature(
                    feature_container,
                    feature_arr,
                    "Adjusted_Canonical_Absolute_Power",
                    channel_name,
                    band_name,
                )

            # Adjusted relative canonical power ==================================
            if feature_categories["Adjusted_Canonical_Relative_Power"] and band_name!= "Broadband":
                feature_arr = rel_canonical_power(
                    psd=flattened_psd, freqs=freqs, fmin=fmin, fmax=fmax
                )
                feature_container = add_feature(
                    feature_container,
                    feature_arr,
                    "Adjusted_Canonical_Relative_Power",
                    channel_name,
                    band_name,
                )

            # OriginalPSD absolute canonical power ==================================
            if feature_categories["OriginalPSD_Canonical_Absolute_Power"]:
                feature_arr = abs_canonical_power(
                    psd=original_psd, freqs=freqs, fmin=fmin, fmax=fmax
                )
                feature_container = add_feature(
                    feature_container,
                    feature_arr,
                    "OriginalPSD_Canonical_Absolute_Power",
                    channel_name,
                    band_name,
                )

            # OriginalPSD relative canonical power ==================================
            if feature_categories["OriginalPSD_Canonical_Relative_Power"] and band_name!= "Broadband":
                feature_arr = rel_canonical_power(
                    psd=original_psd, freqs=freqs, fmin=fmin, fmax=fmax
                )
                feature_container = add_feature(
                    feature_container,
                    feature_arr,
                    "OriginalPSD_Canonical_Relative_Power",
                    channel_name,
                    band_name,
                )

            if band_name != "Broadband" and band_peaks:

                # Adjusted absolute Relative power ==================================
                if feature_categories["Adjusted_Individualized_Absolute_Power"]:
                    feature_arr = abs_individual_power(
                        psd=flattened_psd,
                        freqs=freqs,
                        band_peaks=band_peaks,
                        individualized_band_ranges=individualized_band_ranges,
                        band_name=band_name,
                    )
                    feature_container = add_feature(
                        feature_container,
                        feature_arr,
                        "Adjusted_Individualized_Absolute_Power",
                        channel_name,
                        band_name,
                    )

                # Adjusted relative Relative power ==================================
                if feature_categories["Adjusted_Individualized_Relative_Power"]:
                    feature_arr = rel_individual_power(
                        psd=flattened_psd,
                        freqs=freqs,
                        band_peaks=band_peaks,
                        individualized_band_ranges=individualized_band_ranges,
                        band_name=band_name,
                    )
                    feature_container = add_feature(
                        feature_container,
                        feature_arr,
                        "Adjusted_Individualized_Relative_Power",
                        channel_name,
                        band_name,
                    )

                # OriginalPSD absolute Relative power ==================================
                if feature_categories["OriginalPSD_Individualized_Absolute_Power"]:
                    feature_arr = abs_individual_power(
                        psd=original_psd,
                        freqs=freqs,
                        band_peaks=band_peaks,
                        individualized_band_ranges=individualized_band_ranges,
                        band_name=band_name,
                    )
                    feature_container = add_feature(
                        feature_container,
                        feature_arr,
                        "OriginalPSD_Individualized_Absolute_Power",
                        channel_name,
                        band_name,
                    )

                # OriginalPSD relative Relative power ==================================
                if feature_categories["OriginalPSD_Individualized_Relative_Power"]:
                    feature_arr = rel_individual_power(
                        psd=original_psd,
                        freqs=freqs,
                        band_peaks=band_peaks,
                        individualized_band_ranges=individualized_band_ranges,
                        band_name=band_name,
                    )
                    feature_container = add_feature(
                        feature_container,
                        feature_arr,
                        "OriginalPSD_Individualized_Relative_Power",
                        channel_name,
                        band_name,
                    )

    # # feature summarization ================================================================
    if which_layout:
        feature_container = summarizeFeatures(
            df=feature_container,
            device=device,
            which_layout=which_layout,
            which_sensor=which_sensor,
        )

    # Flatten the DataFrame and create neww column names
    final_df = pd.DataFrame(feature_container.values.flatten()).T
    final_df.columns = [
        f"{index}_{col}"
        for index in feature_container.index
        for col in feature_container.columns
    ]

    logger.info(f"The shape of the extracted features:{final_df.shape}")

    final_df.index = [subject_id]

    return final_df






class SpectralDecomposer(ABC):
    """Abstract base class for spectral decomposition methods"""
    
    @abstractmethod
    def get_aperiodic_params(self):
        """Return aperiodic parameters (offset, exponent, knee if applicable)"""
        pass
    
    @abstractmethod
    def get_periodic_spectrum(self, original_psds):
        """Return isolated periodic component"""
        pass
    
    @abstractmethod
    def get_peak_params(self, fmin, fmax):
        """Return peak parameters"""
        pass
    
    @abstractmethod
    def get_r_squared(self):
        """Return goodness of fit metric"""
        pass


class FOOOFDecomposer(SpectralDecomposer):
    def __init__(self, fooof_model, mode, ch_num):
        self.ch_num = ch_num
        self.model = fooof_model.get_fooof(ind=ch_num)
        self.mode = mode

    def get_aperiodic_params(self):
        
        reordered_params = []
        params = self.model.get_params("aperiodic_params")
        # offset
        reordered_params.append(params[0])

        # exponent
        if self.mode == "knee":
            exponent_index = 2
        elif self.mode == "fixed":
            exponent_index = 1
        else:
            raise ValueError(
                f"Unknown aperiodic_mode: {self.mode}. Expected 'knee' or 'fixed'."
            )
        reordered_params.append(params[exponent_index])
        
        return reordered_params
    
    def get_periodic_spectrum(self, original_psds):
        original_psd = original_psds[self.ch_num, :]
        return original_psd - 10**self.model._ap_fit
    
    def get_peak_params(self, fmin, fmax):

        peaks = self.model.get_params("peak_params")

        # filter peaks: check for NaNs and then within thee frequency band
        band_peaks = [
            peak for peak in peaks if not np.any(np.isnan(peak)) and fmin <= peak[0] <= fmax
        ]

        if not band_peaks:
            return None, None
        
        # Get the dominant peak by selecting the one with the maximum second element (e.g., power)
        dominant_peak = max(band_peaks, key=lambda x: x[1])
        # Return the frequency of the dominant peak (first element of the tuple)

        return dominant_peak, band_peaks
    
    def get_r_squared(self):
        return self.model.r_squared_

class PYRASADecomposer(SpectralDecomposer):
    def __init__(self, model, mode, ch_name, ch_num, aperiodic):
        self.mode = mode
        self.model = model
        self.aperiodic = aperiodic
        self.ch_name = ch_name
        self.ch_num = ch_num


    def get_aperiodic_params(self):
        
        aperiodic_params = self.aperiodic.aperiodic_params
        aperiodic_params_of_interest = aperiodic_params[aperiodic_params["ch_name"] == self.ch_name]

        params = []

        # offset
        params.append(aperiodic_params_of_interest["Offset"].item())
        # exponent 
        params.append(aperiodic_params_of_interest["Exponent_1"].item())

        return params
    
    def get_periodic_spectrum(self, original_psds=None):
        # print(self.model.periodic.get_data().squeeze().shape)
        return self.model.periodic.get_data().squeeze()[self.ch_num, :]
    
    def get_peak_params(self, fmin, fmax):
        return None, None # TODO
        
    def get_r_squared(self):
        gof = self.aperiodic.gof
        return gof[gof["ch_name"] == self.ch_name]["R2"].item()