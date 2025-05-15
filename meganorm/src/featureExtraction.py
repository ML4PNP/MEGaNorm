import numpy as np
import os
import sys
import tqdm
import json
import pickle
import argparse
import pandas as pd
import fooof as f
from typing import Dict, List
from typing import Union

# from layouts import load_specific_layout
from meganorm.utils.IO import make_config
from meganorm.layouts.layouts import load_specific_layout


def offset(fm: f.FOOOF) -> float:
    """
    Extract the offset parameter from the aperiodic component of a FOOOF model.

    Parameters
    ----------
    fm : f.FOOOF
        A FOOOF model object that has been fit to data and contains aperiodic parameters.

    Returns
    -------
    float
        The offset value, which is the first element of the aperiodic parameters.

    Raises
    ------
    TypeError
        Expected a FOOOF model instance.
    """
    if not isinstance(fm, f.FOOOF):
        raise TypeError("Expected a FOOOF model instance.")

    return fm.get_params("aperiodic_params")[0]


def exponent(fm: f.FOOOF, aperiodic_mode: str) -> float:
    """
    Extract the exponent value from the aperiodic component of a FOOOF model.

    Parameters
    ----------
    fm : f.FOOOF
        A FOOOF model object that has been fit to data and contains aperiodic parameters.
    aperiodic_mode : str
        The aperiodic mode that has been used to fit the model. Must be one of ['knee', 'fixed'].

    Returns
    -------
    float
        The exponent value corresponding to the specified mode ('knee' or 'fixed')

    Raises
    ------
    ValueError
        Unknown aperiodic_mode; Expected 'knee' or 'fixed'.
    """
    if aperiodic_mode == "knee":
        exponent_index = 2
    elif aperiodic_mode == "fixed":
        exponent_index = 1
    else:
        raise ValueError(
            f"Unknown aperiodic_mode: {aperiodic_mode}. Expected 'knee' or 'fixed'."
        )

    return fm.get_params("aperiodic_params")[exponent_index]


def find_peak_in_band(
    fm: f.FOOOF, fmin: Union[int, float], fmax: Union[int, float]
) -> list:
    """
    Find peaks in a specified frequency band (determined by fmin and fmax) from the peak parameters of a FOOOF model.

    Parameters
    ----------
    fm : f.FOOOF
        A FOOOF model object that contains peak parameters.
    fmin : Union[int, float]
        The lower frequency of the band.
    fmax : Union[int, float]
        The upper frequency of the band.

    Returns
    -------
    list
        A list of tuples where each tuple represents a peak. In the tuples, the first element is the
        frequency of the corresponding peak, the second element is the peak value (power), and the third element is the width of the peak.
    """
    peaks = fm.get_params("peak_params")

    # filter peaks: check for NaNs and then within thee frequency band
    band_peaks = [
        peak for peak in peaks if not np.any(np.isnan(peak)) and fmin <= peak[0] <= fmax
    ]

    return band_peaks


def peak_center(band_peaks: list):
    """
    Returns the frequency of the center of a dominant peak.

    Parameters
    ----------
    band_peaks : list
        A list of tuples where each tuple represents a peak. This list is the output of 'find_peak_in_band'
        function. In the tuples, the first element is the frequency, the second element is the peak value,
        and the third element is the width of the dominant peak.
    Returns
    -------
    float
        The frequency of the dominant peak, or np.nan if the list is empty.
    """
    if not band_peaks:
        return np.nan

    # Get the dominant peak by selecting the one with the maximum second element (e.g., power)
    dominant_peak = max(band_peaks, key=lambda x: x[1])

    # Return the frequency of the dominant peak (first element of the tuple)
    return dominant_peak[0]


def peak_power(band_peaks: list):
    """
    Returns the power of the center of a dominant peak from a list of peaks.

    Parameters
    ----------
    band_peaks : list
        A list of tuples where each tuple represents a peak. This list is the output of 'find_peak_in_band'
        function. In the tuples, the first element is the frequency, the second element is the peak value,
        and the third element is the width of the dominant peak.

    Returns
    -------
    float
        The power of the dominant peak, or np.nan if the list is empty.
    """
    if not band_peaks:
        return np.nan

    dominant_peak = max(band_peaks, key=lambda x: x[1])
    return dominant_peak[1]


def peak_width(band_peaks: list):
    """
    Returns the width of the dominant peak from a list of peaks.

    Parameters
    ----------
    band_peaks : A list of tuples where each tuple represents a peak. This list is the output of 'find_peak_in_band'
                function. In the tuples, the first element is the frequency, the second
                element is the peak value, and the third element is the width of the dominant peak.

    Returns
    -------
    float
        The width of the dominant peak, or np.nan if the list is empty.
    """
    if not band_peaks:
        return np.nan

    dominant_peak = max(band_peaks, key=lambda x: x[1])
    return dominant_peak[2]


def isolate_periodic(fm: f.FOOOF, psd: np.ndarray) -> np.ndarray:
    """
    Isolates the periodic component of the power spectrum by subtracting the aperiodic fit
    from the original pwer spectrum density.

    Parameters
    ----------
    fm : f.FOOOF
        An already fitted FOOOF model object.
    psd : np.ndarray
        Original power spectrum in linear scale.

    Returns
    -------
    np.ndarray
        A 1D array of the peridic component of the power spectrum.
    """
    return psd - 10**fm._ap_fit


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
    band_power = np.trapz(psd[band_indices], freqs[band_indices])

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
    band_power = np.trapz(psd[band_indices], freqs[band_indices])
    total_power = np.trapz(psd, freqs)

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

    band_power = np.trapz(psd[peak_range_indices], freqs[peak_range_indices])
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

    band_power = np.trapz(psd[peak_range_indices], freqs[peak_range_indices])
    total_power = np.trapz(psd, freqs)

    if total_power == 0:
        return np.nan

    return band_power / total_power


def summarizeFeatures(df, extention, which_layout, which_sensor):
    """
    Summarizes a feature DataFrame by averaging channels based on a specified sensor layout.

    Since sensor positions may differ across datasets recorded with different MEG hardware systems,
    this function enables consistent feature extraction by averaging signals across the whole brain
    or predefined brain regions (e.g., lobes).

    The function computes the mean of selected channels (e.g., MEG, EEG) according to a layout
    specified in a JSON file. The layout file is selected based on the recording extension
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
    extension (e.g., 'FIF.json'). The appropriate key in the JSON (e.g., 'FIF_MEG_LOBE') is constructed
    using `extention`, `which_layout`, and `which_sensor`.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame where each column represents a channel and each row a sample (subject or epoch).
    extention : str
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
            extention.upper() + "_" + modality.upper() + "_" + which_layout.upper()
        )
        layout = load_specific_layout(extention.upper(), layout_name)

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
    powerNumerator = np.trapz(psd[bandIndices], freqs[bandIndices])

    # Denominator
    bandIndices = np.logical_and(
        freqs >= freqRangeDenominator[0], freqs <= freqRangeDenominator[1]
    )
    powerDenominator = np.trapz(psd[bandIndices], freqs[bandIndices])

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
                    feature_names.append(f"{feature}_{freq_band}")
            else:
                # For features that don't need frequency bands
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
    feature_name = feature_name + band_name
    feature_container.at[feature_name, channel_name] = feature_arr

    return feature_container


def feature_extract(
    subject_id: str,
    fmGroup: f.FOOOF,
    psds: np.ndarray,
    feature_categories: Dict[str, bool],
    freqs: np.ndarray,
    freq_bands: Dict[str, tuple],
    channel_names: List[str],
    individualized_band_ranges: Dict[str, tuple],
    extention: str,
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
    fmGroup : f.FOOOF
        Group of FOOOF models, where each model corresponds to a channel and
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
    extention : str
        The extension of the subject's recording (e.g., 'FIF', 'DS'). Used to read the
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
        If `fmGroup` is not an instance of f.FOOOF.
    ValueError
        If `min_r_squared` is not between 0 and 1.
    """

    if aperiodic_mode not in ["knee", "fixed"]:
        raise ValueError(
            f"Unknown aperiodic_mode: {aperiodic_mode}. Expected 'knee' or 'fixed'."
        )
    if not isinstance(fmGroup, f.FOOOF):
        raise TypeError("Expected a FOOOF model instance.")
    if not 0 <= min_r_squared <= 1:
        raise ValueError("Minimum R squared should be between zero and 1")

    # Store features in a pandas DataFrame with channel names as columns
    # and feature names as the index,
    feature_container = create_feature_container(
        feature_categories, freq_bands, channel_names
    )

    for channel_num, channel_name in enumerate(channel_names):

        # getting the fooof model of ith channel
        fm = fmGroup.get_fooof(ind=channel_num)

        # fooof fitness
        # TODO, this needs to go out of feature exctraction
        r_squared = fm.r_squared_
        if r_squared < min_r_squared:
            continue

        # offset ==================================
        if feature_categories["Offset"]:
            feature_arr = offset(fm)
            feature_container = add_feature(
                feature_container, feature_arr, "Offset", channel_name, ""
            )
        # Exponent ==================================
        if feature_categories["Exponent"]:
            feature_arr = exponent(fm, aperiodic_mode)
            feature_container = add_feature(
                feature_container, feature_arr, "Exponent", channel_name, ""
            )

        original_psd = psds[channel_num, :]
        # isolate periodic parts of signals
        flattened_psd = np.asarray(isolate_periodic(fm, original_psd))

        # whenever aperidic activity is higher than periodic activity
        # => set the preiodic acitivity to zero
        flattened_psd = np.array(list(map(lambda x: max(0, x), flattened_psd)))

        # Loop through each frequency band
        for band_name, (fmin, fmax) in freq_bands.items():

            # Peak Features ==================================
            band_peaks = find_peak_in_band(fm, fmin, fmax)

            if feature_categories["Peak_Center"]:
                feature_arr = peak_center(band_peaks)
                feature_container = add_feature(
                    feature_container,
                    feature_arr,
                    "Peak_Center",
                    channel_name,
                    band_name,
                )

            if feature_categories["Peak_Power"]:
                feature_arr = peak_power(band_peaks)
                feature_container = add_feature(
                    feature_container,
                    feature_arr,
                    "Peak_Power",
                    channel_name,
                    band_name,
                )

            if feature_categories["Peak_Width"]:
                feature_arr = peak_width(band_peaks)
                feature_container = add_feature(
                    feature_container,
                    feature_arr,
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
            if feature_categories["Adjusted_Canonical_Relative_Power"]:
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
            if feature_categories["OriginalPSD_Canonical_Relative_Power"]:
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
            extention=extention,
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

    final_df.index = [subject_id]

    return final_df
