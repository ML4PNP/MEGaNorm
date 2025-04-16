import numpy as np
import os
import sys
import tqdm
import json
import pickle
import argparse
import pandas as pd
import fooof as f
from typing import Union
# from layouts import load_specific_layout
from meganorm.utils.IO import make_config
from meganorm.layouts.layouts import load_specific_layout



def offset(fm: f.FOOOF) -> float:
    """
    Extract the offset parameter from the aperiodic component of a FOOOF model.

    Args:
        fm (f.FOOOF): A FOOOF model object that has been fit to data and contains aperiodic parameters.

    Returns:
        (float): The offset value, which is the first element of the aperiodic parameters.
    """
    if not isinstance(fm, f.FOOOF):
        raise TypeError("Expected a FOOOF model instance.")

    return fm.get_params("aperiodic_params")[0] 


def exponent(fm:f.FOOOF, aperiodic_mode:str) -> float:
    """
    Extract the exponent value from the aperiodic component of a FOOOF model, based on the specified mode.

    Args
        fm (f.FOOOF): A FOOOF model object that has been fit to data and contains aperiodic parameters.
        aperiodic_mode (str): The mode to extract the exponent value for. Must be one of ['knee', 'fixed'].

    Returns
        (float): The exponent value corresponding to the specified mode ('knee' or 'fixed').
    """
    if aperiodic_mode == "knee": 
        exponent_index = 2 
    elif aperiodic_mode == "fixed": 
        exponent_index = 1
    else:
        raise ValueError(f"Unknown aperiodic_mode: {aperiodic_mode}. Expected 'knee' or 'fixed'.")

    return fm.get_params("aperiodic_params")[exponent_index]


def find_peak_in_band(fm: f.FOOOF, fmin: Union[int, float], fmax: Union[int, float]) -> list:
    """
    Find peaks in a specified frequency band from the peak parameters of a FOOOF model.

    Args
        fm (f.FOOOF): A FOOOF model object that contains peak parameters.
        fmin (int or float): The minimum frequency of the band.
        fmax (int or float): The maximum frequency of the band.

    Returns
        list: A list of peaks that lie within the specified frequency band. Each peak is represented as a tuple.
    """

    peaks = fm.get_params('peak_params')
    
    # filter peaks: check for NaNs and then within thee frequency band
    band_peaks = [
        peak for peak in peaks
        if not np.any(np.isnan(peak)) and fmin <= peak[0] <= fmax
    ]

    return band_peaks


def peak_center(band_peaks:list):
    """
    Returns the frequency of the dominant peak from a list of band peaks.

    Args
        band_peaks (list): A list of tuples where each tuple represents a peak.
                                     The first element is the frequency and the second element is the peak value.

    Returns
        float: The frequency of the dominant peak, or np.nan if the list is empty.
    """
    if not band_peaks:
        return np.nan

    # Get the dominant peak by selecting the one with the maximum second element (e.g., power)
    dominant_peak = max(band_peaks, key=lambda x: x[1])
    
    # Return the frequency of the dominant peak (first element of the tuple)
    return dominant_peak[0]


def peak_power(band_peaks:list):
    """
    Returns the power of the dominant peak from a list of band peaks.

    Args
        band_peaks (list of tuples): A list of tuples where each tuple represents a peak.
                                     The first element is the frequency and the second element is the peak value
                                     (e.g., power, amplitude, or another relevant measure).

    Returns
        float: The power of the dominant peak, or np.nan if the list is empty.
    """
    if not band_peaks:
        return np.nan

    dominant_peak = max(band_peaks, key=lambda x: x[1])
    return dominant_peak[1]


def peak_width(band_peaks:list):
    """
    Returns the width of the dominant peak from a list of band peaks.

    Args
        band_peaks (list of tuples): A list of tuples where each tuple represents a peak.
                                     The elements are (frequency, power, width).

    Returns
        float: The width of the dominant peak, or np.nan if the list is empty.
    """
    if not band_peaks:
        return np.nan

    dominant_peak = max(band_peaks, key=lambda x: x[1])
    return dominant_peak[2]


def isolate_periodic(fm: f.FOOOF, psd: np.ndarray) -> np.ndarray:
    """
    Isolates the periodic component of the power spectrum by subtracting the aperiodic fit.

    Args
        fm (f.FOOOF): A FOOOF model object that contains the aperiodic fit.
        psd (np.ndarray): A 1D array of the original power spectrum in linear scale.

    Returns
        np.ndarray: A 1D array of the peridic component of the power spectrum.
    """
    return psd - 10**fm._ap_fit


def abs_canonical_power(psd: np.ndarray, freqs: np.ndarray, 
                    fmin: Union[int, float], fmax : Union[int, float]) -> float:
    """
    Calculates absolute canonical band power from a power spectrum.

    Args
        psd (np.ndarray): A 1D array of power spectral density values (in linear scale).
        freqs (np.ndarray): A 1D array of frequency values corresponding to the PSD.
        fmin (int or float): Lower bound of the frequency band.
        fmax (int or float): Upper bound of the frequency band.

    Returns
        float: Log-transformed absolute power in the specified frequency band.
    """
    band_indices = np.logical_and(freqs >= fmin, freqs <= fmax)
    band_power = np.trapz(psd[band_indices], freqs[band_indices])
    
    return np.log10(band_power)


def rel_canonical_power(psd: np.ndarray, freqs: np.ndarray,
                        fmin: Union[int, float], fmax: Union[int, float]) -> float:
    """
    Calculates relative canonical band power from a power spectrum.

    Args
        psd (np.ndarray): A 1D array of power spectral density values (in linear scale).
        freqs (np.ndarray): A 1D array of frequency values corresponding to the PSD.
        fmin (int or float): Lower bound of the frequency band.
        fmax (int or float): Upper bound of the frequency band.

    Returns
        float: Relative power in the specified frequency band. Returns np.nan if total power is zero.
    """
    band_indices = np.logical_and(freqs >= fmin, freqs <= fmax)
    band_power = np.trapz(psd[band_indices], freqs[band_indices])
    total_power = np.trapz(psd, freqs)

    if total_power == 0:
        return np.nan

    return band_power / total_power


def abs_individual_power(psd, freqs, band_peaks, individualized_band_ranges, band_name):
    """
    Calculates absolute power in an individualized frequency band centered around the dominant peak.

    Args:
        psd (np.ndarray): A 1D array of power spectral density values (in linear scale).
        freqs (np.ndarray): A 1D array of frequency values corresponding to the PSD.
        band_peaks (list of tuples): List of peak tuples (frequency, power, width).
        individualized_band_ranges (dict): Dictionary mapping band names to (lower_offset, upper_offset) in Hz.
        band_name (str): Name of the frequency band to compute power for.

    Returns:
        float: Log-transformed absolute power in the individualized frequency band. Returns np.nan if no peaks are found.
    """
    if not band_peaks or band_name not in individualized_band_ranges:
        return np.nan

    # Find the dominant peak
    dominant_peak = max(band_peaks, key=lambda x: x[1])
    peak_freq = dominant_peak[0]
    lower_offset, upper_offset = individualized_band_ranges[band_name]

    # Define the frequency range around the peak and find matching indices
    peak_range_indices = np.logical_and(freqs >= peak_freq + lower_offset,
                                        freqs <= peak_freq + upper_offset)

    band_power = np.trapz(psd[peak_range_indices], freqs[peak_range_indices])
    return np.log10(band_power)


def rel_individual_power(psd, freqs, band_peaks, individualized_band_ranges, band_name):
    """
    Calculates relative power in an individualized frequency band centered around the dominant peak.

    Args:
        psd (np.ndarray): A 1D array of power spectral density values (in linear scale).
        freqs (np.ndarray): A 1D array of frequency values corresponding to the PSD.
        band_peaks (list of tuples): List of peak tuples (frequency, power, width).
        individualized_band_ranges (dict): Dictionary mapping band names to (lower_offset, upper_offset) in Hz.
        band_name (str): Name of the frequency band to compute power for.

    Returns:
        float: Relative power in the individualized frequency band. Returns np.nan if total power is zero or input is invalid.
    """
    if not band_peaks or band_name not in individualized_band_ranges:
        return np.nan

    # Find the dominant peak
    dominant_peak = max(band_peaks, key=lambda x: x[1])
    peak_freq = dominant_peak[0]
    lower_offset, upper_offset = individualized_band_ranges[band_name]

    # Define the range around the peak frequency
    peak_range_indices = np.logical_and(freqs >= peak_freq + lower_offset,
                                        freqs <= peak_freq + upper_offset)

    band_power = np.trapz(psd[peak_range_indices], freqs[peak_range_indices])
    total_power = np.trapz(psd, freqs)

    if total_power == 0:
        return np.nan

    return band_power / total_power


def summarizeFeatures(df, extention, which_layout, which_sensor):
    """
    average features accroding to the layout. 
    """
    df.dropna(axis=0, how="all", inplace=True)
    summrized_df = pd.DataFrame(index=df.index)

    # TODO: If both meg and eeg is True, this won't work!
    if which_layout == "all":
        summrized_df[which_layout] = df.mean(axis=1)

    else:
        modality = [s_type for s_type, if_alculate in which_sensor.items() if if_alculate][0]
        
        layout_name = extention.upper() + "_" + modality.upper() + "_" + which_layout.upper()
        layout = load_specific_layout(extention.upper(), layout_name)
        
        for parcel_name, channels_list in layout.items():
            summrized_df[parcel_name] = df[list(channels_list)].mean(axis=1)
    
    return summrized_df


def psd_ratio(psd, freqs, freqRangeNumerator:float, freqRangeDenominator:float, channelNames:str, name:str, psdType:str):
    """
    this function calculates ratios of canonical frequency bands
    """

    # Numerator
    bandIndices = np.logical_and(freqs >= freqRangeNumerator[0] ,
                                freqs <= freqRangeNumerator[1])
    powerNumerator = np.trapz(psd[bandIndices], freqs[bandIndices])

    # Denominator
    bandIndices = np.logical_and(freqs >= freqRangeDenominator[0] ,
                                freqs <= freqRangeDenominator[1])
    powerDenominator = np.trapz(psd[bandIndices], freqs[bandIndices])

    # ratio
    featRow = np.log10(powerNumerator) / np.log10(powerDenominator)
    featName = f"{psdType}_Canonical_Absolute_Power_{name}_{channelNames}"

    return [featRow], [featName]


def create_feature_container(feature_categories, freq_bands, channel_names):
    # TODO if band_name != "broadband" although not necessary because we fill the
    # data frame with name (df.at())
    no_freq = ["Offset","Exponent","Peak_Center","Peak_Power","Peak_Width"]
    feature_names = []
    for feature, if_calculate in feature_categories.items():
        if if_calculate:
            if feature not in no_freq:
                for freq_band in freq_bands:
                    feature = feature + freq_band
                    feature_names.append(feature)
            else: feature_names.append(feature)
        else: continue

    return pd.DataFrame(columns=channel_names, index=feature_names)


def add_feature(feature_container, feature_arr, feature_name, channel_name, band_name):
    feature_name = feature_name + band_name
    feature_container.at[feature_name, channel_name] = feature_arr
    return feature_container


def feature_extract(subjectId, fmGroup, psds, feature_categories, freqs, freq_bands, 
                    channel_names, individualized_band_ranges, extention,
                    which_layout , which_sensor, aperiodic_mode, min_r_squared):
    """
    extract features from fooof results

    parameters
    -----------
    subjectId: str
    subject ID
    
    fmGroup: fooof object
    here, group represent channels

    psds: array
    original power spectrum (not flattened)

    freqs: list
    frequency values corresponding to each power value

    freq_bands: dictionary

    leastR2: float
    least accpetable r_squared in fitting fooof mdeols

    channel_names: list

    individualized_band_ranges: dict
    individualized frequency ranges

    return
    -------------
    featuresRow: list
    """

    # Store features in a pandas DataFrame with channel names as columns 
    # and feature names as the index,
    
    feature_container = create_feature_container(feature_categories, freq_bands, channel_names)

    for channel_num, channel_name in enumerate(channel_names):

        # getting the fooof model of ith channel
        fm = fmGroup.get_fooof(ind=channel_num)

        # fooof fitness
        # TODO, this needs to go out of feature exctraction
        r_squared = fm.r_squared_ 
        if r_squared < min_r_squared: continue

        # offset ==================================
        if feature_categories["Offset"]:
            feature_arr = offset(fm)
            feature_container = add_feature(feature_container, feature_arr, "Offset", channel_name, "")
        # Exponent ==================================
        if feature_categories["Exponent"]:
            feature_arr = exponent(fm, aperiodic_mode)
            feature_container = add_feature(feature_container, feature_arr, "Exponent", channel_name, "")
        
        original_psd = psds[channel_num, :]
        # isolate periodic parts of signals
        flattened_psd = np.asarray(isolate_periodic(fm, original_psd))
        
        # whenever aperidic activity is higher than periodic activity
        # => set the preiodic acitivity to zero
        flattened_psd = np.array(list(map(lambda x: max(0, x), flattened_psd)))

        #Loop through each frequency band
        for band_name, (fmin, fmax) in freq_bands.items():

            # Peak Features ==================================
            band_peaks = find_peak_in_band(fm, fmin, fmax)

            if feature_categories["Peak_Center"]:
                feature_arr = peak_center(band_peaks)
                feature_container = add_feature(feature_container, feature_arr, "Peak_Center", channel_name, band_name)

            if feature_categories["Peak_Power"]:
                feature_arr = peak_power(band_peaks)
                feature_container = add_feature(feature_container, feature_arr, "Peak_Power", channel_name, band_name)

            if feature_categories["Peak_Width"]:
                feature_arr = peak_width(band_peaks)
                feature_container = add_feature(feature_container, feature_arr, "Peak_Width", channel_name, band_name)

            # Adjusted absolute canonical power ==================================
            if feature_categories["Adjusted_Canonical_Absolute_Power"]:
                feature_arr = abs_canonical_power(psd=flattened_psd, freqs=freqs, fmin=fmin, fmax=fmax)
                feature_container = add_feature(feature_container, feature_arr, "Adjusted_Canonical_Absolute_Power", channel_name, band_name)

            # Adjusted relative canonical power ==================================
            if feature_categories["Adjusted_Canonical_Relative_Power"]:
                feature_arr = rel_canonical_power(psd=flattened_psd, freqs=freqs, fmin=fmin, fmax=fmax)
                feature_container = add_feature(feature_container, feature_arr, "Adjusted_Canonical_Relative_Power", channel_name, band_name)

            # OriginalPSD absolute canonical power ==================================
            if feature_categories["OriginalPSD_Canonical_Absolute_Power"]:
                feature_arr = abs_canonical_power(psd=original_psd, freqs=freqs, fmin=fmin, fmax=fmax)
                feature_container = add_feature(feature_container, feature_arr, "OriginalPSD_Canonical_Absolute_Power", channel_name, band_name)
                
            # OriginalPSD relative canonical power ==================================
            if feature_categories["OriginalPSD_Canonical_Relative_Power"]:
                feature_arr = rel_canonical_power(psd=original_psd, freqs=freqs, fmin=fmin, fmax=fmax)
                feature_container = add_feature(feature_container, feature_arr, "OriginalPSD_Canonical_Relative_Power", channel_name, band_name)

            if band_name != "Broadband" and band_peaks:

                # Adjusted absolute Relative power ==================================
                if feature_categories["Adjusted_Individualized_Absolute_Power"]:
                    feature_arr = abs_individual_power(psd=flattened_psd, freqs=freqs, band_peaks=band_peaks, 
                                individualized_band_ranges=individualized_band_ranges, band_name=band_name)  
                    feature_container = add_feature(feature_container, feature_arr, "Adjusted_Individualized_Absolute_Power", channel_name, band_name)

                # Adjusted relative Relative power ==================================
                if feature_categories["Adjusted_Individualized_Relative_Power"]:
                    feature_arr = rel_individual_power(psd=flattened_psd, freqs=freqs, band_peaks=band_peaks, 
                                individualized_band_ranges=individualized_band_ranges, band_name=band_name)  
                    feature_container = add_feature(feature_container, feature_arr, "Adjusted_Individualized_Relative_Power", channel_name, band_name)
                
                # OriginalPSD absolute Relative power ==================================
                if feature_categories["OriginalPSD_Individualized_Absolute_Power"]:
                    feature_arr = abs_individual_power(psd=original_psd, freqs=freqs, band_peaks=band_peaks, 
                                individualized_band_ranges=individualized_band_ranges, band_name=band_name)    
                    feature_container = add_feature(feature_container, feature_arr, "OriginalPSD_Individualized_Absolute_Power", channel_name, band_name)
     
                # OriginalPSD relative Relative power ==================================
                if feature_categories["OriginalPSD_Individualized_Relative_Power"]:
                    feature_arr = rel_individual_power(psd=original_psd, freqs=freqs, band_peaks=band_peaks, 
                                individualized_band_ranges=individualized_band_ranges, band_name=band_name)
                    feature_container = add_feature(feature_container, feature_arr, "OriginalPSD_Individualized_Relative_Power", channel_name, band_name)

    # # feature summarization ================================================================ 
    if which_layout:
        feature_container = summarizeFeatures(df=feature_container, extention=extention,
                                            which_layout=which_layout, which_sensor=which_sensor)

    # Flatten the DataFrame and create neww column names
    final_df = pd.DataFrame(feature_container.values.flatten()).T 
    final_df.columns = [f"{index}_{col}" for index in feature_container.index for col in feature_container.columns]

    final_df.index = [subjectId]
    
    return final_df
    






if __name__ == "__main__":



    parser = argparse.ArgumentParser()

    parser.add_argument("dir", type=str,
            help="data directory (pickle format)")
    parser.add_argument("savePath", type=str,
            help="where to save data")
	# optional arguments
    parser.add_argument("--configs", type=str, default=None,
        help="Address of configs json file")
    
    args = parser.parse_args()




    with open(args.dir, "rb") as fooofFile:
        counter = 0
        while True:
            try: pickle.load(fooofFile) ; counter += 1
            except: break

    # Loading configs
    if args.configs is not None:
        with open(args.configs, 'r') as f:
            configs = json.load(f)
    else: configs = make_config()
        
    


    with open(args.dir, "rb") as fooofFile:
    
        for j in tqdm.tqdm(range(counter)):
            
            subjectId, (fmGroup, psds, freqs) = next(iter(pickle.load(fooofFile).items()))


            features = featureExtract(subjectId=subjectId,
                                    fmGroup=fmGroup,
                                    psds=psds,
                                    freqs=freqs,
                                    freq_bands=configs["freq_bands"],
                                    channelNames=configs['ch_names'],
                                    bandSubRanges=configs["bandSubRanges"],
                                    featureCategories=configs["featureCategories"])  

            features.to_csv(f"{subjectId}.csv")          
            
    
    

    

