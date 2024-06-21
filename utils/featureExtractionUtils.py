import pandas as pd
import numpy as np
import fooof as f
import sys
import os

# Add utils folder to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
layout_path = os.path.join(parent_dir, 'layouts')
sys.path.append(layout_path)

from layouts import load_specific_layout





def apperiodicFeatures(fm, channelNames, featureCategories):
    """
    returns aperiodic features (exponent and offset) and their names
    """
    featRow, featName = [], []
    
    if "Offset" in featureCategories:
        featRow.append(fm.get_params("aperiodic_params")[0])
        featName.append(f"Offset_{channelNames}")

    if "Exponent" in featureCategories:
        featRow.append(fm.get_params("aperiodic_params")[1])
        featName.append(f"Exponent_{channelNames}")

    return featRow, featName







def peakParameters(fm, fmin, fmax, channelNames, bandName, featureCategories):
    """
    This function returns peak parmaters:
    1. Dominant peak frequency
    2. Dominant peak power
    3. Dominant peak width
    """

    featRow, featName = [], []
    nanFlag = False
    

    # checking for possible nan values
    band_peaks = [peak for peak in 
                fm.get_params('peak_params') if np.any(peak == peak)]

    band_peaks = [peak for peak in band_peaks if fmin <= peak[0] <= fmax]
    
    # If there are peaks within this band, find the one with the highest amplitude
    if band_peaks:
        # Sort the peaks by amplitude (peak[1]) and get the frequency of the highest amplitude peak
        dominant_peak = max(band_peaks, key=lambda x: x[1])

        if "Peak_Center" in featureCategories:
            featRow.append(dominant_peak[0])
            featName.append(f"Peak_Center_{bandName}_{channelNames}")

        if "Peak_Power" in featureCategories:
            featRow.append(dominant_peak[1])
            featName.append(f"Peak_Power_{bandName}_{channelNames}")

        if "Peak_Width" in featureCategories:
            featRow.append(dominant_peak[2])
            featName.append(f"Peak_Width_{bandName}_{channelNames}")

    # in case there is no peak for a certain frequency band
    else:

        if "Peak_Center" in featureCategories:
            featRow.append(np.nan)
            featName.append(f"Peak_Center_{bandName}_{channelNames}")

        if "Peak_Power" in featureCategories:
            featRow.append(np.nan)
            featName.append(f"Peak_Power_{bandName}_{channelNames}")

        if "Peak_Width" in featureCategories:
            featRow.append(np.nan)
            featName.append(f"Peak_Width_{bandName}_{channelNames}")

        dominant_peak = np.nan                 
            

    if not band_peaks: nanFlag = True

    
    # dominant peak is returned in case you want to calcualte 
    # individualized band power
    return featRow, featName, dominant_peak, nanFlag








def isolatePeriodic(fm, psd):
    """
    this function isolate periodic parts of signal
    through subtracting aperiodic fit from original psds
    """
    return psd - 10**fm._ap_fit 






def canonicalPower(psd, freqs, fmin, fmax, channelNames, bandName, psdType, featureCategories):
    """
    calculate adjusted (isolated periodic data) relative and absolute 
    power for canonical band powers.
    """

    totalPower = np.trapz(psd, freqs)

    # Find indices of frequencies within the current band
    bandIndices = np.logical_and(freqs >= fmin, freqs <= fmax)
    
    # Integrate the power within the band on the flattened spectrum
    bandPowerFlattened = np.trapz(psd[bandIndices], freqs[bandIndices])

    featVal, featName = [], []

    # Compute the average relative power for the band on the flattened spectrum
    if "Canonical_Absolute_Power" in featureCategories:
        featVal.append(np.log10(bandPowerFlattened))
        featName.append(f"{psdType}_Canonical_Absolute_Power_{bandName}_{channelNames}")
    # Compute the average relative power for the band on the flattened spectrum
    if "Canonical_Relative_Power" in featureCategories:
        if bandName=="Broadband":
            return featVal, featName
        featVal.append(bandPowerFlattened / totalPower)
        featName.append(f"{psdType}_Canonical_Relative_Power_{bandName}_{channelNames}")

    return featVal, featName







def individulizedPower(psd, dominant_peak, freqs, bandSubRanges, nanFlag, bandName, channelNames, psdType, featureCategories):
    """
    This function calculates total power for individualized 
    frequency range
    """

    featRow, featName = [], []
    
    if nanFlag == False: 

        total_power = np.trapz(psd, freqs)

        # Define the range around the peak frequency and Find indices of frequencies within this range
        peak_range_indices = np.logical_and(freqs >= dominant_peak[0] + bandSubRanges[bandName][0], 
                                            freqs <= dominant_peak[0] + bandSubRanges[bandName][1])
        # Integrate power within the range on the flattened spectrum
        
        #  the power within the band on the flattened spectrum
        avg_power = np.trapz(psd[peak_range_indices], freqs[peak_range_indices])
    
        # Compute the average relative power for the band on the flattened spectrum
        if "Individualized_Relative_Power" in featureCategories:
            featRow.append(avg_power / total_power)
            featName.append(f"{psdType}_Individualized_Relative_Power_{bandName}_{channelNames}")


        # Compute the average absolute power for the band on the flattened spectrum
        if "Individualized_Absolute_Power" in featureCategories:
            featRow.append(np.log10(avg_power))
            featName.append(f"{psdType}_Individualized_Absolute_Power_{bandName}_{channelNames}")
        

    else:
        # Compute the average relative power for the band on the flattened spectrum
        if "Individualized_Relative_Power" in featureCategories:
            featRow.append(np.nan)
            featName.append(f"{psdType}_Individualized_Relative_Power_{bandName}_{channelNames}")


        # Compute the average absolute power for the band on the flattened spectrum
        if "Individualized_Absolute_Power" in featureCategories:
            featRow.append(np.nan)
            featName.append(f"{psdType}_Individualized_Absolute_Power_{bandName}_{channelNames}")

    return featRow, featName





def summarizeFeatures(df, device, layout_name):

    """
    average features across the whole brain. 
    """
    
    layout = load_specific_layout(device, layout_name)
    
    summarized = []

    for layoutName, layoutRow in layout.items():

        sensorType = layoutName.split("_")[0]
        if sensorType in ["MAG", "GRAD"]: sensorType = ""

        parcell = df.loc[:, [col for col in df.columns if col.split("_")[-1] in layoutRow]]
        categories = set()

        for name in parcell.columns:
            if "Offset" in name or "Exponent" in name:
                categories.add(name.split("_")[:-1][0] + f"{sensorType}")
            elif "r_squared" not in name and "participant_ID" not in name:
                categories.add("_".join(name.split("_")[:-1]) + f"_{sensorType}")
        
        if sensorType in ["MAG", "GRAD"]:
            dfs = [parcell.loc[:, parcell.columns.str.startswith(uniqueName[:-5])].mean(axis=1) for uniqueName in categories]
        else:
            dfs = [parcell.loc[:, parcell.columns.str.startswith(uniqueName)].mean(axis=1) for uniqueName in categories]

        averaged = pd.concat(dfs, axis=1); averaged.columns = list(categories)
        

        summarized.append(averaged)
    summarized = pd.concat(summarized, axis=1)

    summarized.sort_index(axis=1, inplace=True)

    return summarized