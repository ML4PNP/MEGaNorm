import numpy as np
import fooof as f







def apperiodicFeatures(fm, channelNames, featureCategories):
    """
    returns aperiodic features (exponent and offset) and their names
    """
    featRow, featName = [], []
    
    if "offset" in featureCategories:
        featRow.append(fm.get_params("aperiodic_params")[0])
        featName.append(f"offset_{channelNames}")

    if "exponent" in featureCategories:
        featRow.append(fm.get_params("aperiodic_params")[1])
        featName.append(f"exponent_{channelNames}")

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

        if "frequency_dominant_peak" in featureCategories:
            featRow.append(dominant_peak[0])
            featName.append(f"frequency_dominant_peak_{bandName}_{channelNames}")

        if "power_dominant_peak" in featureCategories:
            featRow.append(dominant_peak[1])
            featName.append(f"power_dominant_peak_{bandName}_{channelNames}")

        if "width_dominant_peak" in featureCategories:
            featRow.append(dominant_peak[2])
            featName.append(f"width_dominant_peak_{bandName}_{channelNames}")

    # in case there is no peak for a certain frequency band
    else:

        if "frequency_dominant_peak" in featureCategories:
            featRow.append(np.nan)
            featName.append(f"frequency_dominant_peak_{bandName}_{channelNames}")

        if "power_dominant_peak" in featureCategories:
            featRow.append(np.nan)
            featName.append(f"power_dominant_peak_{bandName}_{channelNames}")

        if "width_dominant_peak" in featureCategories:
            featRow.append(np.nan)
            featName.append(f"width_dominant_peak_{bandName}_{channelNames}")

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
    if "Canonical_Relative_Power" in featureCategories:
        featVal.append(bandPowerFlattened / totalPower)
        featName.append(f"Canonical_Relative_Power_{bandName}_{psdType}_{channelNames}")

    # Compute the average relative power for the band on the flattened spectrum
    if "Canonical_Absolute_Power" in featureCategories:
        featVal.append(np.log10(np.abs(bandPowerFlattened)))
        featName.append(f"Canonical_Absolute_Power_{bandName}_{psdType}_{channelNames}")


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
            featName.append(f"Individualized_Relative_Power_{bandName}_{psdType}_{channelNames}")


        # Compute the average absolute power for the band on the flattened spectrum
        if "Individualized_Absolute_Power" in featureCategories:
            featRow.append(np.log10(abs(avg_power)))
            featName.append(f"Individualized_Absolute_Power_{bandName}_{psdType}_{channelNames}")
        

    else:
        # Compute the average relative power for the band on the flattened spectrum
        if "Individualized_Relative_Power" in featureCategories:
            featRow.append(np.nan)
            featName.append(f"Individualized_Relative_Power_{bandName}_{psdType}_{channelNames}")


        # Compute the average absolute power for the band on the flattened spectrum
        if "Individualized_Absolute_Power" in featureCategories:
            featRow.append(np.nan)
            featName.append(f"Individualized_Absolute_Power_{bandName}_{psdType}_{channelNames}")

    return featRow, featName