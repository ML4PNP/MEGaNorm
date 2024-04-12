import h5py
import tqdm
import fooof
import numpy as np
import pickle
import itertools

from dataManagementUtils import readFooofres, subjectList, saveFeatures
from config.config import freqBands, bandSubRanges
from processUtils import Features, isNan


def featureEx(subjectId, fmGroup, psds, freqs, freqBands, leastR2):
    """
    This function extract features from periodic data
    and save them along with aperiodic paramereters
    """

    features = Features.fmFeaturesContainer(psds.shape[0], freqBands)

    for i in range(psds.shape[0]):

        # getting the fooof model of ith channel
        fm = fmGroup.get_fooof(ind=i)

        # if fooof model is overfitted => exclude the channel
        if fm.r_squared_ < leastR2: continue

    
        features['offset'][i] = fm.get_params('aperiodic_params')[0]
        features['exponent'][i] = fm.get_params('aperiodic_params')[1]
        
        # Using linear frequency space when calculating powers 
        flattened_psd = psds[i, :] - 10**fm._ap_fit 
        
        # Compute total power in the flattened spectrum for normalization
        total_power_flattened = np.trapz(flattened_psd, freqs)
        
        # Loop through each frequency band
        for band, (fmin, fmax) in freqBands.items():
            
            if band != 'Broadband':
                ################################# Power Features ##############################
                
                # Find indices of frequencies within the current band
                band_indices = np.logical_and(freqs >= fmin,
                                            freqs <= fmax)
                
                # Integrate the power within the band on the flattened spectrum
                band_power_flattened = np.trapz(flattened_psd[band_indices], 
                                                freqs[band_indices])
                
                # Compute the average relative power for the band on the flattened spectrum
                features['canonical_band_power'][band][i] = band_power_flattened / total_power_flattened
                
                ################################# Peak Features ##############################
            
            # Filter the peaks that fall within the current frequency band
            band_peaks = []
            for peak in fm.get_params('peak_params'):
                if np.any(peak != peak):
                    band_peaks = [np.nan, np.nan, np.nan]
                else: 
                    band_peaks = [peak for peak in 
                                fm.get_params('peak_params') if fmin <= peak[0] <= fmax] 


            # band_peaks = [peak for peak in 
            #               fm.get_params('peak_params') if fmin <= peak[0] <= fmax]

            
            # If there are peaks within this band, find the one with the highest amplitude
            if band_peaks and not np.any(np.array(band_peaks) != np.array(band_peaks)):
                # Sort the peaks by amplitude (peak[1]) and get the frequency of the highest amplitude peak
                dominant_peak = max(band_peaks, key=lambda x: x[1])
                features['dominant_peak_freqs'][band][i] = dominant_peak[0] 
                features['dominant_peak_power'][band][i] = dominant_peak[1]  
                features['dominant_peak_width'][band][i] = dominant_peak[2] 

                # Individualized Band Power 
                if band != 'Broadband':
                    # Define the range around the peak frequency and Find indices of frequencies within this range
                    peak_range_indices = np.logical_and(freqs >= dominant_peak[0] + 
                                                            bandSubRanges[band][0], 
                                                        freqs <= dominant_peak[0] + 
                                                            bandSubRanges[band][1])
                    
                    # Integrate power within the range on the flattened spectrum
                    
                    #  the power within the band on the flattened spectrum
                    avg_power = np.trapz(flattened_psd[peak_range_indices], freqs[peak_range_indices])
                
                    # Compute the average relative power for the band on the flattened spectrum
                    features['individualized_band_power'][band][i] = avg_power / total_power_flattened

            elif not band_peaks and np.any(band_peaks != band_peaks):
                features['dominant_peak_freqs'][band][i] = np.nan
                features['dominant_peak_power'][band][i] = np.nan 
                features['dominant_peak_width'][band][i] = np.nan
                if band != "Broadband":
                    features['individualized_band_power'][band][i] = np.nan

        featureRow = []
        for key1, value1 in features.items():
            print(np.shape(value1))
            # for key2, value2 in value1.items():
            #     featureRow.extend(value2)

    return features





if __name__ == "__main__":

    with open("data/fooofResults/fooofModels.pkl", "rb") as fooofFile:
        counter = 0
        while True:
            try: pickle.load(fooofFile) ; counter += 1
            
            except: break
    
    print(counter)
    savePath = "/home/zamanzad/trial1/data/features/featureMatrix.csv"
    leastR2 = 0.9 # least acceptable R squred of fitted models


    with open("data/fooofResults/fooofModels.pkl", "rb") as fooofFile:
       
        for j in tqdm.tqdm(range(counter)):
            
            subjectId, (fmGroup, psds, freqs) = next(iter(pickle.load(fooofFile).items()))
            print(subjectId)

            featureSet = featureEx(subjectId, fmGroup, psds, freqs, freqBands, leastR2)
            break

    #         saveFeatures(savePath, featureSet)
    
    

    

