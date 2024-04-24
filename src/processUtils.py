from fooof.plts.annotate import plot_annotated_model
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import fooof as f
import json
import mne





class neuralParameterize():

    """
    this class of functions tries to parametrize neural spectrum,
    modeling both periodic and aperiodic parts of M/EEG signals
    """


    @staticmethod
    def fooofModeling(segments, freqRangeLow, freqRangeHigh, min_peak_height,
                     peak_threshold, fs, psdMethod, psd_n_overlap, psd_n_fft):

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
        

        psds, freqs = segments.compute_psd( 
                        method = psdMethod,
                        fmin = freqRangeLow,
                        fmax = freqRangeHigh,
                        n_jobs = -1,
                        average = "mean",
                        # having at least two cycles of 1Hz
                        n_overlap = psd_n_overlap*fs, #1
                        n_fft = psd_n_fft*fs, #2
                        n_per_seg = 2*fs, 
                        verbose=False).average().get_data(return_freqs=True)
        
        NumChannel = segments.get_data().shape[0]       

        # fitting seperate models for each channel
        fooofModels = f.FOOOFGroup(peak_width_limits=[1, 12.0], 
                                   min_peak_height=min_peak_height, 
                                   peak_threshold=peak_threshold, 
                                   aperiodic_mode='fixed')
        fooofModels.fit(freqs, psds, [freqRangeLow, freqRangeHigh], n_jobs=-1)

        return fooofModels, psds, freqs
    




    @staticmethod
    def fooofModeling2(segments, spectrumMethod, fs, freqRange=[1, 60]):

        """
        Deprecated

        
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
        periodics: ndarray
        periodics signals

        aperiodics: ndarray
        aperiodics signals

        freqs: array
        list of frequencies
        """

        
        
        # welch PSD
        if spectrumMethod == "welch":
            psds, freqs = segments.compute_psd( 
                            method = spectrumMethod,
                            fmin = 3,
                            fmax = 40,
                            n_jobs = -1,
                            average = "mean",
                            # having at least two cycles of 1Hz
                            n_overlap = 1*fs,
                            n_fft = 2*fs,
                            n_per_seg = 2*fs, 
                            verbose=False).average().get_data(return_freqs=True)
        

        NumChannel = segments.get_data().shape[0]


        # fitting seperate models for each channel
        foofModels = f.FOOOFGroup()
        foofModels.fit(freqs, psds, freqRange, n_jobs=-1)

        # looping over channels and getting periodic
        # and aperiodic parts
        aperiodics , periodicsPeaks, periodic = [], [], []
        for channelNum in range(NumChannel) : 
            # full model
            fm = foofModels.get_fooof(ind=channelNum)
            
            aperiodic = fm.get_model(component="aperiodic", space="log")
            aperiodics.append(fm.get_model(component="aperiodic"))

            periodicPeak = fm.get_model(component="peak")
            periodicsPeaks.append(periodicPeak)

            # those psd values between freqRange[0] and freqRange[1]
            desiredFreq = np.where(np.logical_and(freqs >= freqRange[0],
                                        freqs <= freqRange[-1]))[0]
            # since we want to save both periodic peaks and 
            # full signal - "periodic peaks"
            periodic.append(np.log10(psds[channelNum, desiredFreq]) - aperiodic) 

            # this is just for the sake of plotting the whole model 
            # for a specific index, therefore, you can comment the next
            # three lines!
            desiredInd = 10
            if channelNum == desiredInd:
                fmDesired = foofModels.get_fooof(ind=channelNum)
            
        periodic = np.vstack(periodic)
        periodicsPeaks = np.vstack(periodicsPeaks)
        aperiodics = np.vstack(aperiodics)
        
        # must be <= and >= to be correct!
        freqs = list(filter(lambda x: freqRange[0]<=x<=freqRange[1], freqs))

        aperiodicParams = foofModels.get_params('aperiodic_params')
        periodicParams = foofModels.get_params('peak_params')

        # optional (just for visualization)
        # neuralParameterize.plotFooof(periodic, aperiodics, 
            #periodicsPeaks, fmDesired, freqs, desiredInd)

        return periodicsPeaks, aperiodics, periodic, freqs, aperiodicParams, periodicParams
    



    
    @staticmethod
    def plotFooof(periodics, aperiodics, periodicsPeaks, fmDesired, freqs, ind):
        """
        Deprecated
        plot the results
        """
        fig, ax = plt.subplots(2,2, figsize=(25, 20))
        
        ax[0,0].plot(freqs, periodicsPeaks[ind,:], color="black")
        ax[0,0].set_title("Periodic peaks")

        ax[0,1].plot(freqs, aperiodics[ind,:], color="black")
        ax[0,1].set_title("Aperiodic signal")


        plot_annotated_model(fmDesired, ax=ax[1,0])
        ax[1,0].set_title("Full annotated model")
        
        ax[1,1].plot(freqs, periodics[ind,:], color="black")
        ax[1,1].set_title("original signal - aperiodic part")

        plt.savefig("pictures/fooof/fooofRes3.png")



def isNan(value):
    """
    if value is nan: True
    else: false
    """
    return value != value




class features:


    @staticmethod
    def bandExtraction(sig, freqs, freqRange:tuple):
        """
        This function return power values related
        a specific frequency band
        """

        freqs = np.asarray(freqs)
        indexes = np.where(np.logical_and(freqs>=freqRange[0],
                                         freqs<=freqRange[-1]))[0].tolist()

        return sig[:, indexes]
    



    def meanPower(sig, freqs, freqBands):
        """
        this function calculate mean or median power
        for different canonical frequency bands

        parameters
        -----------
        sig: ndarray
        PSDS 

        freqs: list
        list of frequencies values

        freqs: dict
        canonical frequency bands

        return:
        ----------
        mean power of each frequency bands

        """
        
        delta = np.mean(features.bandExtraction(sig, freqs, 
                                                freqBands.get("delta")),axis=1)
        theta = np.mean(features.bandExtraction(sig, freqs, 
                                                freqBands.get("theta")), axis=1)
        alpha = np.mean(features.bandExtraction(sig, freqs, 
                                                freqBands.get("alpha")), axis=1)
        beta = np.mean(features.bandExtraction(sig, freqs, 
                                               freqBands.get("beta")), axis=1)
        gamma = np.mean(features.bandExtraction(sig, freqs, 
                                                freqBands.get("gamma")), axis=1)

        return delta, theta, alpha, beta, gamma
    







class normative:
    """
    A class of various functions, related to
    normative modeling
    """

    @staticmethod
    def prepareData(featurePath:str, metaDataPath:str) -> None:
        """
        This function reads the feature and target
        matrices, combines them, and then saves them
        as two separate .txt files for future modeling purposes.
        """

        # in future version, you must have specified features
        # name before getting to this step
        with open ("data/features/featuresNames.json", "r") as file:
            featNames = json.load(file)
        featMat = pd.read_csv(featurePath, header=None, names=featNames)
        metaData = pd.read_csv(metaDataPath, sep="\t")

        featMat = featMat.merge(metaData, on="participant_id", how="left")
        featMat.dropna(axis=0, inplace=True)

        covMat = featMat[["gender_code", "age"]]
        

        # getting average values over channels (nice method, hah?!)
        # remember, including non-int will results in an errror
        featMat = featMat.iloc[:,:-5]
        featMat = featMat.set_index(["participant_id"])
        featMat = featMat.T.groupby(
            lambda x: x.split(" ")[0]).mean(numeric_only=True).T
        
        # print("heeeere: ", featMat.head())

        covPath = "data/normativeInput/covariateNormSample.txt"
        featPath = "data/normativeInput/responseVarNorm.txt"

        covMat.to_csv(covPath,
                      sep=" ",
                      header=False,
                      index=False)
        
        featMat.to_csv(featPath,
                       sep=" ",
                       header=False,
                       index=False)
        


        return covPath, featPath
    

    @staticmethod
    def covForward(min, max, step):
        """
        This function create a covariate matrix for forward
        modeling
        
        parameters
        -------------
        min: int
        the lowest age that your model can represent

        max: int
        the highest age that your model can represent
        
        step:int
        bins

        return
        ------------
        savePath: str
        forward covariate path
        """

        bins = int(((max - min)/ step)+1)

        ageCov = np.arange(min, max, step).tolist()
        sex0 = (np.ones(bins).astype("int32")+1).tolist()
        sex1 = np.ones(bins).astype("int32").tolist()

        covariateForward = {"sex": sex0 + sex1,
                            "age": ageCov + ageCov}
        
        savePath = "data/normativeInput/'covariate_forwardmodel.txt'"
        pd.DataFrame(covariateForward).to_csv(savePath,
                                            sep = ' ',
                                            header = False,
                                            index = False)

        return savePath
    




def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))







class Features:


    @staticmethod
    def apperiodicFeatures(fm, channelNames):
        """
        returns aperiodic features (exponent and offset) and their names
        """
        featRow, featName = [], []

        # feature value
        featRow.extend([fm.get_params("aperiodic_params")[0], fm.get_params("aperiodic_params")[1]])
        # add feature name
        featName.extend([f"offset - {channelNames}", f"exponent - {channelNames}"])

        return featRow, featName


    
    @staticmethod
    def peakParameters(fm, freqs, fmin, fmax, channelNames, bandName):
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
            
            # frequency, power and width of dominant peak
            featRow.extend([dominant_peak[0], dominant_peak[1] , dominant_peak[2]])   
            featName.extend([f"(1) frequency of dominant peak - {bandName} - {channelNames}", 
                                f"(2) power of dominant peak - {bandName} - {channelNames}" , 
                                f"(3) width of dominant peak - {bandName} - {channelNames}"])
        
        # in case there is no peak for a certain frequency band
        else:
            featRow.extend([np.nan, np.nan, np.nan])   
            featName.extend([f"(1) frequency of dominant peak - {bandName} - {channelNames}", 
                                f"(2) power of dominant peak - {bandName} - {channelNames}" , 
                                f"(3) width of dominant peak - {bandName} - {channelNames}"])  
            dominant_peak = np.nan                 
                

        if not band_peaks: nanFlag = True

        
        # dominant peak is returned in case you want to calcualte 
        # individualized band power
        return featRow, featName, dominant_peak, nanFlag
    


    @staticmethod
    def flatPeriodic(fm, psd):
        """
        this function isolate periodic parts of signal
        through subtracting aperiodic fit from original psds
        """
        return psd - 10**fm._ap_fit 
    


    @staticmethod
    def canonicalPower(flattenedPSD, freqs, fmin, fmax, channelNames, bandName, psdType):
        """
        calculate adjusted (isolated periodic data) relative and absolute 
        power for canonical band powers.
        """

        totalPowerFlattened = np.trapz(flattenedPSD, freqs)

        # Find indices of frequencies within the current band
        bandIndices = np.logical_and(freqs >= fmin, freqs <= fmax)
        
        # Integrate the power within the band on the flattened spectrum
        bandPowerFlattened = np.trapz(flattenedPSD[bandIndices], freqs[bandIndices])
        
        # Compute the average relative power for the band on the flattened spectrum
        RelFeatRow = bandPowerFlattened / totalPowerFlattened
        RelFeatName = f"(4) Canonical Relative Power - {bandName} - {channelNames} ({psdType})"
        # Compute the average absolute power for the band on the flattened spectrum
        AbsFeatRow = np.log10(np.abs(bandPowerFlattened))
        AbsFeatName = f"(5) Canonical Absolute Power - {bandName} - {channelNames} ({psdType})"

        return RelFeatRow, RelFeatName, AbsFeatRow, AbsFeatName
    
    

    @staticmethod
    def individulizedPower(flattenedPsd, total_power_flattened, dominant_peak, freqs, bandSubRanges, nanFlag, bandName, channelNames, psdType):
        """
        This function calculates total power for individualized 
        frequency range
        """
        
        if nanFlag == False: 
            # Define the range around the peak frequency and Find indices of frequencies within this range
            peak_range_indices = np.logical_and(freqs >= dominant_peak[0] + bandSubRanges[bandName][0], 
                                                freqs <= dominant_peak[0] + bandSubRanges[bandName][1])
            # Integrate power within the range on the flattened spectrum
            
            #  the power within the band on the flattened spectrum
            avg_power = np.trapz(flattenedPsd[peak_range_indices], freqs[peak_range_indices])
        
            # Compute the average relative power for the band on the flattened spectrum
            RelFeatRow = avg_power / total_power_flattened ; RelFeatName = f"Individualized Relative Power - {bandName} - {channelNames} ({psdType})"

            # Compute the average absolute power for the band on the flattened spectrum
            AbsFeatRow = np.log10(abs(avg_power)) ; AbsFeatName = f"Individualized Absolute Power - {bandName} - {channelNames} ({psdType})" 


        else:
            RelFeatRow = np.nan ; RelFeatName = f"(6) Individualized Relative Power - {bandName} - {channelNames} ({psdType})"
            AbsFeatRow = np.nan ; AbsFeatName = f"(7) Individualized Absolute Power - {bandName} - {channelNames} ({psdType})"

        return RelFeatRow, RelFeatName, AbsFeatRow, AbsFeatName
    




    





    @staticmethod
    def featureEx1(subjectId, fmGroup, psds, freqs, freqBands, leastR2):
        """
        Deprecated
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



