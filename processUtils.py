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
    def fooofModeling(segments, spectrumMethod, fs, freqRange=[1, 60]):

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
        fooofModels = f.FOOOFGroup(peak_width_limits=[1, 12.0], 
                                   min_peak_height=0, 
                                   peak_threshold=2, 
                                   aperiodic_mode='fixed')
        fooofModels.fit(freqs, psds, freqRange, n_jobs=-1)


        return fooofModels
    

    def extractFooofFeatures():
        pass





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
        
        print("hereeeeee: ",np.shape(psds))

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
            periodic.append(np.log10(psds[channelNum, desiredFreq]) - aperiodic) # change periodic to flattened

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
        # neuralParameterize.plotFooof(periodic, aperiodics, periodicsPeaks, fmDesired, freqs, desiredInd)

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









        



