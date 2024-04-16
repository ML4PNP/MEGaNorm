import mne
import h5py
import glob
import tqdm
import fooof
import json
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt

from dataManagementUtils import readFooofres, subjectList, saveFeatures
from config.config import freqBands, bandSubRanges
from processUtils import Features, isNan





def featureEx(subjectId, fmGroup, psds, freqs, freqBands, leastR2, channelNmaes):
    """
    This function extract features from periodic data
    and save them along with aperiodic paramereters
    """


    # in order to save features and their name
    featuresRow, FeaturesName = [], []

    for i in range(psds.shape[0]):
        

        # getting the fooof model of ith channel
        fm = fmGroup.get_fooof(ind=i)

        # if fooof model is overfitted => exclude the channel
        if fm.r_squared_ < leastR2: 
            empty = np.empty(25)
            empty[:] = np.nan
            featuresRow.extend(empty.tolist())
            continue

        # # ################################# exponent and offset ##############################
        featRow, featName = Features.apperiodicFeatures(fm=fm, channelNmaes=channelNmaes[i])
        featuresRow.extend(featRow); FeaturesName.extend(featName)
        # #===================================================================================== 

        # isolate periodic parts of signals
        flattenedPsd = Features.flatPeriodic(fm, psds[i, :])
        
        totalPowerFlattened = np.trapz(flattenedPsd, freqs)

        #Loop through each frequency band
        for bandName, (fmin, fmax) in freqBands.items():

            if bandName != "Broadband": 

                # ################################# Power Features ##############################
                featRow, featName = Features.canonicalBandPower(flattenedPsd, 
                                                                freqs, 
                                                                fmin, 
                                                                fmax, 
                                                                channelNmaes[i], 
                                                                bandName)
                featuresRow.append(featRow); FeaturesName.append(featName)
                #================================================================================ 



        #     ################################# Peak Features ################################
            (featRow, 
            featName, 
            dominant_peak,
            nanFlag) = Features.peakParameters(fm, 
                                                freqs, 
                                                fmin, 
                                                fmax, 
                                                channelNmaes[i], 
                                                bandName)
            featuresRow.extend(featRow); FeaturesName.extend(featName)
        #     #================================================================================


        #     ################################# Individualized band power ################################
            if bandName != "Broadband":
                featRow, featName = Features.individulizedBandPower(flattenedPsd, 
                                                                    totalPowerFlattened, 
                                                                    dominant_peak, 
                                                                    freqs, 
                                                                    bandSubRanges, 
                                                                    nanFlag, 
                                                                    bandName, 
                                                                    channelNmaes[i])
                featuresRow.extend(featRow); FeaturesName.extend(featName)
                # 
        #     #============================================================================================

    if len(FeaturesName) == 7650:
        with open("data/features/featuresNames.josn", "w") as file:
            json.dump(FeaturesName, file)
    
    
    return featuresRow






if __name__ == "__main__":

    with open("data/fooofResults/fooofModels.pkl", "rb") as fooofFile:
        counter = 0
        while True:
            try: pickle.load(fooofFile) ; counter += 1
            except: break

    basPath = "/home/smkia/Data/CamCAN/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/*/*.fif"
    dataPaths = glob.glob(basPath)
    raw = mne.io.read_raw_fif(dataPaths[0]).pick(picks="meg")
    channelNmaes = raw.info['ch_names']
    
    savePath = "data/features/featureMatrix.csv"
    leastR2 = 0.9 # least acceptable R squred of fitted models


    with open("data/fooofResults/fooofModels.pkl", "rb") as fooofFile:
       
        for j in tqdm.tqdm(range(counter)):
            
            subjectId, (fmGroup, psds, freqs) = next(iter(pickle.load(fooofFile).items()))

            featureSet = featureEx(subjectId, fmGroup, psds, freqs, freqBands, leastR2, channelNmaes)
            featureSet.insert(0, subjectId)
            
            saveFeatures(savePath, featureSet)
            
            
    
    

    

