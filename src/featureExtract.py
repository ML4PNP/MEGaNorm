import mne
import h5py
import glob
import tqdm
import fooof
import json
import pickle
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from dataManagementUtils import readFooofres, subjectList, saveFeatures
from processUtils import Features, isNan
import config





def featureEx(subjectId, fmGroup, psds, freqs, freqBands, leastR2, channelNmaes, bandSubRanges):
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

        #     #============================================================================================

    if len(FeaturesName) == 7650:
        with open("data/features/featuresNames.josn", "w") as file:
            json.dump(FeaturesName, file)
    
    
    return featuresRow






if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str,
            help="data directory (pickle format)")
    parser.add_argument("--rawMegData", type=str,
            help="address to where meg data in order to extract channel names")
    parser.add_argument("--leastR2", type=float,
            help="least acceptable R squared for a fooof model")
    parser.add_argument("--savePath", type=str,
            help="where to save data")
    
    args = parser.parse_args()

    # remove the following lines
    args.dir = "data/fooofResults/fooofModels.pkl"
    args.rawMegData = "/home/smkia/Data/CamCAN/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/*/*.fif"
    args.savePath = "data/features/featureMatrix.csv"



    with open(args.dir, "rb") as fooofFile:
        counter = 0
        while True:
            try: pickle.load(fooofFile) ; counter += 1
            except: break

    
    dataPaths = glob.glob(args.rawMegData)
    raw = mne.io.read_raw_fif(dataPaths[0]).pick(picks="meg")
    channelNmaes = raw.info['ch_names']
    
    if not args.leastR2 : args.leastR2 = config.leastR2
    


    with open(args.dir, "rb") as fooofFile:
       
        for j in tqdm.tqdm(range(counter)):
            
            subjectId, (fmGroup, psds, freqs) = next(iter(pickle.load(fooofFile).items()))

            if np.quantile(fmGroup.get_params(name="r_squared"), 0.25) < 0.9 : 
                print(f"The fooof model for the subject: {subjectId} is overfitted")
                continue

            featureSet = featureEx(subjectId,
                                    fmGroup,
                                    psds,
                                    freqs,
                                    config.freqBands,
                                    args.leastR2,
                                    channelNmaes,
                                    config.bandSubRanges)

            featureSet.insert(0, subjectId)
            
            saveFeatures(args.savePath, featureSet)
            
            
    
    

    

