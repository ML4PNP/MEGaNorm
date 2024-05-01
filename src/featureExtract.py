import mne
import glob
import tqdm
import json
import pickle
import argparse
import numpy as np


from dataManagementUtils import saveFeatures
from processUtils import Features
import config





def featureEx(subjectId, fmGroup, psds, freqs, freqBands, channelNames, bandSubRanges, leastR2=0.9):
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

    freqBands: dictionary

    leastR2: float
    least accpetable r_squared in fitting fooof mdeols

    channelNames: list

    bandSubRanges: dict
    individualized frequency ranges

    return
    -------------
    featuresRow: list
    """


    # in order to save features and their name
    featuresRow, FeaturesName = [], []

    for i in range(psds.shape[0]):
        

        # getting the fooof model of ith channel
        fm = fmGroup.get_fooof(ind=i)

        # if fooof model is underfitted => exclude the channel
        if fm.r_squared_ < leastR2: 
            empty = np.empty(49)
            empty[:] = np.nan
            featuresRow.extend(empty.tolist())
            continue

        # # ################################# exponent and offset ##############################
        featRow, featName = Features.apperiodicFeatures(fm=fm, channelNames=channelNames[i])
        featuresRow.extend(featRow); FeaturesName.extend(featName)
        # #===================================================================================== 

        # isolate periodic parts of signals
        flattenedPsd = Features.flatPeriodic(fm, psds[i, :])
        
        totalPowerFlattened = np.trapz(flattenedPsd, freqs)

        #Loop through each frequency band
        for bandName, (fmin, fmax) in freqBands.items():

        #     ################################# Peak Features ################################
            (featRow, 
            featName, 
            dominant_peak,
            nanFlag) = Features.peakParameters(fm, 
                                                freqs, 
                                                fmin, 
                                                fmax, 
                                                channelNames[i], 
                                                bandName)
            featuresRow.extend(featRow); FeaturesName.extend(featName)
        #     #================================================================================

            if bandName != "Broadband": 

                ################################# Power Features  #######################################
                # adjusted
                RelFeatRow, RelFeatName, AbsFeatRow, AbsFeatName = Features.canonicalPower(flattenedPsd, 
                                                                        freqs, 
                                                                        fmin, 
                                                                        fmax, 
                                                                        channelNames[i], 
                                                                        bandName,
                                                                        psdType="adjusted")
                featuresRow.extend([RelFeatRow, AbsFeatRow]); FeaturesName.extend([RelFeatName, AbsFeatName])
                # original psd
                RelFeatRow, RelFeatName, AbsFeatRow, AbsFeatName = Features.canonicalPower(psds[i, :], 
                                                                        freqs, 
                                                                        fmin, 
                                                                        fmax, 
                                                                        channelNames[i], 
                                                                        bandName,
                                                                        psdType="original psd")
                featuresRow.extend([RelFeatRow, AbsFeatRow]); FeaturesName.extend([RelFeatName, AbsFeatName])  
                #=========================================================================================== 



                ################################# Individualized band power ################################
                # adjusted
                RelFeatRow, RelFeatName, AbsFeatRow, AbsFeatName = Features.individulizedPower(flattenedPsd, 
                                                                    totalPowerFlattened, 
                                                                    dominant_peak, 
                                                                    freqs, 
                                                                    bandSubRanges, 
                                                                    nanFlag, 
                                                                    bandName, 
                                                                    channelNames[i],
                                                                    psdType="adjusted")
                featuresRow.extend([RelFeatRow, AbsFeatRow]); FeaturesName.extend([RelFeatName, AbsFeatName]) 
                # original psd
                RelFeatRow, RelFeatName, AbsFeatRow, AbsFeatName = Features.individulizedPower(psds[i, :], 
                                                                    totalPowerFlattened, 
                                                                    dominant_peak, 
                                                                    freqs, 
                                                                    bandSubRanges, 
                                                                    nanFlag, 
                                                                    bandName, 
                                                                    channelNames[i],
                                                                    psdType="original psd")
                featuresRow.extend([RelFeatRow, AbsFeatRow]); FeaturesName.extend([RelFeatName, AbsFeatName]) 
                #============================================================================================


    
    featuresRow.insert(0, subjectId)
    return featuresRow, FeaturesName






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
    channelNames = raw.info['ch_names']
    
    if not args.leastR2 : args.leastR2 = config.leastR2
    


    with open(args.dir, "rb") as fooofFile:
       
        for j in tqdm.tqdm(range(counter)):
            
            subjectId, (fmGroup, psds, freqs) = next(iter(pickle.load(fooofFile).items()))

            if np.quantile(fmGroup.get_params(name="r_squared"), 0.25) < 0.9 : 
                print(f"The fooof model for the subject: {subjectId} is overfitted")
                continue

            featureSet, FeaturesName = featureEx(subjectId,
                                    fmGroup,
                                    psds,
                                    freqs,
                                    config.freqBands,
                                    args.leastR2,
                                    channelNames,
                                    config.bandSubRanges)
            
            
            if len(FeaturesName) == 4998:
                with open("data/features/featuresNames.json", "w") as file:
                    FeaturesName.insert(0, "participant_id")
                    json.dump(FeaturesName, file)            

            
            saveFeatures(args.savePath, featureSet)
            
            
    
    

    

