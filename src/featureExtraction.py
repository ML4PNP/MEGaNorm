import os
import sys
import mne
import glob
import tqdm
import json
import pickle
import argparse
import numpy as np
import pandas as pd

# Add utils folder to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'utils')
sys.path.append(config_path)


from summarizeFeatures import summarizeFeatures
import featureExtractionUtils
from IO import make_config





def featureExtract(subjectId, fmGroup, psds, featureCategories, freqs, freqBands, channelNames, bandSubRanges, sensorsInf, whichSensor):
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
    featuresRow, featuresNames = [], []

    for i in range(psds.shape[0]):

        # getting the fooof model of ith channel
        fm = fmGroup.get_fooof(ind=i)

        # fooof fitness
        r_squared = fm.r_squared_ 
        featuresRow.append(r_squared); featuresNames.append(f"r_squared_{channelNames[i]}")


        #################################### exponent and offset ##############################
        featRow, featName = featureExtractionUtils.apperiodicFeatures(fm=fm, 
                                                            channelNames=channelNames[i], 
                                                            featureCategories=featureCategories)
        featuresRow.extend(featRow); featuresNames.extend(featName)
        #====================================================================================== 


        # isolate periodic parts of signals
        flattenedPsd = featureExtractionUtils.isolatePeriodic(fm, psds[i, :])
        

        #Loop through each frequency band
        for bandName, (fmin, fmax) in freqBands.items():


        #   ################################# Peak Features ###################################
            (featRow, 
            featName, 
            dominant_peak,
            nanFlag) = featureExtractionUtils.peakParameters(fm=fm, 
                                                        fmin=fmin, 
                                                        fmax=fmax, 
                                                        channelNames=channelNames[i], 
                                                        bandName=bandName,
                                                        featureCategories=featureCategories)
            featuresRow.extend(featRow); featuresNames.extend(featName)
        #   #===================================================================================

            if bandName != "Broadband": 

                ################################# Power Features  #######################################
                # adjusted
                featRow, featName = featureExtractionUtils.canonicalPower(psd=flattenedPsd, 
                                                                        freqs=freqs, 
                                                                        fmin=fmin, 
                                                                        fmax=fmax, 
                                                                        channelNames=channelNames[i], 
                                                                        bandName=bandName,
                                                                        psdType="adjusted",
                                                                        featureCategories=featureCategories)
                featuresRow.extend(featRow); featuresNames.extend(featName)
                # original psd
                featRow, featName = featureExtractionUtils.canonicalPower(psd=psds[i, :], 
                                                                        freqs=freqs, 
                                                                        fmin=fmin, 
                                                                        fmax=fmax, 
                                                                        channelNames=channelNames[i], 
                                                                        bandName=bandName,
                                                                        psdType="originalPsd",
                                                                        featureCategories=featureCategories)
                featuresRow.extend(featRow); featuresNames.extend(featName)
                #=========================================================================================== 



                ################################# Individualized band power ################################
                # adjusted
                featRow, featName = featureExtractionUtils.individulizedPower(psd=flattenedPsd, 
                                                                    dominant_peak=dominant_peak, 
                                                                    freqs=freqs, 
                                                                    bandSubRanges=bandSubRanges, 
                                                                    nanFlag=nanFlag, 
                                                                    bandName=bandName, 
                                                                    channelNames=channelNames[i],
                                                                    psdType="adjusted",
                                                                    featureCategories=featureCategories)
                featuresRow.extend(featRow); featuresNames.extend(featName)
                # original psd
                featRow, featName = featureExtractionUtils.individulizedPower(psd=psds[i, :], 
                                                                    dominant_peak=dominant_peak, 
                                                                    freqs=freqs, 
                                                                    bandSubRanges=bandSubRanges, 
                                                                    nanFlag=nanFlag, 
                                                                    bandName=bandName, 
                                                                    channelNames=channelNames[i],
                                                                    psdType="originalPsd",
                                                                    featureCategories=featureCategories)
                featuresRow.extend(featRow); featuresNames.extend(featName)
                #============================================================================================


    features = pd.DataFrame(data = [featuresRow], 
                      columns=featuresNames) 
    
    # feature summarization ================================================================ 
    features = summarizeFeatures(df=features, 
                                sensorsInf=sensorsInf,
                                whichSensor=whichSensor)
    features.index = [subjectId]
    
    return features
    






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
                                    freqBands=configs["freqBands"],
                                    channelNames=configs['ch_names'],
                                    bandSubRanges=configs["bandSubRanges"],
                                    featureCategories=configs["featureCategories"])  

            features.to_csv(f"{subjectId}.csv")          
            
    
    

    

