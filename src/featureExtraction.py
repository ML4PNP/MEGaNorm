import numpy as np
import os
import sys
import math
import tqdm
import json
import pickle
import argparse
import pandas as pd

# Add utils folder to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'utils')
sys.path.append(config_path)

import featureExtractionUtils
from IO import make_config


def featureExtract(subjectId, fmGroup, psds, featureCategories, freqs, freqBands, 
                   channelNames, bandSubRanges, device, layout, aperiodic_mode):
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

    features, names = [], []


    for i in range(psds.shape[0]):
        flage = True

        # in order to save features and their name
        featuresRow, featuresNames = [], []

        # getting the fooof model of ith channel
        fm = fmGroup.get_fooof(ind=i)

        # fooof fitness
        r_squared = fm.r_squared_ 
        if r_squared < 0.9: continue
        featuresRow.append(r_squared); featuresNames.append(f"r_squared_{channelNames[i]}")


        #################################### exponent and offset ##############################
        featRow, featName = featureExtractionUtils.apperiodicFeatures(fm=fm, 
                                                            channelNames=channelNames[i], 
                                                            featureCategories=featureCategories,
                                                            aperiodic_mode=aperiodic_mode)
        featuresRow.extend(featRow); featuresNames.extend(featName)
        #====================================================================================== 


        # isolate periodic parts of signals
        flattenedPsd = np.asarray(featureExtractionUtils.isolatePeriodic(fm, psds[i, :]))

        # whenever aperidic activity is higher than periodic activity
        # => set the preiodic acitivity to zero
        flattenedPsd = np.array(list(map(lambda x: max(0, x), flattenedPsd)))


        #################################### Theta-Beta ratio ###################################
        # featRow, featName = featureExtractionUtils.psd_ratio(psd=flattenedPsd,
        #                                                     freqs=freqs,
        #                                                     freqRangeNumerator=freqBands["Theta"],
        #                                                     freqRangeDenominator=freqBands["Beta"],
        #                                                     channelNames=channelNames[i],
        #                                                     name="Theta_Beta",
        #                                                     psdType="Adjusted")
        # featuresRow.extend(featRow); featuresNames.extend(featName)

        # featRow, featName = featureExtractionUtils.psd_ratio(psd=psds[i, :],
        #                                                     freqs=freqs,
        #                                                     freqRangeNumerator=freqBands["Theta"],
        #                                                     freqRangeDenominator=freqBands["Beta"],
        #                                                     channelNames=channelNames[i],
        #                                                     name="Theta_Beta",
        #                                                     psdType="originalPSD")
        # featuresRow.extend(featRow); featuresNames.extend(featName)
        # =======================================================================================


        #Loop through each frequency band
        for bandName, (fmin, fmax) in freqBands.items():


            ################################# Peak Features ###################################
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
            #===================================================================================
            

            ################################# Power Features  #######################################
            # adjusted
            featRow, featName = featureExtractionUtils.canonicalPower(psd=flattenedPsd, 
                                                                    freqs=freqs, 
                                                                    fmin=fmin, 
                                                                    fmax=fmax, 
                                                                    channelNames=channelNames[i], 
                                                                    bandName=bandName,
                                                                    psdType="Adjusted",
                                                                    featureCategories=featureCategories)
            

            featuresRow.extend(featRow); featuresNames.extend(featName)
            # original psd
            featRow, featName = featureExtractionUtils.canonicalPower(psd=psds[i, :], 
                                                                    freqs=freqs, 
                                                                    fmin=fmin, 
                                                                    fmax=fmax, 
                                                                    channelNames=channelNames[i], 
                                                                    bandName=bandName,
                                                                    psdType="OriginalPsd",
                                                                    featureCategories=featureCategories)
            featuresRow.extend(featRow); featuresNames.extend(featName)
            #=========================================================================================== 


            if bandName != "Broadband": 
                ################################# Individualized band power ################################
                # adjusted
                featRow, featName = featureExtractionUtils.individulizedPower(psd=flattenedPsd, 
                                                                    dominant_peak=dominant_peak, 
                                                                    freqs=freqs, 
                                                                    bandSubRanges=bandSubRanges, 
                                                                    nanFlag=nanFlag, 
                                                                    bandName=bandName, 
                                                                    channelNames=channelNames[i],
                                                                    psdType="Adjusted",
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
                                                                    psdType="OriginalPsd",
                                                                    featureCategories=featureCategories)
                featuresRow.extend(featRow); featuresNames.extend(featName)
                #============================================================================================

        if flage == True: features.extend(featuresRow), names.extend(featuresNames)

    features = pd.DataFrame(data=[features], 
                            columns=names) 
    
    # feature summarization ================================================================ 
    features = featureExtractionUtils.summarizeFeatures(df=features, 
                                                        device=device, 
                                                        layout_name=layout)
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
            
    
    

    

