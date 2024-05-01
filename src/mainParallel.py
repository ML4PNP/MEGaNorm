
import numpy as np
import argparse
from config import make_config
import json
import os
import sys

from preprocess import preprocess
from fooofAnalysis import fooof
from dataManagementUtils import storeFooofModels
from featureExtract import featureEx
from dataManagementUtils import saveFeatures

def mainParallel(*args):
        
        parser = argparse.ArgumentParser()
        # positional Arguments 
        parser.add_argument("dir", 
                help="Address to your data")
        parser.add_argument("saveDir", type=str,
                help="where to save extracted features")
        parser.add_argument("--configs", type=str, default=None,
                help="Address of configs json file")
        
        # saving options
        parser.add_argument("--fooofResSave", type=str,
                    help="if available, fooof results will be saved")

        args = parser.parse_args()
        
        # Loading configs
        if args.configs is not None:
                with open(args.configs, 'r') as f:
                        configs = json.load(f)
        else:
                configs = make_config()

        # subject ID
        subID = args.dir.split("/")[-2]

        # preproces ========================================================================
        filteredData = preprocess(subjectPath = args.dir,
                                targetFS = configs['targetFS'],
                                n_component = configs['n_component'],
                                maxIter = configs['maxIter'],
                                IcaMethod = configs['IcaMethod'],
                                cutoffFreqLow = configs['cutoffFreqLow'],
                                cutoffFreqHigh = configs['cutoffFreqHigh'])

        # fooof analysis ====================================================================
        filteredData = filteredData.pick(picks=[configs['meg_sensors']]) # TODO: this should go to the preprocessing
        fmGroup, psds, freqs = fooof(data = filteredData.pick(picks=[configs['meg_sensors']]),
                                        freqRangeLow = configs['freqRangeLow'],
                                        freqRangeHigh = configs['freqRangeHigh'],
                                        min_peak_height = configs['min_peak_height'],
                                        peak_threshold = configs['peak_threshold'],
                                        fs = configs['targetFS'],
                                        tmin = configs['tmin'],
                                        tmax = configs['tmax'],
                                        segmentsLength = configs['segmentsLength'],
                                        overlap = configs['overlap'],
                                        psdMethod = configs['psdMethod'],
                                        psd_n_overlap = configs['psd_n_overlap'],
                                        psd_n_fft = configs['psd_n_fft'])
        if args.fooofResSave: 
                storeFooofModels(args.fooofResSave, 
                                subID, 
                                fmGroup,
                                psds,
                                freqs)
        

        # feature extraction ==================================================================
        channelNames = filteredData.info['ch_names']

        if np.quantile(fmGroup.get_params(name="r_squared"), 0.25) < 0.9 : 
                print(f"The fooof model for the subject: {subID} is overfitted")
                raise Exception

        featureSet, FeaturesName = featureEx(subjectId = subID,
                                        fmGroup = fmGroup,
                                        psds = psds,
                                        freqs = freqs,
                                        freqBands = configs['freqBands'],
                                        channelNames = channelNames,
                                        bandSubRanges = configs['bandSubRanges'],
                                        leastR2 = configs['leastR2'])


        if len(FeaturesName) == 4998:
                with open(os.path.join(args.saveDir, "featuresNames.json"), "w") as file:
                        FeaturesName.insert(0, "participant_id")
                        json.dump(FeaturesName, file)

        saveFeatures(os.path.join(args.saveDir, f"{subID}.csv"), featureSet)
   

if __name__ == "__main__":
             
        mainParallel(sys.argv[1:])