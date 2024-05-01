
import numpy as np
import argparse
import config
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


        # preproces ========================================================================
        # optional arguments
        parser.add_argument("--targetFS", type=int,
                help="Specify the desired sampling rate for resampling your data.")
        parser.add_argument("--n_component", type=float,
                help="ICA n_components")
        parser.add_argument("--maxIter", type=int,
                help="maximum number of iteration in the ICA algorithm")
        parser.add_argument("--IcaMethod", type=str,
                choices=["fastica", "infomax", "picard"],
                help="which ICA method to use")
        parser.add_argument("--cutoffFreqHigh", type=int,
                help="Cutoff frequency for filtering data prior to feeding it into ICA.")
        parser.add_argument("--cutoffFreqLow", type=int,
                help="Cutoff frequency for filtering data prior to feeding it into ICA.")

        # fooof analysis ====================================================================
        parser.add_argument("--freqRangeLow", type=float,
                help="Desired frequency range to run FOOOF (lower limit)")
        parser.add_argument("--freqRangeHigh", type=float,
                help="Desired frequency range to run FOOOF (higher limit)")
        parser.add_argument("--min_peak_height", type=int,
                help="Absolute threshold for detecting peaks")
        parser.add_argument("--peak_threshold", type=float,
                help="Relative threshold for detecting peaks")
        # segmentation
        parser.add_argument("--fs", type=int,
                help="sampling rate")
        parser.add_argument("--tmin", type=int,
                help="start time of the raw data to use in seconds")
        parser.add_argument("--tmax", type=int,
                help="end time of the raw data to use in seconds")
        parser.add_argument("--segmentsLength", type=int,
                help="length of MEG segments in seconds")
        parser.add_argument("--overlap", type=float,
                help="amount of overlap between MEG sigals segmentation")
        # psd
        parser.add_argument("-psdMethod", type=str,
                choices=["welch", "multitaper"],
                help="Spectral estimation method.")
        parser.add_argument("--psd_n_overlap", type=float,
                help="amount of overlap between windows in Welch's method")
        parser.add_argument("--psd_n_fft", type=float)

        # saving options
        parser.add_argument("--fooofResSave", type=str,
                        help="if available, fooof results will be saved")
        # feature extraction ==================================================================
        parser.add_argument("--leastR2", type=float,
                help="least acceptable R squared for a fooof model")




        args = parser.parse_args()

        # preproces ========================================================================
        if not args.targetFS : args.targetFS = config.targetFS
        if not args.n_component: args.n_component = config.n_component
        if not args.maxIter: args.maxIter = config.maxIter
        if not args.IcaMethod: args.IcaMethod = config.IcaMethod
        if not args.cutoffFreqHigh: args.cutoffFreqHigh = config.cutoffFreqHigh
        if not args.cutoffFreqLow: args.cutoffFreqLow = config.cutoffFreqLow

        # fooof analysis ====================================================================
        # fooof
        if not args.freqRangeLow : args.freqRangeLow = config.freqRangeLow
        if not args.freqRangeHigh : args.freqRangeHigh = config.freqRangeHigh
        if not args.min_peak_height: args.min_peak_height = config.min_peak_height
        if not args.peak_threshold: args.peak_threshold = config.peak_threshold
        # segmentation
        if not args.fs: args.fs = config.fs
        if not args.tmin: args.tmin = config.tmin
        if not args.tmax: args.tmax = config.tmax
        if not args.overlap: args.overlap = config.overlap
        if not args.segmentsLength: args.segmentsLength = config.segmentsLength
        # psd
        if not args.psdMethod: args.psdMethod = config.psdMethod
        if not args.psd_n_fft: args.psd_n_fft = config.psd_n_fft
        if not args.psd_n_overlap: args.psd_n_overlap = config.psd_n_overlap
        # feature extraction ==================================================================
        if not args.leastR2 : args.leastR2 = config.leastR2



        # args.dir = "/home/smkia/Data/CamCAN/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/*/*.fif"
        # args.dir = "/home/smkia/Data/CamCAN/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/sub-CC410226/mf2pt2_sub-CC410226_ses-rest_task-rest_megtransdef.fif"
        # args.saveDir = "data/parallelData"

        # subject ID
        subID = args.dir.split("/")[-2]

        # preproces ========================================================================
        filteredData = preprocess(subjectPath = args.dir,
                                targetFS = args.targetFS,
                                n_component = args.n_component,
                                maxIter = args.maxIter,
                                IcaMethod = args.IcaMethod,
                                cutoffFreqLow = args.cutoffFreqLow,
                                cutoffFreqHigh = args.cutoffFreqHigh)

        # fooof analysis ====================================================================
        filteredData = filteredData.pick(picks=["mag"])
        fmGroup, psds, freqs = fooof(data = filteredData.pick(picks=["mag"]),
                                        freqRangeLow = args.freqRangeLow,
                                        freqRangeHigh = args.freqRangeHigh,
                                        min_peak_height = args.min_peak_height,
                                        peak_threshold = args.peak_threshold,
                                        fs = args.targetFS,
                                        tmin = args.tmin,
                                        tmax = args.tmax,
                                        segmentsLength = args.segmentsLength,
                                        overlap = args.overlap,
                                        psdMethod = args.psdMethod,
                                        psd_n_overlap = args.psd_n_overlap,
                                        psd_n_fft = args.psd_n_fft)
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
                                        freqBands = config.freqBands,
                                        leastR2 = args.leastR2,
                                        channelNames = channelNames,
                                        bandSubRanges = config.bandSubRanges)


        if len(FeaturesName) == 4998:
                with open(os.path.join(args.saveDir, "featuresNames.json"), "w") as file:
                        FeaturesName.insert(0, "participant_id")
                        json.dump(FeaturesName, file)

        saveFeatures(os.path.join(args.saveDir, f"{subID}.csv"), featureSet)
   

if __name__ == "__main__":
             
        mainParallel(sys.argv[1:])