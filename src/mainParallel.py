
import numpy as np
import argparse
import json
import os
import sys




# Add utils folder to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'utils')
sys.path.append(config_path)
# sys.path.insert(0, "../utils")


from IO import make_config, storeFooofModels
from preprocessUtils import segmentEpoch
from psdParameterize import psdParameterize
from preprocess import preprocess
from featureExtraction import featureExtract




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

	# args.dir = "/project/meganorm/Data/camcan/CamCAN/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/sub-CC620090/mf2pt2_sub-CC620090_ses-rest_task-rest_megtransdef.fif"
	# args.saveDir = "/home/meganorm-mznasrabadi/MEGaNorm/dataTest"


	# Loading configs
	if args.configs is not None:
		with open(args.configs, 'r') as f:
			configs = json.load(f)
	else:
		configs = make_config()

	# subject ID
	subID = args.dir.split("/")[-2]

	# preproces ========================================================================
	filteredData, channelNames = preprocess(subjectPath = args.dir,
							fs = configs['fs'],
							n_component = configs['ica_n_component'],
							maxIter = configs['ica_maxIter'],
							IcaMethod = configs['ica_method'],
							cutoffFreqLow = configs['cutoffFreqLow'],
							cutoffFreqHigh = configs['cutoffFreqHigh'],
							whichSensor=configs["whichSensor"])
	
	# segmentation =====================================================================
	segments = segmentEpoch(data=filteredData, 
				fs = configs['fs'],
				tmin = configs['segments_tmin'],
				tmax = configs['segments_tmax'],
				segmentsLength = configs['segments_length'],
				overlap = configs['segments_overlap'])

	# fooof analysis ====================================================================
	fmGroup, psds, freqs = psdParameterize(segments = segments,
									fs = configs['fs'],
									# psd parameters
									psdMethod = configs['psd_method'],
									psd_n_overlap = configs['psd_n_overlap'],
									psd_n_fft = configs['psd_n_fft'],
									n_per_seg = configs["n_per_seg"],
									# fooof parameters
									freqRangeLow = configs['fooof_freqRangeLow'],
									freqRangeHigh = configs['fooof_freqRangeHigh'],
									min_peak_height = configs['fooof_min_peak_height'],
									peak_threshold = configs['fooof_peak_threshold'],
									peak_width_limits = configs["fooof_peak_width_limits"])
	if args.fooofResSave: 
		storeFooofModels(args.fooofResSave, 
						subID, 
						fmGroup,
						psds,
						freqs)

	# feature extraction ==================================================================
	features = featureExtract(subjectId = subID,
							fmGroup = fmGroup,
							psds = psds,
							freqs = freqs,
							freqBands = configs['freqBands'],
							channelNames = channelNames,
							bandSubRanges = configs['bandSubRanges'],
							featureCategories=configs["featuresCategories"],
							sensorsInf=configs["sensorsID"],
							whichSensor=configs["whichSensor"])
	
	features.to_csv(os.path.join(args.saveDir, f"{subID}.csv"))



if __name__ == "__main__":
        
	mainParallel(sys.argv[1:])