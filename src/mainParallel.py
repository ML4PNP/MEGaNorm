
import numpy as np
import argparse
import json
import os
import sys

# Add utils folder to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'utils')
sys.path.append(config_path)

from IO import make_config, storeFooofModels
from preprocess import preprocess
from psdParameterize import psdParameterize
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
							cutoffFreqHigh = configs['cutoffFreqHigh'],
							sensorType=configs["meg"])

	# fooof analysis ====================================================================
	fmGroup, psds, freqs = psdParameterize(data = filteredData.pick(picks=[configs['meg_sensors']]),
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
									psd_n_fft = configs['psd_n_fft'],
									n_per_seg = configs["n_per_seg"],
									peak_width_limits = configs["peak_width_limits"])
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
							channelNames = configs['ch_names'],
							bandSubRanges = configs['bandSubRanges'],
							featureCategories=configs["featureCategories"])


	features.to_csv(os.path.join(args.saveDir, f"{subID}.csv"))



if __name__ == "__main__":
        
	mainParallel(sys.argv[1:])