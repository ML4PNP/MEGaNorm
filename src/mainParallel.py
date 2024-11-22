from pathlib import Path
import argparse
import json
import os
import sys
import mne
import numpy as np
import pathlib
import mne_bids
import pandas as pd

# Add utils folder to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'utils')
sys.path.append(config_path)

from IO import make_config, storeFooofModels
from psdParameterize import psdParameterize
from preprocess import preprocess, segment_epoch, drop_bads
from featureExtraction import feature_extract


def mainParallel(*args):
        
	parser = argparse.ArgumentParser()
	# positional Arguments 
	parser.add_argument("dir", 
				type=str,
				help="Address to your data")
	parser.add_argument("saveDir", 
				type=str,
				help="where to save extracted features")
	parser.add_argument("subject",
				type=str,
				help="participant ID")
	parser.add_argument("--configs", type=str, default=None,
			help="Address of configs json file")
	
	args = parser.parse_args()

	# Loading configs
	if args.configs is not None:
		with open(args.configs, 'r') as f:
			configs = json.load(f)
	else:
		configs = make_config('configs')

	# subject ID
	subID = args.subject

	paths = args.dir.split("*")
	paths = list(filter(lambda x: len(x), paths))
	path = paths[0]

	# read the data ====================================================================
	data = mne.io.read_raw(path, verbose=False, preload=True)

	power_line_freq = data.info.get("line_freq") 
	if not power_line_freq:
		power_line_freq = 60

	# In order to determine the loayout
	extention = path[0].split(".")[-1]

	if configs["which_sensor"] == "eeg":
		channel_file = os.path.dirname(path[0])
		channel_file = os.path.join(channel_file, subID + "_task-rest_channels.tsv")
		channels_df = pd.read_csv(channel_file, sep = '\t')
		channels_types = channels_df.set_index('name')['type'].str.lower().to_dict()
		data.set_channel_types(channels_types)

	which_sensor = {"meg":False,
					"mag":False,
					"grad":False,
					"eeg":False,
					"opm":False}
	for key, values in which_sensor.items():
		if key == configs["which_sensor"]:
			which_sensor[key] = True

	# preproces ========================================================================
	filtered_data, channel_names, sampling_rate = preprocess(data=data,
												n_component = configs['ica_n_component'],
												ica_max_iter = configs['ica_max_iter'],
												IcaMethod = configs['ica_method'],
												cutoffFreqLow = configs['cutoffFreqLow'],
												cutoffFreqHigh = configs['cutoffFreqHigh'],
												which_sensor = which_sensor,
												resampling_rate = configs["resampling_rate"],
												digital_filter = configs["digital_filter"],
												rereference_method = configs['rereference_method'],
												apply_ica = configs["apply_ica"],
												auto_ica_corr_thr = configs["auto_ica_corr_thr"],
												power_line_freq = power_line_freq)
	
	# segmentation =====================================================================
	segments = segment_epoch(data=filtered_data, 
							sampling_rate = sampling_rate,
							tmin = configs['segments_tmin'],
							tmax = configs['segments_tmax'],
							segmentsLength = configs['segments_length'],
							overlap = configs['segments_overlap'])
	
	# drop bad channels ================================================================
	# segments = drop_bads(segments = segments,
	# 					mag_var_threshold = configs["mag_var_threshold"],
	# 					grad_var_threshold = configs["grad_var_threshold"],
	# 					eeg_var_threshold = configs["eeg_var_threshold"],
	# 					mag_flat_threshold = configs["mag_flat_threshold"],
	# 					grad_flat_threshold = configs["grad_flat_threshold"],
	# 					eeg_flat_threshold = configs["eeg_flat_threshold"],
	# 					which_sensor = which_sensor)

	# fooof analysis ====================================================================
	fmGroup, psds, freqs = psdParameterize(segments = segments,
									sampling_rate = sampling_rate,
									# psd parameters
									psdMethod = configs['psd_method'],
									psd_n_overlap = configs['psd_n_overlap'],
									psd_n_fft = configs['psd_n_fft'],
									n_per_seg = configs["psd_n_per_seg"],
									# fooof parameters
									freqRangeLow = configs['fooof_freqRangeLow'],
									freqRangeHigh = configs['fooof_freqRangeHigh'],
									min_peak_height = configs['fooof_min_peak_height'],
									peak_threshold = configs['fooof_peak_threshold'],
									peak_width_limits = configs["fooof_peak_width_limits"],
									aperiodic_mode = configs["aperiodic_mode"])
	
	if configs["fooof_res_save_path"]: 
		storeFooofModels(configs["fooof_res_save_path"], 
						subID, 
						fmGroup,
						psds,
						freqs)

	# # feature extraction ==================================================================
	features = feature_extract(subjectId = subID,
							fmGroup = fmGroup,
							psds = psds,
							freqs = freqs,
							freq_bands = configs['freq_bands'],
							channel_names = channel_names,
							individualized_band_ranges = configs['individualized_band_ranges'],
							feature_categories= configs["feature_categories"],
							extention = extention,
       						which_layout = configs["which_layout"],
							which_sensor = which_sensor,
							aperiodic_mode = configs["aperiodic_mode"],
							min_r_squared = configs["min_r_squared"])
	
	features.to_csv(os.path.join(args.saveDir, f"{subID}.csv"))

if __name__ == "__main__":

	# command = python src/mainParallel.py /project/meganorm/Data/BTNRH/BTNRH/BIDS_data/sub-049/meg/sub-049_task-rest_meg.fif /home/meganorm-mznasrabadi/MEGaNorm/tests
	# command = python src/mainParallel.py /project/meganorm/Data/BTNRH/CAMCAN/BIDS_data/sub-CC221828/meg/sub-CC221828_task-rest_meg.fif /home/meganorm-mznasrabadi/MEGaNorm/tests

	mainParallel(sys.argv[1:])