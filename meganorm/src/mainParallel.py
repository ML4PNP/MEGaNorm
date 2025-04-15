import argparse
import json
import os
import sys
import mne
import pandas as pd
import glob

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

	extention = path[0].split(".")[-1]
	if "4D" in path[0]: extention = "BTI" # TODO: you need to change this

	#Task
	task = path.split("/")[-1].split("_")[-2]

	# read the data ====================================================================

	try:	
		data = mne.io.read_raw(path, verbose=False, preload=True)
	except:
		data = mne.io.read_raw_bti(                   
					pdf_fname=os.path.join(path, "c,rfDC"),     
					config_fname=os.path.join(path, "config"),     
					head_shape_fname=None,  
					preload=True)

	power_line_freq = data.info.get("line_freq") 
	if not power_line_freq:
		power_line_freq = 60

	if configs["which_sensor"] == "eeg":
		base_dir = os.path.dirname(path)
		subID = args.subject
		search_pattern = os.path.join(base_dir, f"**_{task}_channels.tsv")
		channel_files = glob.glob(search_pattern, recursive=True)
		channel_file = channel_files[0]
		channels_df = pd.read_csv(channel_file, sep='\t')
		channels_types = channels_df.set_index('name')['type'].str.lower().to_dict()
		data.set_channel_types(channels_types)	

	if configs["which_sensor"] == "eeg":
		montage = data.get_montage()
		if montage is None: 
			try: 
				search_pattern_montage = os.path.join(base_dir, "*_montage.csv")
				print("Searching for:",search_pattern_montage )
				montage_files = glob.glob(search_pattern_montage, recursive=True)

				if not montage_files:
					raise FileNotFoundError("No montage CSV file found!")

				eeg_montage = montage_files[0]
				montage_df = pd.read_csv(eeg_montage)
				ch_positions = {
            		row['Channel']: [row['X'], row['Y'], row['Z']]
           		for _, row in montage_df.iterrows()
        		}
				eeg_montage = mne.channels.make_dig_montage(ch_pos=ch_positions, coord_frame='head')
				data.set_montage(eeg_montage)
			
			except Exception as e:
				# Log the error and continue without setting the montage
				print(f"Error setting montage: {e}")
				print("Continuing without a montage. This may raise issues for ICA label.")



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

	# command = python src/mainParallel.py /project/meganorm/Data/MOUS/sub-A2021/meg/sub-A2021_task-rest_meg.ds /home/meganorm-mznasrabadi/MEGaNorm/tests sub-A2021
	# command = python src/mainParallel.py /project/meganorm/Data/BTNRH/CAMCAN/BIDS_data/sub-CC221828/meg/sub-CC221828_task-rest_meg.fif /home/meganorm-mznasrabadi/MEGaNorm/tests

	mainParallel(sys.argv[1:])