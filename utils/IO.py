import pickle
import pandas as pd
import json
import os
import re
import glob
from pathlib import Path
import mne 
import numpy as np 

def make_config(project, path=None):

    # preprocess configurations =================================================
    # downsample data
    config = dict()

    # You could also set layout to None to have high 
    # choices: all, lobe, None
    config["which_layout"] = "all"

    # which sensor type should be used
    # choices: meg, mag, grad, eeg, opm
    config["which_sensor"] = "meg"
    # config['fs'] = 1000

    # ICA configuration
    config['ica_n_component'] = 30
    config['ica_max_iter'] = 800
    config['ica_method'] = "fastica"

    # lower and upper cutoff frequencies in a bandpass filter
    config['cutoffFreqLow'] = 1
    config['cutoffFreqHigh'] = 45

    config["resampling_rate"] = 1000
    config["digital_filter"] = True
    config["notch_filter"] = False

    config["apply_ica"] = True

    config["auto_ica_corr_thr"] = 0.9

    # options are "average", "REST", and None 
    config["rereference_method"]= "average"
    
    # variance threshold across time
    config["mag_var_threshold"] = 4e-12
    config["grad_var_threshold"] = 4000e-13
    config["eeg_var_threshold"] = 40e-6
    # flatness threshold across time
    config["mag_flat_threshold"] = 10e-15
    config["grad_flat_threshold"] = 10e-15
    config["eeg_flat_threshold"] = None
    config["eeg_flat_threshold"] = 40e-6
    # variance thershold across channels
    config["zscore_std_thresh"] = 15 # change this

    # segmentation ==============================================
    #start time of the raw data to use in seconds, this is to avoid possible eye blinks in close-eyed resting state. 
    config['segments_tmin'] = 20
    # end time of the raw data to use in seconds, this is to avoid possible eye blinks in close-eyed resting state.
    config['segments_tmax'] = -20
    # length of MEG segments in seconds
    config['segments_length'] = 10
    # amount of overlap between MEG sigals in seconds
    config['segments_overlap'] = 2

    # PSD ==============================================
    # Spectral estimation method
    config['psd_method'] = "welch"
    # amount of overlap between windows in Welch's method
    config['psd_n_overlap'] = 1
    config['psd_n_fft'] = 2
    # number of samples in psd
    config["psd_n_per_seg"] = 2

    # fooof analysis configurations ==============================================
    # Desired frequency range to run FOOOF
    config['fooof_freqRangeLow'] = 3
    config['fooof_freqRangeHigh'] = 40
    # which mode should be used for fitting; choices (knee, fixed)
    config["aperiodic_mode"] = "knee"
    # minimum acceptable peak width in fooof analysis
    config["fooof_peak_width_limits"] = [1.0, 12.0]
    #Absolute threshold for detecting peaks
    config['fooof_min_peak_height'] = 0
    #Relative threshold for detecting peaks
    config['fooof_peak_threshold'] = 2

    # feature extraction ==========================================================
    # Define frequency bands
    config['freq_bands'] = {
                            'Theta': (3, 8),
                            'Alpha': (8, 13),
                            'Beta': (13, 30),
                            'Gamma': (30, 40),
                            # 'Broadband': (3, 40)
                            }

    # Define individualized frequency range over main peaks in each freq band
    config['individualized_band_ranges'] = { 
                                            'Theta': (-2, 3),
                                            'Alpha': (-2, 3), # change to (-4,2)
                                            'Beta': (-8, 9),
                                            'Gamma': (-5, 5)
                                            }

    # least acceptable R squred of fitted models
    config['min_r_squared'] = 0.9 
 
    config['feature_categories'] = {
                                    "Offset":False,
                                    "Exponent":False,
                                    "Peak_Center":False,
                                    "Peak_Power":False,
                                    "Peak_Width":False,
                                    "Adjusted_Canonical_Relative_Power":True, 
                                    "Adjusted_Canonical_Absolute_Power":False,
                                    "Adjusted_Individualized_Relative_Power":False,
                                    "Adjusted_Individualized_Absolute_Power":False,
                                    "OriginalPSD_Canonical_Relative_Power":True, 
                                    "OriginalPSD_Canonical_Absolute_Power":False,
                                    "OriginalPSD_Individualized_Relative_Power":False,
                                    "OriginalPSD_Individualized_Absolute_Power":False,
                                    }
    
    config["fooof_res_save_path"] = None

    config["random_state"] = 42

    if path is not None:
        out_file = open(os.path.join(path, project + ".json"), "w") 
        json.dump(config, out_file, indent = 6) 
        out_file.close()

    return config 


def storeFooofModels(path, subjId, fooofModels, psds, freqs) -> None:
    """
    This function stores the periodic and aperiodic 
    results in a h5py file

    parameters
    ------------
    path: str
    where to save

    subjid: str
    subject ID

    fooofModels: object

    returns
    -------------
    None

    """

    with open(os.path.join(path, subjId + ".pickle"), "ab") as file:
        pickle.dump([fooofModels, psds, freqs], file)


def merge_datasets(datasets):

    subjects = {}
    
    for dataset_name in datasets.keys(): 
        path = datasets[dataset_name]
        dirs = os.listdir(path)
        for dir in dirs:
            full_path = os.path.join(path, dir)
            if os.path.isdir(full_path):
                subjects[dir] = full_path
                
    return subjects


def merge_datasets_with_regex(datasets):
    
    subjects = {}

    for dataset_name, dataset_info in datasets.items():
        base_dir = dataset_info['base_dir']
        regex_pattern = dataset_info['regex']
        
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        # Walk through the base directory to find subject directories
        for subject_dir in dirs:
            formatted_regex = regex_pattern.format(subject_name=subject_dir, base_dir=base_dir)
            if os.path.exists(formatted_regex):
                subjects[subject_dir] = formatted_regex
                
    return subjects


def separate_eyes_open_close_eeglab(input_base_path, output_base_path, annotation_description_open, annotation_description_close, trim_before=5, trim_after=5):
    # Ensure output directory exists
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    search_pattern = os.path.join(input_base_path, "*/eeg/*_task-rest_eeg.set")
    raw_set_paths = glob.glob(search_pattern, recursive=True) # Use glob to find all .set files in the input directory

    # Loop through all found .set files
    for set_path in raw_set_paths:
        subject_id = Path(set_path).parts[-3]  # Extract subject number from the file path
        subject_output_path = os.path.join(output_base_path, subject_id, 'eeg') # Create the subject-specific output path

        # Ensure output directory for the subject exists
        if not os.path.exists(subject_output_path):
            os.makedirs(subject_output_path)

        # Load the raw .set file (EEGLAB format)
        raw = mne.io.read_raw(set_path, preload=True)

        # Extract annotations
        annotations = raw.annotations

        # Separate eyes open and eyes closed events
        eyes_open_events = annotations[annotations.description == annotation_description_open]
        eyes_closed_events = annotations[annotations.description == annotation_description_close]

        # Extract and concatenate eyes open segments
        eyes_open_data = []
        for onset, duration in zip(eyes_open_events.onset, eyes_open_events.duration):
            
            if duration <= trim_before + trim_after:
                print(f"Skipping event with onset {onset} and duration {duration} (invalid after trimming)")
                continue
            
            # Trim the first 5s and last 5s from each event
            trimmed_onset = onset + trim_before
            trimmed_duration = duration - trim_before - trim_after
            start_sample = int(trimmed_onset * raw.info['sfreq'])
            stop_sample = int((trimmed_onset + trimmed_duration) * raw.info['sfreq'])
            eyes_open_data.append(raw[:, start_sample:stop_sample][0])

        if eyes_open_data:
            eyes_open_data_concat = np.concatenate(eyes_open_data, axis=1)
            raw_eyes_open = mne.io.RawArray(eyes_open_data_concat, raw.info)

            # Save eyes open data as a new .set file
            eyes_open_file_path = os.path.join(subject_output_path, f'{subject_id}_task-eyesopen_eeg.set')
            mne.export.export_raw(eyes_open_file_path, raw_eyes_open, fmt='eeglab', overwrite = True)

        # Extract and concatenate eyes closed segments
        eyes_closed_data = []
        for onset, duration in zip(eyes_closed_events.onset, eyes_closed_events.duration):
            
            if duration <= trim_before + trim_after:
                print(f"Skipping event with onset {onset} and duration {duration} (invalid after trimming)")
                continue
            
            trimmed_onset = onset + trim_before
            trimmed_duration = duration - trim_before - trim_after
            start_sample = int(trimmed_onset * raw.info['sfreq'])
            stop_sample = int((trimmed_onset + trimmed_duration) * raw.info['sfreq'])
            eyes_closed_data.append(raw[:, start_sample:stop_sample][0])

        if eyes_closed_data:
            eyes_closed_data_concat = np.concatenate(eyes_closed_data, axis=1)
            raw_eyes_closed = mne.io.RawArray(eyes_closed_data_concat, raw.info)

            # Save eyes closed data as a new .set file
            eyes_closed_file_path = os.path.join(subject_output_path,f'{subject_id}_task-eyesclosed_eeg.set')
            mne.export.export_raw(eyes_closed_file_path, raw_eyes_closed, fmt='eeglab', overwrite = True)


def merge_fidp_demo(datasets_paths:str, features_dir:str, data_set_names:list, include_patients=False, diagnosis="parkinson"):
    """
    Loads demographic data and features, then concatenates them. 
    It assigns a site index for each dataset and normalizes the age range to [0, 1].
    Note that the demographic data must be stored according to MNE BIDS standards.
    
    Parameters:
        datasets_paths (str): Path to the datasets.
        features_dir (str): Path to the extracted features.

    Returns:
        data (pd.DataFrame): Merged data.
    """

    demographic_df = pd.DataFrame({})
    for counter,dataset_path in enumerate(datasets_paths):
        demo = pd.read_csv(os.path.join(dataset_path, "participants_bids.tsv"), 
                                                sep="\t", index_col=0)
        demo.index = demo.index.astype(str)

        if not 'site' in demo.columns:
            demo['site'] = data_set_names[counter] 

        demographic_df = pd.concat([demographic_df, 
                                    demo],
                                    axis=0)

    demographic_df.drop(columns="eyes", inplace=True)
    # demographic_df["eyes"] = pd.factorize(demographic_df["eyes"])[0]

    demographic_df["sex"] = pd.factorize(demographic_df["sex"])[0]
    demographic_df["site"] = pd.factorize(demographic_df["site"])[0]
    
    feature_path = os.path.join(features_dir, "all_features.csv")
    data = pd.read_csv(feature_path, index_col=0)
    data.index = data.index.astype(str)
    data.index.name=None

    data = demographic_df.join(data, how='inner')
    data.index.name=None

    # resacle age range to [0,1]
    data["age"] = data["age"]/100

    #initialize data_patient
    data_patient = None

    if not include_patients:
        data_patient = data[data["diagnosis"].isin(diagnosis)] #changed from data_patient = data[data["diagnosis"] == diagnosis]
        # data_patient["diagnosis"] = pd.factorize(data_patient["diagnosis"])[0]

        data = data[data["diagnosis"] == "control"]
        data.drop(columns="diagnosis", inplace=True)
    elif include_patients:
        data = data.dropna(subset=["diagnosis"]) #Drop rows where diangosis = nan 
        data["diagnosis"] = np.where(data["diagnosis"] == "control", 0, pd.factorize(data["diagnosis"])[0] + 1) 
    
    # Filter out sites with only one subject
    site_counts = data["site"].value_counts()
    valid_sites = site_counts[site_counts > 1].index

    data = data[data["site"].isin(valid_sites)]
    if data_patient is not None:
        data_patient = data_patient[data_patient["site"].isin(valid_sites)]
        data_patient.dropna(inplace=True) #drop rows with nan values 
    
    # Create a mapping for renumbering sites so that numbers are sequentially and none are skipped
    site_mapping = {old_site: new_site for new_site, old_site in enumerate(sorted(valid_sites))}
    
    # Apply the new site numbering
    data["site"] = data["site"].map(site_mapping)
    if data_patient is not None:
        data_patient["site"] = data_patient["site"].map(site_mapping)
        
    return data, data_patient


def merge_datasets_with_glob(datasets):
    
    subjects = {}

    for dataset_name, dataset_info in datasets.items():
        base_dir = dataset_info['base_dir']
        
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        subjects.update({subj:[] for subj in dirs})

        paths = glob.glob(f"{datasets[dataset_name]["base_dir"]}/**/*{datasets[dataset_name]["task"]}*{datasets[dataset_name]["ending"]}", recursive=True)

        # Walk through the base directory to find subject directories
        for subject_dir in dirs:
            pattern = os.path.join(datasets[dataset_name]["base_dir"], subject_dir)
            subjects[subject_dir].extend(list(filter(lambda path: path.startswith(pattern), paths)))

    def join_with_star(lst):
        if len(lst) == 1:
            return lst[0] + '*'  
        return '*'.join(lst) 

    # add this part to main parallel when you want to concatenate 
    # different run
    subjects = dict(filter(lambda item:item[1], subjects.items()))
    subjects = {key: join_with_star(value) for key, value in subjects.items()}

    # 	paths = args.dir.split("*")
	# paths = list(filter(lambda x: len(x), paths))
	# # read the data ====================================================================
	# for path_counter, path in enumerate(paths):
	# 	if path_counter == 0:
	# 		data = mne.io.read_raw(path, verbose=False, preload=True)
	# 		dev_head_t_ref = data.info['dev_head_t']
	# 	else:
	# 		new_data = mne.io.read_raw(path, verbose=False, preload=True)
	# 		new_data = mne.preprocessing.maxwell_filter(
	# 												new_data,
	# 												origin=(0,0,0),
	# 												coord_frame='head',
	# 												destination=dev_head_t_ref
	# 												)
	# 		data = mne.concatenate_raws([data, new_data])

    return subjects