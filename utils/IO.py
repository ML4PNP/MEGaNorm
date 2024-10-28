import pickle
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
    
    config["device"] = "MEGIN"
    config["layout"] = "Megin_MAG_All"
    
    # which sensor type should be used
    # choices: 1. meg: all, 2.mag, 3.grad
    config["whichSensor"] = "mag"
    config['fs'] = 1000

    # ICA configuration
    config['ica_n_component'] = 30
    config['ica_maxIter'] = 800
    config['ica_method'] = "fastica"
    # lower and upper cutoff frequencies in a bandpass filter
    config['cutoffFreqLow'] = 1
    config['cutoffFreqHigh'] = 45

    # Signal space projection
    config["ssp_ngrad"] = 3
    config["ssp_nmag"] = 3

    config["CAMCAN_preprocess"] = {"resampling": False,
                                    "digital_filter": True,
                                    "autoICA": True,
                                    "SSP": False}
    config["BTNRH_preprocess"] = {"resampling": False,
                                    "digital_filter": True,
                                    "autoICA": False,
                                    "SSP":True}

    # fooof analysis configurations ==============================================
    # Desired frequency range to run FOOOF
    config['fooof_freqRangeLow'] = 3
    config['fooof_freqRangeHigh'] = 40
    #start time of the raw data to use in seconds, this is to avoid possible eye blinks in close-eyed resting state. 
    config['segments_tmin'] = 20
    # end time of the raw data to use in seconds, this is to avoid possible eye blinks in close-eyed resting state.
    config['segments_tmax'] = -20
    # length of MEG segments in seconds
    config['segments_length'] = 10
    # amount of overlap between MEG sigals in seconds
    config['segments_overlap'] = 2
    # which mode should be used for fitting; choices (knee, fixed)
    config["aperiodic_mode"] = "knee"


    # Spectral estimation method
    config['psd_method'] = "welch"
    # amount of overlap between windows in Welch's method
    config['psd_n_overlap'] = 1
    config['psd_n_fft'] = 2
    # number of samples in psd
    config["n_per_seg"] = 2
    
    # minimum acceptable peak width in fooof analysis
    config["fooof_peak_width_limits"] = [1.0, 12.0]
    #Absolute threshold for detecting peaks
    config['fooof_min_peak_height'] = 0
    #Relative threshold for detecting peaks
    config['fooof_peak_threshold'] = 2

    # feature extraction ==========================================================
    # Define frequency bands
    config['freqBands'] = {'Theta': (3, 8),
                            'Alpha': (8, 13),
                            'Beta': (13, 30),
                            'Gamma': (30, 40),
                            # 'Broadband': (3, 40)
                            }

    # Define individualized frequency range over main peaks in each freq band
    config['bandSubRanges'] = { 'Theta': (-2, 3),
                                'Alpha': (-2, 3), # change to (-4,2)
                                'Beta': (-8, 9),
                                'Gamma': (-5, 5)}

    # least acceptable R squred of fitted models
    config['leastR2'] = 0.9 


    config['featuresCategories'] = [
                                    # "Offset", # 1
                                    # "Exponent", # 1
                                    # "Peak_Center", # 5,
                                    # "Peak_Power",# 5,
                                    # "Peak_Width", # 5,
                                    "Canonical_Relative_Power", 
                                    # "Canonical_Absolute_Power",
                                    # "Individualized_Relative_Power",
                                    # "Individualized_Absolute_Power",
                                    ]
    config["fooof_res_save_path"] = None

    if path is not None:
        out_file = open(os.path.join(path, project + ".json"), "w") 
        json.dump(config, out_file, indent = 6) 
        out_file.close()

    return config 






def make_ch_Names(sensor_type, ch_names, feature_categories):
    
    if sensor_type == "mag":
        ch_names = [channel for channel in ch_names if channel.endswith("1")]
    if sensor_type == "grad1":
        ch_names = [channel for channel in ch_names if channel.endswith("2")]
    if sensor_type == "grad2":
        ch_names = [channel for channel in ch_names if channel.endswith("3")]
    
    return ch_names
    



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

    search_pattern = os.path.join(input_base_path, "*/eeg/*.set")
    raw_set_paths = glob.glob(search_pattern, recursive=True) # Use glob to find all .set files in the input directory

    # Loop through all found .set files
    for set_path in raw_set_paths:
        subject_id = Path(set_path).parts[-3]  # Extract subject number from the file path
        subject_output_path = os.path.join(output_base_path, subject_id, 'eeg') # Create the subject-specific output path

        # Ensure output directory for the subject exists
        if not os.path.exists(subject_output_path):
            os.makedirs(subject_output_path)

        # Load the raw .set file (EEGLAB format)
        raw = mne.io.read_raw_eeglab(set_path, preload=True)

        # Extract annotations
        annotations = raw.annotations

        # Separate eyes open and eyes closed events
        eyes_open_events = annotations[annotations.description == annotation_description_open]
        eyes_closed_events = annotations[annotations.description == annotation_description_close]

        # Extract and concatenate eyes open segments
        eyes_open_data = []
        for onset, duration in zip(eyes_open_events.onset, eyes_open_events.duration):
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
            mne.export.export_raw(eyes_open_file_path, raw_eyes_open, fmt='eeglab')

        # Extract and concatenate eyes closed segments
        eyes_closed_data = []
        for onset, duration in zip(eyes_closed_events.onset, eyes_closed_events.duration):
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
            mne.export.export_raw(eyes_closed_file_path, raw_eyes_closed, fmt='eeglab')