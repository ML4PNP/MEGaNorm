import os
import glob
from pathlib import Path
import scipy
import mne
import mne_bids
import pandas as pd
from utils.IO import separate_eyes_open_close_eeglab

def preprocess_events_file(file_path):
    """
    Preprocesses the events CSV file to remove repeated headers but keeps the first header.
    """
    clean_lines = []
    header_found = False  # Track if header is already added

    with open(file_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            # Keep the first header, remove repeated ones, and skip empty lines
            if stripped_line.lower() == '"type","latency","urevent",':
                if not header_found:  # Add the header only once
                    clean_lines.append(stripped_line)
                    header_found = True
            elif stripped_line:  # Skip empty lines
                clean_lines.append(stripped_line)

    # Write the cleaned data to a temporary file
    cleaned_file_path = file_path + "_cleaned.csv"
    with open(cleaned_file_path, 'w') as f:
        f.write('\n'.join(clean_lines) + '\n')  # Ensure proper formatting

    return cleaned_file_path

def mne_bids_MIPDB(input_base_path, output_base_path, montage_path):

    """
    This code converges the MIPDB csv files into an mne raw object and then into BIDS format. 
    """

    # Ensure output directory exists
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    #Find all raw resting state files and event files from every participant
    search_pattern = os.path.join(input_base_path, "*/EEG/raw/csv_format/*001.csv")
    raw_path = glob.glob(search_pattern) 
    search_pattern2 = os.path.join(input_base_path, "*/EEG/raw/csv_format/*001_events.csv")
    events_data = glob.glob(search_pattern2) 

    # Loop through all found .raw files
    for raw, events in zip(raw_path, events_data): 

        #extract subject id
        subject_id = Path(raw).parts[-5] 

        #transform csv file into dataframe
        dataframe = pd.read_csv(raw)
        n_channels = len(dataframe) # E1 to E128 (127 channels, so I assume reference CZ is not in there)
        sampling_freq = 500

        #create mne info object
        info = mne.create_info(n_channels, sfreq=sampling_freq)
        raw = mne.io.RawArray(dataframe.values, info)  

        # set channel locations
        montage = mne.channels.read_custom_montage(montage_path)
        raw.info.set_montage(montage)

        # Create a mapping to rename the channels in raw to match the montage
        mapping = {f'{i}': f'E{i+1}' for i in range(n_channels)}  # EEG 000 -> E1, EEG 001 -> E2, ..., EEG 128 -> E129
        #mapping['EEG 128'] = 'Cz'  
        raw.info.rename_channels(mapping)

        # Define channels on nek and chin as misc and channels close to eyes as eog
        eeg_channels = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 
                        'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20',
                        'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30',
                        'E31', 'E32', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39', 'E40',
                        'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E50',
                        'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58', 'E59', 'E60',
                        'E61', 'E62', 'E64', 'E65', 'E66', 'E67', 'E69', 'E70',
                        'E71', 'E72', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80',
                        'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E89', 'E90',
                        'E91', 'E92', 'E93', 'E95', 'E96', 'E97', 'E98', 'E100',
                        'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E110',
                        'E111', 'E112', 'E114', 'E115', 'E116', 'E117', 'E118', 'E120',
                        'E121', 'E122', 'E123', 'E124', 'E125', 'E126']
        misc_channels = ['E48', 'E49', 'E56', 'E63', 'E68', 'E73', 'E81', 'E88', 'E94', 'E99', 'E107', 'E113', 'E119']
        eog_channels = ['E127', 'E126', 'E125'] #Actually also E128 but not present? 

        # Create a dictionary for setting channel types
        channel_types = {ch: 'eeg' for ch in eeg_channels}
        channel_types.update({ch: 'misc' for ch in misc_channels})
        channel_types.update({ch: 'eog' for ch in eog_channels})

        # Apply the channel types to the raw object
        raw.set_channel_types(channel_types)

        ##### Extract event and epoch information to set the annotations
        # Preprocess the events file to remove repeated headers
        cleaned_events_path = preprocess_events_file(events)
        events_data = pd.read_csv(cleaned_events_path)
        
        # Ensure 'latency' column is numeric
        events_data['latency'] = pd.to_numeric(events_data['latency'], errors='coerce')

        # Drop rows with invalid latencies, if any
        events_data = events_data.dropna(subset=['latency'])

        event_type = events_data['type']
        event_types = [str(item) for item in event_type] 
        event_latencies = events_data['latency']

        event_onsets = [latency / sampling_freq for latency in event_latencies]
        event_duration = [event_onsets[i] - event_onsets[i-1] for i in range (1, len(event_onsets))]
        event_duration.append(0)
        
        # Create annotations using onset, duration, and description
        annotations = mne.Annotations(onset=event_onsets, duration=event_duration, description=event_types)

        # Attach annotations to the raw object
        raw.set_annotations(annotations)

        raw_save_path = os.path.join(output_base_path, f"sub-{subject_id}_raw.fif")
        raw.save(raw_save_path, overwrite=True)
        print(f"Raw data saved to {raw_save_path}")

        #extra info that cannot 
        raw.info["line_freq"] = 60
        raw.info["device_info"] = {
            'type': 'EEG',
            'manufacturer': 'Electrical Geodesics',
            'model': 'HydroCel GSN 130'
            }
        

        #convert to BIDS
        bids_path = mne_bids.BIDSPath(subject=subject_id, datatype='eeg', task='rest',
                     root=output_base_path)
        mne_bids.write_raw_bids(raw, bids_path=bids_path, allow_preload=True, format='EEGLAB', overwrite=True,)   
         
    return None
     

if __name__ == "__main__":

    #Convert to bids

    #input_base_path = "/project/meganorm/Data/EEG_MIPDB/EEG/" #Final
    #output_base_path = "/project/meganorm/Data/EEG_MIPDB/EEG_BIDS/" #Final

    input_base_path1 = "/home/meganorm-yverduyn/Dev/MIPDB/EEG"
    output_base_path1 = "/home/meganorm-yverduyn/Dev/MIPDB/EEG_BIDS"
    montage_path = "/project/meganorm/Data/EEG_MIPDB/info/GSN_HydroCel_129.sfp"
    mne_bids_MIPDB(input_base_path1, output_base_path1, montage_path)

    #Separate eyes closed and open trials

    #input_base_path = "/project/meganorm/Data/EEG_MIPDB/EEG_BIDS/"
    #output_base_path = "/project/meganorm/Data/EEG_MIPDB/EEG_BIDS/"
    
    input_base_path2 = "/home/meganorm-yverduyn/Dev/MIPDB/EEG_BIDS"
    output_base_path2 = "/home/meganorm-yverduyn/Dev/MIPDB/EEG_BIDS"
    annotation_description_open = "20"
    annotation_description_close = "30"

    separate_eyes_open_close_eeglab(input_base_path2, output_base_path2, annotation_description_open, annotation_description_close, trim_before=5, trim_after=5)



