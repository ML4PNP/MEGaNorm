import mne_bids
import glob
import shutil
import os
import mne
from pathlib import Path
import scipy
from utils.EEGlab import read_raw_eeglab

def mne_bids_CMI(input_base_path, output_base_path, montage_path):
    # Ensure output directory exists
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    
    search_pattern = os.path.join(input_base_path, "*/*/EEG/raw/mat_format/RestingState.mat")
    raw_mat_path = glob.glob(search_pattern) # Use glob to find all RestingState.mat files in raw/mat_format folder

    # Loop through all found .mat files
    for mat_path in raw_mat_path:
        subject_id = Path(mat_path).parts[-5]  # Extract subject number from the file path

        # read the math file & save the mat_file to extract needed info later on 
        raw = read_raw_eeglab(mat_path) 
        mat_data = scipy.io.loadmat(mat_path)

        #### ADD extra information, that is not in the raw object yet ####

        EEG_data = mat_data['EEG'] # Access the EEG data to extract info within there
        sfreq = EEG_data['srate'][0][0][0][0]    #Get sampling frequency 
        # set channel locations
        montage = mne.channels.read_custom_montage(montage_path)

        # Create a mapping to rename the channels in raw to match the montage
        mapping = {f'EEG {i:03d}': f'E{i+1}' for i in range(128)}  # EEG 000 -> E1, EEG 001 -> E2, ..., EEG 128 -> E129
        mapping['EEG 128'] = 'Cz'  
        raw.info.rename_channels(mapping)
        raw.info.set_montage(montage)

        #set eeg reference 
        raw.set_eeg_reference(ref_channels=['Cz'])

        ##### Extract event and epoch information to set the annotations
        # Access the event info within EEG_data
        events_data = EEG_data['event']

        # Loop through each event entry to extract type, onset and duration needded for annotations 
        for event in events_data[0][0]:
            event_type = event['type']
            event_types = [item[0] for item in event_type]

            event_sample= event['sample']
            event_samples = [item[0][0] for item in event_sample]
            event_onsets = [(sample / sfreq) for sample in event_samples]
            print(event_onsets)

            event_duration = [event_onsets[i] - event_onsets[i-1] for i in range (1, len(event_onsets))]
            event_duration.append(0)

        # Create annotations using onset, duration, and description
        annotations = mne.Annotations(onset=event_onsets, duration=event_duration, description=event_types)

        # Attach annotations to the raw object
        raw.set_annotations(annotations)

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
    input_base_path = "/project/meganorm/Data/EEG_CMI/EEG/"
    output_base_path = "/project/meganorm/Data/EEG_CMI/EEG_BIDS"
    montage_path = "/project/meganorm/Data/EEG_CMI/info/GSN_HydroCel_129.sfp"
    mne_bids_CMI(input_base_path, output_base_path, montage_path)