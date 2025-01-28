import mne_bids
import glob
import shutil
import os
import mne
from pathlib import Path
import scipy
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from utils.EEGlab import read_raw_eeglab

def mne_bids_CMI(input_base_path, output_base_path, montage_path):
    """
    This code converges the CMI dataset into BIDS format. 
    Meanwhile, it defines channels on nek and chin as misc channels and channels around the eyes as eog channels 
    """

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

        # Define channels on nek and chin as misc and channels close to eyes as eog
        misc_channels = ['E48', 'E49', 'E56', 'E63', 'E68', 'E73', 'E81', 'E88', 'E94', 'E99', 'E107', 'E113', 'E119']
        eog_channels = ['E8', 'E14', 'E17', 'E21', 'E25', 'E128', 'E127', 'E126', 'E125'] 

        # Create a dictionary for setting channel types
        channel_types = {ch: 'misc' for ch in misc_channels}
        channel_types.update({ch: 'eog' for ch in eog_channels})

        # Apply the channel types to the raw object
        raw.set_channel_types(channel_types)

        #set eeg reference to match the reference of the recording 
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

        #channel_types = raw.get_channel_types()
        #for ch_name, ch_type in zip(raw.info['ch_names'], channel_types):
            #print(f"Channel: {ch_name}, Type: {ch_type}")

        print(f"created bids for parcipant {subject_id}" )

    return None

def define_eog_ecg_channels_CMI(input_base_path):


    search_pattern = os.path.join(input_base_path, "*/eeg/*_task-eyesclosed_channels.tsv")
    channel_files = glob.glob(search_pattern)
    print("found:", channel_files)

    for file in channel_files: 
        channels_df = pd.read_csv(file, sep="\t")
        #Define the mappings for channel types 
        update_mapping = {
            "E8": "EOG",
            "E14": "EOG",
            "E17": "EOG",
            "E21": "EOG",
            "E25": "EOG",
            "E128": "EOG",
            "E127": "EOG",
            "E126": "EOG",
            "E125": "EOG",
            "E48": "misc",
            "E49": "misc",
            "E56": "misc",
            "E63": "misc",
            "E68": "misc",
            "E73": "misc",
            "E81": "misc",
            "E88": "misc",
            "E94": "misc",
            "E99": "misc",
            "E107": "misc",
            "E113": "misc",
            "E119": "misc"
            }
    
        # Update the 'type' column based on the mapping
        channels_df["type"] = channels_df["name"].apply(lambda x: update_mapping[x] if x in update_mapping else channels_df.loc[channels_df["name"] == x, "type"].values[0])

        # Save the updated DataFrame back to the .tsv file
        channels_df.to_csv(file, sep="\t", index=False)

        print("Channel types updated successfully.")


def load_covariates_CMI(CMI_covariates_path:str, save_dir:str): 

    """Load age, gender, diagnosis and site for CMI dataset.

    Args:
        path (str): path to covariates.

    Returns:
        DataFrame: Pandas dataframe containing age and gender for CMI dataset.
    """

    df = pd.read_csv(CMI_covariates_path, sep=',')
    df = df.rename(columns={df.columns[0]: 'EID'})
    df = df[['EID', 'Age', 'Sex', 'Site', 'Diagnosis']]  # Select necessary columns
    df = df.rename(columns={'EID': 'Subject_ID', 'Age': 'age', 'Sex': 'gender', 'Site':'site', 'Diagnosis':'diagnosis'})  # Rename columns
    #TODO Maybe rename diagnosis to match for every dataset?
    df.dropna(inplace=True)

    # Prepend 'sub-' to all Subject_IDs so that it can easily be merged with the features 
    df['Subject_ID'] = 'sub-' + df['Subject_ID'].astype(str)

    df.to_csv(save_dir, sep='\t', index=False) 

def load_CMI_data(feature_path, covariates_path):
    """Load CMI dataset

    Args:
        feature_path (str): Path to the the feature csv file.
        covariates_path (str): path to the covariates tsv file.

    Returns:
        DataFrame: Pandas dataframe with CMI covariates and features.
    """
      
    CMI_covariates = pd.read_csv(covariates_path, sep='\t', index_col=0)    
    CMI_features = pd.read_csv(feature_path, index_col=0)
    CMI_features.index = CMI_features.index.str.replace('^sub-', '', regex=True)
    CMI_data = CMI_covariates.join(CMI_features, how='inner')
    return CMI_data

if __name__ == "__main__":
    input_base_path = "/home/meganorm-yverduyn/Dev/2_EXAMPLE_SUBJECTS_CMI"
    output_base_path = "/home/meganorm-yverduyn/Dev/2_EXAMPLE_SUBJECTS_CMI/BIDS"
    montage_path = "/project/meganorm/Data/EEG_CMI/info/GSN_HydroCel_129.sfp"
    mne_bids_CMI(input_base_path, output_base_path, montage_path)

    #CMI_demo_path = "/project/meganorm/Data/EEG_CMI/Phenotypes/HBN_R1_1_Pheno.csv" ##for R1
    #CMI_site_path = "/project/meganorm/Data/EEG_CMI/info/Subject-Site_R1_1.xlsx" ##for R1
    #load_covariates_CMI(CMI_demo_path, CMI_site_path)