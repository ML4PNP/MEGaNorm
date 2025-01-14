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
    Meanwhile, it defines channels on nek and chin as misc channels and channels around the eyes as eog channels #TODO
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
        eog_channels = ['E128', 'E127', 'E126', 'E125']

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

        channel_types = raw.get_channel_types()
        for ch_name, ch_type in zip(raw.info['ch_names'], channel_types):
            print(f"Channel: {ch_name}, Type: {ch_type}")

    return None

def load_covariates_CMI(CMI_demo_paths:list, CMI_site_paths:list, save_dir:str): 

    """Load age, gender and site for CMI dataset.

    Args:
        path (str): path to covariates.

    Returns:
        DataFrame: Pandas dataframe containing age and gender for CMI dataset.
    """
    
    # Initialize empty DataFrames
    combined_demo = pd.DataFrame()
    combined_site = pd.DataFrame()

    for path in CMI_demo_paths:
        df_demo = pd.read_csv(path, sep=',')
        df_demo = df_demo[['EID', 'Age', 'Sex']]  # Select necessary columns
        df_demo = df_demo.rename(columns={'EID': 'Subject_ID', 'Age': 'age', 'Sex': 'gender'})  # Rename columns
        combined_demo = pd.concat([combined_demo, df_demo], ignore_index=True)

    for path in CMI_site_paths:
        df_site = pd.read_excel(path)
        df_site.columns = df_site.columns.str.replace(' ', '_')  # Replace spaces with underscores to deal with study_site vs study site
        df_site = df_site[['EID', 'Study_Site']]  # Select necessary columns
        df_site = df_site.rename(columns={'EID': 'Subject_ID', 'Study_Site': 'site'})  # Rename columns
        combined_site = pd.concat([combined_site, df_site], ignore_index=True)


    df = pd.merge(combined_demo, combined_site, on='Subject_ID', how='inner')
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
    input_base_path = "/project/meganorm/Data/EEG_CMI/EEG/"
    output_base_path = "/project/meganorm/Data/EEG_CMI/EEG_BIDS"
    montage_path = "/project/meganorm/Data/EEG_CMI/info/GSN_HydroCel_129.sfp"
    mne_bids_CMI(input_base_path, output_base_path, montage_path)

    CMI_demo_path = "/project/meganorm/Data/EEG_CMI/Phenotypes/HBN_R1_1_Pheno.csv" ##for R1
    CMI_site_path = "/project/meganorm/Data/EEG_CMI/info/Subject-Site_R1_1.xlsx" ##for R1
    load_covariates_CMI(CMI_demo_path, CMI_site_path)