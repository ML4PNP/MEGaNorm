import mne_bids
import glob
import shutil
import os
import mne
from pathlib import Path
import scipy
from utils.EEGlab import read_raw_eeglab
import pandas as pd


def make_demo_file_bids(file_dir:str, save_dir:str, id_col:int, age_col:int, sex_col:int,
                        male_indicator, female_indicator) -> None:

    """
    This function retrieves the address of a demographic file and converts 
    it to a BIDS-compatible format. 
    Ensure the output is saved in a directory structured according to BIDS specifications.

    Parameters:
        file_dir (str): Path to the input demographic file (e.g., CSV).
        save_dir (str): Directory where the BIDS-formatted file should be saved.
        id_col (int): Column index for the participant ID.
        age_col (int): Column index for the age.
        sex_col (int): Column index for the sex/gender.
        male_indicator: Value in the sex column that indicates male.
        female_indicator: Value in the sex column that indicates female.

    Returns:
        None
    """
    col_indices = {"participant_id" : id_col,
                    "age" : age_col,
                    "sex" : sex_col}
    
    if "xlsx" in file_dir[-4:]:
        df = pd.read_excel(file_dir, index_col=None)
    if "csv" in file_dir[-4:]:
        df = pd.read_csv(file_dir, index_col=None)
    if "tsv" in file_dir[-4:]:
        df = pd.read_csv(file_dir, sep='\t', index_col=None)
    

    col_names = df.columns.to_list()
    new_df = pd.DataFrame({})
    # rearrange
    for counter, (col_name, col_id) in enumerate(col_indices.items()): 
        col = df[col_names[col_id]]
        new_df.insert(counter, col_name, col)

    new_df.dropna(inplace=True)
    new_df.replace({"sex": {male_indicator:0, female_indicator:1}}, inplace=True)
    new_df['age'] = new_df['age'].astype(int)

    new_df.to_csv(save_dir, sep='\t', index=False)




if __name__ == "__main__":

    # Preparing demographic data according to mne_bids format
    # BTH
    file_dir = "/project/meganorm/Data/BTNRH/Rempe_Ott_PNAS_2022_Data.xlsx"
    save_dir = "/project/meganorm/Data/BTNRH/BTNRH/BIDS_data/participants.tsv"
    make_demo_file_bids(file_dir, 
                        save_dir, id_col=0, 
                        age_col=1, 
                        sex_col=2, 
                        male_indicator="M", 
                        female_indicator="F")

    # CAMCAN
    file_dir = "/project/meganorm/Data/camcan/CamCAN/cc700/participants.tsv"
    save_dir = "/project/meganorm/Data/BTNRH/CAMCAN/BIDS_data/participants.tsv"
    make_demo_file_bids(file_dir, 
                        save_dir, 
                        id_col=0, 
                        age_col=1, 
                        sex_col=3, 
                        male_indicator="MALE", 
                        female_indicator="FEMALE")
    
    # CMI
    file_dir = "/project/meganorm/Data/EEG_CMI/Phenotypes/HBN_R1_1_Pheno.csv" #For R1
    save_dir = "/project/meganorm/Data/EEG_CMI/EEG_BIDS/participants.tsv"
    make_demo_file_bids(file_dir, 
                        save_dir, 
                        id_col=0, 
                        age_col=2, 
                        sex_col=1, 
                        male_indicator="0", 
                        female_indicator="1")