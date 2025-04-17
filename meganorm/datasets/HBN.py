import mne_bids
import glob
import os
import mne
from pathlib import Path
import scipy
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from meganorm.utils.EEGlab import read_raw_eeglab


def mne_bids_HBN(input_base_path, output_base_path, montage_path):
    """
    This function converges the HBN dataset .mat files into BIDS format.

    It reads EEG recordings from the HBN dataset, adds additional metadata such as 
    channel types and annotations, and writes the data into BIDS-compliant format using `mne-bids`.

    Channels on the neck and chin are classified as 'misc', and those close to the eyes as 'eog'. 
    The function also applies a custom montage, sets Cz as the reference electrode, and embeds event 
    annotations into the raw EEG data.


    Parameters
    ----------
    input_base_path : str
        Path to the base directory containing the HBN dataset
    output_base_path : str
        Directory where the converted BIDS dataset will be saved
    montage_path : _type_
        Path to the custom montage file to be used for setting EEG channel locations

    Returns
    -------
    None
        This function does not return any value. The output is saved to the specified output directory

    Notes
    -----
    - The input `.mat` files are expected to follow the structure: `*/subject/session/EEG/raw/mat_format/RestingState.mat`.
    - The sampling frequency is extracted directly from the `.mat` file.
    - Events are extracted from the EEG structure and converted into MNE annotations.
    - The following channels are treated specially:
        - Misc (neck and chin): E48, E49, E56, E63, E68, E73, E81, E88, E94, E99, E107, E113, E119
        - EOG (eyes): E8, E14, E17, E21, E25, E128, E127, E126, E125
    """

    # Ensure output directory exists
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    # Use glob to find all RestingState.mat files in raw/mat_format folder
    search_pattern = os.path.join(
        input_base_path, "*/*/EEG/raw/mat_format/RestingState.mat"
    )
    raw_mat_path = glob.glob(
        search_pattern
    )  

    # Loop through all found .mat files
    for mat_path in raw_mat_path:
        try:
            # Extract subject number from the file path
            subject_id = Path(mat_path).parts[
                -5
            ]  
            print(subject_id)

            # Read the .mat file & save the mat_file to extract needed info later on
            raw = read_raw_eeglab(mat_path) 
            mat_data = scipy.io.loadmat(mat_path)

            #### ADD extra information, that is not in the raw object yet ####
            EEG_data = mat_data["EEG"]  # Access the EEG data to extract info
            sfreq = EEG_data["srate"][0][0][0][0]  # Get sampling frequency

            # Set channel locations
            montage = mne.channels.read_custom_montage(montage_path)

            # Create a mapping to rename the channels in raw to match the montage
            mapping = {
                f"EEG {i:03d}": f"E{i+1}" for i in range(128)
            }  # EEG 000 -> E1, EEG 001 -> E2, ..., EEG 128 -> E129
            mapping["EEG 128"] = "Cz"
            raw.info.rename_channels(mapping)
            raw.info.set_montage(montage)

            # Define channels on neck and chin as misc and channels close to eyes as eog
            misc_channels = [
                "E48",
                "E49",
                "E56",
                "E63",
                "E68",
                "E73",
                "E81",
                "E88",
                "E94",
                "E99",
                "E107",
                "E113",
                "E119",
            ]
            eog_channels = [
                "E8",
                "E14",
                "E17",
                "E21",
                "E25",
                "E128",
                "E127",
                "E126",
                "E125",
            ]

            # Create a dictionary for setting channel types
            channel_types = {ch: "misc" for ch in misc_channels}
            channel_types.update({ch: "eog" for ch in eog_channels})

            # Apply the channel types to the raw object
            raw.set_channel_types(channel_types)

            # Set EEG reference to match the reference of the recording
            raw.set_eeg_reference(ref_channels=["Cz"])

            ##### Extract event and epoch information to set the annotations #####
            events_data = EEG_data["event"]

            # Loop through each event entry to extract type, onset, and duration needed for annotations
            for event in events_data[0][0]:
                event_type = event["type"]
                event_types = [item[0] for item in event_type]

                event_sample = event["sample"]
                event_samples = [item[0][0] for item in event_sample]
                event_onsets = [(sample / sfreq) for sample in event_samples]
                print(event_onsets)

                event_duration = [
                    event_onsets[i] - event_onsets[i - 1]
                    for i in range(1, len(event_onsets))
                ]
                event_duration.append(0)

            # Create annotations using onset, duration, and description
            annotations = mne.Annotations(
                onset=event_onsets, duration=event_duration, description=event_types
            )

            # Attach annotations to the raw object
            raw.set_annotations(annotations)

            # Extra info
            raw.info["line_freq"] = 60
            raw.info["device_info"] = {
                "type": "EEG",
                "manufacturer": "Electrical Geodesics",
                "model": "HydroCel GSN 130",
            }

            # Convert to BIDS
            bids_path = mne_bids.BIDSPath(
                subject=subject_id, datatype="eeg", task="rest", root=output_base_path
            )
            mne_bids.write_raw_bids(
                raw,
                bids_path=bids_path,
                allow_preload=True,
                format="EEGLAB",
                overwrite=True,
            )

            print(f"Created BIDS for participant {subject_id}")

        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            continue

    return None


def define_eog_ecg_channels_HBN(input_base_path):
    """
    This function updates the channel types for the HBN dataset EEG channel TSV files.
    Note that this function is designed only for the eyesclosed trials. This eyes closed file is created by using the separate_eyes_open_close_eeglab function from utils.io 

    Parameters
    ----------
    input_base_path : str
        Path to the directory containing BIDS-formatted EEG channel TSV files. It should include 
        folders structured like `sub-*/eeg/*_task-eyesclosed_channels.tsv`.

    Returns
    -------
    None
        The function modifies channel TSV files in-place and does not return any value.

    Notes
    -----
    - Channels related to the eyes are labeled as "EOG":
        E8, E14, E17, E21, E25, E128, E127, E126, E125
    - Channels on the neck and chin are labeled as "misc":
        E48, E49, E56, E63, E68, E73, E81, E88, E94, E99, E107, E113, E119
    - All other channels retain their original type(EEG). 
    """    

    search_pattern = os.path.join(
        input_base_path, "*/eeg/*_task-eyesclosed_channels.tsv"
    )
    channel_files = glob.glob(search_pattern)
    print("found:", channel_files)

    for file in channel_files:
        channels_df = pd.read_csv(file, sep="\t")
        # Define the mappings for channel types
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
            "E119": "misc",
        }

        # Update the 'type' column based on the mapping
        channels_df["type"] = channels_df["name"].apply(
            lambda x: (
                update_mapping[x]
                if x in update_mapping
                else channels_df.loc[channels_df["name"] == x, "type"].values[0]
            )
        )

        # Save the updated DataFrame back to the .tsv file
        channels_df.to_csv(file, sep="\t", index=False)

        print("Channel types updated successfully.")


def determine_final_diagnosis(row):
    """
    This functions determines the final diagnosis label for a given participant. 
    Only confirmed diagnosis are kept, otherwise No Diagnosis Given, Diagnosis not confirmed or No info about confirmation is returned.
    If none of the conditions are met, nan is returned

    Parameters
    ----------
    row : pandas.Series
        A row from a pandas DataFrame representing a single participant. Must contain the 
        following keys: 'NoDX', 'DX_01_Sub', and 'DX_01_Confirmed'

    Returns
    -------
    str or float
        A string representing the participant's diagnosis status
    """

    if row["NoDX"] == 1 and row["DX_01_Sub"] == "No Diagnosis Given":
        return "No Diagnosis Given"
    elif row["NoDX"] == 2 and row["DX_01_Confirmed"] == 1:
        return row["DX_01_Sub"]
    elif row["NoDX"] == 3:
        return "dropped out"
    elif row["NoDX"] == 2 and row["DX_01_Confirmed"] == 0:
        return "Diagnosis not confirmed"
    elif row["NoDX"] == 2 and pd.isna(row["DX_01_Confirmed"]):
        return "No info about confirmation"
    else:
        return np.nan


def clean_diagnosis(input_path, save_path):
    """
    This function loads the HBN phenotype file, extracts and renames colums related to clinician consensus diagnoses, and determines the final diagnosis for 
    each subject using the `determine_final_diagnosis` function. The cleaned dataset 
    is saved as a new CSV file.

    Parameters
    ----------
    input_path : str
        Path to the raw phenotype CSV file containing subject diagnosis information.
    save_path :  str
        Path where the cleaned and processed CSV file will be saved.
    """
    

    # Load phenotype file
    file = input_path
    df = pd.read_csv(file)

    subject_id_column = df.columns[0]
    diagnosis_given = "Diagnosis_ClinicianConsensus,NoDX"

    # only keep columns of interest
    columns_to_keep = [subject_id_column, diagnosis_given]
    required_columns = set(columns_to_keep)
    assert required_columns.issubset(df.columns), "Missing expected columns in input CSV"

    for i in range(1, 2):
        prefix = f"Diagnosis_ClinicianConsensus,DX_{i:02d}"
        columns_to_keep.extend(
            [f"{prefix}", f"{prefix}_Confirmed", f"{prefix}_Cat", f"{prefix}_Sub"]
        )

    final_df = df[columns_to_keep]

    # Rename columns and subjectIDs
    final_df = final_df.rename(columns={subject_id_column: "subject"})
    final_df["subject"] = "sub-" + final_df["subject"].str.replace(
        ",assessment", "", regex=False
    )
    final_df.columns = [
        col.replace("Diagnosis_ClinicianConsensus,", "") for col in final_df.columns
    ]

    final_df["DX_01_Sub"] = final_df["DX_01_Sub"].fillna(final_df["DX_01_Cat"])

    final_df["FinalDiagnosis"] = final_df.apply(determine_final_diagnosis, axis=1)

    newfile = save_path
    final_df.to_csv(newfile, index=False)


def load_covariates_HBN(base_path: str, save_dir: str):
    """This info loads all files containing age, gender, diagnosis and site for HBN dataset"
    This function uses the pheno and site files that were available on the HBN website without the need for a data usage agreement
    It assumes that the pheno information is store in "HBN_R*_Pheno.csv" for each release and that the site 
    information is stored in one file "Subject-Site_all_Releases.xlsx" containing site info of all releases

    Parameters
    ----------
    base_path : str
        The file path where all files containing age, gender, diagnosis and site for HBN dataset can be found
    save_dir : str
         DataFrame: Pandas dataframe containing age, gender, site and diagnosis for HBN dataset.
    """

    # Find & concatenate all phenotype files to extract age and gender later
    search_pattern_pheno = os.path.join(base_path, "HBN_R*_Pheno.csv")
    pheno_files = glob.glob(search_pattern_pheno)
    pheno_dfs = []

    for file in pheno_files:
        df = pd.read_csv(file)
        pheno_dfs.append(df)

    full_pheno_df = pd.concat(pheno_dfs, ignore_index=True)
    full_pheno_df.rename(columns={"EID": "subject"}, inplace=True)
    full_pheno_df["subject"] = "sub-" + full_pheno_df["subject"].astype(str)

    # site files
    site_file = os.path.join(base_path, "Subject-Site_all_Releases.xlsx")
    site_df = pd.read_excel(site_file, engine="openpyxl")

    site_mapping = {
        0.0: "HBN0",
        1.0: "HBN1",
        2.0: "HBN2",
        3.0: "HBN3",
        4.0: "HBN4",
        5.0: "HBN5",
    }

    site_df["Study Site"] = site_df["Study Site"].map(site_mapping).astype(str)

    # Keep only rows from pheno_df that match site_df R_number
    merged_pheno_site = pd.merge(site_df, full_pheno_df, on="subject", how="inner")

    # Find cleaned diagnosis file
    search_pattern_diagnosis = os.path.join(base_path, "cleaned_diagnosis.csv")
    diagnosis_files = glob.glob(search_pattern_diagnosis)
    diagnosis_dfs = [pd.read_csv(file) for file in diagnosis_files]
    diagnosis_df = pd.concat(diagnosis_dfs, ignore_index=True)

    # Merge the 3 dataframes
    final_df = pd.merge(merged_pheno_site, diagnosis_df, on="subject", how="inner")

    # Ensure each subject appears only once (keep first occurrence)
    final_df = final_df.drop_duplicates(subset=["subject"], keep="first")

    # Save the final dataframe
    final_df.to_csv(save_dir, index=False)


def load_HBN_data(feature_path, covariates_path):
    """
    Load and merge HBN feature and covariate data.

    This function reads in phenotype covariates and extracted features for participants 
    in the HBN dataset, aligns them by subject ID, and returns 
    a merged DataFrame containing both sets of information.

    Parameters
    ----------
    feature_path : str
        Path to the CSV file containing extracted features. The index should include subject IDs,
        typically prefixed with "sub-".
    covariates_path : str
        Path to the TSV file containing subjects covariates. The first column
        should correspond to subject IDs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame resulting from the inner join of covariates and features,
        indexed by subject ID (without the "sub-" prefix).

    Notes
    -----
    - The subject IDs in the `HBN_features` file are stripped of the "sub-" prefix before merging.

    """
    HBN_covariates = pd.read_csv(covariates_path, sep="\t", index_col=0)
    HBN_features = pd.read_csv(feature_path, index_col=0)
    #remove sub- so that subjectIDs match format of subject IDs in covariate file (without Sub)
    HBN_features.index = HBN_features.index.str.replace("^sub-", "", regex=True)
    HBN_data = HBN_covariates.join(HBN_features, how="inner")
    return HBN_data
