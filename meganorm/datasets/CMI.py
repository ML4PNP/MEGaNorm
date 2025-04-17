import mne_bids
import glob
import shutil
import os
import mne
from pathlib import Path
import scipy
import numpy as np
import pandas as pd
import re
import numpy as np
import pandas as pd
from meganorm.utils.EEGlab import read_raw_eeglab


def mne_bids_CMI(input_base_path, output_base_path, montage_path):
    """
    This code converges the CMI dataset into BIDS format.
    Meanwhile, it defines channels on nek and chin as misc channels and channels around the eyes as eog channels
    """

    # Ensure output directory exists
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    search_pattern = os.path.join(
        input_base_path, "*/*/EEG/raw/mat_format/RestingState.mat"
    )
    raw_mat_path = glob.glob(
        search_pattern
    )  # Use glob to find all RestingState.mat files in raw/mat_format folder

    # Loop through all found .mat files
    for mat_path in raw_mat_path:
        try:
            subject_id = Path(mat_path).parts[
                -5
            ]  # Extract subject number from the file path
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


def define_eog_ecg_channels_CMI(input_base_path):

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
    # Load phenotype file
    file = input_path
    df = pd.read_csv(file)

    subject_id_column = df.columns[0]
    diagnosis_given = df.columns[156]

    # only keep columns of interest
    columns_to_keep = [subject_id_column, diagnosis_given]
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


def load_covariates_CMI(base_path: str, save_dir: str):
    """This info loads all files containing age, gender, diagnosis and site for CMI dataset"

    Args:
        path (str): path to covariates.

    Returns:
        DataFrame: Pandas dataframe containing age, gender, site and diagnosis for CMI dataset.
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
        0.0: "CMI0",
        1.0: "CMI1",
        2.0: "CMI2",
        3.0: "CMI3",
        4.0: "CMI4",
        5.0: "CMI5",
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


def load_CMI_data(feature_path, covariates_path):
    """Load CMI dataset

    Args:
        feature_path (str): Path to the the feature csv file.
        covariates_path (str): path to the covariates tsv file.

    Returns:
        DataFrame: Pandas dataframe with CMI covariates and features.
    """

    CMI_covariates = pd.read_csv(covariates_path, sep="\t", index_col=0)
    CMI_features = pd.read_csv(feature_path, index_col=0)
    CMI_features.index = CMI_features.index.str.replace("^sub-", "", regex=True)
    CMI_data = CMI_covariates.join(CMI_features, how="inner")
    return CMI_data


if __name__ == "__main__":
    input_base_path = "/home/meganorm-yverduyn/Dev/2_EXAMPLE_SUBJECTS_CMI"
    output_base_path = "/home/meganorm-yverduyn/Dev/2_EXAMPLE_SUBJECTS_CMI/BIDS"
    montage_path = "/project/meganorm/Data/EEG_CMI/info/GSN_HydroCel_129.sfp"
    mne_bids_CMI(input_base_path, output_base_path, montage_path)

    base_path = "/project/meganorm/Data/EEG_CMI/info/"
    save_dir = "/project/meganorm/Data/EEG_CMI/info/participants_info.csv"
    load_covariates_CMI(base_path, save_dir)

    input_path = (
        "/project/meganorm/Data/EEG_CMI/Phenotypes/data-2025-01-28T08_15_39.544Z.csv"
    )
    save_path = "/project/meganorm/Data/EEG_CMI/info/cleaned_diagnosis.tsv"
    clean_diagnosis(input_path, save_path)
