import os
import glob
import pandas as pd


def define_eog_ecg_emg_channels_TDBrain(input_base_path):

    """Update channel types for TDBrain EEG channel TSV files (only for restEC task, session 1)

    This function scans the EEG channel metadata files for the TDBrain dataset
    and updates specific channels to appropriate types: "EOG", "ECG", or "EMG".
    The updated TSV files are saved in-place.

    Parameters
    ----------
    input_base_path : _type_
        Path to the root directory of the TDBrain dataset. It should contain
        subject/session folders with EEG data organized under `sub-*/ses-1/eeg/`.
    
    Returns
    -------
    None
        This function performs file operations and does not return any value.
        The modified files are saved directly to disk.
    """

    search_pattern = os.path.join(
        input_base_path, "*/ses-1/eeg/*_ses-1_task-restEC_channels.tsv"
    )
    channel_files = glob.glob(search_pattern)
    
    for file in channel_files:
        channels_df = pd.read_csv(file, sep="\t")
        # Define the mappings for channel types
        update_mapping = {
            "VPVA": "EOG",
            "VNVB": "EOG",
            "HPHL": "EOG",
            "HNHR": "EOG",
            "Erbs": "ECG",
            "Mass": "EMG",
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
