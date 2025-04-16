import os
import glob
import pandas as pd


def define_eog_ecg_emg_channels_TDBrain(input_base_path):

    search_pattern = os.path.join(
        input_base_path, "*/ses-1/eeg/*_ses-1_task-restEC_channels.tsv"
    )
    channel_files = glob.glob(search_pattern)
    print("found:", channel_files)

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


if __name__ == "__main__":
    path = "/project/meganorm/Data/EEG_TDBrain/EEG/"
    define_eog_ecg_emg_channels_TDBrain(path)
