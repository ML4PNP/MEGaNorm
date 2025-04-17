import os
import glob
from pathlib import Path
import scipy
import mne
import mne_bids
import pandas as pd


def preprocess_events_file(file_path):
    """
    Preprocess the events CSV file by removing repeated header rows while retaining the first.

    This function cleans exported EEG event files that may contain duplicated header lines (e.g., `"type","latency","urevent",`). 
    It keeps the first header line, removes all subsequent duplicate headers, and writes the 
    cleaned data to a new file.

    Parameters
    ----------
    file_path : str
        Path to the raw events CSV file that needs preprocessing.

    Returns
    -------
    str
        Path to the cleaned CSV file. The cleaned file is saved at the same location with
        "_cleaned.csv" appended to the original file name.
    """
    clean_lines = []
    header_found = False  # Track if header is already added

    with open(file_path, "r") as f:
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
    with open(cleaned_file_path, "w") as f:
        f.write("\n".join(clean_lines) + "\n")  # Ensure proper formatting

    return cleaned_file_path


def mne_bids_MIPDB(input_base_path, output_base_path, montage_path):
    """
    This function converges the MIPDB dataset .csv files into BIDS format.
    
    This function processes EEG recordings from the MIPDB dataset that are stored in CSV files.
    It converts them into MNE Raw objects, applies channel mappings and a custom montage, 
    sets specific channel types (e.g., 'eog', 'misc'), embeds event annotations, 
    and saves the data in BIDS-compliant format using the `mne-bids` library.

    Parameters
    ----------
    input_base_path : str
        Path to the base directory containing the raw MIPDB dataset
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
    - Raw EEG recordings are expected in CSV format, with filenames ending in `001.csv` and corresponding event files ending in in `001_events.csv`
    - The following channels are treated specially:
        - Misc (neck and chin): E48, E49, E56, E63, E68, E73, E81, E88, E94, E99, E107, E113, E119
        - EOG (eyes): E8, E14, E17, E21, E25, E128, E127, E126, E125
    """

    # Ensure output directory exists
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
        print("directory created")

    # Find all raw resting state files and event files from every participant
    search_pattern = os.path.join(input_base_path, "*/EEG/raw/csv_format/*001.csv")
    raw_path = glob.glob(search_pattern)
    search_pattern2 = os.path.join(
        input_base_path, "*/EEG/raw/csv_format/*001_events.csv"
    )
    events_data = glob.glob(search_pattern2)

    # Loop through all found .raw files
    for raw, events in zip(raw_path, events_data):

        # extract subject id
        subject_id = Path(raw).parts[-5]

        # transform csv file into dataframe
        dataframe = pd.read_csv(raw, header=None)
        n_channels = len(dataframe)
        sampling_freq = 500

        # create mne info object
        info = mne.create_info(n_channels, sfreq=sampling_freq, ch_types="eeg")
        raw = mne.io.RawArray(dataframe.values, info)

        # set channel locations
        montage = mne.channels.read_custom_montage(montage_path)

        # Create a mapping to rename the channels in raw to match the montage
        mapping = {f"{i}": f"E{i+1}" for i in range(n_channels)}  # E1, E2, ...
        raw.info.rename_channels(mapping)
        raw.info.set_montage(montage)

        # Define channels on nek and chin as misc and channels close to eyes as eog
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
            "E127",
            "E126",
            "E125",
            "E128",
        ]

        # Create a dictionary for setting channel types
        channel_types = {ch: "misc" for ch in misc_channels}
        channel_types.update({ch: "eog" for ch in eog_channels})

        # Apply the channel types to the raw object
        raw.set_channel_types(channel_types)

        print("channel Types", raw.get_channel_types)
        print("channel names", raw.ch_names)
        print("montage", raw.get_montage)

        ##### Extract event and epoch information to set the annotations
        # Preprocess the events file to remove repeated headers
        cleaned_events_path = preprocess_events_file(events)
        events_data = pd.read_csv(cleaned_events_path)

        # Ensure 'latency' column is numeric
        events_data["latency"] = pd.to_numeric(events_data["latency"], errors="coerce")

        # Drop rows with invalid latencies, if any
        events_data = events_data.dropna(subset=["latency"])

        event_type = events_data["type"]
        event_types = [str(item) for item in event_type]
        event_latencies = events_data["latency"]

        event_onsets = [latency / sampling_freq for latency in event_latencies]
        event_duration = [
            event_onsets[i] - event_onsets[i - 1] for i in range(1, len(event_onsets))
        ]
        event_duration.append(0)

        # Create annotations using onset, duration, and description
        annotations = mne.Annotations(
            onset=event_onsets, duration=event_duration, description=event_types
        )

        # Attach annotations to the raw object
        raw.set_annotations(annotations)

        # extra info that cannot
        raw.info["line_freq"] = 60
        raw.info["device_info"] = {
            "type": "EEG",
            "manufacturer": "Electrical Geodesics",
            "model": "HydroCel GSN 130",
        }

        # convert to BIDS
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

    return None
