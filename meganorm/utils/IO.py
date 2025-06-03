import pickle
import pandas as pd
import json
import os
import re
import glob
from pathlib import Path
import mne
import numpy as np


def make_config(path=None):
    """
    Create a configuration dictionary for a neuroimaging preprocessing pipeline.

    This function generates configuration settings for preprocessing, feature extraction,
    spectral analysis, and other relevant parameters used in processing EEG/MEG data.
    Optionally, it saves the generated configuration to a JSON file in the specified path.

    Parameters
    ----------
    path : str, optional
        The directory path where the configuration file should be saved. If not provided,
        the configuration is not saved to a file.

    Returns
    -------
    config : dict
        The configuration dictionary containing settings for preprocessing, feature extraction,
        and analysis.

    Notes
    -----
    - The generated configuration includes settings for ICA preprocessing, spectral
      estimation, and feature extraction for EEG/MEG data.
    - Default values are provided for the majority of settings.
    - If `path` is provided, a `.json` file containing the configuration will be saved.
    """

    # preprocess configurations =================================================
    # downsample data
    config = dict()

    # You could also set layout to None to have high
    # choices: all, lobe, None
    config["which_layout"] = "all"

    # which sensor type should be used
    # choices: meg, mag, grad, eeg, opm
    config["which_sensor"] = "meg"
    # config['fs'] = 1000

    # ICA configuration
    config["ica_n_component"] = 30
    config["ica_max_iter"] = 800
    config["ica_method"] = "fastica"

    # lower and upper cutoff frequencies in a bandpass filter
    config["cutoffFreqLow"] = 1
    config["cutoffFreqHigh"] = 45

    config["resampling_rate"] = 1000
    config["digital_filter"] = True
    config["notch_filter"] = False

    config["apply_ica"] = True

    config["auto_ica_corr_thr"] = 0.9

    # options are "average", "REST", and None
    config["rereference_method"] = "average"

    # variance threshold across time
    config["mag_var_threshold"] = 4e-12
    config["grad_var_threshold"] = 4000e-13
    config["eeg_var_threshold"] = 40e-6
    # flatness threshold across time
    config["mag_flat_threshold"] = 10e-15
    config["grad_flat_threshold"] = 10e-15
    config["eeg_flat_threshold"] = 40e-6
    # variance thershold across channels
    config["zscore_std_thresh"] = 15  # change this

    # segmentation ==============================================
    # start time of the raw data to use in seconds, this is to avoid possible eye blinks in close-eyed resting state.
    config["segments_tmin"] = 20
    # end time of the raw data to use in seconds, this is to avoid possible eye blinks in close-eyed resting state.
    config["segments_tmax"] = -20
    # length of MEG segments in seconds
    config["segments_length"] = 10
    # amount of overlap between MEG sigals in seconds
    config["segments_overlap"] = 2

    # PSD ==============================================
    # Spectral estimation method
    config["psd_method"] = "welch"
    # amount of overlap between windows in Welch's method
    config["psd_n_overlap"] = 1
    config["psd_n_fft"] = 2
    # number of samples in psd
    config["psd_n_per_seg"] = 2

    # fooof analysis configurations ==============================================
    # Desired frequency range to run FOOOF
    config["fooof_freq_range_low"] = 3
    config["fooof_freq_range_high"] = 40
    config["fooof_freq_range_low"] = 3
    config["fooof_freq_range_high"] = 40
    # which mode should be used for fitting; choices (knee, fixed)
    config["aperiodic_mode"] = "knee"
    # minimum acceptable peak width in fooof analysis
    config["fooof_peak_width_limits"] = [1.0, 12.0]
    # Absolute threshold for detecting peaks
    config["fooof_min_peak_height"] = 0
    # Relative threshold for detecting peaks
    config["fooof_peak_threshold"] = 2

    # feature extraction ==========================================================
    # Define frequency bands
    config["freq_bands"] = {
        "Theta": (3, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 40),
        # 'Broadband': (3, 40)
    }

    # Define individualized frequency range over main peaks in each freq band
    config["individualized_band_ranges"] = {
        "Theta": (-2, 3),
        "Alpha": (-2, 3),  # change to (-4,2)
        "Beta": (-8, 9),
        "Gamma": (-5, 5),
    }

    # least acceptable R squred of fitted models
    config["min_r_squared"] = 0.9

    config["feature_categories"] = {
        "Offset": False,
        "Exponent": False,
        "Peak_Center": False,
        "Peak_Power": False,
        "Peak_Width": False,
        "Adjusted_Canonical_Relative_Power": True,
        "Adjusted_Canonical_Absolute_Power": False,
        "Adjusted_Individualized_Relative_Power": False,
        "Adjusted_Individualized_Absolute_Power": False,
        "OriginalPSD_Canonical_Relative_Power": False,
        "OriginalPSD_Canonical_Absolute_Power": False,
        "OriginalPSD_Individualized_Relative_Power": False,
        "OriginalPSD_Individualized_Absolute_Power": False,
    }

    config["fooof_res_save_path"] = None

    config["random_state"] = 42

    if path is not None:
        out_file = open(os.path.join(path, "configuration.json"), "w")
        json.dump(config, out_file, indent=6)
        out_file.close()

    return config


def storeFooofModels(path, subjId, fooofModels, psds, freqs) -> None:
    """
    Stores the periodic and aperiodic results from FOOOF analysis in a pickle file.

    This function saves the FOOOF models, the power spectral densities (PSDs),
    and the associated frequency data for a given subject into a `.pickle` file.
    The data is appended to the file for each subject.

    Parameters
    ----------
    path : str
        Directory path where the results will be saved.

    subjId : str
        The subject ID for which the results are saved.

    fooofModels : object
        The FOOOF model object containing the periodic and aperiodic components.

    psds : ndarray
        Power Spectral Densities (PSDs) calculated for the subject.

    freqs : ndarray
        Frequency values corresponding to the PSDs.

    Returns
    -------
    None
        This function does not return any value; it writes the results to a file.

    """
    with open(os.path.join(path, subjId + ".pickle"), "wb") as file:
        pickle.dump([fooofModels, psds, freqs], file)


def separate_eyes_open_close_eeglab(
    input_base_path,
    output_base_path,
    annotation_description_open,
    annotation_description_close,
    trim_before=5,
    trim_after=5,
):
    # Ensure output directory exists
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    search_pattern = os.path.join(input_base_path, "*/eeg/*_task-rest_eeg.set")
    raw_set_paths = glob.glob(
        search_pattern, recursive=True
    )  # Use glob to find all .set files in the input directory

    # Loop through all found .set files
    for set_path in raw_set_paths:
        subject_id = Path(set_path).parts[
            -3
        ]  # Extract subject number from the file path
        subject_output_path = os.path.join(
            output_base_path, subject_id, "eeg"
        )  # Create the subject-specific output path

        # Ensure output directory for the subject exists
        if not os.path.exists(subject_output_path):
            os.makedirs(subject_output_path)

        # Load the raw .set file (EEGLAB format)
        raw = mne.io.read_raw(set_path, preload=True)

        # Extract annotations
        annotations = raw.annotations

        # Separate eyes open and eyes closed events
        eyes_open_events = annotations[
            annotations.description == annotation_description_open
        ]
        eyes_closed_events = annotations[
            annotations.description == annotation_description_close
        ]

        # Extract and concatenate eyes open segments
        eyes_open_data = []
        for onset, duration in zip(eyes_open_events.onset, eyes_open_events.duration):

            if duration <= trim_before + trim_after:
                print(
                    f"Skipping event with onset {onset} and duration {duration} (invalid after trimming)"
                )
                continue

            # Trim the first 5s and last 5s from each event
            trimmed_onset = onset + trim_before
            trimmed_duration = duration - trim_before - trim_after
            start_sample = int(trimmed_onset * raw.info["sfreq"])
            stop_sample = int((trimmed_onset + trimmed_duration) * raw.info["sfreq"])
            eyes_open_data.append(raw[:, start_sample:stop_sample][0])

        if eyes_open_data:
            eyes_open_data_concat = np.concatenate(eyes_open_data, axis=1)
            raw_eyes_open = mne.io.RawArray(eyes_open_data_concat, raw.info)

            # Save eyes open data as a new .set file
            eyes_open_file_path = os.path.join(
                subject_output_path, f"{subject_id}_task-eyesopen_eeg.set"
            )
            mne.export.export_raw(
                eyes_open_file_path, raw_eyes_open, fmt="eeglab", overwrite=True
            )

        # Extract and concatenate eyes closed segments
        eyes_closed_data = []
        for onset, duration in zip(
            eyes_closed_events.onset, eyes_closed_events.duration
        ):

            if duration <= trim_before + trim_after:
                print(
                    f"Skipping event with onset {onset} and duration {duration} (invalid after trimming)"
                )
                continue

            trimmed_onset = onset + trim_before
            trimmed_duration = duration - trim_before - trim_after
            start_sample = int(trimmed_onset * raw.info["sfreq"])
            stop_sample = int((trimmed_onset + trimmed_duration) * raw.info["sfreq"])
            eyes_closed_data.append(raw[:, start_sample:stop_sample][0])

        if eyes_closed_data:
            eyes_closed_data_concat = np.concatenate(eyes_closed_data, axis=1)
            raw_eyes_closed = mne.io.RawArray(eyes_closed_data_concat, raw.info)

            # Save eyes closed data as a new .set file
            eyes_closed_file_path = os.path.join(
                subject_output_path, f"{subject_id}_task-eyesclosed_eeg.set"
            )
            mne.export.export_raw(
                eyes_closed_file_path, raw_eyes_closed, fmt="eeglab", overwrite=True
            )


def merge_fidp_demo(
    datasets_paths: list,
    features_dir: str,
    dataset_names: list,
    drop_columns: list = ["eyes"],
):
    """
    Merge demographic metadata and extracted features into a single DataFrame.

    This function loads demographic data and feature data,
    assigns a site label to each participant if missing, removes unnecessary columns,
    and merges demographic information with corresponding extracted features.

    Parameters
    ----------
    datasets_paths : list
        List of paths to the dataset directories containing demographic files
        ('participants_bids.tsv').
    features_dir : str
        Path to the directory containing the extracted features ('all_features.csv').
    dataset_names : list of str
        List of dataset names corresponding to each dataset path. Used to populate
        missing 'site' information if necessary.
    drop_columns : list of str, optional
        Columns to drop from the demographic data before merging. Default is ["eyes"].

    Returns
    -------
    data : pandas.DataFrame
        Merged DataFrame containing both demographic information and feature data,
        with participants indexed as strings.

    Raises
        ------
        FileNotFoundError
            If the 'participants_bids.tsv' file is missing in any of the dataset paths or
            the 'all_features.csv' file is missing in the provided features directory.
    """

    # Initialize empty DataFrame
    demographic_df = pd.DataFrame()

    # Loop through dataset paths
    for counter, dataset_path in enumerate(datasets_paths):
        demo_path = os.path.join(dataset_path, "participants_bids.tsv")
        if not os.path.exists(demo_path):
            raise FileNotFoundError(
                f"The file 'participants_bids.tsv' is missing from the directory: {dataset_path}. "
                "This file must be created using the 'make_demo_file_bids' function and placed in "
                "the corresponding dataset directory."
            )
        demo = pd.read_csv(demo_path, sep="\t", index_col=0)
        demo.index = demo.index.astype(str)

        if "site" not in demo.columns:
            demo["site"] = dataset_names[counter]

        demographic_df = pd.concat([demographic_df, demo], axis=0)

    # Drop unnecessary columns
    demographic_df.drop(columns=drop_columns, errors="ignore", inplace=True)

    # Load features
    feature_path = os.path.join(features_dir, "all_features.csv")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"The file 'all_features.csv' is missing in the directory: {features_dir}."
        )
    features_df = pd.read_csv(feature_path, index_col=0)
    features_df.index = features_df.index.astype(str)

    # Merge demographic and features
    data = demographic_df.join(features_df, how="inner")
    data.index.name = None

    return data


def factorize_columns(df: pd.DataFrame, columns: list):
    """
    Factorizes specified columns in the DataFrame.
    For the 'diagnosis' column, it assigns 0 to 'control' and factorizes the rest.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the columns to be factorized.
    columns : list
        List of column names to be factorized.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with factorized columns.
    """

    for col in columns:
        if col in df.columns:
            if col == "diagnosis":
                # Drop rows where diagnosis is NaN
                df = df.dropna(subset=["diagnosis"])
                # Assign 0 to 'control' and factorize the rest
                df["diagnosis"] = np.where(
                    df["diagnosis"] == "control",
                    0,
                    pd.factorize(df["diagnosis"])[0] + 1,
                )
            else:
                # Factorize other columns
                df[col] = pd.factorize(df[col])[0]

    return df


def normalize_column(df, column="age", normalizer=100):
    """
    Normalizes a specified column in the DataFrame by dividing its values by the given normalizer.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the column to be normalized.
    column : str, optional
        The column to be normalized (default is "age").
    normalizer : float or None, optional
        The value by which the column will be divided. If None, the column will not be normalized.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with the normalized column.

    Raises
    ------
    KeyError
        If the specified column does not exist in the DataFrame.
    ValueError
        If the normalizer is not a positive numeric value.
    """

    # Check if column exists in DataFrame
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    # Check if the normalizer is a valid positive number
    if normalizer is not None and (
        not isinstance(normalizer, (int, float)) or normalizer <= 0
    ):
        raise ValueError(
            f"Normalizer should be a positive numeric value, got {normalizer}."
        )

    # Normalize the column if a valid normalizer is provided
    if normalizer:
        df[column] = df[column] / normalizer

    return df


def separate_patient_data(df, diagnosis: list):
    """
    Separates patients' data from control data based on the diagnosis column.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the patient data.
    diagnosis : list of str
        A list of diagnosis values used to separate patients' data.

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame containing only control data (after dropping the 'diagnosis' column).
    df_patient : pandas.DataFrame
        The DataFrame containing the patient data.

    Raises
    ------
    KeyError
        If the 'diagnosis' column is not found in the DataFrame.
    """

    # Ensure the 'diagnosis' column exists in the DataFrame
    if "diagnosis" not in df.columns:
        raise KeyError("The 'diagnosis' column is missing in the DataFrame.")

    # Separate the patient data based on the 'diagnosis' column
    df_patient = df[df["diagnosis"].isin(diagnosis)]

    # Filter the control data and drop the 'diagnosis' column
    df = df[df["diagnosis"] == "control"].drop(columns="diagnosis", errors="ignore")

    return df, df_patient


def merge_datasets_with_glob(datasets):
    """
    Merges file paths across multiple datasets using glob pattern matching.

    This function walks through the provided datasets' base directories to find
    subject folders and file paths matching a specified task and file ending. It
    creates a dictionary mapping each subject to a glob pattern that can be used
    to aggregate files across multiple runs or sessions.

    Parameters
    ----------
    datasets : dict
        Dictionary where each key is a dataset name, and each value is a dictionary
        with the following keys:
            - "base_dir" (str): Base directory containing subject subdirectories.
            - "task" (str): Task keyword to search for in filenames.
            - "ending" (str): File ending (e.g., '.nii.gz') to filter relevant files.

    Returns
    -------
    dict
        A dictionary mapping subject IDs to a glob-style path string that aggregates
        all matching files for that subject. Only subjects with at least one matched
        file are included.

    Notes
    -----
    This function is designed to assist in scenarios where each subject may have
    multiple files (e.g., different runs or sessions), and the goal is to create
    a single pattern that can be used to load all related files for a subject.
    """
    subjects = {}

    for dataset_name, dataset_info in datasets.items():
        base_dir = dataset_info["base_dir"]

        dirs = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]
        subjects.update({subj: [] for subj in dirs})

        paths = glob.glob(
            f"{datasets[dataset_name]["base_dir"]}/**/*{datasets[dataset_name]["task"]}*{datasets[dataset_name]["ending"]}",
            recursive=True,
        )

        # Walk through the base directory to find subject directories
        for subject_dir in dirs:
            pattern = os.path.join(datasets[dataset_name]["base_dir"], subject_dir)
            subjects[subject_dir].extend(
                list(filter(lambda path: path.startswith(pattern), paths))
            )

    def join_with_star(lst):
        if len(lst) == 1:
            return lst[0] + "*"
        return "*".join(lst)

    # add this part to main parallel when you want to concatenate
    # different run
    subjects = dict(filter(lambda item: item[1], subjects.items()))
    subjects = {key: join_with_star(value) for key, value in subjects.items()}

    return subjects


def make_demo_file_bids(
    file_dir: str, save_dir: str, id_col: int, age_col: int, *columns
) -> None:
    """
    Convert formats of demographic data into a single format so it can be used
    in later stages.

    Parameters
    ----------
    file_dir : str
        Path to the input demographic file (supports CSV, TSV, or XLSX).
    save_dir : str
        Path where the BIDS-formatted demographic file will be saved (as TSV).
    id_col : int
        Column index containing the participant ID.
    age_col : int
        Column index containing participant age.
    *extra_columns : dict
        Additional column definitions. While age and participants id were defined
        using positional arguments, extra coulmn modification (e.g., sex and eyes
        condition) can be revised and converted to a single format across dataset
        using this function. Each dict can contain:
            - 'col_name': str, required name for the output column. This does not
                necessarly match the column name before being passed to this function.
            - 'col_id': int, index of the column that the revision should be applied to.
            - 'single_value': value to assign to all rows if no col_id and mapping are given.
                This can be helpful when all subjects in a dataset have the same properties
                e.g., eyes open condition.
            - 'mapping': dict, if single value is not defined, value mapping can be passed
                to map the initial values to the target values.

    Returns
    -------
    None
    """
    for col in columns:
        if col.get("single_value") and col.get("mapping"):
            raise ValueError(
                "'single_value' and 'mapping' can not be both defined. One of them must be None; see the documentation!"
            )

    # Load input file based on extension
    if file_dir.endswith(".xlsx"):
        df = pd.read_excel(file_dir)
    elif file_dir.endswith(".csv"):
        df = pd.read_csv(file_dir)
    elif file_dir.endswith(".tsv"):
        df = pd.read_csv(file_dir, sep="\t")
    else:
        raise ValueError(f"Unsupported file type for: {file_dir}")

    # Initialize new dataframe with required fields
    new_df = pd.DataFrame(
        {"participant_id": df.iloc[:, id_col], "age": df.iloc[:, age_col]}
    )

    for col in columns:
        col_name = col.get("col_name")
        col_id = col.get("col_id")
        mapping = col.get("mapping")
        single_value = col.get("single_value")

        if col_name is None:
            raise ValueError("Each column dictionary must contain a 'col_name'.")

        if col_id is not None:
            new_df[col_name] = df.iloc[:, col_id]
            if mapping:
                new_df[col_name] = new_df[col_name].map(mapping)
        elif single_value is not None:
            new_df[col_name] = single_value
        else:
            raise ValueError(
                f"Column '{col_name}' must have either 'col_id' or 'single_value'."
            )

        # Special case handling
        if col_name == "diagnosis":
            new_df[col_name] = new_df[col_name].fillna("nan")

    # Remove duplicate participants
    new_df = new_df.drop_duplicates(subset="participant_id", keep="first")

    # Save as BIDS-compatible TSV
    new_df.to_csv(save_dir, sep="\t", index=False)
