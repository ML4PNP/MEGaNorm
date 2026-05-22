import pickle
import pandas as pd
import json
import os
import re
import glob
from pathlib import Path
import mne
import numpy as np
from typing import Literal, Dict, Tuple, List, Optional
from typing import Optional
import warnings
from typing import Union
from typing import ClassVar
from pydantic import BaseModel, Field, PositiveInt, confloat, conint, conlist, field_validator, NegativeInt, model_validator

class BandRatio(BaseModel):
    numerator: Literal["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    denominator: Literal["Delta", "Theta", "Alpha", "Beta", "Gamma"]


class Config(BaseModel):
    """
    Configuration class for preprocessing, feature extraction, and analysis of 
    neurophysiological signals (e.g., MEG, EEG, OPM). Provides a comprehensive set 
    of parameters for filtering, ICA, artifact detection, source localization, 
    power spectral density (PSD) estimation, FOOOF analysis, and feature extraction.

    Attributes
    ----------
    which_layout : {"all", "lobe", "None"}, default="all"
        Sensor layout selection.
    which_sensor : {"mag", "grad", "meg", "eeg", "opm"}, default="meg"
        Sensor type to process.

    ICA Parameters
    --------------
    ica_n_component : int, default=30
        Number of ICA components to compute.
    ica_max_iter : int, default=800
        Maximum iterations for ICA algorithm.
    ica_method : {"fastica", "infomax", "picard"}, default="fastica"
        ICA algorithm to use.

    Filtering Parameters
    -------------------
    cutoffFreqLow : int, default=1
        Low cutoff frequency for bandpass filter (Hz).
    cutoffFreqHigh : int, default=40
        High cutoff frequency for bandpass filter (Hz).
    resampling_rate : int, default=1000
        Sampling rate for resampling (Hz).
    digital_filter : bool, default=True
        Apply digital bandpass filtering.
    notch_filter : bool, default=True
        Apply notch filter to remove line noise.

    Artifact Detection
    -----------------
    muscle_activity_thr : int, default=4
        Threshold for muscle artifact detection.
    muscle_activity_min_length_good : float, default=0.1
        Minimum clean segment length after artifact removal (s).
    muscle_activity_filter_freq : tuple[int, int], default=(110, 140)
        Frequency range for muscle artifact detection (Hz).

    apply_ica : bool, default=True
        Apply ICA artifact correction.
    auto_ica_corr_thr : float, default=0.9
        Correlation threshold for automatic ICA component removal.

    EEG Reference
    -------------
    rereference_method : {"average", "REST", "None"}, default="average"
        EEG re-referencing method.

    Signal Quality Thresholds
    -------------------------
    mag_var_threshold : float, default=4e-12
        Variance threshold for MEG magnetometers.
    grad_var_threshold : float, default=4000e-13
        Variance threshold for MEG gradiometers.
    eeg_var_threshold : float, default=40e-6
        Variance threshold for EEG.
    mag_flat_threshold : float, default=10e-15
        Flatline threshold for MEG magnetometers.
    grad_flat_threshold : float, default=10e-15
        Flatline threshold for MEG gradiometers.
    eeg_flat_threshold : float, default=40e-6
        Flatline threshold for EEG.
    zscore_std_thresh : int, default=15
        Standard deviation threshold for z-score outlier rejection.

    Segmentation
    ------------
    segments_tmin : int, default=20
        Start time relative to event (s).
    segments_tmax : int, default=-20
        End time relative to event (s).
    segments_length : int, default=10
        Segment length (s).
    segments_overlap : int, default=2
        Segment overlap (s).

    Source Localization
    -------------------
    apply_source_localization : bool, default=False
        Whether to perform source localization.
    SL_source_space : {"surface", "volumetric"}, default="surface"
        Type of source space.
    SL_conductivity : tuple[float, ...], default=(0.3,)
        Conductivity values for head model layers.
    SL_inverse_operator : {"lcmv"}, default="lcmv"
        Inverse operator method.
    source_space_spacing : {"ico3", "ico4", "ico5", "ico6", "oct5", "oct6"}, default="ico4"
        Source space resolution.
    source_space_spacing_number : {3, 4, 5, 6}, default=4
        Resolution number corresponding to `source_space_spacing`.

    Beamformer Parameters
    --------------------
    beamformer_pick_ori : {None, "normal", "max-power", "vector"}, default="max-power"
        Orientation selection for beamformer.
    beamformer_weight_norm : {None, "unit-noise-gain", "nai", "unit-noise-gain-invariant"}, default="unit-noise-gain"
        Weight normalization method.
    beamforme_depth : float, optional
        Scaling factor to correct for head-center bias.
    inverse_regularization_value : float, default=0.05
        Regularization for data covariance matrix.

    Parcellation
    ------------
    parcellation_parc : {None, "aparc.a2009s", "parac"}, default="aparc.a2009s"
        Predefined parcellation.
    parcellation_annot_fname : Path or None
        Custom parcellation file.

    PSD Parameters
    --------------
    psd_method : {"multitaper", "welch"}, default="welch"
        PSD estimation method.
    psd_n_overlap : int, default=1
        Number of overlapping samples.
    psd_n_fft : int, default=2
        FFT length.
    psd_n_per_seg : int, default=2
        Segment length for PSD.

    FOOOF Analysis
    --------------
    fooof_freq_range_low : int, default=3
        Lower frequency bound (Hz).
    fooof_freq_range_high : int, default=40
        Upper frequency bound (Hz).
    aperiodic_mode : {"knee", "fixed"}, default="knee"
        Model for aperiodic component.
    fooof_peak_width_limits : list[float], default=[1.0, 12.0]
        Peak width limits (Hz).
    fooof_min_peak_height : int, default=0
        Minimum peak height.
    fooof_peak_threshold : int, default=2
        Peak detection threshold.
    fooof_res_save_path : str, optional
        Path to save FOOOF results.

    Feature Extraction
    ------------------
    freq_bands : dict[str, tuple[int, int]]
        Canonical frequency bands.
    individualized_band_ranges : dict[str, tuple[int, int]]
        Offsets for individualized bands.
    min_r_squared : float, default=0.9
        Minimum R² for model fit.
    feature_categories : dict[str, bool]
        Flags indicating which features to extract.

    Miscellaneous
    -------------
    random_state : int, default=42
        Random seed for reproducibility.

    Methods
    -------
    save(path)
        Save configuration to JSON.
    load(path) -> Config
        Load configuration from JSON.
    muscle_activity_thr_fv(v)
        Validator for muscle activity threshold.
    muscle_activity_filter_freq_fv(v)
        Validator for muscle artifact frequency.
    SL_conductivity_mv()
        Validator for EEG conductivity.
    """

    which_meg_session : int = 0 # the first session

    which_layout: Literal["all", "lobe", None] = "all"
    which_sensor: Literal["mag", "grad", "meg", "eeg", "opm"] = "meg"

    drop_noisy_flat_channel: bool = True

    # ICA
    apply_ica_elbow_detection: bool = False
    ica_n_component: Optional[PositiveInt] = None
    ica_max_iter: PositiveInt = 800
    ica_method: Literal["fastica", "infomax", "picard"] = "fastica"

    cutoffFreqLow: float = 1.0
    cutoffFreqHigh: PositiveInt = 80

    resampling_rate: PositiveInt = 1000
    digital_filter: bool = True
    notch_filter: bool = True

    apply_oversampled_temporal_projection: bool = True

    apply_Head_movement_correction: bool = True
    Head_movement_limit_from_mean: float = 0.0015

    apply_chpi_filter: bool = False

    # gedai settings
    apply_gedai: bool = True
    gedai_method: Literal["both", "spectral", "broadband"] = "both"
    sensai_method: Literal["optimize", "gridsearch"] = "optimize"
    gedai_duration: Union[float, int] = 12
    gedai_overlap: Union[float, int] = 0.5
    gedai_preliminary_broadband_noise_multiplier: float = 6.0
    gedai_noise_multiplier: float = 3.0
    gedai_wavelet_type: str ="haar"
    gedai_wavelet_level: Union[Literal["auto"], PositiveInt, Literal[0]] = "auto"
    gedai_wavelet_low_cutoff: Union[None, float] = None
    gedai_epoch_size_in_cycles: PositiveInt = 12
    gedai_highpass_cutoff: float = 0.1

    muscle_activity_thr: int = 4
    muscle_activity_min_length_good: float = 0.1
    muscle_activity_filter_freq: Tuple[int, int] = (110, 140)

    apply_environmental_noise_correction: bool = True
    ctf_gradient_comp_level: PositiveInt = 3
    apply_environmental_noise_ssp_with_eroom: bool = False
    apply_environmental_noise_ica_with_ref_meg: bool = True
    environmental_noise_ica_with_ref_meg_thr: float = 2.5
    ica_if_reject_by_annotation: bool = True
    environmental_noise_ica_with_ref_meg_method: Literal["together", "separate"] = "separate"
    environmental_noise_ica_with_ref_meg_measure: Literal["zscore", "correlation"] = "zscore"
    
    apply_ica: bool = True
    auto_ica_corr_thr: confloat(ge=0, le=1) = 0.5

    rereference_method: Literal["average", "REST", "None"] = "average"

    bad_segment_removal_method: Literal["autoreject", "fixed_thr", None] = "autoreject"
    mag_var_threshold: float = 5000e-15
    grad_var_threshold: float = 5000e-13
    eeg_var_threshold: float = 40e-6
    mag_flat_threshold: float = 10e-15
    grad_flat_threshold: float = 10e-13
    eeg_flat_threshold: float = 40e-6
    zscore_std_thresh: PositiveInt = 15

    segments_tmin: PositiveInt = 20
    segments_tmax: NegativeInt = -20
    segments_length: PositiveInt = 10
    segments_overlap: int = 2

    # autoreject
    autoreject_n_interpolates: List[int] = [1, 4, 8, 16, 32]
    autoreject_consensus_percs: List[float] = list(np.linspace(0, 1.0, 11))
    autoreject_cv: Union[int, Literal["auto"]] = "auto"
    autoreject_thresh_method: Literal['bayesian_optimization', "random_search"] = "bayesian_optimization"

    # Source localization
    apply_source_localization: bool = False
    apply_empty_room_recording: bool = True
    apply_mri_QC: bool = False
    SL_source_space: Literal["surface", "volumetric"] = "volumetric"
    SL_conductivity: Tuple[float, ...] = (0.3,)
    SL_inverse_operator: Literal["lcmv"] = "lcmv"

    # the spacing to use for source space specificatin
    source_space_spacing:  Literal["ico3", "ico4", "ico5", "ico6", "oct5", "oct6"] = "ico4"
    source_space_spacing_number: Literal[3, 4, 5, 6]=4

    coregisteration_final_n_iterations: int = 20
    coregisteration_final_nasion_weight: float = 10.0
    covariance_method: str = "empirical"

    # Determines whether to keep vectors for all source orientations
    # or to select a single fixed orientation, depending on the chosen algorithm.
    beamformer_pick_ori: Literal[None, "normal", "max-power", "vector"] = "max-power"
    beamformer_weight_norm: Literal[None, "unit-noise-gain", "nai", "unit-noise-gain-invariant"] = "unit-noise-gain"

    # This parameter scales the activation to correct for head-center bias.
    beamforme_depth: confloat(ge=0, le=1) = 0.08

    # this is used for regularaizing the data covariance (shifting the matrix)
    inverse_regularization_value: confloat(ge=0, le=1) = 0.05

    apply_morphing: bool = False

    # the pacellation to use
    parcellation_parc: Literal[None, "aparc.a2009s", "parac"] = "aparc.a2009s"

    # A custom parcellation file
    parcellation_annot_fname: Optional[Path] = None


    # PSD
    psd_method: Literal["multitaper", "welch"] = "welch"
    psd_n_overlap: PositiveInt = 1
    psd_n_fft: PositiveInt = 2
    psd_n_per_seg: PositiveInt = 2

    parametrization_method: Literal["fooof", "irasa"] = "irasa"
    # PYRASA
    irasa_hset: Tuple[float, float, float] = (1.05, 2.0, 0.05)

    # FOOOF analysis
    fooof_freq_range_low: PositiveInt = 3
    fooof_freq_range_high: PositiveInt = 40
    aperiodic_mode: Literal["knee", "fixed"] = "knee"
    fooof_peak_width_limits: List[float] = [1.0, 12.0]
    fooof_min_peak_height: int = 0
    fooof_peak_threshold: PositiveInt = 2
    
    save_source_localized_epochs: bool = False
    save_psds : bool = False

    # Feature extraction
    freq_bands: Dict[str, Tuple[int, int]] = {
        "Theta": (3, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 40),
    }

    individualized_band_ranges: Dict[str, Tuple[int, int]] = {
        "Theta": (-2, 3),
        "Alpha": (-2, 3),
        "Beta": (-8, 9),
        "Gamma": (-5, 5),
    }

    power_band_ratios_list: List[BandRatio] = [
        BandRatio(numerator="Theta", denominator="Beta"),
        BandRatio(numerator="Theta", denominator="Alpha"),
        BandRatio(numerator="Alpha", denominator="Beta"),
        BandRatio(numerator="Delta", denominator="Beta"),
        BandRatio(numerator="Delta", denominator="Alpha"),
        BandRatio(numerator="Delta", denominator="Theta"),
        BandRatio(numerator="Beta", denominator="Gamma"),
        BandRatio(numerator="Alpha", denominator="Gamma"),
        BandRatio(numerator="Theta", denominator="Gamma"),
        BandRatio(numerator="Delta", denominator="Gamma"),
    ]

    min_r_squared: confloat(ge=0, le=1) = 0.9

    feature_categories: Dict[str, bool] = {
        "Offset": True,
        "Exponent": True,
        "Peak_Center": False,
        "Peak_Power": False,
        "Peak_Width": False,
        "Adjusted_Canonical_Relative_Power": True,
        "Adjusted_Canonical_Absolute_Power": True,
        "Adjusted_Individualized_Relative_Power": False,
        "Adjusted_Individualized_Absolute_Power": False,
        "OriginalPSD_Canonical_Relative_Power": True,
        "OriginalPSD_Canonical_Absolute_Power": True,
        "OriginalPSD_Individualized_Relative_Power": False,
        "OriginalPSD_Individualized_Absolute_Power": False,
        "Adjusted_Band_Ratio" : True, 
        "OriginalPSD_Band_Ratio": True,
        "Hemispheric_Asymmetry_index": True
    }

    fooof_res_save_path: Optional[str] = None
    random_state: int = 42


    @field_validator("muscle_activity_thr")
    def muscle_activity_thr_fv(cls, v):
        if v < 3:
            warnings.warn("Select a higher threshold for muscle activity artifacts. Low values " \
            "remove clean data.")
        return v


    @ field_validator("muscle_activity_filter_freq")
    def muscle_activity_filter_freq_fv(cls, v):
        if v[0] < 100:
            warnings.warn(f"Muscle activity artifact affects higher frequencies than {v[0]} Hz.")
        return v


    @model_validator(mode="after")
    def SL_conductivity_mv(self):
        if len(self.SL_conductivity) == 1 and self.which_sensor == "eeg":
            raise ValueError("In the case of EEG, you must have a three layers conductivity model due to volume conduction.")
        return self
    

    @model_validator(mode="after")
    def source_space_res(self):
        if int(self.source_space_spacing[-1]) != self.source_space_spacing_number:
            raise ValueError("The source_space_spacing and source_space_spacing_number should match")
        return self


    @model_validator(mode="after")
    def beamformer_arg_check(self):
        if self.beamformer_pick_ori == "vector" and self.beamformer_weight_norm != "unit-noise-gain-invariant":

            error_msg = "If you wish to compute a vector beamformer, it is necessary to use" \
                        " unit-noise-gain-invariant for weight_norm argument. This is for addressing" \
                        " the center of head bias where deeper sources can have larger scale than" \
                        " superfacial sources."
            
            raise ValueError(error_msg)
        return self

    @model_validator(mode="after")
    def center_head_bias_scale_check(self):
        if not self.beamforme_depth and self.SL_source_space == "volumetric":
            error_msg = "If you want to use volumetric source space (interested in deeper sources)," \
            " please define beamforme_depth as positive float number, i.e., 0.8. This is used to address" \
            " the center of head bias."
            raise ValueError(error_msg)
        return self
    
    @model_validator(mode="after")
    def ica_e_noise_removal(self):
        if (self.environmental_noise_ica_with_ref_meg_measure == "correlation" and not
            0<self.environmental_noise_ica_with_ref_meg_thr<1):
            
            error_mg = "If the threshold method for removing environmental noise using " \
            "ICA and ref-MEG is correlation, the corresponding measure must be between 0 and 1."
            raise ValueError(error_mg)
        return self    


    @model_validator(mode="after")
    def pacellation_checker(self):
        if not self.parcellation_parc and not self.parcellation_annot_fname:
            raise ValueError("Parcellation should be passed. Otherwise pass a custom parcellation file (.annot)")
        return self

    @model_validator(mode="after")
    def gedai_params_check(self):
        method = self.gedai_method
        wavelet_level = self.gedai_wavelet_level
        duration = self.gedai_duration
        broadband_multiplier = self.gedai_preliminary_broadband_noise_multiplier

        if method == "broadband" and wavelet_level != 0:
            raise ValueError("broadband method requires wavelet_level=0")
        if method == "broadband" and not duration:
            raise ValueError("broadband method requires gedai_duration")
        if method == "spectral" and wavelet_level == 0:
            raise ValueError("spectral method requires wavelet_level > 0")
        if method == "both" and not broadband_multiplier:
            raise ValueError("both method requires gedai_preliminary_broadband_noise_multiplier")
        
        return self

    def save(self, save_path:str, overwrite=False):
        "save the configurations to a JSON file"

        if os.path.exists(save_path) and overwrite == False:
            err_msg = f"A configuration file already exists in this directory: {save_path}. Set the overwrite to True."
            raise FileExistsError(err_msg)
            
        with open(save_path, "w") as file:
            json.dump(self.model_dump(), file, indent=4)


    @classmethod
    def load(cls, path: str):
        # Load configuration from a JSON file
        with open(path, "r") as file:
            cfg = json.load(file)
        return cls(**cfg)
    


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

    def join_with_star(lst):
        if not lst:
            return None
        if len(lst) == 1:
            return lst[0] + "*"
        return "*".join(lst)
    
    subjects = {}

    for dataset_name, dataset_info in datasets.items():
        base_dir = dataset_info["base_dir"]
        task = dataset_info["task"]
        ending = dataset_info["ending"]
        line_freq = dataset_info["line_freq"]
        empty_room_task = dataset_info["empty_room_task"]
        empty_room_path = dataset_info["empty_room_path"]
        surfaces = dataset_info["surfaces_dir"]
        

        dirs = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]
        
        for subj in dirs:
        
            rs_record_paths = glob.glob(
                f"{base_dir}/{subj}/**/*{task}*{ending}",
                recursive=True
                )
            
            if empty_room_task:
                er_record_paths = glob.glob(
                    f"{empty_room_path}/{subj}/**/*{empty_room_task}*{ending}",
                    recursive=True
                    )
            else: 
                er_record_paths = None
            
            if surfaces:
                if os.path.isdir(os.path.join(surfaces, subj)):
                    surface = surfaces
                else:
                    surface = None
            else: 
                surface = None

            subjects.update(
                {subj: 
                    {
                    "rest_record": join_with_star(rs_record_paths),
                    "line_freq": line_freq,
                    "empty_room_record":join_with_star(er_record_paths),
                    "mri_surface":surface
                    }
                }
            )
            
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


def set_path(project_dir):
    """
    Create and initialize directory structure for a given project.

    This function generates a set of predefined directories for
    feature extraction and normative modeling workflows within the
    specified project directory. If any of these directories do not
    exist, they will be created. The function returns the path to the
    features log directory.

    Parameters
    ----------
    project_dir : str
        Path to the root project directory where the folder structure
        will be created.

    Returns
    -------
    str
        Absolute path to the 'log' directory inside the 'Features'
        folder.

    Notes
    -----
    The function creates the following directory structure:

    - ``Features/``  
      - ``log/`` (for saving logs of feature extraction)
      - ``temp/`` (for temporarily storing extracted features)
      - ``figures/`` (for saving generated figures)

    - ``Normative modeling/``  
      - ``Runs/`` (for saving model run outputs)
      - ``Figures/`` (for visual outputs related to modeling)
      - ``Models summary/`` (for summaries of model results)
    """
    def make_folder(path):
        if not os.path.isdir(path):
            os.makedirs(path)

    # Feature extraction
    features_dir = os.path.join(project_dir, 'Features')
    features_log_path = os.path.join(features_dir, 'log_slurm_jobs')
    features_temp_path = os.path.join(features_dir,'temp')
    figures_dir = os.path.join(features_dir, "figures")
    exluded_participants_path = os.path.join(features_dir, "excluded_participants")
    saved_outputs_path = os.path.join(features_dir, "Saved_outputs")
    save_epochs_path = os.path.join(saved_outputs_path, "Epochs")
    save_psds_path = os.path.join(saved_outputs_path, "PSDs")
    configurations = os.path.join(features_dir, "Configurations")

    make_folder(features_dir)
    make_folder(features_log_path)
    make_folder(features_temp_path)
    make_folder(figures_dir)
    make_folder(saved_outputs_path)
    make_folder(save_epochs_path)
    make_folder(save_psds_path)
    make_folder(configurations)
    make_folder(exluded_participants_path)

    # Normative models
    nm_dir = os.path.join(project_dir, "Normative_models")
    make_folder(nm_dir)

    return features_dir, features_log_path



def clean_nan_columns(df, nan_threshold):
    """
    Remove columns with more NaNs than `nan_threshold`,
    otherwise impute NaNs with the column median.

    Parameters:
        df (pd.DataFrame): Input dataframe
        nan_threshold (int): Max allowed NaNs per column

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df = df.copy()

    for col in df.columns:
        nan_count = df[col].isna().sum()

        if nan_count > nan_threshold:
            # Drop column
            df.drop(columns=col, inplace=True)
        else:
            # Impute with median (only for numeric columns)
            if pd.api.types.is_numeric_dtype(df[col]):
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)

    return df



def find_other_mri_session(base_mri_path, missing_mri_subjects, str_mri_ending, which_session):

    new_paths = {}
    for subject in missing_mri_subjects:
        mri_paths = glob.glob(f"{base_mri_path}/{subject}/**/*{str_mri_ending}", recursive=True)

        if len(mri_paths) > which_session-1:
            new_paths.update({subject: mri_paths[which_session-1]})
        
    return new_paths

def find_failed_meg_subjects(log_path):
    
    missing_meg_subjects = []
    paths = os.scandir(log_path)
    paths = list(filter(lambda x: "err" in x.name, paths))
    for path in paths:
        with open(path, "r") as f:
            content = f.read()
            if "error" in content:
                subject = os.path.basename(path).split(".")[0].split("_")[0]
                missing_meg_subjects.append(subject)

    return set(missing_meg_subjects)

def find_other_meg_session(base_meg_path,
                           missing_meg_subjects,
                           str_meg_ending,
                           task_name,
                           which_session):
    
    new_paths = {}
    for subject in missing_meg_subjects:
        rs_record_paths = glob.glob(
                    f"{base_meg_path}/{subject}/**/*{task_name}*{str_meg_ending}",
                    recursive=True
                    )
        if len(rs_record_paths) > which_session - 1:
            new_paths.update({subject: rs_record_paths[which_session - 1]})
    return new_paths