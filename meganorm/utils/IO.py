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
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    confloat,
    conint,
    conlist,
    field_validator,
    NegativeInt,
    model_validator,
)


class BandRatio(BaseModel):
    numerator: Literal["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    denominator: Literal["Delta", "Theta", "Alpha", "Beta", "Gamma"]


class Config(BaseModel):
    """
    Configuration for preprocessing, artifact correction, source localization,
    and spectral feature extraction of neurophysiological signals (MEG/EEG/OPM).

    Parameters
    ----------
    which_meg_session : int, default=0
        Index of the MEG session to process.
    which_layout : {"all", "lobe", None}, default="all"
        Sensor layout grouping used for reporting/analysis.
    which_sensor : {"mag", "grad", "meg", "eeg", "opm"}, default="meg"
        Sensor type to process.
    drop_noisy_flat_channel : bool, default=True
        Drop channels flagged as noisy or flat before further processing.

    Filtering & Resampling
    -----------------------
    cutoffFreqLow, cutoffFreqHigh : float, PositiveInt, default=1.0, 80
        Bandpass filter cutoff frequencies (Hz).
    resampling_rate : PositiveInt, default=1000
        Target sampling rate (Hz).
    digital_filter, notch_filter : bool, default=True
        Apply bandpass / line-noise notch filtering.
    apply_oversampled_temporal_projection : bool, default=True
        Apply oversampled temporal projection (OTP) denoising.
    apply_Head_movement_correction : bool, default=True
        Correct for head movement during recording.
    Head_movement_limit_from_mean : float, default=0.0015
        Maximum allowed deviation from mean head position (m).
    apply_chpi_filter : bool, default=False
        Filter out cHPI coil signals.

    ICA
    ---
    apply_ica : bool, default=True
        Apply ICA-based artifact correction.
    apply_ica_elbow_detection : bool, default=False
        Automatically select the number of ICA components via elbow detection.
    ica_n_component : PositiveInt or None, default=None
        Number of ICA components to compute.
    ica_max_iter : PositiveInt, default=800
        Maximum ICA iterations.
    ica_method : {"fastica", "infomax", "picard"}, default="fastica"
        ICA algorithm.
    ica_if_reject_by_annotation : bool, default=True
        Exclude annotated bad segments when fitting ICA.
    auto_ica_corr_thr : float, default=0.5
        Correlation threshold (0-1) for automatic ICA component rejection.

    GEDAI Artifact Removal
    -----------------------
    apply_gedai : bool, default=True
        Apply GEDAI-based artifact removal.
    gedai_method : {"both", "spectral", "broadband"}, default="both"
        GEDAI denoising strategy.
    sensai_method : {"optimize", "gridsearch"}, default="optimize"
        Parameter search strategy for SensAI.
    gedai_duration, gedai_overlap : float or int, default=12, 0.5
        Window duration (s) and overlap fraction for GEDAI.
    gedai_preliminary_broadband_noise_multiplier : float, default=6.0
        Noise multiplier for preliminary broadband detection.
    gedai_noise_multiplier : float, default=3.0
        Noise multiplier used in GEDAI thresholding.
    gedai_wavelet_type : str, default="haar"
        Wavelet family used for spectral GEDAI.
    gedai_wavelet_level : "auto", PositiveInt, or 0, default="auto"
        Wavelet decomposition level.
    gedai_wavelet_low_cutoff : float or None, default=None
        Low-frequency cutoff for wavelet-based denoising.
    gedai_epoch_size_in_cycles : PositiveInt, default=12
        Epoch size expressed in number of cycles.
    gedai_highpass_cutoff : float, default=0.1
        High-pass cutoff applied before GEDAI (Hz).

    Muscle Artifact Detection
    ---------------------------
    muscle_activity_thr : int, default=4
        Detection threshold for muscle artifacts.
    muscle_activity_min_length_good : float, default=0.1
        Minimum length of a clean segment retained after removal (s).
    muscle_activity_filter_freq : tuple[int, int], default=(110, 140)
        Frequency band used for muscle artifact detection (Hz).

    Environmental Noise Correction
    --------------------------------
    apply_environmental_noise_correction : bool, default=True
        Apply environmental/reference-based noise correction.
    ctf_gradient_comp_level : PositiveInt, default=3
        CTF gradient compensation level.
    apply_environmental_noise_ssp_with_eroom : bool, default=False
        Use empty-room SSP projectors for noise correction.
    apply_environmental_noise_ica_with_ref_meg : bool, default=True
        Use reference-MEG-guided ICA for environmental noise removal.
    environmental_noise_ica_with_ref_meg_thr : float, default=2.5
        Threshold for ref-MEG-guided ICA component rejection.
    environmental_noise_ica_with_ref_meg_method : {"together", "separate"}, default="separate"
        Whether to process reference channels jointly or separately.
    environmental_noise_ica_with_ref_meg_measure : {"zscore", "correlation"}, default="zscore"
        Metric used to score ICA components against reference channels.

    EEG Reference & Bad-Segment Rejection
    ----------------------------------------
    rereference_method : {"average", "REST", None}, default="average"
        EEG re-referencing scheme.
    bad_segment_removal_method : {"autoreject", "fixed_thr", None}, default="autoreject"
        Method for rejecting bad data segments.
    mag_var_threshold, grad_var_threshold, eeg_var_threshold : float
        Variance-based rejection thresholds per channel type.
    mag_flat_threshold, grad_flat_threshold, eeg_flat_threshold : float
        Flatline-detection thresholds per channel type.
    zscore_std_thresh : PositiveInt, default=15
        Z-score threshold for outlier rejection.
    autoreject_n_interpolates : list[int], default=[1, 4, 8, 16, 32]
        Candidate interpolation counts for Autoreject.
    autoreject_consensus_percs : list[float]
        Candidate consensus percentages for Autoreject (11 values, 0-1).
    autoreject_cv : int or "auto", default="auto"
        Cross-validation folds for Autoreject.
    autoreject_thresh_method : {"bayesian_optimization", "random_search"}, default="bayesian_optimization"
        Threshold search strategy for Autoreject.

    Segmentation
    ------------
    segments_tmin, segments_tmax : PositiveInt, NegativeInt, default=20, -20
        Segment start/end times relative to event (s).
    segments_length, segments_overlap : int, default=10, 2
        Segment length and overlap (s).

    Source Localization
    --------------------
    apply_source_localization : bool, default=False
        Whether to perform source localization.
    apply_empty_room_recording : bool, default=True
        Use empty-room recordings for noise covariance estimation.
    apply_mri_QC : bool, default=False
        Run quality control on MRI/FreeSurfer output.
    apply_mri_template : bool, default=False
        Use a template MRI instead of subject-specific anatomy.
    freesurfer_template_path, freesurfer_home, freesurfer_license : str or None
        Paths to FreeSurfer template derivatives, installation, and license.
    force_new_watershed_bem : bool, default=False
        Recompute the watershed BEM surfaces.
    gcaatlas : bool, default=True
        Use the GCA atlas for subcortical segmentation.
    SL_source_space : {"surface", "volumetric"}, default="volumetric"
        Type of source space.
    SL_conductivity : tuple[float, ...], default=(0.3,)
        Head-model layer conductivities (three values required for EEG).
    SL_inverse_operator : {"lcmv"}, default="lcmv"
        Inverse operator method.
    source_space_spacing : {"ico3"..."ico6", "oct5", "oct6"}, default="ico4"
        Source space resolution.
    source_space_spacing_number : {3, 4, 5, 6}, default=4
        Numeric resolution; must match `source_space_spacing`.
    coregisteration_final_n_iterations : int, default=20
        Iterations for the final coregistration refinement.
    coregisteration_final_nasion_weight : float, default=10.0
        Weight applied to the nasion fiducial during coregistration.
    covariance_method : str, default="empirical"
        Method for noise covariance estimation.

    Beamformer & Parcellation
    ----------------------------
    beamformer_pick_ori : {None, "normal", "max-power", "vector"}, default="max-power"
        Source orientation constraint.
    beamformer_weight_norm : {None, "unit-noise-gain", "nai", "unit-noise-gain-invariant"}, default="unit-noise-gain"
        Beamformer weight normalization.
    beamforme_depth : float, default=0.08
        Depth-weighting factor correcting for center-of-head bias.
    inverse_regularization_value : float, default=0.05
        Regularization applied to the data covariance matrix.
    apply_morphing : bool, default=False
        Morph source estimates to a common template brain.
    parcellation_parc : {None, "aparc.a2009s", "parac"}, default="aparc.a2009s"
        Predefined cortical parcellation.
    parcellation_annot_fname : Path or None
        Custom parcellation (.annot) file, used if `parcellation_parc` is None.

    PSD & Spectral Parametrization
    ---------------------------------
    psd_method : {"multitaper", "welch"}, default="welch"
        PSD estimation method.
    psd_n_overlap, psd_n_fft, psd_n_per_seg : PositiveInt, default=1, 2, 2
        Welch/multitaper PSD parameters.
    parametrization_method : {"fooof", "irasa"}, default="irasa"
        Method for separating aperiodic and periodic spectral components.
    irasa_hset : tuple[float, float, float], default=(1.05, 2.0, 0.05)
        Resampling factor range/step for IRASA.
    fooof_freq_range_low, fooof_freq_range_high : PositiveInt, default=3, 40
        Frequency range for FOOOF fitting (Hz).
    aperiodic_mode : {"knee", "fixed"}, default="knee"
        Aperiodic component model.
    fooof_peak_width_limits : list[float], default=[1.0, 12.0]
        Allowed peak width range (Hz).
    fooof_min_peak_height : int, default=0
        Minimum peak height for detection.
    fooof_peak_threshold : PositiveInt, default=2
        Peak detection threshold (in SD of the flattened spectrum).
    fooof_res_save_path : str or None
        Path to save FOOOF results.
    save_source_localized_epochs, save_psds : bool, default=False
        Persist intermediate source-localized epochs / PSDs to disk.

    Feature Extraction
    -------------------
    freq_bands : dict[str, tuple[int, int]]
        Canonical frequency band definitions (Theta, Alpha, Beta, Gamma).
    individualized_band_ranges : dict[str, tuple[int, int]]
        Per-band offsets (Hz) used to individualize canonical bands.
    power_band_ratios_list : list[BandRatio]
        Band-power ratios to compute (e.g. Theta/Beta).
    min_r_squared : float, default=0.9
        Minimum R² required to accept a spectral model fit.
    feature_categories : dict[str, bool]
        Flags selecting which feature families to extract (offset, exponent,
        peak parameters, canonical/individualized band power, band ratios,
        hemispheric asymmetry).

    Miscellaneous
    -------------
    random_state : int, default=42
        Random seed for reproducibility.

    Methods
    -------
    save(save_path, overwrite=False)
        Serialize the configuration to a JSON file.
    load(path)
        Load a configuration from a JSON file.

    Notes
    -----
    Model validators enforce cross-field consistency, e.g.: a three-layer
    `SL_conductivity` is required for EEG source localization; `beamformer_pick_ori
    == "vector"` requires `beamformer_weight_norm == "unit-noise-gain-invariant"`;
    `source_space_spacing` must match `source_space_spacing_number`; GEDAI
    parameters must be consistent with the chosen `gedai_method`; and MRI
    template use is mutually exclusive with MRI QC.
    """

    model_config = {"extra": "forbid"}

    which_meg_session: int = 0  # the first session

    which_layout: Literal["all", "lobe", None] = "all"
    which_sensor: Literal["mag", "grad", "meg", "eeg", "opm"] = "meg"

    drop_noisy_flat_channel: bool = True

    remove_nonfinite_segment_threshold: PositiveInt = 5

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
    gedai_wavelet_type: str = "haar"
    gedai_wavelet_level: Union[Literal["auto"], PositiveInt, Literal[0]] = "auto"
    gedai_wavelet_low_cutoff: Union[None, float] = None
    gedai_epoch_size_in_cycles: PositiveInt = 12
    gedai_highpass_cutoff: float = 0.1

    muscle_activity_thr: int = 4
    muscle_activity_min_length_good: float = 0.1
    muscle_activity_filter_freq: Tuple[int, int] = (110, 140)

    apply_environmental_noise_correction: bool = True
    same_environmental_noise_removal: bool = False
    ctf_gradient_comp_level: PositiveInt = 3
    apply_environmental_noise_ssp_with_eroom: bool = False
    apply_environmental_noise_ica_with_ref_meg: bool = True
    environmental_noise_ica_with_ref_meg_thr: float = 2.5
    ica_if_reject_by_annotation: bool = True
    environmental_noise_ica_with_ref_meg_method: Literal["together", "separate"] = (
        "separate"
    )
    environmental_noise_ica_with_ref_meg_measure: Literal["zscore", "correlation"] = (
        "zscore"
    )

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
    autoreject_thresh_method: Literal["bayesian_optimization", "random_search"] = (
        "bayesian_optimization"
    )

    # Source localization
    apply_source_localization: bool = False
    apply_empty_room_recording: bool = True
    apply_mri_QC: bool = False
    take_screenshot_of_coregisteration: bool = True
    apply_mri_template: bool = False
    freesurfer_template_path: Optional[str] = None
    freesurfer_home: Optional[str] = None
    freesurfer_license: Optional[str] = None
    coregisteration_scale_mode: Literal["uniform", "3-axis", None] = None
    force_new_watershed_bem: bool = False
    gcaatlas: bool = True
    SL_source_space: Literal["surface", "volumetric"] = "volumetric"
    SL_conductivity: Tuple[float, ...] = (0.3,)
    SL_inverse_operator: Literal["lcmv"] = "lcmv"

    # the spacing to use for source space specificatin
    source_space_spacing: Literal["ico3", "ico4", "ico5", "ico6", "oct5", "oct6"] = (
        "ico4"
    )
    source_space_spacing_number: Literal[3, 4, 5, 6] = 4

    save_transformation_FIF_file: bool = False
    coregisteration_final_n_iterations: int = 20
    coregisteration_final_nasion_weight: float = 10.0
    covariance_method: str = "empirical"

    # Determines whether to keep vectors for all source orientations
    # or to select a single fixed orientation, depending on the chosen algorithm.
    beamformer_pick_ori: Literal[None, "normal", "max-power", "vector"] = "max-power"
    beamformer_weight_norm: Literal[
        None, "unit-noise-gain", "nai", "unit-noise-gain-invariant"
    ] = "unit-noise-gain"

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
    save_psds: bool = False

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
        "Adjusted_Band_Ratio": True,
        "OriginalPSD_Band_Ratio": True,
        "Hemispheric_Asymmetry_index": True,
    }

    fooof_res_save_path: Optional[str] = None
    random_state: int = 42

    @field_validator("muscle_activity_thr")
    def muscle_activity_thr_fv(cls, v):
        if v < 3:
            warnings.warn(
                "Select a higher threshold for muscle activity artifacts. Low values "
                "remove clean data."
            )
        return v

    @field_validator("muscle_activity_filter_freq")
    def muscle_activity_filter_freq_fv(cls, v):
        if v[0] < 100:
            warnings.warn(
                f"Muscle activity artifact affects higher frequencies than {v[0]} Hz."
            )
        return v

    @model_validator(mode="after")
    def SL_conductivity_mv(self):
        if (
            len(self.SL_conductivity) == 1
            and self.which_sensor == "eeg"
            and self.apply_source_localization
        ):
            raise ValueError(
                "In the case of EEG, you must have a three layers conductivity model due to volume conduction."
            )
        return self

    @model_validator(mode="after")
    def source_space_res(self):
        if int(self.source_space_spacing[-1]) != self.source_space_spacing_number:
            raise ValueError(
                "The source_space_spacing and source_space_spacing_number should match"
            )
        return self

    @model_validator(mode="after")
    def beamformer_arg_check(self):
        if (
            self.beamformer_pick_ori == "vector"
            and self.beamformer_weight_norm != "unit-noise-gain-invariant"
        ):

            error_msg = (
                "If you wish to compute a vector beamformer, it is necessary to use"
                " unit-noise-gain-invariant for weight_norm argument. This is for addressing"
                " the center of head bias where deeper sources can have larger scale than"
                " superfacial sources."
            )

            raise ValueError(error_msg)
        return self

    @model_validator(mode="after")
    def center_head_bias_scale_check(self):
        if not self.beamforme_depth and self.SL_source_space == "volumetric":
            error_msg = (
                "If you want to use volumetric source space (interested in deeper sources),"
                " please define beamforme_depth as positive float number, i.e., 0.8. This is used to address"
                " the center of head bias."
            )
            raise ValueError(error_msg)
        return self

    @model_validator(mode="after")
    def ica_e_noise_removal(self):
        if (
            self.environmental_noise_ica_with_ref_meg_measure == "correlation"
            and not 0 < self.environmental_noise_ica_with_ref_meg_thr < 1
        ):

            error_mg = (
                "If the threshold method for removing environmental noise using "
                "ICA and ref-MEG is correlation, the corresponding measure must be between 0 and 1."
            )
            raise ValueError(error_mg)
        return self

    @model_validator(mode="after")
    def pacellation_checker(self):
        if not self.parcellation_parc and not self.parcellation_annot_fname:
            raise ValueError(
                "Parcellation should be passed. Otherwise pass a custom parcellation file (.annot)"
            )
        return self

    @model_validator(mode="after")
    def mri_template_check(self):
        if self.apply_mri_template and not self.freesurfer_template_path:
            err_msg = (
                "In order to apply template MRI for source localization, a path to already freesurfer-preprocessed"
                "derivatives is necessary"
            )
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def either_template_mri_or_mri_qc(self):
        if self.apply_mri_QC and self.apply_mri_template:
            err_msg = (
                "You can not apply MRI QC on already preprocessed freesurfer template"
            )
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def env_noise_removal_same(self):
        if self.same_environmental_noise_removal:
            if (
                not self.apply_environmental_noise_ssp_with_eroom
                or not self.apply_environmental_noise_ica_with_ref_meg
            ):
                err_msg = (
                    "If you intened to apply the same environmental noise removal must choose between using"
                    " ref_meg or empty room recording. You can not apply gradient compensation or maxwell filter across all scanners."
                )
            raise ValueError(err_msg)
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
            raise ValueError(
                "both method requires gedai_preliminary_broadband_noise_multiplier"
            )

        return self

    def save(self, save_path: str, overwrite=False):
        "save the configurations to a JSON file"

        if os.path.exists(save_path) and overwrite == False:
            err_msg = f"A configuration file already exists in this directory: {save_path}. Set the overwrite to True."
            raise FileExistsError(err_msg)

        with open(save_path, "w") as file:
            file.write(self.model_dump_json(indent=4))

    @classmethod
    def load(cls, path: str):
        # Load configuration from a JSON file
        with open(path, "r") as file:
            cfg = json.load(file)
        return cls(**cfg)


def load_recording(device, path, empty_room_recording_path, configs, logger):
    if device == "CTF":
        data = mne.io.read_raw_ctf(path, preload=True)
        if empty_room_recording_path and configs.apply_source_localization:
            empty_room_recording = mne.io.read_raw_ctf(
                empty_room_recording_path, preload=True
            )
            logger.info("Empty room recording was found")
        elif not empty_room_recording_path and configs.apply_source_localization:
            empty_room_recording = None
            logger.info("No empty room recording was found")
        else:
            empty_room_recording = None

    elif device == "BTI":
        temp_hs_file_path = os.path.join(path, "hs_file")
        hs_file = temp_hs_file_path if os.path.exists(temp_hs_file_path) else None
        data = mne.io.read_raw_bti(
            pdf_fname=os.path.join(path, "c,rfDC"),
            config_fname=os.path.join(path, "config"),
            head_shape_fname=hs_file,
            preload=True,
        )
        if empty_room_recording_path and configs.apply_source_localization:
            empty_room_recording = mne.io.read_raw_bti(
                pdf_fname=os.path.join(empty_room_recording_path, "c,rfDC"),
                config_fname=os.path.join(empty_room_recording_path, "config"),
                head_shape_fname=None,
                preload=True,
            )
            logger.info("Empty room recording was found")
        elif not empty_room_recording_path and configs.apply_source_localization:
            empty_room_recording = None
            logger.info("No empty room recording was found")
        else:
            empty_room_recording = None

    elif device == "ARTEMIS123":
        if configs.apply_source_localization:
            temp = str(Path(path).parent)
            pos_files = glob.glob(f"{temp}/*.pos")
            if not pos_files:
                err_msg = f"No .pos file found next to the Artemis recording in {temp}."
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)
            pos_file = pos_files[0]
            data = mne.io.read_raw_artemis123(
                path,
                preload=True,
                pos_fname=pos_file,
                add_head_trans=True,
            )
        else:
            data = mne.io.read_raw_artemis123(
                path,
                preload=True,
            )
        if empty_room_recording_path and configs.apply_source_localization:
            empty_room_recording = mne.io.read_raw_artemis123(
                empty_room_recording_path,
                preload=True,
            )
            logger.info("Empty room recording was found")
        elif not empty_room_recording_path and configs.apply_source_localization:
            empty_room_recording = None
            logger.info("No empty room recording was found")
        else:
            empty_room_recording = None

    else:
        data = mne.io.read_raw(path, preload=True)
        if empty_room_recording_path and configs.apply_source_localization:
            empty_room_recording = mne.io.read_raw(
                empty_room_recording_path, preload=True
            )
            logger.info("Empty room recording was found")
        elif not empty_room_recording_path and configs.apply_source_localization:
            empty_room_recording = None
            logger.info("No empty room recording was found")
        else:
            empty_room_recording = None

    return data, empty_room_recording


def separate_eyes_open_close_eeglab(
    input_base_path,
    output_base_path,
    annotation_description_open,
    annotation_description_close,
    trim_before=5,
    trim_after=5,
):
    """
    Split resting-state EEGLAB recordings into separate eyes-open and
    eyes-closed files based on annotations.

    Scans `input_base_path` for BIDS-style resting-state ``.set`` files,
    extracts and trims annotated eyes-open and eyes-closed segments,
    concatenates each condition's segments, and writes them out as new
    EEGLAB ``.set`` files under a subject-specific folder in
    `output_base_path`.

    Parameters
    ----------
    input_base_path : str
        Root directory containing subject subfolders with resting-state
        EEGLAB recordings, matched via the pattern
        ``*/eeg/*_task-rest_eeg.set``.
    output_base_path : str
        Root directory where the separated eyes-open and eyes-closed
        files will be saved, created if it does not already exist.
    annotation_description_open : str
        Annotation description label marking eyes-open segments.
    annotation_description_close : str
        Annotation description label marking eyes-closed segments.
    trim_before : float, optional
        Duration in seconds to trim from the start of each annotated
        segment. Default is 5.
    trim_after : float, optional
        Duration in seconds to trim from the end of each annotated
        segment. Default is 5.

    Returns
    -------
    None
    """
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

        device = dataset_info["device_type"]

        line_freq = dataset_info.get("line_freq", 50)

        empty_room_task = dataset_info.get("empty_room_task", None)
        empty_room_path = dataset_info.get("empty_room_path", None)
        empty_room_ending = dataset_info.get("empty_room_ending", None)

        surfaces = dataset_info.get("surfaces_dir", None)

        event_file_path = dataset_info.get("event_file_path", None)
        event_file_task = dataset_info.get("event_file_task", None)
        event_file_ending = dataset_info.get("event_file_ending", None)
        event_of_interest = dataset_info.get("event_of_interest", None)

        dirs = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        for subj in dirs:

            # resting state data
            rs_record_paths = glob.glob(
                f"{base_dir}/{subj}/**/*{task}*{ending}", recursive=True
            )

            # empty room record
            if empty_room_task:
                er_record_paths = glob.glob(
                    f"{empty_room_path}/{subj}/**/*{empty_room_task}*{empty_room_ending}",
                    recursive=True,
                )
            else:
                er_record_paths = None

            # freesurfer surfaces
            if surfaces:
                if os.path.isdir(os.path.join(surfaces, subj)):
                    surface = surfaces
                else:
                    surface = None
            else:
                surface = None

            # event file
            if event_file_task and event_file_path:
                event_record_paths = glob.glob(
                    f"{event_file_path}/{subj}/**/*{event_file_task}*{event_file_ending}",
                    recursive=True,
                )
            else:
                event_record_paths = None

            subjects.update(
                {
                    subj: {
                        "rest_record": join_with_star(rs_record_paths),
                        "device": device,
                        "line_freq": str(line_freq),
                        "empty_room_record": join_with_star(er_record_paths),
                        "mri_surface": surface,
                        "dataset_name": dataset_name,
                        "event_record": join_with_star(event_record_paths),
                        "event_of_interest": str(event_of_interest),
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

    save_dir = os.path.join(save_dir, "participants_bids.tsv")
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
    features_dir = os.path.join(project_dir, "Features")
    features_log_path = os.path.join(features_dir, "log_slurm_jobs")
    features_temp_path = os.path.join(features_dir, "temp")
    figures_dir = os.path.join(features_dir, "figures")
    exluded_participants_path = os.path.join(features_dir, "excluded_participants")
    saved_outputs_path = os.path.join(features_dir, "Saved_outputs")
    save_epochs_path = os.path.join(saved_outputs_path, "Epochs")
    save_psds_path = os.path.join(saved_outputs_path, "PSDs")
    save_coregistration_QC_path = os.path.join(saved_outputs_path, "coregistration_QC")
    save_transformation_path = os.path.join(saved_outputs_path, "transformation_FIF_file")
    configurations = os.path.join(features_dir, "Configurations")
    mri_templates = os.path.join(features_dir, "MRI_templates")

    make_folder(features_dir)
    make_folder(features_log_path)
    make_folder(features_temp_path)
    make_folder(figures_dir)
    make_folder(saved_outputs_path)
    make_folder(save_epochs_path)
    make_folder(save_psds_path)
    make_folder(save_coregistration_QC_path)
    make_folder(save_transformation_path)
    make_folder(configurations)
    make_folder(exluded_participants_path)
    make_folder(mri_templates)

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


def find_other_mri_session(
    base_mri_path, missing_mri_subjects, str_mri_ending, which_session
):
    """
    Find an alternative MRI file for each subject at a given session index.

    Searches recursively under each subject's directory for files matching
    the given filename ending, then selects the file at the requested
    session index. Intended for locating a fallback MRI session (e.g. a
    second scan) for subjects whose primary MRI is missing or unusable.

    Parameters
    ----------
    base_mri_path : str or Path
        Root directory containing per-subject MRI folders.
    missing_mri_subjects : iterable of str
        Subject IDs to search for.
    str_mri_ending : str
        Filename suffix/pattern used to match MRI files (e.g. "T1w.nii.gz").
    which_session : int
        1-based index of the session to select from each subject's matched
        file list.

    Returns
    -------
    dict[str, str]
        Mapping of subject ID to the matched MRI file path. Subjects with
        fewer than `which_session` matching files are omitted.
    """

    new_paths = {}
    for subject in missing_mri_subjects:
        mri_paths = glob.glob(
            f"{base_mri_path}/{subject}/**/*{str_mri_ending}", recursive=True
        )

        if len(mri_paths) > which_session - 1:
            new_paths.update({subject: mri_paths[which_session - 1]})

    return new_paths


def find_failed_meg_subjects(log_path):
    """
    Identify subjects whose MEG processing failed, based on log files.

    Scans the given directory for error log files (files with "err" in the
    name) and flags any subject whose log contains the string "error".

    Parameters
    ----------
    log_path : str or Path
        Directory containing per-subject log files.

    Returns
    -------
    set of str
        Unique subject IDs (parsed from the log filename, before the first
        underscore) whose logs indicate a processing error.
    """

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


def find_other_meg_session(
    base_meg_path, missing_meg_subjects, str_meg_ending, task_name, which_session
):
    """
    Find an alternative MEG recording for each subject at a given session index.

    Searches recursively under each subject's directory for files matching
    the given task name and filename ending, then selects the file at the
    requested session index. Intended for locating a fallback MEG session
    for subjects whose primary recording is missing or failed processing.

    Parameters
    ----------
    base_meg_path : str or Path
        Root directory containing per-subject MEG folders.
    missing_meg_subjects : iterable of str
        Subject IDs to search for.
    str_meg_ending : str
        Filename suffix/pattern used to match MEG files (e.g. "raw.fif").
    task_name : str
        Task identifier expected to appear in the filename (e.g. "rest").
    which_session : int
        1-based index of the session to select from each subject's matched
        file list.

    Returns
    -------
    dict[str, str]
        Mapping of subject ID to the matched MEG file path. Subjects with
        fewer than `which_session` matching files are omitted.
    """
    new_paths = {}
    for subject in missing_meg_subjects:
        rs_record_paths = glob.glob(
            f"{base_meg_path}/{subject}/**/*{task_name}*{str_meg_ending}",
            recursive=True,
        )
        if len(rs_record_paths) > which_session - 1:
            new_paths.update({subject: rs_record_paths[which_session - 1]})
    return new_paths
