import argparse
import json
import numpy as np
import os
import sys
import mne
import logging
import pandas as pd
from pathlib import Path
import glob
from datetime import datetime
from meganorm.src.source_localization import source_localization, numpy_to_mne_epoch
from meganorm.utils.IO import Config, load_recording
from meganorm.src.preprocess import (
    preprocess,
    segment_epoch,
    drop_noisy_meg_channels,
    prepare_eeg_data,
    auto_reject_segmentation,
)
from meganorm.src.psdParameterize import parameterize_psds
from meganorm.src.featureExtraction import feature_extract


def main_argparser(args=None):
    """
    Build and parse command-line arguments for the feature extraction script.

    Defines the positional and optional arguments needed to run a single
    subject through the preprocessing, source localization, and spectral
    feature extraction pipeline, and parses them into a namespace.

    Parameters
    ----------
    args : list of str or None, optional
        List of argument strings to parse. If None, arguments are read
        from `sys.argv` (default behavior for command-line invocation).

    Returns
    -------
    argparse.Namespace
        Namespace containing the parsed arguments:

        dir : str
            Path to the subject's raw data. May contain a glob pattern
            (using "*" as a separator) when multiple recordings/sessions
            are present; the session used is selected via
            `configs.which_meg_session`.
        save_dir : str
            Directory where extracted features (and, depending on config,
            logs, PSDs, and source-localized epochs) will be saved.
        subject : str
            Subject/participant identifier, used for file naming, logging,
            and locating subject-specific FreeSurfer data.
        configs : str
            Path to a JSON file (loadable via `Config.load`) specifying
            preprocessing, segmentation, source localization, PSD, and
            feature extraction parameters.
        line_freq : int or str, default=60
            Power line frequency (Hz) used for notch filtering if it cannot
            be auto-detected from the recording. Pass the string "None" to
            disable an explicit override.
        surfaces_dir : str or None, default=None
            Path to the FreeSurfer `subjects` directory, required when
            `apply_source_localization` is enabled in the config (unless
            an MRI template is used).
        empty_room_recording_path : str or None, default=None
            Path to the subject's empty-room recording, used for noise
            pre-whitening. May contain a glob pattern. Particularly
            relevant when source localization is applied to recordings
            with both magnetometers and gradiometers.
        event_record : str or None, default=None
            Path to the event file for this subject, resolved via glob if
            a pattern is provided. Used together with `event_of_interest`
            to extract epochs around specific events.
        event_of_interest : str or None, default=None
            Event ID to extract epochs around (e.g. "16"). Only used if
            `event_record` is also provided.
        device_type: Device type.
            For MEG, only BTI, CTF, ARTEMIS123 and MEGIN are supported.

    Notes
    -----
    Several optional arguments accept the literal string "None" as a way
    to explicitly disable a default from the command line; `main` converts
    these strings to Python `None` after parsing.
    """
    parser = argparse.ArgumentParser()

    # Positional Arguments
    parser.add_argument("dir", type=str, help="Address to your data")
    parser.add_argument("save_dir", type=str, help="Where to save extracted features")
    parser.add_argument("subject", type=str, help="Participant ID")
    parser.add_argument(
        "configs", type=str
    )  # TODO: make it optional for both sequential and parallel computing
    # Optional arguments
    parser.add_argument(
        "--line_freq",
        default=60,
        help="The line power frequency; This will be used for notch filter."
        " If None is passed, 60 Hz will be used.",
    )
    parser.add_argument(
        "--surfaces_dir",
        type=str,
        default=None,
        help="If you need to apply source localization, set this to the"
        "directory in which freesurfer results are saved",
    )
    parser.add_argument(
        "--empty_room_recording_path",
        type=str,
        default=None,
        help="This the path to subjetc's empty room recording. Although not"
        " it can be helpful for pre-whitennin the data. Note that empty room"
        " room recordings are necessary for applying source localization"
        " to recordings with both magnetometer and gradiometer.",
    )
    parser.add_argument(
        "--event_record",
        type=str,
        default=None,
        help="Path to the event file for this subject (glob-resolved).",
    )
    parser.add_argument(
        "--event_of_interest",
        type=str,
        default=None,
        help="Event ID of interest for epoch extraction (e.g., '16').",
    )
    parser.add_argument(
        "--device_type",
        type=str,
        default=None,
        help="MEG device type (e.g., MEGIN, CTF, BTI). Overrides inference "
        "from the file path when provided.",
    )

    return parser.parse_args(args)


def set_logger(args, pakcages_to_silent):
    """
    Configure a per-subject file logger and silence noisy dependencies.

    Removes any existing root logging handlers, then configures logging to
    write INFO-level (and above) messages to a subject-specific log file,
    while restricting specified third-party packages to WARNING level to
    reduce log verbosity from dependencies like MNE.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (as returned by `main_argparser`), of which this
        function uses:

        save_dir : str
            Base output directory. The log file is written to
            `<parent of save_dir>/Saved_outputs/log_summary/`.
        subject : str
            Subject identifier, used to name the log file as
            `subject_<subject>_report.log`.
    pakcages_to_silent : list of str
        Names of logging namespaces (typically third-party package names,
        e.g. "mne", "numexpr", "dipy") whose log level should be raised to
        WARNING so their INFO/DEBUG messages are suppressed.

    Returns
    -------
    logging.Logger
        A logger for the current module (`__name__`), writing to the
        configured subject-specific log file in write mode (overwriting
        any prior log for the same subject).

    Notes
    -----
    The log directory is created if it does not already exist. Because
    root handlers are cleared before reconfiguration, calling this
    function more than once per process will reset logging for the whole
    run, not just for this module.
    """
    save_dir = os.path.join(Path(args.save_dir).parent, "Saved_outputs", "log_summary")

    os.makedirs(save_dir, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # set the logger
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(save_dir, f"subject_{args.subject}_report.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(funcName)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    for package in pakcages_to_silent:
        logging.getLogger(package).setLevel(logging.WARNING)

    return logger


def main(args):
    """
    Run the full spectral feature extraction pipeline for one subject.

    Loads raw MEG/EEG/OPM data and processes it end-to-end: channel
    cleanup, filtering, ICA/GEDAI and environmental noise correction,
    head-movement correction, segmentation with optional Autoreject
    bad-segment removal, optional source localization, PSD computation
    with FOOOF/IRASA spectral parametrization, and band-power feature
    extraction. Extracted features are saved to a per-subject CSV.

    Parameters
    ----------
    args : list of str
        Raw command-line arguments (typically `sys.argv[1:]`), parsed
        via `main_argparser`. See that function for full argument details.

    Returns
    -------
    None

    Notes
    -----
    - Device is inferred from the file path/extension (BTI/4D, CTF, or
      MEGIN/fif); only one matched session is used even if the glob
      pattern resolves multiple recordings.
    - Line frequency comes from recording metadata, then `--line_freq`,
      defaulting to 60 Hz.
    """
    # parse the arguments
    args = main_argparser(args)
    logger = set_logger(args, ["mne", "numexpr", "dipy"])

    start_time = datetime.now()
    logger.info(f"Starting the process for the subject {args.subject} at {start_time}:")

    if args.line_freq == "None":
        args.line_freq = None
    if args.empty_room_recording_path == "None":
        args.empty_room_recording_path = None
    if args.surfaces_dir == "None":
        args.surfaces_dir = None
    if args.event_record == "None":
        args.event_record = None
    if args.event_of_interest == "None":
        args.event_of_interest = None
    if args.device_type == "None":
        args.device_type = None

    configs = Config.load(args.configs)

    if configs.apply_source_localization and not configs.apply_mri_template:
        freesurfer_data_path = os.path.join(args.surfaces_dir, args.subject)
        if not os.path.isdir(freesurfer_data_path):
            error_msg = f"The Freesurfer file corresponding to this subject is not found in {freesurfer_data_path}."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    paths = args.dir.split("*")
    paths = list(filter(lambda x: len(x), paths))
    path = paths[configs.which_meg_session]

    current = Path(args.save_dir)
    project_dir = current.parent

    if args.empty_room_recording_path:
        empty_room_recording_paths = args.empty_room_recording_path.split("*")
        empty_room_recording_paths = list(
            filter(lambda x: len(x), empty_room_recording_paths)
        )
        empty_room_recording_path = empty_room_recording_paths[0]
    else:
        empty_room_recording_path = None

    if args.event_record:
        event_record_paths = args.event_record.split("*")
        event_record_paths = list(filter(lambda x: len(x), event_record_paths))
        event_record = event_record_paths[0]
        event_of_interest = int(args.event_of_interest)
    else:
        event_record = None
        event_of_interest = None

    logger.warning(
        f"{len(paths)} recordings were detected for this subject. The first one"
        " will be used in this analysis."
    )

    if not configs.which_sensor == "eeg":
        if args.device_type:
            device = args.device_type
        elif "4D" in path:
            device = "BTI"
        elif path.split(".")[-1] == "ds":
            device = "CTF"
        elif path.split(".")[-1] == "fif":
            device = "MEGIN"
        elif path.split(".")[-1] == "bin":
            device = "ARTEMIS123"
        else:
            err_msg = "The provided MEG recording is not supported yet."
            logger.error(err_msg)
            raise ValueError(err_msg)
    else:
        device = path.split(".")[
            -1
        ]  # TODO: it was originaly path[0]. Check if this correction is correct.

    device = device.upper()
    # ------------------------------------------------------------ load data
    data, empty_room_recording = load_recording(
        device, path, empty_room_recording_path, configs, logger
    )

    # ------------------------------------------------------------
    power_line_freq = data.info.get("line_freq")
    if not power_line_freq:
        if args.line_freq is not None:
            power_line_freq = int(args.line_freq)
        else:
            logger.warning(
                "Power line frequency could not be detected; defaulting to 60 Hz."
            )
            power_line_freq = 60

    # ------------------------------------------------------------
    if configs.which_sensor == "eeg":
        data = prepare_eeg_data(data, path)

    which_sensor_dict = dict.fromkeys(["meg", "mag", "grad", "eeg", "opm"], False)
    which_sensor_dict[configs.which_sensor] = True

    # ------------------------------------------------------------
    if (
        configs.which_sensor in ["meg", "grad", "mag"]
        and configs.drop_noisy_flat_channel
    ):
        data, empty_room_recording = drop_noisy_meg_channels(
            data=data,
            subID=args.subject,
            args=args,
            device=device,
            which_sensor=which_sensor_dict,
            empty_room_recording=empty_room_recording,
        )

    # ------------------------------------------------------------
    (
        filtered_data,
        channel_names,
        sampling_rate,
        empty_room_recording,
        _,
        segment_events,
    ) = preprocess(
        data=data,
        device=device,
        subject=args.subject,
        freesurfer_dir=args.surfaces_dir,
        n_component=configs.ica_n_component,
        ica_max_iter=configs.ica_max_iter,
        IcaMethod=configs.ica_method,
        cutoffFreqLow=configs.cutoffFreqLow,
        cutoffFreqHigh=configs.cutoffFreqHigh,
        which_sensor=which_sensor_dict,
        resampling_rate=configs.resampling_rate,
        digital_filter=configs.digital_filter,
        rereference_method=configs.rereference_method,
        apply_ica=configs.apply_ica,
        auto_ica_corr_thr=configs.auto_ica_corr_thr,
        power_line_freq=power_line_freq,
        empty_room_recording=empty_room_recording,
        muscle_activity_min_length_good=configs.muscle_activity_min_length_good,
        muscle_activity_filter_freq=configs.muscle_activity_filter_freq,
        muscle_activity_thr=configs.muscle_activity_thr,
        apply_ica_elbow_detection=configs.apply_ica_elbow_detection,
        apply_oversampled_temporal_projection=configs.apply_oversampled_temporal_projection,
        apply_Head_movement_correction=configs.apply_Head_movement_correction,
        Head_movement_limit_from_mean=configs.Head_movement_limit_from_mean,
        apply_chpi_filter=configs.apply_chpi_filter,
        apply_environmental_noise_correction=configs.apply_environmental_noise_correction,
        ctf_gradient_comp_level=configs.ctf_gradient_comp_level,
        apply_environmental_noise_ssp_with_eroom=configs.apply_environmental_noise_ssp_with_eroom,
        apply_environmental_noise_ica_with_ref_meg=configs.apply_environmental_noise_ica_with_ref_meg,
        environmental_noise_ica_with_ref_meg_thr=configs.environmental_noise_ica_with_ref_meg_thr,
        ica_if_reject_by_annotation=configs.ica_if_reject_by_annotation,
        environmental_noise_ica_with_ref_meg_method=configs.environmental_noise_ica_with_ref_meg_method,
        environmental_noise_ica_with_ref_meg_measure=configs.environmental_noise_ica_with_ref_meg_measure,
        apply_gedai=configs.apply_gedai,
        gedai_method=configs.gedai_method,
        sensai_method=configs.sensai_method,
        conductivity=configs.SL_conductivity,
        source_space=configs.SL_source_space,
        gedai_duration=configs.gedai_duration,
        gedai_overlap=configs.gedai_overlap,
        gedai_preliminary_broadband_noise_multiplier=configs.gedai_preliminary_broadband_noise_multiplier,
        gedai_noise_multiplier=configs.gedai_noise_multiplier,
        gedai_wavelet_type=configs.gedai_wavelet_type,
        gedai_wavelet_level=configs.gedai_wavelet_level,
        gedai_wavelet_low_cutoff=configs.gedai_wavelet_low_cutoff,
        gedai_epoch_size_in_cycles=configs.gedai_epoch_size_in_cycles,
        gedai_highpass_cutoff=configs.gedai_highpass_cutoff,
        source_space_spacing=configs.source_space_spacing,
        source_space_spacing_number=configs.source_space_spacing_number,
        event_record=event_record,
        event_of_interest=event_of_interest,
        segments_length=configs.segments_length,
        overlap=configs.segments_overlap,
        same_environmental_noise_removal=configs.same_environmental_noise_removal,
    )

    # Remove UADC001 annotations - temp
    annot = filtered_data.annotations
    filtered_data.set_annotations(annot[annot.description != "UADC001"])

    # ------------------------------------------------------------
    if configs.bad_segment_removal_method in [None, "fixed_thr"]:
        segments = segment_epoch(
            data=filtered_data,
            which_sensor=which_sensor_dict,
            sampling_rate=sampling_rate,
            tmin=configs.segments_tmin,
            tmax=configs.segments_tmax,
            segments_length=configs.segments_length,
            overlap=configs.segments_overlap,
            ica_if_reject_by_annotation=configs.ica_if_reject_by_annotation,
            bad_segment_removal_method=configs.bad_segment_removal_method,
            mag_var_threshold=configs.mag_var_threshold,
            grad_var_threshold=configs.grad_var_threshold,
            eeg_var_threshold=configs.eeg_var_threshold,
            mag_flat_threshold=configs.mag_flat_threshold,
            grad_flat_threshold=configs.grad_flat_threshold,
            eeg_flat_threshold=configs.eeg_flat_threshold,
            segment_events=segment_events,
        )

    elif configs.bad_segment_removal_method == "autoreject":
        segments, _ = auto_reject_segmentation(
            raw=filtered_data,
            sampling_rate=sampling_rate,
            tmin=configs.segments_tmin,
            tmax=configs.segments_tmax,
            segments_length=configs.segments_length,
            overlap=configs.segments_overlap,
            ica_if_reject_by_annotation=configs.ica_if_reject_by_annotation,
            n_interpolates=configs.autoreject_n_interpolates,
            consensus_percs=configs.autoreject_consensus_percs,
            cv=configs.autoreject_cv,
            thresh_method=configs.autoreject_thresh_method,
            random_state=configs.random_state,
            segment_events=segment_events,
        )

    # ------------------------------------------------------------
    if configs.apply_source_localization:
        logger.info("Starting the source localization")
        stc, labels = source_localization(
            recording_path=path,
            project_dir=project_dir,
            subject=args.subject,
            subjects_dir=args.surfaces_dir,
            subject_to="fsaverage",
            data=filtered_data,
            segments=segments,
            empty_room_recording=empty_room_recording,
            source_space=configs.SL_source_space,
            conductivity=configs.SL_conductivity,
            inverse_operator=configs.SL_inverse_operator,
            figures_path=os.path.join(args.save_dir, "figures"),
            which_sensor_dict=which_sensor_dict,
            plot_3d=False,
            **configs.model_dump(),
        )
        segments = numpy_to_mne_epoch(stc, labels, "mag", sampling_rate)
        channel_names = segments.info["ch_names"]

    if configs.save_source_localized_epochs:
        save_epoch_path = os.path.join(
            Path(args.save_dir).parent, "Saved_outputs", "Epochs", args.subject
        )
        if not os.path.exists(save_epoch_path):
            os.mkdir(save_epoch_path)
        segments.save(f"{save_epoch_path}/{args.subject}-SL-epo.fif", overwrite=True)

    # ------------------------------------------------------------
    spectral_models, psds, freqs = parameterize_psds(
        segments=segments,
        sampling_rate=sampling_rate,
        # psd parameters
        psd_method=configs.psd_method,
        psd_n_overlap=configs.psd_n_overlap,
        psd_n_fft=configs.psd_n_fft,
        n_per_seg=configs.psd_n_per_seg,
        # parametrization method
        parametrization_method=configs.parametrization_method,
        aperiodic_mode=configs.aperiodic_mode,
        freq_range_low=configs.fooof_freq_range_low,
        freq_range_high=configs.fooof_freq_range_high,
        # fooof parameters
        min_peak_height=configs.fooof_min_peak_height,
        peak_threshold=configs.fooof_peak_threshold,
        peak_width_limits=configs.fooof_peak_width_limits,
        # pyrasa parameters
        irasa_hset=configs.irasa_hset,
    )

    if configs.save_psds:
        save_psds_path = os.path.join(
            Path(args.save_dir).parent, "Saved_outputs", "PSDs", args.subject
        )
        if not os.path.exists(save_psds_path):
            os.mkdir(save_psds_path)
        np.save(f"{save_psds_path}/{args.subject}-regional-psd.npy", psds)
        np.save(f"{save_psds_path}/{args.subject}-freqs.npy", freqs)

    # ------------------------------------------------------------
    features = feature_extract(
        subject_id=args.subject,
        spectral_models=spectral_models,
        psds=psds,
        freqs=freqs,
        freq_bands=configs.freq_bands,
        channel_names=channel_names,
        individualized_band_ranges=configs.individualized_band_ranges,
        feature_categories=configs.feature_categories,
        device=device,
        which_layout=configs.which_layout,
        which_sensor=which_sensor_dict,
        aperiodic_mode=configs.aperiodic_mode,
        min_r_squared=configs.min_r_squared,
        power_band_ratios_list=configs.power_band_ratios_list,
    )

    features.to_csv(os.path.join(args.save_dir, f"{args.subject}.csv"))

    logger.info(
        f"The feature extraction process for the subject {args.subject} is complete."
    )
    end_time = datetime.now()
    elapsed = end_time - start_time
    logger.info(f"Script ended at {end_time}")
    logger.info(f"Total elapsed time: {elapsed}")


if __name__ == "__main__":

    main(sys.argv[1:])
