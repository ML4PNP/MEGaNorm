import argparse
import json
import os
import sys
import mne
import logging
import pandas as pd
import glob
from datetime import datetime
from meganorm.src.source_localization import source_localization, numpy_to_mne_raw
from meganorm.utils.IO import make_config, storeFooofModels, Config
from meganorm.src.psdParameterize import psdParameterize
from meganorm.src.preprocess import (
    preprocess,
    segment_epoch,
    drop_noisy_meg_channels,
    prepare_eeg_data,
)
from meganorm.src.featureExtraction import feature_extract



def main_argparser(args=None):
    """
    Create and parse command-line arguments for the feature extraction script.

    Parameters
    ----------
    args : list of str or None, optional
        List of arguments to parse. If None, parses arguments from `sys.argv`.

    Returns
    -------
    argparse.Namespace
        Namespace containing the parsed command-line arguments:
        - `dir` : str
            Path to the input data directory.
        - `save_dir` : str
            Directory where extracted features will be saved.
        - `subject` : str
            Participant identifier.
        - `surfaces_dir` : str or None
            Path to the FreeSurfer surfaces directory (used for source localization). Default is None.
        - `empty_room_recording_path` : str or None
            Path to subject's empty room recording for pre-whitening. Default is None.
        - `configs` : str or None
            Path to optional configuration file. Default is None.

    Notes
    -----
    - The empty room recording is particularly useful for recordings with both magnetometer
      and gradiometer sensors when performing source localization.
    """
    parser = argparse.ArgumentParser()
    
    # Positional Arguments
    parser.add_argument("dir", type=str, help="Address to your data")
    parser.add_argument("save_dir", type=str, help="Where to save extracted features")
    parser.add_argument("subject", type=str, help="Participant ID")
    parser.add_argument("configs", type=str) # TODO: make it optional for both sequential and parallel computing
    
    parser.add_argument("--surfaces_dir", type=str, default=None,
                        help="If you need to apply source localization, set this to the"\
                        "directory in which freesurfer results are saved")
    parser.add_argument("--empty_room_recording_path", type=str, default=None,
                        help="This the path to subjetc's empty room recording. Although not" \
                        " it can be helpful for pre-whitennin the data. Note that empty room" \
                        " room recordings are necessary for applying source localization" \
                        " to recordings with both magnetometer and gradiometer.")
    

    return parser.parse_args(args)


def set_logger(args, pakcages_to_silent):
    """
    Set up a logger for the experiment or script, writing logs to a file 
    and silencing specified packages.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments namespace containing at least `save_dir` and `subject` attributes.
        - `args.save_dir` : str
            Base directory where log files will be saved.
        - `args.subject` : str
            Subject identifier, used to name the log file.
    pakcages_to_silent : list of str
        List of package names whose logging level should be set to WARNING 
        to reduce verbosity.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    - The logger writes to a file named `subject_<subject>_report.log` 
      in a `log_summary` folder inside `args.save_dir`.
    - Existing root handlers are removed before setting up the new logger.
    - Silenced packages will not log INFO or DEBUG messages.
    """
    save_dir = os.path.join(args.save_dir, "log_summary")
    os.makedirs(save_dir, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # set the logger
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(save_dir, f"subject_{args.subject}_report.log"),
        filemode="w",
        format='%(name)s - %(levelname)s - %(funcName)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    for package in pakcages_to_silent:
        logging.getLogger(package).setLevel(logging.WARNING)

    return logger


def main(args):
    """
    Main function for running a complete spectral feature extraction pipeline
    using serialized or parallelized workflows.

    This function processes raw MEG/EEG recordings through a pipeline that includes
    preprocessing, segmentation, PSD computation, spectral parameterization using FOOOF,
    and feature extraction. The resulting features are saved to a CSV file.

    Positional Arguments (from command line)
    ----------------------------------------
    dir : str
        Path to the raw MEG/EEG data file or directory.
    save_dir : str
        Directory where the extracted features will be saved.
    subject : str
        Subject or participant identifier used for file naming and tracking.

    Optional Arguments
    ------------------
    --configs : str, optional
        Path to a JSON configuration file specifying preprocessing, segmentation,
        PSD, and FOOOF parameters. If not provided, a default configuration is used.

    Workflow Overview
    -----------------
    1. Loads raw MEG/EEG data.
    2. Applies channel type mapping and sets EEG montage (if applicable).
    3. Removes bad channels using Maxwell filtering (for MEG).
    4. Applies preprocessing steps such as bandpass filtering and ICA.
    5. Segments the continuous data into epochs.
    6. Computes the Power Spectral Density (PSD) for each epoch and channel.
    7. Fits FOOOF models to each PSD to decompose into periodic and aperiodic components.
    8. Extracts spectral features across predefined frequency bands.
    9. Saves the extracted features as a CSV file to the specified output directory.

    Notes
    -----
    - Supports both EEG and MEG modalities.
    - Compatible with various MEG/EEG file formats supported by MNE.
    - Can be run in serial mode or in parallel environments (e.g., SLURM-based clusters).

    Raises
    ------
    FileNotFoundError
        If required montage or channel information is missing.
    ValueError
        If an unsupported sensor type or PSD method is defined in the configuration.
    RuntimeError
        If data loading fails due to unsupported or corrupted formats.
    """
    # parse the arguments
    args = main_argparser(args)
    logger = set_logger(args, ["mne", "numexpr", "dipy"])

    start_time = datetime.now()
    logger.info(f"Starting the process for the subject {args.subject} at {start_time}:")

    if args.empty_room_recording_path == "None":
        args.empty_room_recording_path = None
    if args.surfaces_dir == "None":
        args.surfaces_dir = None

    configs = Config.load(args.configs)

    if configs.apply_source_localization:
        freesurfer_data_path = os.path.join(args.surfaces_dir, args.subject)
        if not os.path.isdir(freesurfer_data_path):
            error_msg = f"The Freesurfer file corresponding to this subject is not found in {freesurfer_data_path}."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    paths = args.dir.split("*")
    paths = list(filter(lambda x: len(x), paths))
    path = paths[0]
    logger.warning(f"{len(paths)} recordings were detected for this subject. The first one" \
                    " will be used in this analysis.")
    
    # Extracting file format (device) for loading layout
    if not configs.which_sensor == "eeg":
        if "4D" in path: # TODO: it was originaly path[0]. Check if this correction is correct.
            device = "BTI" 
        elif path.split(".")[-1] == "ds":
            device = "CTF"
        elif path.split(".")[-1] == "fif":
            device = "MEGIN"
        else:
            err_msg = "The provided MEG recording is not supported yet."
            logger.error(err_msg)
            raise ValueError(err_msg)
    else:
        device = path.split(".")[-1] # TODO: it was originaly path[0]. Check if this correction is correct.
    
    # read the data
    # ------------------------------------------------------------
    if not device == "BTI":
        data = mne.io.read_raw(path, preload=True)
        if args.empty_room_recording_path:
            empty_room_recording = mne.io.read_raw(args.empty_room_recording_path, preload=True)
        else:
            empty_room_recording = None

    else:
        data = mne.io.read_raw_bti(
            pdf_fname=os.path.join(path, "c,rfDC"),
            config_fname=os.path.join(path, "config"),
            head_shape_fname=None,
            preload=True
        )
        if args.empty_room_recording_path:
            empty_room_recording = mne.io.read_raw_bti(
                pdf_fname=os.path.join(args.empty_room_recording_path, "c,rfDC"),
                config_fname=os.path.join(args.empty_room_recording_path, "config"),
                head_shape_fname=None,
                preload=True
            )
        else:
            empty_room_recording = None

    # extract power line freq
    # ------------------------------------------------------------
    power_line_freq = data.info.get("line_freq")
    if not power_line_freq:
        logger.warning("Power line frequency was not detected" \
        "; therefore, it was set to 60 Hz")
        power_line_freq = 60

    # set eeg info (channel types and electrode montage) when it is not there yet
    # ------------------------------------------------------------
    if configs.which_sensor == "eeg":
        data = prepare_eeg_data(data, path)

    # drop noisy channels for MEG
    # ------------------------------------------------------------
    if configs.which_sensor in ["meg", "grad", "mag"] and configs.drop_noisy_flat_channel:
        data, empty_room_recording = drop_noisy_meg_channels(data=data, 
                                            subID=args.subject, 
                                            args=args, 
                                            configs=configs, 
                                            empty_room_recording=empty_room_recording)
    
    which_sensor_dict = dict.fromkeys(["meg", "mag", "grad", "eeg", "opm"], False)
    which_sensor_dict[configs.which_sensor] = True

    # preproces
    # ------------------------------------------------------------
    filtered_data, channel_names, sampling_rate, empty_room_recording, number_of_reduced_ic = preprocess(
        data=data,
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
        device=device,
        apply_ica_elbow_detection=configs.apply_ica_elbow_detection,
        apply_oversampled_temporal_projection = configs.apply_oversampled_temporal_projection,
        apply_Head_movement_correction=configs.apply_Head_movement_correction,
        Head_movement_limit_from_mean = configs.Head_movement_limit_from_mean,
        apply_chpi_filter = configs.apply_chpi_filter,
        apply_environmental_noise_correction = configs.apply_environmental_noise_correction,
        ctf_gradient_comp_level = configs.ctf_gradient_comp_level,
        apply_environmental_noise_ssp_with_eroom = configs.apply_environmental_noise_ssp_with_eroom,
        apply_environmental_noise_ica_with_ref_meg = configs.apply_environmental_noise_ica_with_ref_meg,
        environmental_noise_ica_with_ref_meg_thr = configs.environmental_noise_ica_with_ref_meg_thr,
        ica_if_reject_by_annotation = configs.ica_if_reject_by_annotation,
        environmental_noise_ica_with_ref_meg_method = configs.environmental_noise_ica_with_ref_meg_method,
        environmental_noise_ica_with_ref_meg_measure = configs.environmental_noise_ica_with_ref_meg_measure
    )


    # segmentation 
    # ------------------------------------------------------------
    segments = segment_epoch(
        data = filtered_data,
        which_sensor=which_sensor_dict,
        sampling_rate = sampling_rate,
        tmin = configs.segments_tmin,
        tmax = configs.segments_tmax,
        segments_length = configs.segments_length,
        overlap = configs.segments_overlap,
        ica_if_reject_by_annotation = configs.ica_if_reject_by_annotation,
        remove_bad_segments = configs.remove_bad_segments,
        mag_var_threshold = configs.mag_var_threshold,
        grad_var_threshold = configs.grad_var_threshold,
        eeg_var_threshold = configs.eeg_var_threshold,
        mag_flat_threshold = configs.mag_flat_threshold,
        grad_flat_threshold = configs.grad_flat_threshold,
        eeg_flat_threshold = configs.eeg_flat_threshold,
    )

    # Source localization 
    # ------------------------------------------------------------
    if configs.apply_source_localization:
        logger.info("Starting the source localization")
        stc, labels = source_localization(
                subject=args.subject,
                subjects_dir=args.surfaces_dir, 
                subject_to="fsaverage",
                data=filtered_data,
                empty_room_recording=empty_room_recording,
                source_space=configs.SL_source_space,
                conductivity=configs.SL_conductivity,
                inverse_operator=configs.SL_inverse_operator,
                figures_path=os.path.join(args.save_dir, "figures"),
                number_of_reduced_ic=number_of_reduced_ic,
                which_sensor_dict=which_sensor_dict,
                plot_3d=False,
                **configs.model_dump()
            )
        filtered_data = numpy_to_mne_raw(stc, labels, "mag", sampling_rate)
        channel_names = filtered_data.info["ch_names"]


    # fooof analysis 
    # ------------------------------------------------------------
    fmGroup, psds, freqs = psdParameterize(
        segments=segments,
        sampling_rate=sampling_rate,
        # psd parameters
        psd_method=configs.psd_method,
        psd_n_overlap=configs.psd_n_overlap,
        psd_n_fft=configs.psd_n_fft,
        n_per_seg=configs.psd_n_per_seg,
        # fooof parameters
        freq_range_low=configs.fooof_freq_range_low,
        freq_range_high=configs.fooof_freq_range_high,
        min_peak_height=configs.fooof_min_peak_height,
        peak_threshold=configs.fooof_peak_threshold,
        peak_width_limits=configs.fooof_peak_width_limits,
        aperiodic_mode=configs.aperiodic_mode,
    )

    if configs.fooof_res_save_path:
        storeFooofModels(configs.fooof_res_save_path, args.subject, fmGroup, psds, freqs)

    # feature extraction 
    # ------------------------------------------------------------
    features = feature_extract(
        subject_id=args.subject,
        fmGroup=fmGroup,
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
    )

    features.to_csv(os.path.join(args.save_dir, f"{args.subject}.csv"))

    logger.info(f"The feature extraction process for the subject {args.subject} is complete.")
    end_time = datetime.now()
    elapsed = start_time - end_time
    logger.info(f"Script ended at {end_time}")
    logger.info(f"Total elapsed time: {elapsed}")

if __name__ == "__main__":

    main(sys.argv[1:])
