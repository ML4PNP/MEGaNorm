import argparse
import json
import os
import sys
import mne
import logging
import pandas as pd
import glob
from meganorm.src.source_localization import source_localization, numpy_to_mne_raw
from meganorm.utils.IO import make_config, storeFooofModels
from meganorm.src.psdParameterize import psdParameterize
from meganorm.src.preprocess import (
    preprocess,
    segment_epoch,
    drop_noisy_meg_channels,
    prepare_eeg_data,
)
from meganorm.src.featureExtraction import feature_extract


def main_argparser(args=None):
    parser = argparse.ArgumentParser()
    
    # Positional Arguments
    parser.add_argument("dir", type=str, help="Address to your data")
    parser.add_argument("save_dir", type=str, help="Where to save extracted features")
    parser.add_argument("subject", type=str, help="Participant ID")
    

    parser.add_argument("--surfaces_dir", type=str, default=None,
                        help="If you need to apply source localization, set this to the"\
                        "directory in which freesurfer results are saved")
    parser.add_argument("--empty_room_recording_path", type=str, default=None,
                        help="This the path to subjetc's empty room recording. Although not" \
                        " it can be helpful for pre-whitennin the data. Note that empty room" \
                        " room recordings are necessary for applying source localization" \
                        " to recordings with both magnetometer and gradiometer.")
    parser.add_argument("--configs", type=str, default=None)

    return parser.parse_args(args)


def set_logger(args, pakcages_to_silent):

    save_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_dir, exist_ok=True)

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

    # Loading configs
    # *******************************************************
    if args.configs is not None:
        with open(args.configs, "r") as f:
            configs = json.load(f)
    else:
        configs = make_config("configs")

    # subject ID
    subID = args.subject
    paths = args.dir.split("*")
    paths = list(filter(lambda x: len(x), paths))
    if len(paths) > 0:
        logger.warning(f"{len(paths)} recordings were detected for this subject. The first one" \
                       " will be used in this analysis.")
    path = paths[0]

    # Extracting file format (extention) for loading layout
    extention = path[0].split(".")[-1]
    if "4D" in path[0]:
        extention = "BTI"  # TODO: you need to change this

    logger.info(f"Starting the process for the subject {subID}:")
    # read the data
    # *******************************************************
    try:
        data = mne.io.read_raw(path, preload=True)
        if args.empty_room_recording_path:
            empty_room_recording = mne.io.read_raw(args.empty_room_recording_path, preload=True)
        else:
            empty_room_recording = None

    except:
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
    # *******************************************************
    power_line_freq = data.info.get("line_freq")
    if not power_line_freq:
        logger.warning("Power line frequency was not detected" \
        "; therefore, it was set to 60 Hz")
        power_line_freq = 60

    # set eeg info (channel types and electrode montage) when it is not there yet
    # *******************************************************
    if configs["which_sensor"] == "eeg":
        data = prepare_eeg_data(data, path)

    # drop noisy channels for MEG
    # *******************************************************
    if configs["which_sensor"] in ["meg", "grad", "mag"]:
        data, empty_room_recording = drop_noisy_meg_channels(data=data, 
                                            subID=subID, 
                                            args=args, 
                                            configs=configs, 
                                            empty_room_recording=empty_room_recording)
    

    which_sensor = dict.fromkeys(["meg", "mag", "grad", "eeg", "opm"], False)
    which_sensor[configs.get("which_sensor")] = True

    # preproces
    # *******************************************************
    filtered_data, channel_names, sampling_rate, empty_room_recording = preprocess(
        data=data,
        n_component=configs["ica_n_component"],
        ica_max_iter=configs["ica_max_iter"],
        IcaMethod=configs["ica_method"],
        cutoffFreqLow=configs["cutoffFreqLow"],
        cutoffFreqHigh=configs["cutoffFreqHigh"],
        which_sensor=which_sensor,
        resampling_rate=configs["resampling_rate"],
        digital_filter=configs["digital_filter"],
        rereference_method=configs["rereference_method"],
        apply_ica=configs["apply_ica"],
        auto_ica_corr_thr=configs["auto_ica_corr_thr"],
        power_line_freq=power_line_freq,
        empty_room_recording=empty_room_recording,
        ctf_gradient_comp_level=configs["ctf_gradient_comp_level"],
        muscle_activity_filter_freq=configs["muscle_activity_filter_freq"],
        extention=extention
    )

    # Source localization 
    # *******************************************************
    kwargs = {"source_space_spacing":"ico6",
              "spacing":5}

    if configs["apply_source_localization"]:
        logger.info("Starting the source localization")
        stc, labels = source_localization(
                subject=subID,
                subjects_dir=args.surfaces_dir, 
                subject_to="fsaverage",
                data=filtered_data,
                empty_room_recording=empty_room_recording,
                source_space=configs["SL_source_space"],
                conductivity=configs["SL_conductivity"],
                inverse_operator=configs["SL_inverse_operator"],
                figures_path=os.path.join(args.save_dir, "figures"),
                plot_3d=False,
                **kwargs
            )
        sl_data = numpy_to_mne_raw(stc, labels, "mag", sampling_rate)

    # segmentation 
    # *******************************************************
    segments = segment_epoch(
        data=sl_data,
        sampling_rate=sampling_rate,
        tmin=configs["segments_tmin"],
        tmax=configs["segments_tmax"],
        segmentsLength=configs["segments_length"],
        overlap=configs["segments_overlap"],
    )

    # fooof analysis 
    # *******************************************************
    fmGroup, psds, freqs = psdParameterize(
        segments=segments,
        sampling_rate=sampling_rate,
        # psd parameters
        psd_method=configs["psd_method"],
        psd_n_overlap=configs["psd_n_overlap"],
        psd_n_fft=configs["psd_n_fft"],
        n_per_seg=configs["psd_n_per_seg"],
        # fooof parameters
        freq_range_low=configs["fooof_freq_range_low"],
        freq_range_high=configs["fooof_freq_range_high"],
        min_peak_height=configs["fooof_min_peak_height"],
        peak_threshold=configs["fooof_peak_threshold"],
        peak_width_limits=configs["fooof_peak_width_limits"],
        aperiodic_mode=configs["aperiodic_mode"],
    )

    if configs["fooof_res_save_path"]:
        storeFooofModels(configs["fooof_res_save_path"], subID, fmGroup, psds, freqs)

    # feature extraction 
    # *******************************************************
    features = feature_extract(
        subject_id=subID,
        fmGroup=fmGroup,
        psds=psds,
        freqs=freqs,
        freq_bands=configs["freq_bands"],
        channel_names=channel_names,
        individualized_band_ranges=configs["individualized_band_ranges"],
        feature_categories=configs["feature_categories"],
        extention=extention,
        which_layout=configs["which_layout"],
        which_sensor=which_sensor,
        aperiodic_mode=configs["aperiodic_mode"],
        min_r_squared=configs["min_r_squared"],
    )

    features.to_csv(os.path.join(args.save_dir, f"{subID}.csv"))

    logger.info(f"The process for the subject {subID} is complete.")

if __name__ == "__main__":

    main(sys.argv[1:])
