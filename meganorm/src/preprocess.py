import os
import mne
from mne_icalabel import label_components
import json
import numpy as np
import glob
import logging
from typing import Any, Dict
import pandas as pd
from scipy.stats import zscore
import warnings
from kneed import KneeLocator

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def find_ica_component(ica, data, physiological_signal, auto_ica_corr_thr):
    """
    Identifies independent components that their correlation with physiological
    signals (ECG or EOG) is higher than a threshold.

    Parameters
    ----------
    ica : object
        The fitted ICA object using MNE.
    data : mne.io.Raw
        The raw MEG/EEG data used to extract independent components.
    physiological_signal : np.ndarray
        The physiological signal (ECG or EOG) to compare with independent componentss.
    auto_ica_corr_thr : float
        Pearson correlation threshold (between 0 and 1) for accepting a component as
        noise.

    Returns
    -------
    list
        Index of the component with the highest correlation if it exceeds the threshold.
        Returns an empty list if no component meets the criterion.
    """
    components = ica.get_sources(data.copy()).get_data()

    if components.shape[1] != len(physiological_signal):
        raise ValueError(
            "Length of physiological signal must match the number of time points in the data."
        )

    corr = np.corrcoef(components, physiological_signal)[-1, :-1]

    if np.max(corr) >= auto_ica_corr_thr:
        componentIndx = [int(np.argmax(corr))]
    else:
        componentIndx = []

    return componentIndx


def auto_ica(
    data,
    physiological_sensor,
    n_components=30,
    ica_max_iter=1000,
    IcaMethod="fastica",
    which_sensor={"meg": True, "eeg": True},
    auto_ica_corr_thr=0.9,
):
    """
    Performs automated ICA for artifact removal by identifying components that
    correlate highly with physiological signals (ECG or EOG) which is
    determined by 'auto_ica_corr_thr'.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG/EEG data.
    physiological_sensor : str
        Name of the physiological sensor ('ECG' or 'EOG').
    n_components : int or float
        Number of ICA components to retain.
    ica_max_iter : int
        Maximum number of iterations for the ICA algorithm.
    IcaMethod : str
        ICA algorithm to use (e.g., 'fastica', 'picard', 'infomax').
    which_sensor : dict
        Dictionary indicating sensor types to include (e.g., {'meg': True, 'eeg': True}).
    auto_ica_corr_thr : float
        Threshold for accepting independent component as noisy based
        on correlation with the corresponding physiological recording (ECG or EOG).

    Returns
    -------
    data : mne.io.Raw
        Raw data with bad ICA components removed (in-place modification).
    ICA_flag : bool
        True if no bad components were found, False otherwise.
    """
    # Get physiological signal
    physiological_signal = data.copy().pick(picks=physiological_sensor).get_data()

    # Pick MEG/EEG for ICA
    data = data.pick_types(
        meg=which_sensor.get("meg", False),
        eeg=which_sensor.get("eeg", False),
        ref_meg=False,
        eog=True,
        ecg=True,
    )

    # ICA initialization
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        max_iter=ica_max_iter,
        method=IcaMethod,
        random_state=42,
        verbose=False,
    )
    ica.fit(data, verbose=False, picks=["eeg", "meg"])

    # Detect components correlated with physiological signal
    bad_components = []
    for sensor in physiological_signal:
        bad_components.extend(
            find_ica_component(
                ica=ica,
                data=data,
                physiological_signal=sensor,
                auto_ica_corr_thr=auto_ica_corr_thr,
            )
        )

    logger.info(f"Number of bad ICA components: {len(bad_components)}")

    if bad_components:
        ica.exclude = bad_components.copy()
        ica.apply(data, verbose=False)
        ICA_flag = False
    else:
        ICA_flag = True

    return data, ICA_flag, len(bad_components)


def auto_ica_with_mean(
    data,
    n_components=30,
    ica_max_iter=1000,
    IcaMethod="fastica",
    which_sensor={"meg": True, "eeg": True},
    auto_ica_corr_thr=0.9,
):
    """
    Performs ICA-based artifact rejection using MNE’s built-in ECG correlation method.
    This function creates a synthetic ECG signal (by avergaing across magnetometers
    or Gradiometers) and use it to find and remove the noisy independent component.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG/EEG data.
    n_components : int, optional
        Number of ICA components to retain, by default 30.
    ica_max_iter : int, optional
        Maximum number of iterations for the ICA algorithm, by default 1000.
    IcaMethod : str, optional
        ICA algorithm to use (e.g., 'fastica', 'picard', 'infomax'), by default "fastica".
    which_sensor : dict, optional
        Dictionary specifying sensor types to include (e.g., {"meg": True, "eeg": True}), by default {"meg": True, "eeg": True}.
    auto_ica_corr_thr : float, optional
        Correlation threshold for detecting ECG-related components, by default 0.9.

    Returns
    -------
    mne.io.Raw
        Raw data with ECG-related ICA components removed.
    """
    data = data.pick_types(
        meg=which_sensor["meg"] | which_sensor["mag"] | which_sensor["grad"],
        eeg=which_sensor["eeg"],
        ref_meg=False,
        eog=True,
        ecg=True,
    )

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        max_iter=ica_max_iter,
        method=IcaMethod,
        random_state=42,
        verbose=False,
    )
    ica.fit(data, verbose=False, picks=["eeg", "meg"])

    ecg_indices, _ = ica.find_bads_ecg(
        data, method="correlation", threshold=auto_ica_corr_thr, measure="correlation"
    )

    logger.info(f"Number of bad ICA components detected by creating synthetic ECG signal: {len(ecg_indices)}")
    ica.exclude = ecg_indices
    ica.apply(data, verbose=False)

    return data, len(ecg_indices)


def AutoIca_with_IcaLabel(
    data,
    physiological_noise_type,
    n_components=30,
    ica_max_iter=1000,
    IcaMethod="infomax",
    iclabel_thr=0.8,
):

    if physiological_noise_type == "ecg":
        physiological_noise_type = "heart beat"
    if physiological_noise_type == "eog":
        physiological_noise_type = "eye blink"

    # fit ICA
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        max_iter=ica_max_iter,
        method=IcaMethod,
        random_state=42,
        fit_params=dict(extended=True),
        verbose=False,
    )  # fit_params=dict(extended=True) bc icalabel is trained with this
    ica.fit(data, verbose=False, picks=["eeg"])

    # apply ICLabel
    labels = label_components(data, ica, method="iclabel")

    # Identify and exclude artifact components based on probability threshold of being an artifact
    bad_components = []
    for idx, label in enumerate(labels["labels"]):
        if (
            label == physiological_noise_type
            and labels["y_pred_proba"][idx] > iclabel_thr
        ):
            bad_components.append(idx)

    logger.info("Number of bad Components identified by ICALabel:", len(bad_components))
    ica.exclude = bad_components.copy()
    ica.apply(data, verbose=False)

    return data, len(bad_components)


def prepare_eeg_data(data, path):
    """
    Prepare EEG data by setting channel types and electrode montage when they are not in the data yet

    Parameters
    ----------
    data : mne.io.Raw
        The raw EEG data.
    path : str
        Path to the EEG recording file.

    Returns
    -------
    mne.io.Raw
        The EEG data with updated channel types and montage (if available).
    """
    task = path.split("/")[-1].split("_")[-2]
    base_dir = os.path.dirname(path)

    # Set channel types
    search_pattern = os.path.join(base_dir, f"**_{task}_channels.tsv")
    channel_files = glob.glob(search_pattern, recursive=True)
    if channel_files:
        channels_df = pd.read_csv(channel_files[0], sep="\t")
        channels_types = channels_df.set_index("name")["type"].str.lower().to_dict()
        data.set_channel_types(channels_types)

    # Set montage if not already set
    montage = data.get_montage()
    if montage is None:
        try:
            search_pattern_montage = os.path.join(base_dir, "*_montage.csv")
            montage_files = glob.glob(search_pattern_montage, recursive=True)

            if not montage_files:
                raise FileNotFoundError("No montage CSV file found!")

            montage_df = pd.read_csv(montage_files[0])
            ch_positions = {
                row["Channel"]: [row["X"], row["Y"], row["Z"]]
                for _, row in montage_df.iterrows()
            }
            eeg_montage = mne.channels.make_dig_montage(
                ch_pos=ch_positions, coord_frame="head"
            )
            data.set_montage(eeg_montage)

        except Exception as e:
            print(f"Error setting montage: {e}")
            print(
                "Continuing without a montage. This may raise issues for ICA labeling."
            )

    return data


def segment_epoch(
    data: mne.io.Raw,
    tmin: float,
    tmax: float,
    sampling_rate: float,
    segmentsLength: float,
    overlap: float,
):
    """
    Segments continuous raw data into epochs of fixed length.

    Parameters
    ----------
    data : mne.io.Raw
        MEG/EEG recording.
    tmin : float
        Start time (in seconds) for cropping the raw data.
    tmax : float
        End time (in seconds) for cropping the raw data. 'tmax' must be a
        negative number, indicating the time difference between the crop
        end point and the total recording duration.
    sampling_rate : float
        Sampling rate of the data (Hz).
    segmentsLength : float
        Length of each epoch in seconds.
    overlap : float
        Overlap between successive epochs in seconds.

    Returns
    -------
    mne.Epochs
        Segmented data with fixed-length segments.
    """
    if tmax > 0:
        raise ValueError("The 'tmax' must be a negative number")

    # Calculate absolute tmax based on data duration and trim beginning/end
    tmax = int(np.shape(data.get_data())[1] / sampling_rate + tmax)

    # Crop 20 seconds from both ends to avoid eye-open/close artifacts
    data.crop(tmin=tmin, tmax=tmax)

    # Create fixed-length overlapping epochs
    segments = mne.make_fixed_length_epochs(
        data,
        duration=segmentsLength,
        overlap=overlap,
        reject_by_annotation=True,
        verbose=False,
    )

    return segments


def preprocess(
    data,
    extention,
    which_sensor: dict,
    empty_room_recording=None,
    resampling_rate: int = 1000,
    digital_filter=True,
    rereference_method="average",
    n_component: int = 30,
    ica_max_iter: int = 800,
    IcaMethod: str = "fastica",
    cutoffFreqLow: int = 1,
    cutoffFreqHigh: int = 45,
    apply_ica=True,
    power_line_freq: int = 60,
    auto_ica_corr_thr: float = 0.9,
    muscle_activity_thr=4.0,
    muscle_activity_min_length_good=0.1,
    muscle_activity_filter_freq=(110, 140),
    ica_apply_elbow_detection = False, 
    apply_oversampled_temporal_projection = True,
    apply_Head_movement_correction=True,
    Head_movement_limit_from_mean = 0.0015,
    apply_chpi_filter=False,
    ctf_gradient_comp_level=3,
    apply_environmental_noise_ssp_with_eroom=False,
    apply_environmental_noise_ica_with_ref_meg=False,
    environmental_noise_ica_with_ref_meg_thr=2.5,
    ica_if_reject_by_annotation=True,
    environmental_noise_ica_with_ref_meg_method="together",
    environmental_noise_ica_with_ref_meg_measure="zscore"
):
    """
    Applies a preprocessing pipeline on MEG/EEG data, including filtering, re-referencing (for EEG),
    ICA for artifact removal, and optional downsampling.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG/EEG data.
    extention : str
        The extension of the subject's recording (e.g., 'FIF', 'DS'). Used to read the
        appropriate layout file from the layout directory.
    which_sensor : dict
        Dictionary specifying which sensor types to include (e.g., {'meg': True, 'eeg': True}).
    empty_room_recording : mne.io.Raw, optional
        Empty room recording.
    resampling_rate : int, optional
        Target sampling rate for resampling. If None, resampling is skipped; by default 1000.
    digital_filter : bool, optional
        Whether to apply a bandpass FIR filter to the data; by default True.
    rereference_method : str, optional
        EEG re-referencing method. Supported: "average", "REST"; by default "average".
    n_component : int, optional
        Number of independent component to retain in ICA; by default 30.
    ica_max_iter : int, optional
        Maximum number of iterations for ICA; by default 800.
    IcaMethod : str, optional
        ICA algorithm to use. Supported: 'fastica', 'picard', 'infomax'; by default "fastica".
    cutoffFreqLow : int, optional
        Low cutoff frequency for bandpass filtering; by default 1.
    cutoffFreqHigh : int, optional
        High cutoff frequency for bandpass filtering; by default 45.
    apply_ica : bool, optional
        Whether to apply ICA to remove artifacts; by default True.
    power_line_freq : int, optional
        Power line frequency (for notch filtering); by default 60.
    auto_ica_corr_thr : float, optional
        Correlation threshold for automatic ICA artifact rejection; by default 0.9. That is,
        the correlation between identified independent components and physiological signals (ECG
        and EOG) must be higher than 'auto_ica_corr_thr'
    ctf_gradient_comp_level: int, optional
        The gradient compensation level to apply. Valid values typically include:
        -1 (disable), 0 (raw data), 1, 2, 3 (increasing levels of compensation).
        Default is 3.
    muscle_activity_thr : float, optional
        The threshold for a segment to be considered as artifact (z-scores)
    muscle_activity_min_length_good: float, optional
        The minimum required duration (in seconds) of valid data between consecutive annotations.
    muscle_activity_filter_freq: tuple, optional
        Cutoff frequencies for the band-pass filter used in muscle activity detection.
        Muscle activity is typically more prominent in higher frequency ranges (e.g., 110–140 Hz).
    ica_apply_elbow_detection: bool: False


    Returns
    -------
    mne.io.Raw
        Preprocessed MEG/EEG data.

    Raises
    ------
    ValueError
        auto_ica_corr_thr must be between 0 and 1.
    ValueError
        ICA method must be one of: 'fastica', 'picard', 'infomax'.
    """
    if not 0 < auto_ica_corr_thr <= 1:
        err_msg = "auto_ica_corr_thr must be between 0 and 1."
        logger.error(err_msg)
        raise ValueError(err_msg)
    if IcaMethod not in ["fastica", "picard", "infomax"]:
        err_msg = "ICA method must be one of: 'fastica', 'picard', 'infomax'."
        logger.error(err_msg)
        raise ValueError(err_msg)


    # since pick_channels can not seperate mag and grad signals
    # if not (which_sensor["meg"] or which_sensor["eeg"]):
    if which_sensor["grad"] or which_sensor["meg"]:
        data, empty_room_recording = drop_mag_or_grad(data, empty_room_recording, which_sensor)

    channel_types = set(data.get_channel_types())

    # resample -------------------------------------
    sampling_rate = data.info["sfreq"]
    if resampling_rate and resampling_rate != sampling_rate:
        data.resample(int(resampling_rate), verbose=False, n_jobs=-1)
        sampling_rate = data.info["sfreq"]
        # resampling empty room recording
        if empty_room_recording:
            empty_room_recording.resample(int(resampling_rate), verbose=False, n_jobs=-1)

    # flux jumps (SQUID jumps) ---------------------
    if apply_oversampled_temporal_projection:
        data = mne.preprocessing.oversampled_temporal_projection(data)
    
    # power line -----------------------------------
    data.notch_filter(
        freqs=np.arange(
            int(power_line_freq), 4 * int(power_line_freq) + 1, int(power_line_freq)
        ),
        n_jobs=-1,
    )
    if empty_room_recording:
        empty_room_recording.notch_filter(
            freqs=np.arange(
                int(power_line_freq), 4 * int(power_line_freq) + 1, int(power_line_freq)
            ),
            n_jobs=-1,
        )

    # head motion correction ----------------------
    if which_sensor != "eeg" and apply_Head_movement_correction:
        data, empty_room_recording = head_motion_correction(
            data,
            empty_room_recording,
            extention,
            Head_movement_limit_from_mean=Head_movement_limit_from_mean
            )

    # remove cHPI noise ---------------------------
    if which_sensor != "eeg" and apply_chpi_filter:
        
        # if chpi data is available
        # TODO: use mne.chpi.extract_chpi_locs_ctf or extract_chpi_locs_kit
        if len(mne.chpi.get_chpi_info(data.info)):
            data = mne.chpi.filter_chpi(data,
                                        include_line=False)
            logger.info("Filtering CHPI noise.")
        # if chpi data is inavailable and cutoffFreqHigh > 80
        else:
            if cutoffFreqHigh > 80: # TODO check this
                logger.warning("cHPI info are missing; Therefore"\
                " cHPI noise can not be filtered. In case you have cHPI coils, please put"
                "this information in the data, otherwise you can ignore this error.")

    # digital filter --------------------------------
    if digital_filter:
        data.filter(
            l_freq=int(cutoffFreqLow),
            h_freq=int(cutoffFreqHigh),
            n_jobs=-1,
            verbose=False,
        )
        if empty_room_recording:
            empty_room_recording.filter(
                l_freq=int(cutoffFreqLow),
                h_freq=int(cutoffFreqHigh),
                n_jobs=-1,
                verbose=False,
            )      

    # Muscle artifact detection ---------------------
    if cutoffFreqHigh > muscle_activity_filter_freq[0]:
        muscle_annot, _ = mne.preprocessing.annotate_muscle_zscore(data,
                                                min_length_good=muscle_activity_min_length_good,
                                                filter_freq=muscle_activity_filter_freq,
                                                threshold=muscle_activity_thr)
        # ICA will ignore these and later will be removed in segmentation
        data.set_annotations(muscle_annot)
        logger.info(f"Muscle artifact rejection alg removed {sum(muscle_annot.duration)} seconds of"\
                    " the signal.")
        # TODO: MNE doc: The type of sensors to use. If None it will take the first type in mag, grad, eeg.
        # You need to apply muscle artifact detection on both
        # MAG and Grad seperately.
        # MNE documents: Choose one channel type, if there are axial gradiometers and magnetometers,
        # select magnetometers as they are more sensitive to muscle activity.

    # rereference -----------------------------------
    if which_sensor["eeg"] and rereference_method:
        data = data.set_eeg_reference(rereference_method)
        if empty_room_recording:
            empty_room_recording = empty_room_recording.set_eeg_reference(rereference_method)

    # remove environmental noise ---------------------
    remove_environmental_noise(data,
                               extention,
                               empty_room_recording=empty_room_recording,
                               ctf_gradient_comp_level=ctf_gradient_comp_level,
                               apply_environmental_noise_ssp_with_eroom=apply_environmental_noise_ssp_with_eroom,
                               apply_environmental_noise_ica_with_ref_meg=apply_environmental_noise_ica_with_ref_meg,
                               environmental_noise_ica_with_ref_meg_thr=environmental_noise_ica_with_ref_meg_thr,
                               ica_if_reject_by_annotation=ica_if_reject_by_annotation,
                               environmental_noise_ica_with_ref_meg_method=environmental_noise_ica_with_ref_meg_method,
                               environmental_noise_ica_with_ref_meg_measure=environmental_noise_ica_with_ref_meg_measure)


    # physiological noise ----------------------------
    ICA_flag = True  # initialize flag

    physiological_electrods = {
        channel: channel in channel_types for channel in ["ecg", "eog"]
    }

    if ica_apply_elbow_detection:
        n_component = pca_elbow_locator(data, which_sensor)

    for phys_activity_type, if_elec_exist in physiological_electrods.items():
        
        # number of reduced independent components for rank computation
        number_of_reduced_ic = 0
        
        if which_sensor["meg"] or which_sensor["mag"] or which_sensor["grad"]:  
            # ======================================================================
            # 1
            if if_elec_exist and apply_ica:
                data, _, number_of_reduced_ic  = auto_ica(
                    data=data,
                    n_components=n_component,
                    ica_max_iter=ica_max_iter,
                    IcaMethod=IcaMethod,
                    which_sensor=which_sensor,
                    physiological_sensor=phys_activity_type,
                    auto_ica_corr_thr=auto_ica_corr_thr,
                )
            # 2
            elif not if_elec_exist and apply_ica and phys_activity_type == "ecg":
                data, number_of_reduced_ic = auto_ica_with_mean(
                    data=data,
                    n_components=n_component,
                    ica_max_iter=ica_max_iter,
                    IcaMethod=IcaMethod,
                    which_sensor=which_sensor,
                    auto_ica_corr_thr=auto_ica_corr_thr,
                )

        if which_sensor[
            "eeg"
        ]:  # ======================================================================
            # 1
            if if_elec_exist and apply_ica:
                data, ICA_flag, number_of_reduced_ic = auto_ica(
                    data=data,
                    n_components=n_component,
                    ica_max_iter=ica_max_iter,
                    IcaMethod=IcaMethod,
                    which_sensor=which_sensor,
                    physiological_sensor=phys_activity_type,
                    auto_ica_corr_thr=auto_ica_corr_thr,
                )
            # 2
            elif not if_elec_exist and apply_ica and ICA_flag:
                data, number_of_reduced_ic = AutoIca_with_IcaLabel(
                    data=data,
                    n_components=n_component,
                    ica_max_iter=ica_max_iter,
                    IcaMethod=IcaMethod,
                    iclabel_thr=auto_ica_corr_thr,
                    physiological_noise_type=phys_activity_type,
                )

    data = data.pick_types(
        meg=which_sensor["meg"] | which_sensor["mag"] | which_sensor["grad"],
        eeg=which_sensor["eeg"],
        ref_meg=False,
        eog=False,
        ecg=False,
    )

    if empty_room_recording:
        empty_room_recording = empty_room_recording.pick_types(
            meg=which_sensor["meg"] | which_sensor["mag"] | which_sensor["grad"],
            eeg=which_sensor["eeg"],
            ref_meg=False,
            eog=False,
            ecg=False,
        )

    return data, data.info["ch_names"], int(sampling_rate), empty_room_recording, number_of_reduced_ic


def drop_noisy_meg_channels(
    data: Any, subID: str, args: Any, configs: Dict[str, str], empty_room_recording=None
) -> Any:
    """
    Identifies and removes noisy or flat MEG/EEG channels using Maxwell filtering,
    and logs the number of dropped channels for each subject.

    Parameters
    ----------
    data : instance of `mne.io.Raw`
        The MEG/EEG recording to process.

    subID : str
        Identifier for the subject, used in naming the log file.

    args : argparse.Namespace or similar
        Object containing runtime arguments, including 'saveDir'.

    configs : dict
        Configuration dictionary containing:
            - 'which_sensor': one of {"meg", "mag", "grad", "eeg", "opm"}

    empty_room_recording: instance of `mne.io.Raw`
        Empty room recording.

    Returns
    -------
    data_cleaned : instance of `mne.io.Raw`
        The cleaned data with noisy/flat channels removed.

    Notes
    -----
    If Maxwell filtering has already been applied (e.g., SSS step),
    the function will skip bad channel detection and proceed to drop
    previously marked bad channels.

    The number of dropped channels is saved to a JSON log file in
    a directory derived from `args.saveDir`, replacing 'temp' with
    'log_droped_channels'.
    """
    logger = logging.getLogger(__name__)

    which_sensor = dict.fromkeys(["meg", "mag", "grad", "eeg", "opm"], False)
    which_sensor[configs.which_sensor] = True

    if check_tsss(data):
        msg = "Maxwell filter has already been applied. " \
            "Therefore, bad channel detection using maxwell will be not applied."
        logger.info(msg)
        
    else:
        auto_noisy_chs, auto_flat_chs = mne.preprocessing.find_bad_channels_maxwell(
            data, return_scores=False, verbose=True, coord_frame="meg"
        )
        data.info["bads"] += auto_noisy_chs + auto_flat_chs

    if empty_room_recording:
        empty_room_recording.info["bads"] = data.info["bads"].copy()

    dropped_data = data.copy().drop_channels(data.info["bads"])
    empty_room_recording = empty_room_recording.copy().drop_channels(empty_room_recording.info["bads"])

    # Always proceed to log and drop marked bads
    droped_ch_len = len(data.info["bads"])
    logger.warning(f"{droped_ch_len} channels were droped from the subject's recording")

    return dropped_data, empty_room_recording


def apply_chpi(meg_data, movement_limit, head_pos_save_path, extention):

    if meg_data.info["hpi_results"]:

        # BTi/4D MEG recordings do not support cHPI and don't have 
        # real time recordings  of the brain pos
        if extention == "fif":
            amp = mne.chpi.compute_chpi_amplitudes(meg_data)
            locs = mne.chpi.compute_chpi_locs(meg_data.info, amp)

        if extention == "ds":
            locs = mne.chpi.extract_chpi_locs_ctf(meg_data)

        head_pos = mne.chpi.compute_head_pos(meg_data.info, locs)

        if list(head_pos):
            movement_annot = mne.preprocessing.annotate_movement(meg_data, 
                                                                pos=head_pos, 
                                                                mean_distance_limit=movement_limit)

            moved_times = sum(movement_annot.duration)
            moved_inteval_percentage = moved_times/meg_data.duration[-1]*100

            logger.warning(f"{moved_inteval_percentage} percent of the recording exceeds the mean distance"\
                        "limit for the head motion. Consider using tSSS.")
        else:
            logger.info("Unable to find a reliable solution for any of the coils")

    else:
        logger.info("No cHPI coil was found for the current subject.")

    mne.chpi.write_head_pos(head_pos_save_path, head_pos)


def apply_gradient_comp(ctf_meg_data, grade=3):
    """
    Interpolate bad channels and apply gradient compensation to CTF MEG data.

    This function first interpolates bad MEG channels using the minimum norm method
    to improve signal quality, then applies CTF-style gradient compensation at the
    specified grade level. Gradient compensation is specific to CTF MEG systems and
    helps remove interference from distant sources.

    Parameters
    ----------
    ctf_meg_data : mne.io.Raw
        The raw MEG data object (from a CTF system), loaded with `preload=True`.
        Must contain CTF-specific gradient compensation information.
    grade : int, optional
        The gradient compensation level to apply. Valid values typically include:
        -1 (disable), 0 (raw data), 1, 2, 3 (increasing levels of compensation).
        Default is 3.

    Returns
    -------
    ctf_meg_data : mne.io.Raw
        The same Raw object with interpolated bad channels and gradient compensation applied.

    Notes
    -----
    - Only the minimum norm method (`method={"meg": "MNE"}`) is used for interpolation,
      which is currently the only supported method for MEG in MNE.
    - Bad channels are not reset after interpolation (`reset_bads=False`) so that
      they remain marked in the data structure.
    - This function assumes that the input data is from a CTF MEG system.

    See Also
    --------
    mne.io.Raw.interpolate_bads : Interpolate bad channels in MEG/EEG data.
    mne.io.Raw.apply_gradient_compensation : Apply gradient compensation to CTF MEG data.
    """
    # Only minimum norm method is supported by MNE
    # for interpolating MEG signals; Therefore this
    # argument is not added to the config
    method = {"meg": "MNE"}
    # Bad channels are first interpolated for the
    # sake of SNR
    # reset_bads should remain False to keep track
    # of bad channels so we can use them later in
    # source localization.
    ctf_meg_data.interpolate_bads(reset_bads=False,
                                  method=method)
    
    ctf_meg_data.apply_gradient_compensation(grade=grade) 

    logger.info(f"Gradient compensation with level of {grade} has been applied to the data.")

    return ctf_meg_data


def apply_tsss(
        data,
        cross_talk_path,
        calibration_path,
        head_pos_path=None,
        empty_room_record=None,
        st_duration=10.0,
        st_correlation=0.98
):
    """
    Apply temporal Signal Space Separation (tSSS) to MEG data with optional 
    head position correction and empty-room noise processing.

    This function uses MNE-Python's Maxwell filtering implementation to 
    suppress environmental noise and remove cross-talk between sensors.
    If a head position file is provided, movement compensation will be 
    applied. Optionally, an empty-room recording can be processed with 
    the same parameters for noise estimation.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG data to be processed.
    cross_talk_path : str or path-like
        Path to the cross-talk compensation file (CTF), typically provided
        by the MEG system.
    calibration_path : str or path-like
        Path to the fine-calibration file (CAL), typically provided by the
        MEG system.
    head_pos_path : str or path-like, optional
        Path to the head position file (.pos) obtained from cHPI processing.
        If provided, movement compensation will be applied.
    empty_room_record : mne.io.Raw, optional
        Raw empty-room MEG recording to process for noise estimation. If 
        provided, it will be Maxwell filtered with the same parameters as 
        the main data.
    st_duration : float, default=10.0
        Window duration in seconds for the temporal SSS (tSSS) projection.
        Shorter windows can better track non-stationary interference but 
        may remove more brain signal.
    st_correlation : float, default=0.98
        Correlation limit between SSS basis functions across time windows. 
        Values closer to 1.0 remove less brain signal but may be less 
        effective at removing artifacts.

    Returns
    -------
    data_tsss : mne.io.Raw
        The Maxwell-filtered MEG data with optional movement compensation.
    empty_room_record : mne.io.Raw or None
        The Maxwell-filtered empty-room recording if provided, else None.

    Notes
    -----
    - Maxwell filtering is sensitive to accurate calibration and cross-talk
      compensation files; ensure the provided files match the MEG system used
      for the recording.
    - The `st_duration` and `st_correlation` parameters control the aggressiveness 
      of the temporal projection; inappropriate values can either leave 
      environmental noise in the data or attenuate brain signal.
    - If `head_pos_path` is provided, continuous head position data will be
      used to apply movement compensation during filtering.

    References
    ----------
    .. [1] Taulu, S., Simola, J. (2006). Spatiotemporal signal space separation method 
       for rejecting nearby interference in MEG measurements. Physics in Medicine 
       and Biology, 51(7), 1759.
    .. [2] MNE-Python documentation: 
       https://mne.tools/stable/generated/mne.preprocessing.maxwell_filter.html
    """
    
    if head_pos_path:
        head_pos = mne.chpi.read_head_pos(head_pos_path)
    else:
        head_pos = None

    data_tsss = mne.preprocessing.maxwell_filter(
        raw=data,
        calibration=calibration_path,
        cross_talk=cross_talk_path,
        st_duration=st_duration,
        st_correlation=st_correlation,
        head_pos=head_pos
    )

    if empty_room_record:
        
        empty_room_record = mne.preprocessing.maxwell_filter_prepare_emptyroom(
            raw_er=empty_room_record
        )

        empty_room_record = mne.preprocessing.maxwell_filter(
            raw=empty_room_record,
            calibration=calibration_path,
            cross_talk=cross_talk_path,
            st_duration=st_duration,
            st_correlation=st_correlation,
            head_pos=head_pos
        )

    return data_tsss, empty_room_record


def drop_noisy_segments(
        segments,
        z_thr
):
    """
    Drop noisy data segments based on the z-scored standard deviation 
    across time.

    This function computes the standard deviation of each segment across 
    the time axis, converts these values to z-scores, and removes segments 
    whose z-score exceeds a given threshold. This is useful for discarding 
    artifacts or unusually high-variance data before further processing.

    Parameters
    ----------
    segments : mne.Epochs or mne.Epochs-like
        The segmented MEG/EEG data object. Must have an attribute 
        ``_data`` of shape (n_segments, n_channels, n_times) and a 
        ``drop(indices)`` method to remove segments.
    z_thr : float
        The z-score threshold. Segments with a standard deviation 
        z-score greater than this value will be dropped.

    Returns
    -------
    segments : mne.Epochs or mne.Epochs-like
        The input object with noisy segments removed.

    Notes
    -----
    - This method assumes that noise manifests as abnormally high 
      variance in one or more channels of a segment.
    - Z-scores are computed per channel across all segments, so 
      segments may be flagged for removal if any channel exceeds 
      the threshold.
    - The function logs the number of dropped segments and the 
      remaining count.

    """

    z_scores = zscore(
        np.std(
            segments._data, axis=2
            ),
        axis=0)
    
    bad_segments = np.where(
        z_scores>z_thr
        )[0]
    
    segments=segments.drop(indices=bad_segments)

    logger.info(f"Dropping {len(bad_segments)} segments due to Z > Z_threshold. "\
                f"The final number of used segments: {segments.__len__()}")
    
    return segments


def check_tsss(meg_data):
    """
    Check if Maxwell filtering (tSSS) was applied to raw/epochs data.

    This inspects the processing history for presence of maxfilter info.

    Parameters
    ----------
    meg_data : mne.io.BaseRaw | mne.Epochs
        The MEG data object.

    Returns
    -------
    bool
        True if tSSS has been applied, False otherwise.
    """
    proc_history = meg_data.info.get('proc_history', [])
    if not proc_history:
        return False
    max_info = proc_history[0].get('max_info', {})
    sss_cal = max_info.get('sss_cal', [])
    return len(sss_cal) > 0



def pca_elbow_locator(raw, which_sensor):

    raw = raw.copy().pick_types(
        meg=which_sensor["meg"] or which_sensor["mag"] or which_sensor["grad"],
        eeg=which_sensor["eeg"],
        ref_meg=False,
        eog=False,
        ecg=False,
    )

    # Use sklearn PCA via MNE
    from sklearn.decomposition import PCA
    pca = PCA(n_components=None, whiten=True)
    _ = pca.fit_transform(raw.get_data().T)

    explained_var = pca.explained_variance_ratio_

    # Find elbow
    knee_locator = KneeLocator(
        x=np.arange(1, len(explained_var) + 1),
        y=explained_var,
        curve="concave",
        direction="decreasing"
    )

    elbow_index = knee_locator.knee

    # Handle None and enforce min
    if elbow_index is None or elbow_index < 15:
        elbow_index = 30

    return int(elbow_index)



def drop_mag_or_grad(data, empty_room_recording, which_sensor):

    # since pick_channels can not seperate mag and grad signals
    # if not (which_sensor["meg"] or which_sensor["eeg"]):
    dropping_channels = []

    if which_sensor["grad"]:
        logger.info("Dropping magnetometer sensors.")
        dropping_channels = [
            ch
            for ch, ch_type in zip(data.ch_names, data.get_channel_types())
            if ch_type == "mag"
        ]

    elif which_sensor["mag"]:
        logger.info("Dropping gradiometer sensors.")
        dropping_channels = [
            ch
            for ch, ch_type in zip(data.ch_names, data.get_channel_types())
            if ch_type == "grad"
        ]
    
    else:
        raise Exception
    
    data.drop_channels(dropping_channels)
    if empty_room_recording:
        empty_room_recording.drop_channels(dropping_channels)

    return data, empty_room_recording


def head_motion_correction(data,
                empty_room_recording,
                extention,
                Head_movement_limit_from_mean=0.0015):
    """
    Perform head-motion–correction.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG recording from a subject. Must contain cHPI channels if
        head-movement estimation is desired.

    empty_room_recording : mne.io.Raw | None
        Empty-room MEG recording corresponding to the subject data.
        If provided, it will be prepared and Maxwell-filtered using the
        sensor geometry of the subject data.

    extention : str
        File extension identifying the MEG vendor. 

    Head_position_limit_from_mean : float, default=0.0015
        Threshold (in meters) for annotating excessive head movement.
        Time segments where the head position deviates from the mean
        position by more than this value will be annotated.

    Returns
    -------
    data : mne.io.Raw
        The processed subject MEG data. If cHPI data are present, the output
        includes movement annotations and an updated ``dev_head_t`` based on
        the average head position.

    empty_room_recording : mne.io.Raw | None
        The processed empty-room recording, filtered using the same Maxwell
        filtering parameters as the subject data but without movement
        compensation. Returned unchanged if ``None`` was provided.

    """

    
    if check_tsss(data):
        return data, empty_room_recording
    
    # MEGIN devices
    if extention == "fif":
        
        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(data)
        chpi_locs = mne.chpi.compute_chpi_locs(data.info, chpi_amplitudes)
        head_pos = mne.chpi.compute_head_pos(data, 
                                            chpi_locs,
                                            verbose=False)

        data = mne.preprocessing.maxwell_filter(
            data,
            head_pos=head_pos,
            cross_talk=None, # TODO: this should be changed to the real cross_talk file
            calibration=None, # TODO: this should be changed to the real calibration file
        )

        if empty_room_recording:
            empty_room_recording = mne.preprocessing.maxwell_filter_prepare_emptyroom(
                empty_room_recording,
                raw=data,
                bads="keep"
            )

            empty_room_recording = mne.preprocessing.maxwell_filter(
                empty_room_recording,
                head_pos=None, #If array, movement compensation will be performed. 
                cross_talk=None, # TODO: this should be changed to the real cross_talk file
                calibration=None, # TODO: this should be changed to the real calibration file
            )

        # check if cHPI data is available and then apply annotate_movement func
        # TODO: use mne.chpi.extract_chpi_locs_ctf or extract_chpi_locs_kit
        if len(mne.chpi.get_chpi_info(data.infp)):

            chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(data)
            chpi_locs = mne.chpi.compute_chpi_locs(data.info, chpi_amplitudes)
            head_pos = mne.chpi.compute_head_pos(data, 
                                                chpi_locs,
                                                verbose=False)

            movement_annotation, Head_position_over_time = mne.preprocessing.annotate_movement(
                data,
                head_pos=head_pos,
                mean_distance_limit=Head_movement_limit_from_mean
            )

            data.set_annotation(movement_annotation)

            # Calculate the new device head transformation
            new_dev_head_t = mne.preprocessing.compute_average_dev_head_t(data, head_pos)
            data.info["dev_head_t"] = new_dev_head_t

    return data, empty_room_recording



def remove_environmental_noise(data,
                               extention,
                               empty_room_recording=None,
                               ctf_gradient_comp_level=3,
                               apply_environmental_noise_ssp_with_eroom=False,
                               apply_environmental_noise_ica_with_ref_meg=False,
                               environmental_noise_ica_with_ref_meg_thr=2.5,
                               ica_if_reject_by_annotation=True,
                               environmental_noise_ica_with_ref_meg_method="together",
                               environmental_noise_ica_with_ref_meg_measure="zscore"):
        

    # gradient compensation for CTF datasets
    if extention == "CTF":
        data = apply_gradient_comp(data, grade=ctf_gradient_comp_level)

    # If MEGIN device, apply tsss
    elif extention == "fif":
        if not check_tsss(data):
            pass # TODO: to be added

    elif apply_environmental_noise_ssp_with_eroom:        
        if empty_room_recording:
            empty_room_projs = mne.compute_proj_raw(empty_room_recording, n_grad=3, n_mag=3)
            data.add_proj(empty_room_projs)
            data.apply_proj()
            msg = f"Number of detected SSP projectors on Empty_room_recording for removing environmental noise: {len(data.info["projs"])}"
        else:
            msg = "Empty_room_recording is inavailable to perform SSP for environmental noise suppression." \
            " Please, use another method to remove environmental noise."
            logger.info(msg)

    elif apply_environmental_noise_ica_with_ref_meg:

        has_ref_meg = "ref_meg" in data.get_channel_types()
        
        if has_ref_meg:
            all_picks = mne.pick_types(data.info, meg=True, ref_meg=True)
            ica = mne.preprocessing.ICA(n_components=60, max_iter="auto", allow_ref_meg=True)
            ica.fit(data, picks=all_picks)

            bad_ic, scores = ica.find_bads_ref(
                data, 
                threshold = environmental_noise_ica_with_ref_meg_thr,
                reject_by_annotation = ica_if_reject_by_annotation,
                method = environmental_noise_ica_with_ref_meg_method,
                measure = environmental_noise_ica_with_ref_meg_measure)

            data = ica.apply(data, exclude=bad_ic)

            logger.info(f"Number of components, removed by ICA for suppressing environmental noise using ref MEG: {len(bad_ic)}")

    return data
