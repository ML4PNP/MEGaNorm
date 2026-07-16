import os
import mne
import json
import glob
import logging
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, Dict
from scipy.stats import zscore
from kneed import KneeLocator
import matplotlib.pyplot as plt
from gedai.gedai.gedai import (
    Gedai,
)  # TODO: This needs to be changed when meg branch is released
from mne_icalabel import label_components
from meganorm.src.source_localization import check_tsss
from meganorm.utils import data_specific_utils
from gedai.viz import plot_mne_style_overlay_interactive
from meganorm.src.source_localization import corregistration, forward_solution
from autoreject import AutoReject, set_matplotlib_defaults
import autoreject

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

    if np.max(np.abs(corr)) >= auto_ica_corr_thr:
        componentIndx = [int(np.argmax(np.abs(corr)))]
        max_corr = [corr[componentIndx][0]]
    else:
        componentIndx = []
        max_corr = []

    return componentIndx, max_corr


def auto_ica_with_corr(
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
        Name of the physiological sensor ('ecg' or 'eog').
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
        meg=which_sensor.get("meg", False)
        | which_sensor.get("mag", False)
        | which_sensor.get("grad", False),
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
    ica.fit(data, verbose=False, picks=["eeg", "meg", "grad", "mag"])

    # Detect components correlated with physiological signal
    bad_components = []
    max_corrs = []
    for sensor in physiological_signal:

        bad_component, max_corr = find_ica_component(
            ica=ica,
            data=data,
            physiological_signal=sensor,
            auto_ica_corr_thr=auto_ica_corr_thr,
        )
        bad_components.extend(bad_component)
        max_corrs.extend(max_corr)

    if bad_components:
        most_noisy_comp_ind = bad_components[int(np.argmax(np.abs(max_corrs)))]
        logger.info(
            f"In ICA for removing physiological artifacts, component {most_noisy_comp_ind} was removed. "
            f"Its correlation with a {physiological_sensor} channels was: {max_corrs[np.argmax(np.abs(max_corrs))]}",
        )
    else:
        logger.info(
            "In ICA for removing physiological artifacts, no component had a"
            f" high correlation with {physiological_sensor} channels"
        )

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
    ica.fit(data, verbose=False, picks=["eeg", "meg", "mag", "grad"])

    ecg_indices, ecg_scores = ica.find_bads_ecg(
        data, method="correlation", threshold=auto_ica_corr_thr, measure="correlation"
    )

    ica.exclude = ecg_indices
    ica.apply(data, verbose=False)

    if ecg_indices:
        logger.info(
            f"One cardiac-related ICA components was detected and removed by creating synthetic ECG signal. The correlation was: {ecg_scores[ecg_indices]}"
        )

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


def apply_auto_ica_pipeline(
    data,
    channel_types,
    which_sensor,
    n_component,
    ica_max_iter,
    IcaMethod,
    auto_ica_corr_thr,
):
    """
    Apply ICA automatically depending on available physiological channels
    and sensor types (MEG / EEG).


    Parameters
    ----------
    data : mne.raw
        mne.raw data.
    channel_types : list of str
        List of channel type names present in the data (e.g., ``["eeg", "ecg", "eog"]``).
    which_sensor : dict
        Dictionary specifying available sensor modalities.
    n_component : int
        Number of ICA components to compute.
    ica_max_iter : int
        Maximum number of iterations for ICA convergence.
    IcaMethod : str
        ICA algorithm to use (e.g., ``"fastica"``, ``"picard"``).
    auto_ica_corr_thr : float
        Threshold used for correlation-based artifact detection or ICLabel
        classification.

    Returns
    -------
    data : object
        The input data after automatic ICA artifact removal.
    number_of_reduced_ic : int
        Total number of ICA components identified and removed.
    """

    physiological_electrods = {
        channel: channel in channel_types for channel in ["ecg", "eog"]
    }

    ICA_flag = True
    number_of_reduced_ic = 0

    for phys_activity_type, if_elec_exist in physiological_electrods.items():

        # -------- MEG / MAG / GRAD --------
        if (
            which_sensor.get("meg")
            or which_sensor.get("mag")
            or which_sensor.get("grad")
        ):

            if if_elec_exist:
                logger.info(
                    f"Removing {phys_activity_type.upper()} noise using auto_ica_with_corr function."
                )
                data, _, number_of_reduced_ic = auto_ica_with_corr(
                    data=data,
                    n_components=n_component,
                    ica_max_iter=ica_max_iter,
                    IcaMethod=IcaMethod,
                    which_sensor=which_sensor,
                    physiological_sensor=phys_activity_type,
                    auto_ica_corr_thr=auto_ica_corr_thr,
                )

            elif not if_elec_exist and phys_activity_type == "ecg":
                logger.info(
                    f"Removing {phys_activity_type.upper()} noise using auto_ica_with_mean function."
                )
                data, number_of_reduced_ic = auto_ica_with_mean(
                    data=data,
                    n_components=n_component,
                    ica_max_iter=ica_max_iter,
                    IcaMethod=IcaMethod,
                    which_sensor=which_sensor,
                    auto_ica_corr_thr=auto_ica_corr_thr,
                )

        # -------- EEG --------
        if which_sensor.get("eeg"):

            if if_elec_exist:
                logger.info(
                    f"Removing {phys_activity_type.upper()} noise using auto_ica_with_corr function."
                )
                data, ICA_flag, number_of_reduced_ic = auto_ica_with_corr(
                    data=data,
                    n_components=n_component,
                    ica_max_iter=ica_max_iter,
                    IcaMethod=IcaMethod,
                    which_sensor=which_sensor,
                    physiological_sensor=phys_activity_type,
                    auto_ica_corr_thr=auto_ica_corr_thr,
                )

            elif not if_elec_exist and ICA_flag:
                logger.info(
                    f"Removing {phys_activity_type.upper()} noise using AutoIca_with_IcaLabel function."
                )
                data, number_of_reduced_ic = AutoIca_with_IcaLabel(
                    data=data,
                    n_components=n_component,
                    ica_max_iter=ica_max_iter,
                    IcaMethod=IcaMethod,
                    iclabel_thr=auto_ica_corr_thr,
                    physiological_noise_type=phys_activity_type,
                )

    return data, number_of_reduced_ic


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
    which_sensor: dict,
    sampling_rate: float,
    tmin: float = 20,
    tmax: float = -20,
    segments_length: float = 10,
    overlap: float = 0,
    ica_if_reject_by_annotation: bool = True,
    bad_segment_removal_method="fixed_thr",
    mag_var_threshold: float = 5000e-15,
    grad_var_threshold: float = 5000e-13,
    eeg_var_threshold: float = 40e-6,
    mag_flat_threshold: float = 10e-15,
    grad_flat_threshold: float = 10e-13,
    eeg_flat_threshold: float = 40e-6,
    segment_events=None,
):
    """
    Segment continuous MEG/EEG data into fixed-length overlapping epochs.

    This function crops the input raw data to a specified time window, removes
    edge portions to avoid artifacts, and then segments the data into
    fixed-length epochs using MNE-Python. Automatic rejection and flat-channel
    criteria can be applied separately for MEG magnetometers, gradiometers,
    and EEG channels.

    Parameters
    ----------
    data : mne.io.Raw
        Continuous MEG/EEG recording.
    tmin : float
        Start time (in seconds) for cropping the raw data.
    tmax : float
        End time (in seconds) for cropping the raw data. Must be a negative
        value, indicating the offset (in seconds) from the end of the
        recording.
    sampling_rate : float
        Sampling rate of the data in Hz.
    segments_length : float, optional
        Length of each epoch in seconds. Default is 10.
    overlap : float, optional
        Overlap between successive epochs in seconds. Default is 0.
    ica_if_reject_by_annotation : bool, optional
        Whether to reject epochs based on annotations (e.g., ICA-identified
        artifacts). Passed to ``reject_by_annotation`` in ``mne.Epochs``.
        Default is True.
    remove_bad_segments : bool, optional
        Whether to apply amplitude and flatness thresholds to reject bad
        epochs. Default is True.
    mag_var_threshold : float, optional
        Peak-to-peak amplitude threshold for rejecting epochs containing
        artifacts in magnetometer channels (in Tesla). Default is 5000e-15.
    grad_var_threshold : float, optional
        Peak-to-peak amplitude threshold for rejecting epochs containing
        artifacts in gradiometer channels (in Tesla/m). Default is 5000e-13.
    eeg_var_threshold : float, optional
        Peak-to-peak amplitude threshold for rejecting epochs containing
        artifacts in EEG channels (in Volts). Default is 40e-6.
    mag_flat_threshold : float, optional
        Flatness threshold for magnetometer channels (in Tesla). Epochs with
        signals below this threshold are rejected. Default is 10e-15.
    grad_flat_threshold : float, optional
        Flatness threshold for gradiometer channels (in Tesla/m). Default is
        10e-13.
    eeg_flat_threshold : float, optional
        Flatness threshold for EEG channels (in Volts). Default is 40e-6.

    Returns
    -------
    segments : mne.Epochs
        An ``Epochs`` object containing fixed-length segments extracted from
        the continuous data.
    rejection_summary : dict
        A dictionary summarising epoch retention and rejection, with keys:

        - ``total_epochs`` : int, total epochs before rejection.
        - ``retained_epochs`` : int, number of epochs kept.
        - ``discarded_epochs`` : int, number of epochs removed.
        - ``pct_discarded`` : float, percentage of epochs discarded.
        - ``signal_retained_s`` : float, seconds of signal retained.
        - ``signal_total_s`` : float, total seconds before rejection.
        - ``drop_reasons`` : Counter, counts per channel name or annotation
          label that caused rejection. Channel names indicate threshold-based
          rejection; annotation labels (e.g. ``'IGNORED'``) indicate
          annotation-based rejection.

    Raises
    ------
    ValueError
        If ``tmax`` is not a negative number.
    Exception
        If all epochs are rejected, with a summary of drop reasons included
        in the message.
    """
    if tmax > 0:
        raise ValueError("The 'tmax' must be a negative number")

    if bad_segment_removal_method == "fixed_thr":
        # which_sensor["eeg"] is False for MEG-only data
        if not which_sensor["eeg"]:
            ch_types = data.get_channel_types()
            if "mag" in ch_types:
                if "grad" in ch_types:
                    reject = dict(mag=mag_var_threshold, grad=grad_var_threshold)
                    flat = dict(mag=mag_flat_threshold, grad=grad_flat_threshold)
                else:
                    reject = dict(mag=mag_var_threshold)
                    flat = dict(mag=mag_flat_threshold)
            else:
                reject = dict(grad=grad_var_threshold)
                flat = dict(grad=grad_flat_threshold)
        else:
            reject = dict(eeg=eeg_var_threshold)
            flat = dict(eeg=eeg_flat_threshold)
    else:
        reject = None
        flat = None

    tmax = int(np.shape(data.get_data())[1] / sampling_rate + tmax)

    if segment_events is not None:
        events = segment_events.copy()
    else:
        data.crop(tmin=tmin, tmax=tmax)
        events = mne.make_fixed_length_events(
            raw=data,
            duration=segments_length,
            overlap=overlap,
        )
    total_epochs = len(events)

    segments = mne.Epochs(
        data,
        events,
        reject=reject,
        flat=flat,
        reject_by_annotation=ica_if_reject_by_annotation,
        verbose=False,
        tmin=0,
        tmax=segments_length - 1 / sampling_rate,
        baseline=None,
    )

    segments.load_data()

    if bad_segment_removal_method == "fixed_thr":
        retained_epochs = segments.get_data().shape[0]
        discarded_epochs = total_epochs - retained_epochs
        pct_discarded = (
            (discarded_epochs / total_epochs) * 100 if total_epochs > 0 else 0.0
        )

        # drop_log entries are channel names (threshold rejection) or
        # annotation labels; empty tuple means the epoch was kept
        all_reasons = [reason for reasons in segments.drop_log for reason in reasons]
        drop_reasons = Counter(all_reasons)

        rejection_summary = dict(
            total_epochs=total_epochs,
            retained_epochs=retained_epochs,
            discarded_epochs=discarded_epochs,
            pct_discarded=round(pct_discarded, 2),
            signal_retained_s=retained_epochs * segments_length,
            signal_total_s=total_epochs * segments_length,
            drop_reasons=drop_reasons,
        )

        log_msg = (
            f"Epoch rejection summary:\n"
            f"  Total epochs : {total_epochs}\n"
            f"  Retained     : {retained_epochs} "
            f"({100 - pct_discarded:.1f}% | {retained_epochs * segments_length:.1f}s)\n"
            f"  Discarded    : {discarded_epochs} "
            f"({pct_discarded:.1f}% | {discarded_epochs * segments_length:.1f}s)\n"
            f"  Drop reasons : {dict(drop_reasons)}"
        )

        if retained_epochs == 0:
            err_msg = (
                "All epochs were rejected. Every segment was identified as either "
                "noisy or flat. Further processing for this participant cannot proceed.\n"
                + log_msg
            )
            logger.error(err_msg)
            raise Exception(err_msg)

        logger.info(log_msg)

    return segments


def preprocess(
    data,
    device,
    subject,
    freesurfer_dir,
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
    apply_ica_elbow_detection=False,
    apply_oversampled_temporal_projection=True,
    apply_Head_movement_correction=True,
    Head_movement_limit_from_mean=0.0015,
    apply_chpi_filter=False,
    apply_environmental_noise_correction=True,
    ctf_gradient_comp_level=3,
    apply_environmental_noise_ssp_with_eroom=False,
    apply_environmental_noise_ica_with_ref_meg=False,
    environmental_noise_ica_with_ref_meg_thr=2.5,
    ica_if_reject_by_annotation=True,
    environmental_noise_ica_with_ref_meg_method="together",
    environmental_noise_ica_with_ref_meg_measure="zscore",
    apply_gedai=True,
    gedai_method="both",
    sensai_method="optimize",
    conductivity=(0.3,),
    source_space="volumetric",
    gedai_duration=None,
    gedai_overlap=0.5,
    gedai_preliminary_broadband_noise_multiplier=6.0,
    same_environmental_noise_removal=False,
    gedai_noise_multiplier=3.0,
    gedai_wavelet_type="haar",
    gedai_wavelet_level="auto",
    gedai_wavelet_low_cutoff=None,
    gedai_epoch_size_in_cycles=12,
    gedai_highpass_cutoff=0.1,
    source_space_spacing="ico4",
    source_space_spacing_number=4,
    event_record=None,
    event_of_interest=None,
    segments_length=10,
    overlap=5,
):
    """
    Applies a preprocessing pipeline on MEG/EEG data.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG/EEG data.
    device : str
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
    apply_ica_elbow_detection: bool: False


    Returns
    -------
    mne.io.Raw
        Preprocessed MEG/EEG data.
    """
    # since pick_channels can not seperate mag and grad signals
    # if not (which_sensor["meg"] or which_sensor["eeg"]):
    if which_sensor["grad"] or which_sensor["mag"]:
        data, empty_room_recording = drop_mag_or_grad(
            data, empty_room_recording, which_sensor
        )

    channel_types = set(data.get_channel_types())

    # Before resampling, we need to find events
    # TODO: we need to remove this Hard-coded part ASAP. But for now,
    # given that each aston MEG recording is composed of both eyes closed
    # and eyes open, I seperated them like this:
    if "sub-ast_1" in subject:
        data = data_specific_utils._ast_get_rs_block(data, block_index=0)
        events = None
    elif event_record and event_of_interest:
        if device == "MEGIN":
            events = mne.read_events(event_record)
        elif device == "CTF":
            events = mne.find_events(data, stim_channel="UPPT001")
        else:
            events = None # TODO
    else:
        events = None

    # head motion correction ----------------------
    movement_dur = None
    if apply_Head_movement_correction and not which_sensor.get("eeg", False):
        data_temp = data.copy()
        empty_room_recording_temp = (
            empty_room_recording.copy() if empty_room_recording else None
        )

        try:
            data, empty_room_recording, movement_dur = head_motion_correction(
                data_temp,
                empty_room_recording_temp,
                device,
                Head_movement_limit_from_mean=Head_movement_limit_from_mean,
            )
        except Exception as e:
            logger.warning(f"Head motion correction failed: {e}")

    if movement_dur is not None:
        total_dur = data.n_times / data.info["sfreq"]
        usable_dur = total_dur - movement_dur
        step = segments_length - overlap
        needed_dur = 2 * step + segments_length
        if usable_dur < needed_dur:
            msg = (
                f"Only {usable_dur:.1f}s usable after movement annotations; "
                f"need {needed_dur:.1f}s for 3 segments "
                f"(length={segments_length}s, overlap={overlap}s)."
            )
            logger.error(msg)
            raise ValueError(msg)

    # resample -------------------------------------
    sampling_rate = data.info["sfreq"]
    orig_sampling_rate = sampling_rate
    if resampling_rate and resampling_rate != sampling_rate:
        data.resample(int(resampling_rate), verbose=False, n_jobs=-1)
        sampling_rate = resampling_rate
        # resampling empty room recording
        if empty_room_recording:
            empty_room_recording.resample(
                int(resampling_rate), verbose=False, n_jobs=-1
            )

    # flux jumps (SQUID jumps) ---------------------
    if apply_oversampled_temporal_projection and not which_sensor.get("eeg", False):
        data = mne.preprocessing.oversampled_temporal_projection(data)
        msg = "Flux jumps were removed using oversampled temporal projection."
        logger.info(msg)

    # power line -----------------------------------
    nyquist = sampling_rate / 2
    freqs = np.arange(
        int(power_line_freq), 4 * int(power_line_freq) + 1, int(power_line_freq)
    )
    freqs = freqs[freqs <= nyquist]  # keep only valid frequencies

    data.notch_filter(freqs=freqs, n_jobs=-1)

    if empty_room_recording:
        empty_room_recording.notch_filter(freqs=freqs, n_jobs=-1)

    # remove cHPI noise ---------------------------
    try:
        has_chpi = bool(
            mne.chpi.get_chpi_info(data.info, on_missing="ignore")[0].tolist()
        )
    except (KeyError, IndexError):
        has_chpi = False

    if apply_chpi_filter and has_chpi and not which_sensor.get("eeg", False):
        data = mne.chpi.filter_chpi(data, include_line=False)
        logger.info("cHPI filter was applied.")

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

    if apply_gedai:
        data = gedai_preprocess(
            data=data,
            subject=subject,
            freesurfer_dir=freesurfer_dir,
            which_sensor_dict=which_sensor,
            gedai_method=gedai_method,
            sensai_method=sensai_method,
            conductivity=conductivity,
            source_space=source_space,
            gedai_duration=gedai_duration,
            gedai_overlap=gedai_overlap,
            gedai_preliminary_broadband_noise_multiplier=gedai_preliminary_broadband_noise_multiplier,
            gedai_noise_multiplier=gedai_noise_multiplier,
            gedai_wavelet_type=gedai_wavelet_type,
            gedai_wavelet_level=gedai_wavelet_level,
            gedai_wavelet_low_cutoff=gedai_wavelet_low_cutoff,
            gedai_epoch_size_in_cycles=gedai_epoch_size_in_cycles,
            gedai_highpass_cutoff=gedai_highpass_cutoff,
            source_space_spacing=source_space_spacing,
            source_space_spacing_number=source_space_spacing_number,
        )

    # Muscle artifact detection ---------------------
    if cutoffFreqHigh > muscle_activity_filter_freq[0]:
        muscle_annot, _ = mne.preprocessing.annotate_muscle_zscore(
            data,
            min_length_good=muscle_activity_min_length_good,
            filter_freq=muscle_activity_filter_freq,
            threshold=muscle_activity_thr,
        )
        # ICA will ignore these and later will be removed in segmentation
        data.set_annotations(data.annotations + muscle_annot)
        logger.info(
            f"Muscle artifact rejection alg removed {sum(muscle_annot.duration)} seconds of"
            " the signal."
        )

    # rereference -----------------------------------
    if which_sensor["eeg"] and rereference_method:
        data = data.set_eeg_reference(rereference_method)
        if empty_room_recording:
            empty_room_recording = empty_room_recording.set_eeg_reference(
                rereference_method
            )

    # remove environmental noise ---------------------
    if apply_environmental_noise_correction:
        data, empty_room_recording = remove_environmental_noise(
            data,
            device,
            empty_room_recording=empty_room_recording,
            ctf_gradient_comp_level=ctf_gradient_comp_level,
            apply_environmental_noise_ssp_with_eroom=apply_environmental_noise_ssp_with_eroom,
            apply_environmental_noise_ica_with_ref_meg=apply_environmental_noise_ica_with_ref_meg,
            environmental_noise_ica_with_ref_meg_thr=environmental_noise_ica_with_ref_meg_thr,
            ica_if_reject_by_annotation=ica_if_reject_by_annotation,
            environmental_noise_ica_with_ref_meg_method=environmental_noise_ica_with_ref_meg_method,
            environmental_noise_ica_with_ref_meg_measure=environmental_noise_ica_with_ref_meg_measure,
            same_environmental_noise_removal=same_environmental_noise_removal,
        )

    # Remove unwanted epochs associated with some events
    if events is not None:
        logger.info(f"Trial rejection started; event of interest: {event_of_interest}")
        scale = sampling_rate / orig_sampling_rate
        if scale != 1:
            events = events.copy()
            events[:, 0] = np.round(events[:, 0] * scale).astype(int)

        data, segment_events = extract_rs_blocks(
            raw=data,
            events=events,
            rs_id=event_of_interest,
            sampling_rate=sampling_rate,
            segments_length=segments_length,
            overlap=overlap,
        )
    else:
        segment_events = None

    # physiological noise ----------------------------
    if apply_ica:
        if apply_ica_elbow_detection:
            n_component = pca_elbow_locator(data, which_sensor)

        data, number_of_reduced_ic = apply_auto_ica_pipeline(
            data,
            channel_types,
            which_sensor,
            n_component,
            ica_max_iter,
            IcaMethod,
            auto_ica_corr_thr,
        )
    else:
        number_of_reduced_ic = 0

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

    logger.info("Preprocessing is finished.")
    return (
        data,
        data.info["ch_names"],
        int(sampling_rate),
        empty_room_recording,
        number_of_reduced_ic,
        segment_events,
    )


def drop_noisy_meg_channels(
    data: Any,
    subID: str,
    args: Any,
    device: str,
    which_sensor,
    empty_room_recording=None,
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

    which_sensor : dict
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

    if check_tsss(data):
        msg = (
            "Maxwell filter has already been applied. "
            "Therefore, bad channel detection using maxwell will be not applied."
        )
        logger.info(msg)
        auto_noisy_chs = []
        auto_flat_chs = []

    else:
        if device == "CTF":
            data.apply_gradient_compensation(0)

        auto_noisy_chs, auto_flat_chs = mne.preprocessing.find_bad_channels_maxwell(
            data, return_scores=False, verbose=True, coord_frame="meg", ignore_ref=True
        )
        data.info["bads"] += auto_noisy_chs + auto_flat_chs
        if empty_room_recording:
            data.info["bads"] += empty_room_recording.info["bads"]

        logger.warning(
            f"Number of noisy channels that were droped from the subject's recording: {len(auto_noisy_chs)}"
        )
        logger.warning(
            f"Number of flat channels that were droped from the subject's recording: {len(auto_flat_chs)}"
        )

    bads = data.info["bads"][:]
    data.drop_channels(bads)
    if empty_room_recording:
        empty_room_recording.drop_channels(bads)

    return data, empty_room_recording


def apply_chpi(meg_data, movement_limit, head_pos_save_path, device):
    """
    Estimate and save continuous head position from cHPI coils.

    Computes cHPI coil amplitudes and locations (using the appropriate
    method for MEGIN vs. CTF systems), derives head position over
    time, annotates excessive movement relative to `movement_limit`,
    and writes the head position estimates to disk.

    Parameters
    ----------
    meg_data : mne.io.Raw
        Raw MEG data containing cHPI coil information.
    movement_limit : float
        Mean distance limit (in meters) used to annotate periods of
        excessive head movement.
    head_pos_save_path : str
        Path where the computed head position data will be written.
    device : {"fif", "ds"}
        Recording system type, used to select the appropriate cHPI
        location extraction method.

    Returns
    -------
    None
    """

    if meg_data.info["hpi_results"]:

        # BTi/4D MEG recordings do not support cHPI and don't have
        # real time recordings  of the brain pos
        if device == "fif":
            amp = mne.chpi.compute_chpi_amplitudes(meg_data)
            locs = mne.chpi.compute_chpi_locs(meg_data.info, amp)

        if device == "ds":
            locs = mne.chpi.extract_chpi_locs_ctf(meg_data)

        head_pos = mne.chpi.compute_head_pos(meg_data.info, locs)

        if list(head_pos):
            movement_annot = mne.preprocessing.annotate_movement(
                meg_data, pos=head_pos, mean_distance_limit=movement_limit
            )

            moved_times = sum(movement_annot.duration)
            moved_inteval_percentage = moved_times / meg_data.duration[-1] * 100

            logger.warning(
                f"{moved_inteval_percentage} percent of the recording exceeds the mean distance"
                "limit for the head motion. Consider using tSSS."
            )
        else:
            logger.info("Unable to find a reliable solution for any of the coils")

    else:
        logger.info("No cHPI coil was found for the current subject.")

    mne.chpi.write_head_pos(head_pos_save_path, head_pos)


def apply_gradient_comp(ctf_meg_data, empty_room_recording=None, grade=3):
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
    ctf_meg_data.interpolate_bads(reset_bads=False, method=method)

    ctf_meg_data.apply_gradient_compensation(grade=grade)
    if empty_room_recording:
        empty_room_recording.apply_gradient_compensation(grade=grade)

    logger.info(
        f"Gradient compensation with level of {grade} has been applied to the data."
    )

    return ctf_meg_data, empty_room_recording


def apply_tsss(
    data,
    cross_talk_path,
    calibration_path,
    head_pos_path=None,
    empty_room_record=None,
    st_duration=10.0,
    st_correlation=0.98,
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
        head_pos=head_pos,
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
            head_pos=head_pos,
        )

    return data_tsss, empty_room_record


def drop_noisy_segments(segments, z_thr):
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

    z_scores = zscore(np.std(segments._data, axis=2), axis=0)

    bad_segments = np.where(z_scores > z_thr)[0]

    segments = segments.drop(indices=bad_segments)

    logger.info(
        f"Dropping {len(bad_segments)} segments due to Z > Z_threshold. "
        f"The final number of used segments: {segments.__len__()}"
    )

    return segments


def pca_elbow_locator(raw, which_sensor):
    """
    Estimate the number of ICA/PCA components to retain using the
    explained-variance elbow of a PCA decomposition.

    Selects MEG and/or EEG channels per `which_sensor`, fits a PCA on
    the resulting data, and locates the "elbow" of the explained
    variance ratio curve using the Kneedle algorithm.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG/EEG data.
    which_sensor : dict
        Dictionary indicating which sensor types to include (e.g.,
        {'meg': True, 'mag': True, 'grad': True, 'eeg': False}).

    Returns
    -------
    int
        The estimated number of components at the elbow of the
        explained variance curve, with a minimum enforced value of 30.
    """

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
        direction="decreasing",
    )

    elbow_index = knee_locator.knee

    # Handle None and enforce min
    if elbow_index is None or elbow_index < 15:
        elbow_index = 30

    return int(elbow_index)


def drop_mag_or_grad(data, empty_room_recording, which_sensor):
    """
    Drop either magnetometer or gradiometer channels, keeping only one
    MEG channel type.

    If `which_sensor['grad']` is True, magnetometer channels are
    dropped; if `which_sensor['mag']` is True, gradiometer channels
    are dropped. Applied to both `data` and, if provided,
    `empty_room_recording`.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG data containing both magnetometer and gradiometer
        channels.
    empty_room_recording : mne.io.Raw or None
        Corresponding empty-room recording, or None.
    which_sensor : dict
        Dictionary indicating which single MEG channel type to retain
        via keys 'mag' and 'grad'.

    Returns
    -------
    data : mne.io.Raw
        Data with the unwanted MEG channel type removed.
    empty_room_recording : mne.io.Raw or None
        Empty-room recording with the same channels removed, or None
        if not provided.
    """

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

    data.drop_channels(dropping_channels)
    if empty_room_recording:
        empty_room_recording.drop_channels(dropping_channels)

    return data, empty_room_recording


def _chpi_usable(data, device):
    """
    Check whether continuous head-position tracking can be estimated
    from the data for the given device.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG data to check for usable cHPI information.
    device : {"MEGIN", "CTF", "BTI"}
        Recording system type.

    Returns
    -------
    bool
        True if cHPI information usable for head-position estimation
        is present, False otherwise.
    """
    if device == "MEGIN":
        try:
            hpi_freqs, _, _ = mne.chpi.get_chpi_info(data.info, on_missing="ignore")
        except (KeyError, IndexError, ValueError):
            return False
        if hpi_freqs is None or len(hpi_freqs) == 0:
            return False
        return True

    elif device == "CTF":
        hlc = [ch for ch in data.ch_names if ch.startswith("HLC")]
        return len(hlc) > 0

    elif device == "BTI":
        # no channel-name signature; just try the extractor
        try:
            mne.chpi.extract_chpi_locs_kit(data, verbose=False)
            return True
        except Exception:
            return False

    return False


def head_motion_correction(
    data, empty_room_recording, device, Head_movement_limit_from_mean=0.0015
):
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

    device : str
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
    movement_dur = None

    if check_tsss(data):
        msg = "Head motion correction was not applied since tSSS has already been applied to the data."
        logger.info(msg)
        return data, empty_room_recording, movement_dur

    # MEGIN devicesEpoch
    if device == "MEGIN" and _chpi_usable(data, device=device):

        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(data)
        chpi_locs = mne.chpi.compute_chpi_locs(data.info, chpi_amplitudes)
        head_pos = mne.chpi.compute_head_pos(data.info, chpi_locs, verbose=False)

        data = mne.preprocessing.maxwell_filter(
            data,
            head_pos=head_pos,
            cross_talk=None,  # TODO: this should be changed to the real cross_talk file
            calibration=None,  # TODO: this should be changed to the real calibration file
        )

        logger.info(
            "Movement compensation was applied for the subject using maxwell filter."
        )

        if empty_room_recording:
            empty_room_recording = mne.preprocessing.maxwell_filter_prepare_emptyroom(
                empty_room_recording, raw=data, bads="keep"
            )

            empty_room_recording = mne.preprocessing.maxwell_filter(
                empty_room_recording,
                head_pos=head_pos,  # head_pos must match the rs-data
                cross_talk=None,  # TODO: this should be changed to the real cross_talk file
                calibration=None,  # TODO: this should be changed to the real calibration file
            )

    # TODO, expand this if new device comes in!
    elif device in ["CTF", "BTI"] and _chpi_usable(data, device=device):
        if device == "CTF":
            chpi_locs = mne.chpi.extract_chpi_locs_ctf(data, verbose=False)

        elif device == "BTI":
            chpi_locs = mne.chpi.extract_chpi_locs_kit(data, verbose=False)

        else:
            chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(data)
            chpi_locs = mne.chpi.compute_chpi_locs(data.info, chpi_amplitudes)

        head_pos = mne.chpi.compute_head_pos(data.info, chpi_locs, verbose=False)

        movement_annotation, Head_position_over_time = (
            mne.preprocessing.annotate_movement(
                data,
                pos=head_pos,
                mean_distance_limit=Head_movement_limit_from_mean,
            )
        )

        data.set_annotations(movement_annotation)
        movement_dur = sum(movement_annotation.duration)
        logger.info(
            f"Movement annotation algorithm using cHPI coils detected {movement_dur}"
            " seconds of motion."
        )

        # Calculate the new device head transformation
        new_dev_head_t = mne.preprocessing.compute_average_dev_head_t(
            data, head_pos, verbose=False
        )
        data.info["dev_head_t"] = new_dev_head_t

    else:
        logger.info("Movemet correction was not done for the subject.")

    return data, empty_room_recording, movement_dur


def remove_environmental_noise(
    data,
    device,
    empty_room_recording=None,
    ctf_gradient_comp_level=3,
    apply_environmental_noise_ssp_with_eroom=False,
    apply_environmental_noise_ica_with_ref_meg=False,
    environmental_noise_ica_with_ref_meg_thr=2.5,
    ica_if_reject_by_annotation=True,
    environmental_noise_ica_with_ref_meg_method="together",
    environmental_noise_ica_with_ref_meg_measure="zscore",
    same_environmental_noise_removal=False,
):
    """
    Suppress environmental (external) noise using a device-appropriate
    strategy.

    For CTF data, applies gradient compensation. For MEGIN data,
    relies on tSSS if already applied. Otherwise, environmental noise
    can be suppressed via SSP projectors computed from an empty-room
    recording, or via reference-MEG-based ICA.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG/EEG data to clean.
    device : {"CTF", "MEGIN", ...}
        Recording system type, determining the default noise-removal
        strategy.
    empty_room_recording : mne.io.Raw, optional
        Empty-room recording used for gradient compensation or SSP
        projector estimation.
    ctf_gradient_comp_level : int, optional
        Gradient compensation grade to apply for CTF data. Default is 3.
    apply_environmental_noise_ssp_with_eroom : bool, optional
        Whether to compute and apply SSP projectors from the empty-room
        recording. Default is False.
    apply_environmental_noise_ica_with_ref_meg : bool, optional
        Whether to use reference-MEG-based ICA to remove environmental
        noise. Default is False.
    environmental_noise_ica_with_ref_meg_thr : float, optional
        Threshold used by `find_bads_ref` for identifying bad
        components. Default is 2.5.
    ica_if_reject_by_annotation : bool, optional
        Whether to reject data by annotation when fitting the
        reference-MEG ICA. Default is True.
    environmental_noise_ica_with_ref_meg_method : {"together", "separate"}, optional
        Strategy for combining reference-MEG and data channels during
        ICA-based artifact detection. Default is "together".
    environmental_noise_ica_with_ref_meg_measure : str, optional
        Scoring measure used by `find_bads_ref`. Default is "zscore".

    Returns
    -------
    data : mne.io.Raw
        Data with environmental noise suppressed.
    empty_room_recording : mne.io.Raw or None
        Empty-room recording, updated if gradient compensation was
        applied.
    """
    # gradient compensation for CTF datasets
    if device == "CTF" and not same_environmental_noise_removal:
        data, empty_room_recording = apply_gradient_comp(
            data,
            empty_room_recording=empty_room_recording,
            grade=ctf_gradient_comp_level,
        )
        msg = "The data was preprocessed for environmental noise using gradient compensation."
        logger.info(msg)

    # If MEGIN device, apply tsss
    elif device == "MEGIN" and not same_environmental_noise_removal:
        if not check_tsss(data):
            pass  # TODO: to be added
        else:
            msg = "The data has already been preprocessed for environmental noise using tSSS."
            logger.info(msg)

    elif apply_environmental_noise_ssp_with_eroom:
        if empty_room_recording:
            empty_room_projs = mne.compute_proj_raw(
                empty_room_recording, n_grad=3, n_mag=3
            )
            data.add_proj(empty_room_projs)
            data.apply_proj()
            msg = f"Number of detected SSP projectors on Empty_room_recording for removing environmental noise: {len(data.info['projs'])}"
        else:
            msg = (
                "Empty_room_recording is inavailable to perform SSP for environmental noise suppression."
                " Please, use another method to remove environmental noise."
            )
            logger.info(msg)

    elif apply_environmental_noise_ica_with_ref_meg:

        has_ref_meg = "ref_meg" in data.get_channel_types()
        if has_ref_meg:
            data, bad_ic, scores = find_ref_meg_artifact(
                data,
                environmental_noise_ica_with_ref_meg_thr=environmental_noise_ica_with_ref_meg_thr,
                ica_if_reject_by_annotation=ica_if_reject_by_annotation,
                environmental_noise_ica_with_ref_meg_method=environmental_noise_ica_with_ref_meg_method,
                environmental_noise_ica_with_ref_meg_measure=environmental_noise_ica_with_ref_meg_measure,
            )

            logger.info(
                "Number of components removed by ICA for suppressing environmental noise using reference MEG: %d",
                len(bad_ic),
            )

    return data, empty_room_recording


def find_ref_meg_artifact(
    data,
    environmental_noise_ica_with_ref_meg_thr,
    ica_if_reject_by_annotation=True,
    environmental_noise_ica_with_ref_meg_method="together",
    environmental_noise_ica_with_ref_meg_measure="zscore",
):
    """
    Identify and remove environmental-noise ICA components using
    reference MEG channels.

    Fits ICA jointly on MEG and reference-MEG channels (or separately,
    depending on `environmental_noise_ica_with_ref_meg_method`) and
    uses `ICA.find_bads_ref` to detect components correlated with
    reference-channel activity.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG data containing reference MEG channels.
    environmental_noise_ica_with_ref_meg_thr : float
        Threshold passed to `find_bads_ref` for flagging bad
        components.
    ica_if_reject_by_annotation : bool, optional
        Whether to reject data by annotation during ICA fitting.
        Default is True.
    environmental_noise_ica_with_ref_meg_method : {"together", "separate"}, optional
        If "together", ICA is fit jointly on MEG and reference
        channels. If "separate", a separate ICA is fit on reference
        channels and its sources are added to the data before
        artifact detection. Default is "together".
    environmental_noise_ica_with_ref_meg_measure : str, optional
        Scoring measure used by `find_bads_ref`. Default is "zscore".

    Returns
    -------
    data : mne.io.Raw
        Data with identified environmental-noise components removed.
    bad_comps : list of int
        Indices of ICA components excluded as environmental noise.
    scores : ndarray
        Scores computed by `find_bads_ref` for each component.
    """
    data_tog = data.copy()

    all_picks = mne.pick_types(data_tog.info, meg=True, ref_meg=True)
    tog_ica = mne.preprocessing.ICA(
        n_components=20, max_iter="auto", allow_ref_meg=True
    )
    tog_ica.fit(data_tog, picks=all_picks)
    bad_comps, scores = tog_ica.find_bads_ref(
        data_tog,
        reject_by_annotation=ica_if_reject_by_annotation,
        method="together",
        threshold=environmental_noise_ica_with_ref_meg_thr,
        measure=environmental_noise_ica_with_ref_meg_measure,
    )

    if environmental_noise_ica_with_ref_meg_method == "separate":

        data_sep = data.copy()
        ref_picks = mne.pick_types(data_sep.info, meg=False, ref_meg=True)
        ref_ica = mne.preprocessing.ICA(
            n_components=2, max_iter="auto", allow_ref_meg=True
        )
        ref_ica.fit(data_sep, picks=ref_picks)

        ica_sep = tog_ica.copy()
        ref_comps = ref_ica.get_sources(data_sep)
        for ic in ref_comps.ch_names:
            ref_comps.rename_channels({ic: "REF_ICA" + ic})
        data_sep.add_channels([ref_comps])

        bad_comps, scores = ica_sep.find_bads_ref(
            data_sep,
            method="separate",
        )

        data = ica_sep.apply(data_sep, exclude=bad_comps)

    else:
        data = tog_ica.apply(data_tog, exclude=bad_comps)

        # TODO: data_clean.drop_channels(ref_comps.ch_names)

    return data, bad_comps, scores


def _validate_gedai_params(method, wavelet_level, duration, broadband_multiplier):
    """
    Validate parameter combinations for the GEDAI preprocessing method.

    Parameters
    ----------
    method : {"broadband", "spectral", "both"}
        GEDAI artifact removal strategy.
    wavelet_level : int or "auto"
        Number of wavelet decomposition levels.
    duration : float or None
        Segment duration required for broadband suppression.
    broadband_multiplier : float or None
        Noise multiplier required for the preliminary broadband pass
        when `method` is "both".

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the parameter combination is invalid for the chosen method.
    """
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


def _gedai_clean_sensor_type(data, signal_type, fwd, gedai_params, plot=False):
    """
    Apply GEDAI artifact suppression to a single sensor type.

    Selects channels of the given `signal_type`, fits and applies the
    GEDAI algorithm using the supplied forward solution and
    parameters, and optionally plots the fit diagnostics and an
    overlay of cleaned vs. original signal.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG/EEG data containing the sensor type to clean.
    signal_type : {"mag", "grad", "eeg"}
        Sensor type to isolate and clean.
    fwd : mne.Forward or str
        Forward solution (leadfield) used as the reference covariance
        for GEDAI, or a placeholder string if unavailable.
    gedai_params : dict
        Dictionary of GEDAI configuration parameters (wavelet type,
        wavelet level, cutoV frequencies, duration, overlap, noise
        multipliers, etc.).
    plot : bool, optional
        If True, display GEDAI fit diagnostics and an interactive
        before/after overlay. Default is False.

    Returns
    -------
    mne.io.Raw
        Cleaned data restricted to the given sensor type.
    """
    temp_data = data.copy()
    if signal_type in ["mag", "grad"]:
        temp_data.pick_types(meg=signal_type)
    elif signal_type == "eeg":
        temp_data.pick_types(eeg=True)

    logger.info(
        f"Applying GEDAI on {signal_type} signals; number of channels: {temp_data.get_data().shape}"
    )
    gedai = Gedai(
        wavelet_type=gedai_params["wavelet_type"],  # Default
        wavelet_level=gedai_params["wavelet_level"],  # TODO
        wavelet_low_cutoff=gedai_params[
            "wavelet_low_cutoff"
        ],  # This should be set to lower cutoff frequency band in the highpass filter
        epoch_size_in_cycles=gedai_params[
            "epoch_size_in_cycles"
        ],  # 12 is the default for their matlab code, this ensures at least 12 cycles per frequency range
        signal_type="auto",  # default
        highpass_cutoff=gedai_params["highpass_cutoff"],  # default
        preliminary_broadband_noise_multiplier=gedai_params[
            "preliminary_broadband_noise_multiplier"
        ],
    )

    gedai.fit_raw(
        temp_data,
        duration=gedai_params["duration"],
        overlap=gedai_params["overlap"],
        reference_cov=fwd,
        sensai_method=gedai_params["sensai_method"],
        noise_multiplier=gedai_params["noise_multiplier"],
        verbose=False,
        reject_by_annotation=True,
        n_jobs=-1,
    )

    data_corrected = gedai.transform_raw(
        temp_data,
        duration=gedai_params["duration"],
        overlap=gedai_params["overlap"],
        verbose=False,
    )
    logger.info(f"GEDAI was successfuly applied on the {signal_type} signals")

    if plot:
        data_viz = data.copy()
        fig = gedai.plot_fit()
        plt.show()
        plot_mne_style_overlay_interactive(temp_data, data_corrected, duration=10)
        plt.show()

    return data_corrected


def gedai_preprocess(
    data,
    subject,
    freesurfer_dir,
    which_sensor_dict,
    gedai_method="both",
    sensai_method="optimize",
    conductivity=(0.3,),
    source_space="volumetric",
    gedai_duration=None,
    gedai_overlap=0.5,
    gedai_preliminary_broadband_noise_multiplier=6.0,
    gedai_noise_multiplier=3.0,
    gedai_wavelet_type="haar",
    gedai_wavelet_level="auto",
    gedai_wavelet_low_cutoff=None,
    gedai_epoch_size_in_cycles=12,
    gedai_highpass_cutoff=0.1,
    source_space_spacing="ico4",
    source_space_spacing_number=4,
    plot=False,
):
    """
    Preprocess MEG/EEG data using GEDAI artifact removal.

    Performs coregistration, computes a forward solution, and applies GEDAI
    artifact suppression separately for each sensor type (magnetometers,
    gradiometers, EEG). Non-MEG/EEG channels are preserved and recombined
    in the output.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG/EEG recording to be cleaned.
    subject : str
        Subject identifier, must match the corresponding FreeSurfer subject
        directory name.
    freesurfer_dir : str or path-like
        Path to the FreeSurfer subjects directory. If None, a template will be used.
    which_sensor_dict : dict
        Dictionary specifying which sensor types to include in the forward
        solution.
    gedai_method : {"both", "broadband", "spectral"}, optional
        GEDAI artifact removal strategy. "broadband" applies suppression
        across the full frequency range (requires wavelet_level=0 and
        gedai_duration). "spectral" applies suppression per frequency band
        (requires wavelet_level > 0). "both" runs a preliminary broadband
        pass followed by spectral suppression. Default is "both".
    sensai_method : str, optional
        Method used by SensAI for noise estimation. Options are:
        "optimize" and "gridsearch". Default is "optimize".
    conductivity : tuple of float, optional
        Conductivity values (in S/m) for the BEM layers. Use a 1-tuple for
        a single-shell model (MEG only) or a 3-tuple for a three-layer model
        (MEG+EEG). Default is (0.3,).
    source_space : {"volumetric", "surface"}, optional
        Type of source space to use for the forward solution.
        Default is "volumetric".
    gedai_duration : float or None, optional
        Duration (in seconds) of each data segment used during fitting.
        Required when gedai_method is "broadband". Default is None.
    gedai_overlap : float, optional
        Fractional overlap between consecutive segments, between 0 and 1.
        Default is 0.5.
    gedai_preliminary_broadband_noise_multiplier : float, optional
        Noise multiplier for the preliminary broadband suppression pass when
        gedai_method is "both". Default is 6.0.
    gedai_noise_multiplier : float, optional
        Noise multiplier threshold for the main GEDAI suppression step.
        Default is 3.0.
    gedai_wavelet_type : str, optional
        Wavelet family to use for the spectral decomposition. Default is
        "haar".
    gedai_wavelet_level : int or "auto", optional
        Number of wavelet decomposition levels. Set to 0 for broadband mode,
        or "auto" to determine the level automatically. Default is "auto".
    gedai_wavelet_low_cutoff : float or None, optional
        Lower cutoff frequency (in Hz) for the wavelet decomposition. Should
        match the highpass filter cutoff. Default is None.
    gedai_epoch_size_in_cycles : int, optional
        Minimum number of cycles per frequency band used to determine epoch
        length. Default is 12.
    gedai_highpass_cutoff : float, optional
        Highpass filter cutoff frequency (in Hz) applied before GEDAI fitting.
        Default is 0.1.
    source_space_spacing : str, optional
        Spacing parameter for surface source spaces (e.g. "ico4").
        Default is "ico4".
    source_space_spacing_number : int, optional
        Numeric spacing value corresponding to source_space_spacing.
        Default is 4.
    plot : bool, optional
        If True, displays GEDAI fit diagnostics and an interactive overlay
        of the cleaned vs. original signal for each sensor type.
        Default is False.

    Returns
    -------
    mne.io.Raw
        Cleaned raw recording with MEG/EEG channels replaced by their
        GEDAI-suppressed counterparts. All other channels are unchanged.
    """
    logger.info("Preprocessing the data using the gedai algorithm.")

    _validate_gedai_params(
        gedai_method,
        gedai_wavelet_level,
        gedai_duration,
        gedai_preliminary_broadband_noise_multiplier,
    )

    if freesurfer_dir:
        transformation_matrix = corregistration(
            data,
            subject=subject,
            subjects_dir=freesurfer_dir,
            plot_3d=False,
        )

        fwd, _ = forward_solution(
            subject=subject,
            subjects_dir=freesurfer_dir,
            data=data,
            transformation_matrix=transformation_matrix.trans,
            conductivity=conductivity,
            source_space=source_space,
            which_sensor_dict=which_sensor_dict,
            source_space_spacing=source_space_spacing,
            source_space_spacing_number=source_space_spacing_number,
        )
    else:
        fwd = "Leadfield"

    gedai_params = {
        "wavelet_type": gedai_wavelet_type,
        "wavelet_level": gedai_wavelet_level,
        "wavelet_low_cutoff": gedai_wavelet_low_cutoff,
        "epoch_size_in_cycles": gedai_epoch_size_in_cycles,
        "highpass_cutoff": gedai_highpass_cutoff,
        "preliminary_broadband_noise_multiplier": gedai_preliminary_broadband_noise_multiplier,
        "duration": gedai_duration,
        "overlap": gedai_overlap,
        "sensai_method": sensai_method,
        "noise_multiplier": gedai_noise_multiplier,
    }

    meg_eeg_chs = mne.pick_types(data.info, meg=True, eeg=True, ref_meg=False)
    other_signals = data.copy()
    if len(meg_eeg_chs) != len(data.ch_names):
        other_signals.drop_channels([data.ch_names[i] for i in meg_eeg_chs])
    else:
        other_signals = None
    # TODO: meg=True should be changed for EEG as well
    data.pick_types(meg=True, eeg=which_sensor_dict["eeg"], ref_meg=False)
    sensor_types_of_interest = np.unique(data.get_channel_types()).tolist()

    logger.info(
        f"Detected signal types for the GEDAI algorithms are: {sensor_types_of_interest}"
    )
    cleaned_signals = [
        _gedai_clean_sensor_type(data, signal_type, fwd, gedai_params, plot=plot)
        for signal_type in sensor_types_of_interest
    ]

    if other_signals is not None:
        return other_signals.add_channels(cleaned_signals, force_update_info=True)
    elif len(cleaned_signals) == 1:
        return cleaned_signals[0]
    else:
        return cleaned_signals[0].add_channels(
            cleaned_signals[1:], force_update_info=True
        )


def annotate_noisy_raw(raw, reject=None, flat=None, window=1.0, step=0.5):
    """
    Annotate noisy or flat segments of raw data using a sliding-window
    peak-to-peak amplitude criterion.

    Slides a fixed-length window across the recording and flags
    windows where any channel's peak-to-peak amplitude exceeds a
    rejection threshold ('BAD_peak') or falls below a flatness
    threshold ('BAD_flat'), separately for each specified channel
    type.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG/EEG data to scan for artifacts.
    reject : dict or None, optional
        Peak-to-peak amplitude rejection thresholds per channel type
        (e.g., {'mag': 5000e-15, 'eeg': 40e-6}). If None, peak
        rejection is skipped.
    flat : dict or None, optional
        Peak-to-peak amplitude flatness thresholds per channel type.
        If None, flatness rejection is skipped.
    window : float, optional
        Window length in seconds used to evaluate each segment.
        Default is 1.0.
    step : float, optional
        Step size in seconds between successive windows. Default is 0.5.

    Returns
    -------
    mne.Annotations
        Annotations marking 'BAD_peak' and 'BAD_flat' segments. Empty
        if both `reject` and `flat` are None.
    """
    if reject is None and flat is None:
        return mne.Annotations(onset=[], duration=[], description=[])

    sfreq = raw.info["sfreq"]
    win_samples = int(window * sfreq)
    step_samples = int(step * sfreq)
    n_samples = len(raw.times)  # fixed

    ch_types_in_raw = set(raw.get_channel_types())
    types_to_check = set()
    if reject:
        types_to_check.update(k for k in reject if k in ch_types_in_raw)
    if flat:
        types_to_check.update(k for k in flat if k in ch_types_in_raw)

    type_data = {}
    for ch_type in types_to_check:
        if ch_type in ("mag", "grad"):
            picks = mne.pick_types(raw.info, meg=ch_type, ref_meg=False, exclude="bads")
        elif ch_type == "eeg":
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(picks) == 0:
            continue
        type_data[ch_type] = raw.get_data(picks=picks)

    times = raw.times
    n_samples = raw._data.shape[1] if raw.preload else len(times)

    bad_peak_onsets = []
    bad_flat_onsets = []

    i = 0
    while i + win_samples <= n_samples:
        t_onset = times[i]
        is_peak_bad = False
        is_flat_bad = False

        for ch_type, data_arr in type_data.items():
            segment = data_arr[:, i : i + win_samples]
            ptp = np.ptp(segment, axis=1)  # peak-to-peak per channel, shape (n_ch,)

            if reject and ch_type in reject:
                if np.any(ptp > reject[ch_type]):
                    print(ptp)
                    is_peak_bad = True

            if flat and ch_type in flat:
                if np.any(ptp < flat[ch_type]):
                    print("no")
                    is_flat_bad = True

        if is_peak_bad:
            bad_peak_onsets.append(t_onset)
        elif is_flat_bad:
            bad_flat_onsets.append(t_onset)

        i += step_samples

    # Build annotations
    onsets = bad_peak_onsets + bad_flat_onsets
    durations = [window] * len(onsets)
    descriptions = ["BAD_peak"] * len(bad_peak_onsets) + ["BAD_flat"] * len(
        bad_flat_onsets
    )

    return mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=raw.info["meas_date"],
    )


def auto_reject_segmentation(
    raw,
    sampling_rate: float,
    tmin: float = 20,
    tmax: float = -20,
    segments_length: float = 10,
    overlap: float = 0,
    ica_if_reject_by_annotation: bool = True,
    n_interpolates=np.array([1, 4, 8, 16, 32]),
    consensus_percs=np.linspace(0, 1.0, 11),
    cv="auto",
    thresh_method="bayesian_optimization",
    random_state=42,
    segment_events=None,
):
    """
    Segment continuous data into fixed-length epochs and clean them
    using AutoReject.

    Crops the raw data (or uses precomputed segment events), builds
    fixed-length epochs, and fits AutoReject to automatically
    interpolate or reject noisy epochs.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous MEG/EEG recording.
    sampling_rate : float
        Sampling rate of the data in Hz.
    tmin : float, optional
        Start time (in seconds) for cropping the raw data when
        `segment_events` is None. Default is 20.
    tmax : float, optional
        End time offset (in seconds, must be negative) from the end of
        the recording, used for cropping when `segment_events` is
        None. Default is -20.
    segments_length : float, optional
        Length of each epoch in seconds. Default is 10.
    overlap : float, optional
        Overlap between successive fixed-length events in seconds.
        Default is 0.
    ica_if_reject_by_annotation : bool, optional
        Whether to reject data by annotation when building epochs.
        Default is True.
    n_interpolates : ndarray, optional
        Candidate numbers of channels to interpolate, passed to
        AutoReject. Default is [1, 4, 8, 16, 32].
    consensus_percs : ndarray, optional
        Candidate consensus percentages, passed to AutoReject. Default
        is `np.linspace(0, 1.0, 11)`.
    cv : int or "auto", optional
        Number of cross-validation folds for AutoReject. If "auto",
        set based on the number of epochs (clamped between 2 and 10).
        Default is "auto".
    thresh_method : str, optional
        Threshold optimization method used by AutoReject. Default is
        "bayesian_optimization".
    random_state : int, optional
        Random seed for AutoReject. Default is 42.
    segment_events : ndarray or None, optional
        Precomputed MNE-style events array defining epoch onsets. If
        provided, `tmin`/`tmax` cropping is skipped.

    Returns
    -------
    epochs_clean : mne.Epochs
        Epochs after AutoReject interpolation/rejection.
    reject_log : autoreject.RejectLog
        Log describing which epochs/channels were interpolated or
        dropped.

    Raises
    ------
    ValueError
        If `tmax` is not negative, if no epochs could be created, or
        if fewer than 3 epochs are available for AutoReject.
    """

    if tmax >= 0:
        raise ValueError("The 'tmax' must be a negative number")

    tmax = int(np.shape(raw.get_data())[1] / sampling_rate + tmax)

    if segment_events is None:
        raw.crop(tmin=tmin, tmax=tmax)
        events = mne.make_fixed_length_events(
            raw=raw,
            duration=segments_length,
            overlap=overlap,
        )
    else:
        events = segment_events

    epochs = mne.Epochs(
        raw,
        events=events,
        tmin=0,
        tmax=segments_length - (1.0 / sampling_rate),
        baseline=None,
        preload=True,
        reject_by_annotation=ica_if_reject_by_annotation,
    )

    if len(epochs) == 0:
        err_msg = (
            f"No epochs were created. The length of the signal ({raw.times[-1]}) seconds "
            f"is shorter than the segment length of {segments_length} seconds "
            f"after rejecting the annotations which was {raw.annotations.duration} seconds."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    elif 0 < len(epochs) < 3:
        err_msg = (
            f"Only {len(epochs)} epoch found — need at least 3 for " f"autoreject."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    if cv == "auto":
        cv = max(2, min(10, len(epochs)))  # clamp between 2 and 10
        logger.info(f"The number of CV in autoreject was set to {cv} .")

    ar = AutoReject(
        n_interpolate=n_interpolates,
        consensus=consensus_percs,
        cv=cv,
        thresh_method=thresh_method,
        random_state=random_state,
        n_jobs=1,
        verbose=True,
    )

    epochs.load_data()

    ar.fit(epochs)
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)

    total_epochs = len(epochs)
    retained_epochs = len(epochs_clean)
    discarded_epochs = total_epochs - retained_epochs
    pct_discarded = (discarded_epochs / total_epochs) * 100 if total_epochs > 0 else 0.0
    interpolated_epochs = int(np.sum(np.any(reject_log.labels == 1, axis=1)))
    pct_interpolated = (
        (interpolated_epochs / total_epochs) * 100 if total_epochs > 0 else 0.0
    )

    log_msg = (
        f"Epoch rejection summary:\n"
        f"  Total epochs   : {total_epochs}\n"
        f"  Retained       : {retained_epochs} "
        f"({100 - pct_discarded:.1f}% | {retained_epochs * segments_length:.1f}s)\n"
        f"  Interpolated   : {interpolated_epochs} "
        f"({pct_interpolated:.1f}% | {interpolated_epochs * segments_length:.1f}s)\n"
        f"  Discarded      : {discarded_epochs} "
        f"({pct_discarded:.1f}% | {discarded_epochs * segments_length:.1f}s)"
    )

    if retained_epochs == 0:
        logger.error(log_msg)
        raise ValueError(
            "All epochs were rejected by AutoReject. "
            "Every segment was too noisy to repair."
        )

    logger.info(log_msg)
    return epochs_clean, reject_log


def extract_rs_blocks(
    raw, events, rs_id, sampling_rate, segments_length, overlap, seg_event_id=1
):
    """
    Extract and concatenate resting-state blocks from continuous data
    and generate fixed-length segment events within them.

    Identifies contiguous blocks bounded by events matching `rs_id`,
    discards blocks shorter than `segments_length`, concatenates the
    retained blocks, and generates a new fixed-length events array
    (with overlap) within each retained block.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous MEG/EEG recording.
    events : ndarray, shape (n_events, 3)
        MNE-style events array marking block boundaries.
    rs_id : int
        Event ID marking the start of a resting-state block of interest.
    sampling_rate : float
        Sampling rate of the data in Hz.
    segments_length : float
        Desired length of each output segment, in seconds. Blocks
        shorter than this are discarded.
    overlap : float
        Overlap between successive segments within a block, in seconds.
    seg_event_id : int, optional
        Event ID to assign to the generated segment events. Default is 1.

    Returns
    -------
    rs_raw : mne.io.Raw
        Concatenated raw data containing only the retained
        resting-state blocks.
    seg_events : ndarray, shape (n_segments, 3)
        MNE-style events array marking fixed-length segment onsets
        within `rs_raw`.
    """
    first_samp = raw.first_samp
    max_time = raw.times[-1]

    pieces, block_durations = [], []
    for i, (samp, _, eid) in enumerate(events):
        if eid != rs_id:
            continue
        seg_end = events[i + 1, 0] if i + 1 < len(events) else raw.last_samp
        tmin = (samp - first_samp) / sampling_rate
        tmax = (seg_end - first_samp) / sampling_rate
        tmax = min(tmax, max_time)

        dur = tmax - tmin
        if dur < segments_length:
            logger.info(
                f"drop RS: {tmin:6.2f}s -> {tmax:6.2f}s  ({dur:5.2f}s)  [too short]"
            )
            continue
        logger.info(f"keep RS: {tmin:6.2f}s -> {tmax:6.2f}s  ({dur:5.2f}s)")
        p = raw.copy().crop(tmin=tmin, tmax=tmax)
        pieces.append(p)
        block_durations.append(p.times[-1] + 1 / sampling_rate)

    if not pieces:
        err_msg = f"No RS blocks (id={rs_id}) longer than {segments_length}s found."
        logger.error(err_msg)
        raise ValueError(err_msg)

    rs_raw = mne.concatenate_raws(pieces)

    step = segments_length - overlap
    onsets_s = []
    block_start = 0.0
    for dur in block_durations:
        t = block_start
        while t + segments_length <= block_start + dur + 1e-9:
            onsets_s.append(t)
            t += step
        block_start += dur

    rs_first = rs_raw.first_samp
    seg_events = np.array(
        [[int(round(o * sampling_rate)) + rs_first, 0, seg_event_id] for o in onsets_s],
        dtype=int,
    )

    logger.info(f"\nKept {len(pieces)} RS block(s), total {rs_raw.times[-1]:.1f}s")
    logger.info(f"Built {len(seg_events)} epoch event(s) of {segments_length:.0f}s")
    return rs_raw, seg_events
