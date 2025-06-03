import os
import mne
from mne_icalabel import label_components
import json
import numpy as np
import glob
from typing import Any, Dict
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


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

    print("Bad Components identified by auto ICA:", bad_components)

    if bad_components:
        ica.exclude = bad_components.copy()
        ica.apply(data, verbose=False)
        ICA_flag = False
    else:
        ICA_flag = True

    return data, ICA_flag


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

    ica.exclude = ecg_indices
    ica.apply(data, verbose=False)

    return data


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

    print("Bad Components identified by ICALabel:", bad_components)
    ica.exclude = bad_components.copy()
    ica.apply(data, verbose=False)

    return data


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
    which_sensor: dict,
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
):
    """
    Applies a preprocessing pipeline on MEG/EEG data, including filtering, re-referencing (for EEG),
    ICA for artifact removal, and optional downsampling.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG/EEG data.
    which_sensor : dict
        Dictionary specifying which sensor types to include (e.g., {'meg': True, 'eeg': True}).
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
        raise ValueError("auto_ica_corr_thr must be between 0 and 1.")
    if IcaMethod not in ["fastica", "picard", "infomax"]:
        raise ValueError("ICA method must be one of: 'fastica', 'picard', 'infomax'.")

    # since pick_channels can not seperate mag and grad signals
    if not (which_sensor["meg"] or which_sensor["eeg"]):
        if not which_sensor["mag"]:
            mag_channels = [
                ch
                for ch, ch_type in zip(data.ch_names, data.get_channel_types())
                if ch_type == "mag"
            ]
        elif not which_sensor["grad"]:
            mag_channels = [
                ch
                for ch, ch_type in zip(data.ch_names, data.get_channel_types())
                if ch_type == "grad"
            ]
        data.drop_channels(mag_channels)

    channel_types = set(data.get_channel_types())

    sampling_rate = data.info["sfreq"]

    # resample & band pass filter
    if resampling_rate and resampling_rate != sampling_rate:
        data.resample(int(resampling_rate), verbose=False, n_jobs=-1)
        sampling_rate = data.info["sfreq"]

    data.notch_filter(
        freqs=np.arange(
            int(power_line_freq), 4 * int(power_line_freq) + 1, int(power_line_freq)
        ),
        n_jobs=-1,
    )

    if digital_filter:
        data.filter(
            l_freq=int(cutoffFreqLow),
            h_freq=int(cutoffFreqHigh),
            n_jobs=-1,
            verbose=False,
        )

    # rereference
    if which_sensor["eeg"] and rereference_method:
        data = data.set_eeg_reference(rereference_method)

    ICA_flag = True  # initialize flag

    physiological_electrods = {
        channel: channel in channel_types for channel in ["ecg", "eog"]
    }

    for phys_activity_type, if_elec_exist in physiological_electrods.items():

        if which_sensor[
            "meg"
        ]:  # ======================================================================
            # 1
            if if_elec_exist and apply_ica:
                data, _ = auto_ica(
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
                data = auto_ica_with_mean(
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
                data, ICA_flag = auto_ica(
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
                data = AutoIca_with_IcaLabel(
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

    return data, data.info["ch_names"], int(sampling_rate)


def drop_noisy_meg_channels(
    data: Any, subID: str, args: Any, configs: Dict[str, str]
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
    which_sensor = dict.fromkeys(["meg", "mag", "grad", "eeg", "opm"], False)
    which_sensor[configs.get("which_sensor")] = True

    try:
        auto_noisy_chs, auto_flat_chs = mne.preprocessing.find_bad_channels_maxwell(
            data, return_scores=False, verbose=True, coord_frame="meg"
        )
        data.info["bads"] += auto_noisy_chs + auto_flat_chs

    except RuntimeError as e:
        if "Maxwell filtering SSS step has already been applied" in str(e):
            print("Skipping: SSS already applied.")
        else:
            raise

    # Always proceed to log and drop marked bads
    droped_ch_len = len(data.info["bads"])
    log_path = args.saveDir.replace("temp", "log_droped_channels")

    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, f"{subID}.json"), "w") as file:
        json.dump(droped_ch_len, file)

    return data.copy().drop_channels(data.info["bads"])
