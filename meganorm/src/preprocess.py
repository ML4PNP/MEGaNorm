import os
import sys
import mne
from mne_icalabel import label_components
import tqdm
import json
import numpy as np
import argparse
from glob import glob
import warnings

warnings.filterwarnings("ignore")
from mne.preprocessing import find_bad_channels_maxwell


def find_ica_component(ica, data, physiological_signal, auto_ica_corr_thr):
    """
    Identifies the ICA component most correlated with a physiological signal (e.g., ECG or EOG).

    Args:
        ica (object): The fitted ICA object.
        data (mne.io.Raw): The raw MEG/EEG data used to extract ICA components.
        physiological_signal (np.ndarray): The physiological signal (e.g., ECG or EOG) to compare with ICA components.
        auto_ica_corr_thr (float): Threshold for accepting a component based on Pearson correlation.

    Returns:
        list: Index of the component with the highest correlation if it exceeds the threshold.
              Returns an empty list if no component meets the criterion.
    """
    components = ica.get_sources(data.copy()).get_data()

    if components.shape[1] != len(physiological_signal):
        raise ValueError("Length of physiological signal must match the number of time points in the data.")

    corr = np.corrcoef(components, physiological_signal)[-1, :-1]

    if np.max(corr) >= auto_ica_corr_thr:
        componentIndx = [int(np.argmax(corr))]
    else:
        componentIndx = []

    return componentIndx




def auto_ica(data, physiological_sensor, n_components=30, ica_max_iter=1000, 
             IcaMethod="fastica", which_sensor={"meg": True, "eeg": True}, 
             auto_ica_corr_thr=0.9):
    """
    Performs automated ICA for artifact removal by identifying components that
    correlate highly with physiological signals (e.g., ECG or EOG).

    Args:
        data (mne.io.Raw): Raw MEG/EEG data.
        physiological_sensor (str): Name of the physiological sensor (e.g., 'ECG' or 'EOG').
        n_components (int or float): Number of ICA components to retain.
        ica_max_iter (int): Maximum number of iterations for ICA algorithm.
        IcaMethod (str): ICA algorithm to use (e.g., 'fastica', 'picard', 'infomax').
        which_sensor (dict): Dictionary indicating sensor types to include (e.g., {'meg': True, 'eeg': True}).
        auto_ica_corr_thr (float): Threshold for accepting ICA component based on correlation.

    Returns:
        tuple:
            - data (mne.io.Raw): Raw data with bad ICA components removed (in-place modification).
            - ICA_flag (bool): True if no bad components were found, False otherwise.
    """
    # Get physiological signal
    physiological_signal = data.copy().pick(picks=physiological_sensor).get_data()

    # Pick MEG/EEG for ICA
    data = data.pick_types(
        meg=which_sensor.get("meg", False),
        eeg=which_sensor.get("eeg", False),
        ref_meg=False,
        eog=True,
        ecg=True)

    # ICA initialization
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        max_iter=ica_max_iter,
        method=IcaMethod,
        random_state=42,
        verbose=False)
    ica.fit(data, verbose=False, picks=["eeg", "meg"])

    # Detect components correlated with physiological signal
    bad_components = []
    for sensor in physiological_signal:
        bad_components.extend(find_ica_component(
            ica=ica,
            data=data,
            physiological_signal=sensor,
            auto_ica_corr_thr=auto_ica_corr_thr))

    print("Bad Components identified by auto ICA:", bad_components)

    if bad_components:
        ica.exclude = bad_components.copy()
        ica.apply(data, verbose=False)
        ICA_flag = False
    else:
        ICA_flag = True

    return data, ICA_flag



def auto_ica_with_mean(data, n_components=30, ica_max_iter=1000, 
                       IcaMethod="fastica", 
                       which_sensor={"meg": True, "eeg": True}, 
                       auto_ica_corr_thr=0.9):
    """
    Performs ICA-based artifact rejection using MNE’s built-in ECG correlation method.

    Args:
        data (mne.io.Raw): Raw MEG/EEG data.
        n_components (int or float): Number of ICA components to retain.
        ica_max_iter (int): Maximum number of iterations for ICA algorithm.
        IcaMethod (str): ICA algorithm to use (e.g., 'fastica', 'picard', 'infomax').
        which_sensor (dict): Dictionary specifying sensor types to include (e.g., {"meg": True, "eeg": True}).
        auto_ica_corr_thr (float): Correlation threshold for detecting ECG-related components.

    Returns:
        mne.io.Raw: Raw data with ECG-related ICA components removed.
    """

    data = data.pick_types(meg = which_sensor["meg"] | which_sensor["mag"] | which_sensor["grad"], 
                            eeg = which_sensor["eeg"],
                            ref_meg = False,
                            eog = True,
                            ecg = True)
    
    ica = mne.preprocessing.ICA(n_components=n_components,
                                max_iter=ica_max_iter,
                                method=IcaMethod,
                                random_state=42,
                                verbose=False)
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


def segment_epoch(data, tmin, tmax, sampling_rate, segmentsLength, overlap):
    """
    Segments continuous raw data into overlapping epochs of fixed length.

    Args:
        data (mne.io.Raw): Continuous MEG/EEG recording.
        tmin (float): Start time (in seconds) for cropping the raw data.
        tmax (float): End time offset (in seconds) to subtract from total duration.
        sampling_rate (float): Sampling rate of the data (Hz).
        segmentsLength (float): Length of each epoch in seconds.
        overlap (float): Overlap between successive epochs in seconds.

    Returns:
        mne.Epochs: Epoched data with fixed-length segments.
    """
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
        verbose=False
    )

    return segments


def drop_bads(
    segments,
    mag_var_threshold,
    grad_var_threshold,
    eeg_var_threshold,
    mag_flat_threshold,
    grad_flat_threshold,
    eeg_flat_threshold,
    which_sensor,
):

    if which_sensor["meg"]:
        reject_criteria = dict(mag=mag_var_threshold, grad=grad_var_threshold)
        flat_criteria = dict(mag=mag_flat_threshold, grad=grad_flat_threshold)

    if which_sensor["mag"]:
        reject_criteria = dict(mag=mag_var_threshold)
        flat_criteria = dict(mag=mag_flat_threshold)

    if which_sensor["grad"]:
        reject_criteria = dict(grad=grad_var_threshold)
        flat_criteria = dict(grad=grad_flat_threshold)

    if which_sensor["eeg"]:
        reject_criteria = dict(eeg=eeg_var_threshold)
        flat_criteria = dict(eeg=eeg_flat_threshold)

    segments.drop_bad(
        reject=reject_criteria, flat=flat_criteria
    )  ##CHANGE!!! but figure out var threshold

    return segments


def preprocess(data, which_sensor: dict, resampling_rate=None, digital_filter=True,
               rereference_method="average", n_component: int = 30, ica_max_iter: int = 800,
               IcaMethod: str = "fastica", cutoffFreqLow: float = 1, cutoffFreqHigh: float = 45,
               apply_ica=True, power_line_freq: int = 60, auto_ica_corr_thr: float = 0.9):
    """
    Applies a preprocessing pipeline on MEG/EEG data, including filtering, re-referencing,
    ICA for artifact removal, and optional downsampling.

    Args:
        data (mne.io.Raw): Raw MEG/EEG data.
        which_sensor (dict): Dictionary specifying which sensor types to include (e.g., {'meg': True, 'eeg': True}).
        resampling_rate (int, optional): Target sampling rate for resampling. If None, resampling is skipped.
        digital_filter (bool): Whether to apply a bandpass filter to the data.
        rereference_method (str): EEG re-referencing method. Supported: "average", "REST".
        n_component (int): Number of ICA components to retain.
        ica_max_iter (int): Maximum number of iterations for ICA.
        IcaMethod (str): ICA algorithm to use. Supported: 'fastica', 'picard', 'infomax'.
        cutoffFreqLow (float): Low cutoff frequency for bandpass filtering.
        cutoffFreqHigh (float): High cutoff frequency for bandpass filtering.
        apply_ica (bool): Whether to apply ICA to remove artifacts.
        power_line_freq (int): Power line frequency (for notch filtering if added later).
        auto_ica_corr_thr (float): Correlation threshold for automatic ICA artifact rejection.

    Returns:
        mne.io.Raw: Preprocessed MEG/EEG data.
    """
    if not 0 < auto_ica_corr_thr <= 1:
        raise ValueError("auto_ica_corr_thr must be between 0 and 1.")
    if IcaMethod not in ['fastica', 'picard', 'infomax']:
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
        data.resample(resampling_rate, verbose=False, n_jobs=-1)
        sampling_rate = data.info["sfreq"]

    data.notch_filter(
        freqs=np.arange(power_line_freq, 4 * power_line_freq + 1, power_line_freq),
        n_jobs=-1,
    )

    if digital_filter:
        data.filter(
            l_freq=cutoffFreqLow, h_freq=cutoffFreqHigh, n_jobs=-1, verbose=False
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # positional Arguments (remove --)
    parser.add_argument("dir", help="Address to your data")
    parser.add_argument("saveDir", help="Address to where save the result")
    # optional arguments
    parser.add_argument(
        "--configs", type=str, default=None, help="Address of configs json file"
    )

    args = parser.parse_args()

    # Loading configs
    if args.configs is not None:
        with open(args.configs, "r") as f:
            configs = json.load(f)
    else:
        configs = make_config()

    dataPaths = glob(args.dir)
    # loop over all of data
    for count, subjectPath in enumerate(tqdm.tqdm(dataPaths[:])):

        subID = subjectPath.split("/")[-1]

        filteredData = preprocess(
            subjectPath=subjectPath,
            fs=configs["fs"],
            n_component=configs["n_component"],
            maxIter=configs["maxIter"],
            IcaMethod=configs["IcaMethod"],
            cutoffFreqLow=configs["cutoffFreqLow"],
            cutoffFreqHigh=configs["cutoffFreqHigh"],
            sensorType=configs["sensorType"],
        )

        filteredData.save(f"{args.saveDir}/{subID}.fif", overwrite=True)
