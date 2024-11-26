import os
import sys
import mne
from mne_icalabel import label_components
import tqdm
import json
import numpy as np
import argparse
from glob import glob
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add utils folder to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'utils')
sys.path.append(config_path)


def find_ica_component(ica, data, physiological_signal, auto_ica_corr_thr):
    """
    parameters
    -----------
    ica: object
    ica model

    data: mne.raw
    meg data

    physiological_signal: array
    either ECG or 

    return
    ------------
    componentIndx: int
    index of the component with the highest Pearson 
    correlation with the signal of interest.
    """
    components = ica.get_sources(data.copy()).get_data()
    corr = np.corrcoef(components, physiological_signal)[:-1, -1]
    if np.max(corr) >= auto_ica_corr_thr:
        componentIndx = np.argmax(corr)
    else:
        componentIndx = []

    return componentIndx


def auto_ica(data, physiological_sensor, n_components=30, ica_max_iter=1000, IcaMethod="fastica", which_sensor=["meg", "eeg"], auto_ica_corr_thr=0.9):

    """
    This function serves as an automated noise detection tool
    in ICA components. Essentially, it computes the correlation
    between each component and either the ECG or EOG signals, 
    and then returns the component with the highest Pearson 
    correlation coefficient with that signal.

    parameters
    -----------
    data: mne.raw
    meg data

    
    n_components:int - float
    ICA n_components

    max_iter: int
    maximum number of iteration

    IcaMethod: str
    ICA method

    cutoffFreq: list
    cutoff frequency for filtering data before feeding it to ICA.
    Note: this function does not filter data in place

    return
    -----------
    ica: object
    final ica model
    """

    physiological_signal = data.copy().pick(picks=physiological_sensor).get_data()

    data = data.pick_types(meg = which_sensor["meg"] | which_sensor["mag"] | which_sensor["grad"], 
                            eeg = which_sensor["eeg"],
                            ref_meg = False,
                            eog = True,
                            ecg = True)

    # ICA
    ica = mne.preprocessing.ICA(n_components=n_components,
                                max_iter=ica_max_iter,
                                method=IcaMethod,
                                random_state=42,
                                verbose=False)
    ica.fit(data, verbose=False, picks=["eeg", "meg"])
    
    # calculating bad ica components using automatic method
    badComponents = []
    for sensor in physiological_signal:
        badComponents.extend(find_ica_component(ica=ica, data=data, physiological_signal=sensor, 
                                           auto_ica_corr_thr=auto_ica_corr_thr))
        # TODO test if this happens three times for camcan

    ica.exclude = badComponents.copy()
    # ica.apply() changes the Raw object in-place
    ica.apply(data, verbose=False)

    return data


def auto_ica_with_mean(data, n_components=30, ica_max_iter=1000, IcaMethod="fastica", which_sensor=["meg", "eeg"], auto_ica_corr_thr=0.9):

    data = data.pick_types(meg = which_sensor["meg"] | which_sensor["mag"] | which_sensor["grad"], 
                            eeg = which_sensor["eeg"],
                            ref_meg = False,
                            eog = True,
                            ecg = True)
    
    # ICA
    ica = mne.preprocessing.ICA(n_components=n_components,
                                max_iter=ica_max_iter,
                                method=IcaMethod,
                                random_state=42,
                                verbose=False)
    ica.fit(data, verbose=False, picks=["eeg", "meg"])
    
    ecg_indices, _ = ica.find_bads_ecg(data, method="correlation", threshold=auto_ica_corr_thr)

    ica.exclude = ecg_indices
    # ica.apply() changes the Raw object in-place
    ica.apply(data, verbose=False)

    return data


def AutoIca_with_IcaLabel(data, physiological_noise_type, n_components=30, ica_max_iter=1000, IcaMethod="infomax", iclabel_thr=0.8):

    if physiological_noise_type == "ecg": physiological_noise_type = "heart beat"
    if physiological_noise_type == "eog": physiological_noise_type = "eye blink"

    #fit ICA
    ica = mne.preprocessing.ICA(n_components=n_components, 
                                max_iter=ica_max_iter, 
                                method=IcaMethod, 
                                random_state=42, 
                                fit_params=dict(extended=True),
                                verbose=False) #fit_params=dict(extended=True) bc icalabel is trained with this
    ica.fit(data, verbose=False, picks=["eeg"])

    #apply ICLabel
    labels = label_components(data, ica, method='iclabel')

    #Identify and exclude artifact components based on probability threshold of being an artifact
    bad_components = []
    for idx, label in enumerate(labels['labels']):
        if label==physiological_noise_type and labels['y_pred_proba'][idx] > iclabel_thr:
            bad_components.append(idx)

    ica.exclude = bad_components.copy()
    ica.apply(data, verbose=False)

    return data
     

def segment_epoch(data, tmin, tmax, sampling_rate, segmentsLength, overlap):

    # We exclude 20s from both begining and end of signals 
    # since participants usually open and close their eyes
    # in this time interval
    tmax = int(np.shape(data.get_data())[1]/sampling_rate + tmax)
    data.crop(tmin=tmin, tmax=tmax)
    segments = mne.make_fixed_length_epochs(data,
                                            duration=segmentsLength,
                                            overlap=overlap,
                                            reject_by_annotation=True,
                                            verbose=False)

    return segments


def drop_bads(segments, mag_var_threshold, grad_var_threshold, eeg_var_threshold, mag_flat_threshold, 
              grad_flat_threshold, eeg_flat_threshold, which_sensor):
     
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

    segments.drop_bad(reject=reject_criteria, flat=flat_criteria) ##CHANGE!!! but figure out var threshold
        
    return segments


def preprocess(data, which_sensor:dict, resampling_rate=None, digital_filter=True, rereference_method = "average", 
               n_component:int=30, ica_max_iter:int=800, 
                IcaMethod:str="fastica", cutoffFreqLow:float=1, cutoffFreqHigh:float=45, 
                apply_ica=True, power_line_freq:int=60, auto_ica_corr_thr:float=0.9):

    """
    Apply preprocessing pipeline (ICA and downsampling) on MEG signals.

    parameters
    -----------
    subjectPath: str
    MEG signal path

    subIdPosition:int
    indicates which part of the subjectPath contain subject ID.

    targetFS: int
    MEG signals are resampled to this sampling rate

    n_component: float
    numper of component in ICA

    rereference_method: str
    choices: ["average", "REST"]

    ica_max_iter: int
    maximum number of iteration in ICA

    IcaMethod: str
    ICA method
    choices : ["fastica", "picard", "infomax"]

    cutoffFreqLow: int
    lower limit of cutoff frequency in FIR bandpass filter

    cutoffFreqHigh
    higher limit of cutoff frequency in FIR bandpass filter

    returns
    --------------
    datamne raw data
    """
    # since pick_channels can not seperate mag and grad signals
    if not (which_sensor['meg'] or which_sensor['eeg']):
        if not which_sensor['mag']:
            mag_channels = [ch for ch, ch_type in zip(data.ch_names, data.get_channel_types()) if ch_type == 'mag']
        elif not which_sensor['grad']:
            mag_channels = [ch for ch, ch_type in zip(data.ch_names, data.get_channel_types()) if ch_type == 'grad']
        data.drop_channels(mag_channels)

    channel_types = set(data.get_channel_types())

    sampling_rate = data.info["sfreq"]

    # resample & band pass filter
    if resampling_rate and resampling_rate != sampling_rate:
        data.resample(resampling_rate, verbose=False, n_jobs=-1)
        sampling_rate = data.info["sfreq"]

    data.notch_filter(freqs=np.arange(power_line_freq, 4*power_line_freq+1, power_line_freq),
                    n_jobs=-1)
    
    if digital_filter:
        data.filter(l_freq=cutoffFreqLow, 
                    h_freq=cutoffFreqHigh, 
                    n_jobs=-1, 
                    verbose=False)

    #rereference
    if which_sensor['eeg'] and rereference_method:
        data = data.set_eeg_reference(rereference_method)

    physiological_electrods = {channel: channel in channel_types for channel in ["ecg", "eog"]}

    for phys_activity_type, if_elec_exist in physiological_electrods.items():

        if which_sensor['meg']: # ======================================================================
            # 1
            if if_elec_exist and apply_ica:
                data = auto_ica(data=data, 
                            n_components=n_component, 
                            ica_max_iter=ica_max_iter,
                            IcaMethod = IcaMethod,
                            which_sensor=which_sensor,
                            physiological_sensor=phys_activity_type,
                            auto_ica_corr_thr=auto_ica_corr_thr)
            # 2
            elif not if_elec_exist and apply_ica and phys_activity_type=="ecg":
                data = auto_ica_with_mean(data=data, 
                                    n_components=n_component, 
                                    ica_max_iter=ica_max_iter,
                                    IcaMethod = IcaMethod,
                                    which_sensor=which_sensor,
                                    auto_ica_corr_thr=auto_ica_corr_thr)

        if which_sensor['eeg']: # ======================================================================
            # 1
            if if_elec_exist and apply_ica:
                data = auto_ica(data=data, 
                            n_components=n_component, 
                            ica_max_iter=ica_max_iter,
                            IcaMethod = IcaMethod,
                            which_sensor=which_sensor,
                            physiological_sensor=phys_activity_type,
                            auto_ica_corr_thr=auto_ica_corr_thr)
            # 2
            elif not if_elec_exist and apply_ica:
                data = AutoIca_with_IcaLabel(data = data, 
                                        n_components=n_component, 
                                        ica_max_iter=ica_max_iter, 
                                        IcaMethod=IcaMethod,
                                        iclabel_thr=auto_ica_corr_thr,
                                        physiological_noise_type=phys_activity_type) 

    data = data.pick_types(meg = which_sensor["meg"] | which_sensor["mag"] | which_sensor["grad"], 
                            eeg = which_sensor["eeg"],
                            ref_meg = False,
                            eog = False,
                            ecg = False)
    return data, data.info["ch_names"], int(sampling_rate)

    
    
if __name__ == "__main__":
   
	parser = argparse.ArgumentParser()
	# positional Arguments (remove --)
	parser.add_argument("dir", 
				help="Address to your data")
	parser.add_argument("saveDir",
				help="Address to where save the result")
	# optional arguments
	parser.add_argument("--configs", type=str, default=None,
		help="Address of configs json file")

	args = parser.parse_args()

	# Loading configs
	if args.configs is not None:
		with open(args.configs, 'r') as f:
			configs = json.load(f)
	else: configs = make_config()

	dataPaths = glob(args.dir)
	# loop over all of data 
	for count, subjectPath in enumerate(tqdm.tqdm(dataPaths[:])):

		subID = subjectPath.split("/")[-1] 

		filteredData = preprocess(subjectPath=subjectPath,
								fs=configs["fs"],
								n_component=configs["n_component"],
								maxIter=configs["maxIter"],
								IcaMethod=configs["IcaMethod"],
								cutoffFreqLow=configs["cutoffFreqLow"],
								cutoffFreqHigh=configs["cutoffFreqHigh"],
								sensorType=configs["sensorType"])
		
		filteredData.save(f'{args.saveDir}/{subID}.fif', overwrite=True)