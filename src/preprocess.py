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


def findComponent(ica, data, physiological_signal):
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
    componentIndx = np.argmax(corr)

    return componentIndx


def autoICA(data, channel_types, n_components=30, ica_max_iter=1000, IcaMethod="fastica", which_sensor=["meg", "eeg"]):

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
    
    physiological_sensors = [x for x in channel_types if x in ["eog" , "ecg"]] 
    physiological_signal = data.copy().pick(picks=physiological_sensors).get_data()

    data = data.pick_types(meg=which_sensor["meg"] | which_sensor["mag"] | which_sensor["grad"], 
                                eeg=which_sensor["eeg"],
                                ref_meg=False)
    data = data.pick(picks=[sensor for sensor, if_calculate in which_sensor.items() if if_calculate])

    # ICA
    ica = mne.preprocessing.ICA(n_components=n_components,
                                max_iter=ica_max_iter,
                                method=IcaMethod,
                                random_state=42,
                                verbose=False)
    ica.fit(data, verbose=False)
    
    # calculating bad ica components using automatic method
    badComponents = []
    for sensor in physiological_signal:
        badComponents.append(findComponent(ica=ica, data=data, physiological_signal=sensor))
        # TODO test if this happens three times for camcan

    ica.exclude = badComponents.copy()
    # ica.apply() changes the Raw object in-place
    ica.apply(data, verbose=False)

    return data

def AutoIca_with_IcaLabel(data, n_components=30, ica_max_iter=1000, IcaMethod="infomax", artifact_threshold=0.8):

    #fit ICA
    ica = mne.preprocessing.ICA(n_components=n_components, 
                                max_iter=ica_max_iter, 
                                method=IcaMethod, 
                                random_state=42, 
                                fit_params=dict(extended=True),
                                verbose=False) #fit_params=dict(extended=True) bc icalabel is trained with this
    ica.fit(data, verbose=False)

    #apply ICLabel
    labels = label_components(data, ica, method='iclabel')

    #Identify and exclude artifact components based on probability threshold of being an artifact
    bad_components = []
    for idx, label in enumerate(labels['labels']):
        if label not in ['brain', 'other'] and labels['y_pred_proba'][idx] > artifact_threshold:
            bad_components.append(idx)
    
    print("bad components based on iclabel:",bad_components) #DEBUG 
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
              grad_flat_threshold, eeg_flat_threshold, zscore_std_thresh, which_sensor):
     
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

    segments.drop_bad(reject=None, flat=flat_criteria) ##CHANGE!!! but figure out var threshold

    if zscore_std_thresh:
        z_scores = stats.zscore(np.std(segments.get_data(), axis=0), axis=0)
        bad_epochs = np.where(z_scores>zscore_std_thresh)[0]
        segments.drop(indices=bad_epochs)
        
    return segments


def preprocess(data, which_sensor:dict, resampling_rate=None, digital_filter=True, apply_rereference = False, rereference_method = "average", n_component:int=30, ica_max_iter:int=800, 
                IcaMethod:str="fastica", cutoffFreqLow:float=1, cutoffFreqHigh:float=45, 
                ssp_ngrad:int=3, ssp_nmag:int=3, apply_ica=True, apply_ssp=True):

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

    ssp_flage = True

    channel_types = set(data.get_channel_types())

    sampling_rate = data.info["sfreq"]

    # resample & band pass filter
    if resampling_rate and resampling_rate != sampling_rate:
        data.resample(resampling_rate, verbose=False, n_jobs=-1)
        sampling_rate = data.info["sfreq"]

    # TODO: notch filter
    
    if digital_filter:
        data.filter(l_freq=cutoffFreqLow, 
                    h_freq=cutoffFreqHigh, 
                    n_jobs=-1, 
                    verbose=False)
        
    #rereference
    if which_sensor['eeg'] and apply_rereference and rereference_method == "average":
        data = data.set_eeg_reference("average") 

    if which_sensor['eeg'] and apply_rereference and rereference_method == "REST":
        data = data.set_eeg_reference("REST") 

    # apply automated ICA
    #If meg, apply the autoICA 
    if which_sensor['meg'] and apply_ica and ("ecg" in channel_types or "eog" in channel_types):
        data = autoICA(data=data, 
                    n_components=n_component, # FLUX default
                    ica_max_iter=ica_max_iter, # FLUX default,
                    IcaMethod = IcaMethod,
                    which_sensor=which_sensor,
                    channel_types=channel_types)
        ssp_flage = False


    # if eeg, apply automated ICA and ICALabel, based on a determined flag. If eog/ecg channels are both present only autoICA is applied
    # if only one of them is present, autoICA is applied on that channel and afterwards ICALabel is applied on the outcome
    # if none of them is present, only ICALabel is applied 

    if which_sensor['eeg'] and apply_ica: 

        autoICA_flag = False
        if "ecg" in channel_types and "eog" in channel_types:
            autoICA_flag = "both"
        elif "ecg" in channel_types and "eog" not in channel_types:
            autoICA_flag = "ecg_only"
        elif "eog" in channel_types and "ecg" not in channel_types:
            autoICA_flag = "eog_only"

        if apply_ica and autoICA_flag == 'both':
            data = autoICA(data=data, 
                        n_components=n_component, # FLUX default
                        ica_max_iter=ica_max_iter, # FLUX default,
                        IcaMethod = IcaMethod,
                        which_sensor=which_sensor,
                        channel_types=channel_types)
            ssp_flage = False

        elif apply_ica and autoICA_flag == "ecg_only": 
            data = autoICA(data=data, 
                    n_components=n_component,
                    ica_max_iter=ica_max_iter,
                    IcaMethod=IcaMethod,
                    which_sensor=which_sensor,
                    channel_types=["ecg"])

            data = AutoIca_with_IcaLabel(data=data,
                    n_components=n_component,
                    ica_max_iter=ica_max_iter,
                    IcaMethod=IcaMethod,
                    artifact_threshold=0.8) #TODO: Make it more general by not hardcoding the threshold           
        
            ssp_flage = False

        elif apply_ica and autoICA_flag == "eog_only":
            data = autoICA(data=data, 
                    n_components=n_component,
                    ica_max_iter=ica_max_iter,
                    IcaMethod=IcaMethod,
                    which_sensor=which_sensor,
                    channel_types=["eog"])

            data = AutoIca_with_IcaLabel(data=data,
                    n_components=n_component,
                    ica_max_iter=ica_max_iter,
                    IcaMethod=IcaMethod,
                    artifact_threshold=0.8) #TODO: Make it more general by not hardcoding the threshold           
            ssp_flage = False

    # If neither ECG nor EOG is present only IcaLabel
        elif apply_ica:
            data = AutoIca_with_IcaLabel(data = data, 
                    n_components=n_component, 
                    ica_max_iter=ica_max_iter, 
                    IcaMethod=IcaMethod,
                    artifact_threshold= 0.8) #TODO: Make it more general by not hardcoding the threshold                              
            ssp_flage = False

    if apply_ssp and ssp_flage:
    # Note: If no ECG recording is provided, the ECG vector will
    # be automatically calculated using the magnetometer sensors.
        projs, event = mne.preprocessing.compute_proj_ecg(data, n_grad=ssp_ngrad, n_mag=ssp_nmag)
        data.add_proj(projs)
        data.apply_proj()
        # TODO what will happen if we put the following code before SSP, because CTF has ref_meg?
        # For discarding ref_meg
        data = data.pick_types(meg=which_sensor["meg"] | which_sensor["mag"] | which_sensor["grad"], 
                                      eeg=which_sensor["eeg"],
                                      ref_meg=False)
        # For discarding mag or meg
        data = data.pick(picks=[sensor for sensor, if_calculate in which_sensor.items() if if_calculate])
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