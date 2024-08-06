import pickle
import json
import os

def make_config(project, path=None):

    # preprocess configurations =================================================
    # downsample data
    config = dict()
    
    config["device"] = "MEGIN"
    config["layout"] = "Megin_MAG_All"
    
    # which sensor type should be used
    # choices: 1. meg: all, 2.mag, 3.grad
    config["whichSensor"] = "mag"
    config['fs'] = 1000

    # ICA configuration
    config['ica_n_component'] = 30
    config['ica_maxIter'] = 800
    config['ica_method'] = "fastica"
    # lower and upper cutoff frequencies in a bandpass filter
    config['cutoffFreqLow'] = 1
    config['cutoffFreqHigh'] = 45

    # Signal space projection
    config["ssp_ngrad"] = 3
    config["ssp_nmag"] = 3

    config["CAMCAN_preprocess"] = {"resampling": False,
                                    "digital_filter": True,
                                    "autoICA": True,
                                    "SSP": False}
    config["BTNRH_preprocess"] = {"resampling": False,
                                    "digital_filter": True,
                                    "autoICA": False,
                                    "SSP":True}

    # fooof analysis configurations ==============================================
    # Desired frequency range to run FOOOF
    config['fooof_freqRangeLow'] = 3
    config['fooof_freqRangeHigh'] = 40
    #start time of the raw data to use in seconds, this is to avoid possible eye blinks in close-eyed resting state. 
    config['segments_tmin'] = 20
    # end time of the raw data to use in seconds, this is to avoid possible eye blinks in close-eyed resting state.
    config['segments_tmax'] = -20
    # length of MEG segments in seconds
    config['segments_length'] = 10
    # amount of overlap between MEG sigals in seconds
    config['segments_overlap'] = 2
    # which mode should be used for fitting; choices (knee, fixed)
    config["aperiodic_mode"] = "knee"


    # Spectral estimation method
    config['psd_method'] = "welch"
    # amount of overlap between windows in Welch's method
    config['psd_n_overlap'] = 1
    config['psd_n_fft'] = 2
    # number of samples in psd
    config["n_per_seg"] = 2
    
    # minimum acceptable peak width in fooof analysis
    config["fooof_peak_width_limits"] = [1.0, 12.0]
    #Absolute threshold for detecting peaks
    config['fooof_min_peak_height'] = 0
    #Relative threshold for detecting peaks
    config['fooof_peak_threshold'] = 2

    # feature extraction ==========================================================
    # Define frequency bands
    config['freqBands'] = {'Theta': (3, 8),
                            'Alpha': (8, 13),
                            'Beta': (13, 30),
                            'Gamma': (30, 40),
                            'Broadband': (3, 40)}

    # Define individualized frequency range over main peaks in each freq band
    config['bandSubRanges'] = { 'Theta': (-2, 3),
                                'Alpha': (-2, 3), # change to (-4,2)
                                'Beta': (-8, 9),
                                'Gamma': (-5, 5)}

    # least acceptable R squred of fitted models
    config['leastR2'] = 0.9 


    config['featuresCategories'] = ["Offset", # 1
                                    "Exponent", # 1
                                    "Peak_Center", # 5,
                                    "Peak_Power",# 5,
                                    "Peak_Width", # 5,
                                    "Canonical_Relative_Power", 
                                    "Canonical_Absolute_Power",
                                    "Individualized_Relative_Power",
                                    "Individualized_Absolute_Power",
                                    ]
    config["fooof_res_save_path"] = False

    # feature summarize
    # minimum acceptable number of INFs for each subject
    # if number of INFs < min_thr_inf => skip INFs 
    config["min_thr_inf"] = 10

    if path is not None:
        out_file = open(os.path.join(path, project + ".json"), "w") 
        json.dump(config, out_file, indent = 6) 
        out_file.close()

    return config 






def make_ch_Names(sensor_type, ch_names, feature_categories):
    
    if sensor_type == "mag":
        ch_names = [channel for channel in ch_names if channel.endswith("1")]
    if sensor_type == "grad1":
        ch_names = [channel for channel in ch_names if channel.endswith("2")]
    if sensor_type == "grad2":
        ch_names = [channel for channel in ch_names if channel.endswith("3")]
    
    return ch_names
    



def storeFooofModels(path, subjId, fooofModels, psds, freqs) -> None:
    """
    This function stores the periodic and aperiodic 
    results in a h5py file

    parameters
    ------------
    path: str
    where to save

    subjid: str
    subject ID

    fooofModels: object

    returns
    -------------
    None

    """

    with open(os.path.join(path, subjId + ".pickle"), "ab") as file:
        pickle.dump([fooofModels, psds, freqs], file)


def merge_datasets(datasets):

    subjects = {}
    
    for dataset_name in datasets.keys(): 
        path = datasets[dataset_name]
        dirs = os.listdir(path)
        for dir in dirs:
            full_path = os.path.join(path, dir)
            if os.path.isdir(full_path):
                subjects[dir] = full_path
                
    return subjects