from itertools import chain
import pandas as pd
import pickle
import glob
import json
import os

def make_config(path=None):

    # preprocess configurations =================================================
    # downsample data
    config = dict()
    
    # which sensor type should be used
    # choices: 1. meg: all, 2.mag, 3.grad
    config["sensorType"] = "meg"

    config['targetFS'] = 1000

    # ICA configuration
    config['n_component'] = 30
    config['maxIter'] = 800
    config['IcaMethod'] = "fastica"
    # lower and upper cutoff frequencies in a bandpass filter
    config['cutoffFreqLow'] = 1
    config['cutoffFreqHigh'] = 45


    # fooof analysis configurations ==============================================
    # Desired frequency range to run FOOOF
    config['freqRangeLow'] = 3
    config['freqRangeHigh'] = 40
    # sampling rate
    config['fs'] = config['targetFS'] # TODO: Should be removed
    #start time of the raw data to use in seconds, this is to avoid possible eye blinks in close-eyed resting state. 
    config['tmin'] = 20
    # end time of the raw data to use in seconds, this is to avoid possible eye blinks in close-eyed resting state.
    config['tmax'] = -20
    # length of MEG segments in seconds
    config['segmentsLength'] = 10
    # amount of overlap between MEG sigals in seconds
    config['overlap'] = 2
    #Absolute threshold for detecting peaks
    config['min_peak_height'] = 0
    #Relative threshold for detecting peaks
    config['peak_threshold'] = 2
    # Spectral estimation method
    config['psdMethod'] = "welch"
    # amount of overlap between windows in Welch's method
    config['psd_n_overlap'] = 1
    config['psd_n_fft'] = 2
    # number of samples in psd
    config["n_per_seg"] = 2
    # minimum acceptable peak width in fooof analysis
    config["peak_width_limits"] = [1.0, 12.0]


    # feature extraction ==========================================================
    # Define frequency bands
    config['freqBands'] = {
                                'Theta': (4, 8),
                                'Alpha': (8, 13),
                                'Beta': (13, 30),
                                'Gamma': (30, 40),
                                'Broadband': (3, 40)
                            }

    # Define individualized frequency range over main peaks in each freq band
    config['bandSubRanges'] = {
                                'Theta': (-1, 1),
                                'Alpha': (-2, 3), # change to (-4,2)
                                'Beta': (-7, 7),
                                'Gamma': (-5, 5),
                            }

    # least acceptable R squred of fitted models
    config['leastR2'] = 0.9 



    config['channels_spatial_layouts'] = {'MAG_frontal_left':['MEG0121', 'MEG0341','MEG0311','MEG0321','MEG0511','MEG0541','MEG0331','MEG0521','MEG0531','MEG0611','MEG0641','MEG0621','MEG0821'],
                    'MAG_frontal_right':['MEG1411', 'MEG1221','MEG1211','MEG1231','MEG0921','MEG0931','MEG1241','MEG0911','MEG0941','MEG1021','MEG1031','MEG0811','MEG1011'],
                    'MAG_temporal_left':['MEG0111', 'MEG0131','MEG0211','MEG0221','MEG0141','MEG1511','MEG0241','MEG0231','MEG1541','MEG1521','MEG1611','MEG1621','MEG1531'],
                    'MAG_temporal_right':['MEG1421', 'MEG1311','MEG1321','MEG1441','MEG1431','MEG1341','MEG1331','MEG2611','MEG2411','MEG2421','MEG2641','MEG2621','MEG2631'],
                    'MAG_parietal_left':['MEG0411', 'MEG0421','MEG0631','MEG0441','MEG0431','MEG0711','MEG1811','MEG1821','MEG0741','MEG1631','MEG1841','MEG1831','MEG2011'],
                    'MAG_parietal_right':['MEG1041', 'MEG1111','MEG1121','MEG0721','MEG1141','MEG1131','MEG0731','MEG2211','MEG2221','MEG2241','MEG2231','MEG2441','MEG2021'],
                    'MAG_occipital_left':['MEG1641', 'MEG1721','MEG1711','MEG1911','MEG1941','MEG1731','MEG2041','MEG1921','MEG1931','MEG1741','MEG2111','MEG2141'],
                    'MAG_occipital_right':['MEG2431', 'MEG2521','MEG2531','MEG2311','MEG2321','MEG2511','MEG2031','MEG2341','MEG2331','MEG2541','MEG2121','MEG2131'],
                    'GRAD1_frontal_left':['MEG0122', 'MEG0342','MEG0312','MEG0322','MEG0512','MEG0542','MEG0332','MEG0522','MEG0532','MEG0612','MEG0642','MEG0622','MEG0822'],
                    'GRAD1_frontal_right':['MEG1412', 'MEG1222','MEG1212','MEG1232','MEG0922','MEG0932','MEG1242','MEG0912','MEG0942','MEG1022','MEG1032','MEG0812','MEG1012'],
                    'GRAD1_temporal_left':['MEG0112', 'MEG0132','MEG0212','MEG0222','MEG0142','MEG1512','MEG0242','MEG0232','MEG1542','MEG1522','MEG1612','MEG1622','MEG1532'],
                    'GRAD1_temporal_right':['MEG1422', 'MEG1312','MEG1322','MEG1442','MEG1432','MEG1342','MEG1332','MEG2612','MEG2412','MEG2422','MEG2642','MEG2622','MEG2632'],
                    'GRAD1_parietal_left':['MEG0412', 'MEG0422','MEG0632','MEG0442','MEG0432','MEG0712','MEG1812','MEG1822','MEG0742','MEG1632','MEG1842','MEG1832','MEG2012'],
                    'GRAD1_parietal_right':['MEG1042', 'MEG1112','MEG1122','MEG0722','MEG1142','MEG1132','MEG0732','MEG2212','MEG2222','MEG2242','MEG2232','MEG2442','MEG2022'],
                    'GRAD1_occipital_left':['MEG1642', 'MEG1722','MEG1712','MEG1912','MEG1942','MEG1732','MEG2042','MEG1922','MEG1932','MEG1742','MEG2112','MEG2142'],
                    'GRAD1_occipital_right':['MEG2432', 'MEG2522','MEG2532','MEG2312','MEG2322','MEG2512','MEG2032','MEG2342','MEG2332','MEG2542','MEG2122','MEG2132'],
                    'GRAD2_frontal_left':['MEG0123', 'MEG0343','MEG0313','MEG0323','MEG0513','MEG0543','MEG0333','MEG0523','MEG0533','MEG0613','MEG0643','MEG0623','MEG0823'],
                    'GRAD2_frontal_right':['MEG1413', 'MEG1223','MEG1213','MEG1233','MEG0923','MEG0933','MEG1243','MEG0913','MEG0943','MEG1023','MEG1033','MEG0813','MEG1013'],
                    'GRAD2_temporal_left':['MEG0113', 'MEG0133','MEG0213','MEG0223','MEG0143','MEG1513','MEG0243','MEG0233','MEG1543','MEG1523','MEG1613','MEG1623','MEG1533'],
                    'GRAD2_temporal_right':['MEG1423', 'MEG1313','MEG1323','MEG1443','MEG1433','MEG1343','MEG1333','MEG2613','MEG2413','MEG2423','MEG2643','MEG2623','MEG2633'],
                    'GRAD2_parietal_left':['MEG0413', 'MEG0423','MEG0633','MEG0443','MEG0433','MEG0713','MEG1813','MEG1823','MEG0743','MEG1633','MEG1843','MEG1833','MEG2013'],
                    'GRAD2_parietal_right':['MEG1043', 'MEG1113','MEG1123','MEG0723','MEG1143','MEG1133','MEG0733','MEG2213','MEG2223','MEG2243','MEG2233','MEG2443','MEG2023'],
                    'GRAD2_occipital_left':['MEG1643', 'MEG1723','MEG1713','MEG1913','MEG1943','MEG1733','MEG2043','MEG1923','MEG1933','MEG1743','MEG2113','MEG2143'],
                    'GRAD2_occipital_right':['MEG2433', 'MEG2523','MEG2533','MEG2313','MEG2323','MEG2513','MEG2033','MEG2343','MEG2333','MEG2543','MEG2123','MEG2133']
                    }
    
    # TODO check if vertical and horizontal IDs are correct!
    config["sensorsID"] = {"vgrad": "3",
                            "hgrad": "2", 
                            "magne": "1"}


    config['featuresCategories'] = ["offset", # 1
                                    "exponent", # 1
                                    "frequency_dominant_peak", # 5,
                                    "power_dominant_peak",# 5,
                                    "width_dominant_peak", # 5,
                                    "Canonical_Relative_Power", 
                                    "Canonical_Absolute_Power",
                                    "Individualized_Relative_Power ",
                                    "Individualized_Absolute_Power",
                                    ]
    


    config["features_names"], ch_names = make_features_Names(sensor_type = config["sensorType"],
                                                   ch_names = list(chain.from_iterable(config['channels_spatial_layouts'].values())),
                                                   feature_categories = config['featuresCategories'])
    
    config["ch_names"] = ch_names
    


    if path is not None:
        out_file = open(os.path.join(path, "configs.json"), "w") 
        json.dump(config, out_file, indent = 6) 
        out_file.close()

    return config 






def make_features_Names(sensor_type, ch_names, feature_categories):
    
    if sensor_type == "mag":
        ch_names = [channel for channel in ch_names if channel.endswith("1")]
    if sensor_type == "grad1":
        ch_names = [channel for channel in ch_names if channel.endswith("2")]
    if sensor_type == "grad2":
        ch_names = [channel for channel in ch_names if channel.endswith("3")]
    
    return sum([list(map(lambda feat: feat + ch, feature_categories)) for ch  in ch_names ], []), ch_names
    



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

    with open(path, "ab") as file:
        pickle.dump({subjId: [fooofModels, psds, freqs]}, file)



def mergeDataframes(path):
    """
    this function merges all extracted feature dataframes (.CSV) into
    a single .csv file"""
    
    paths = glob.glob(f"{path}/*.csv")
    dfs = [pd.read_csv(path) for path in paths]
    return pd.concat(dfs)






