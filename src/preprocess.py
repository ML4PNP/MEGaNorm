
import os
import sys
import mne
import tqdm
import json
import argparse
from glob import glob


# Add utils folder to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'utils')
sys.path.append(config_path)


from IO import make_config
from preprocessUtils import autoICA

import warnings
warnings.filterwarnings('ignore')



def preprocess(subjectPath:str, targetFS:int=1000, n_component:int=30,
        maxIter:int=800, IcaMethod:str="fastica", cutoffFreqLow:float=1, 
        cutoffFreqHigh:float=45, sensorType="meg"):
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

    maxIter: int
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
    
    

    # read the data
    data = mne.io.read_raw_fif(subjectPath,
                               verbose=False,
                               preload=True)
    
    # resample & band pass filter
    data.resample(targetFS, verbose=False, n_jobs=-1)
    data.filter(l_freq=cutoffFreqLow, 
				h_freq=cutoffFreqHigh, 
                n_jobs=-1, 
				verbose=False)

    # apply automated ICA
    data = autoICA(data=data, 
                n_components=n_component, # FLUX default
                max_iter=maxIter, # FLUX default,
                IcaMethod = IcaMethod,
                cutoffFreq=[cutoffFreqLow, cutoffFreqHigh],
                sensorType=sensorType)

    return data

    

    
    
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
								targetFS=configs["targetFS"],
								n_component=configs["n_component"],
								maxIter=configs["maxIter"],
								IcaMethod=configs["IcaMethod"],
								cutoffFreqLow=configs["cutoffFreqLow"],
								cutoffFreqHigh=configs["cutoffFreqHigh"],
								sensorType=configs["sensorType"])
		
		filteredData.save(f'{args.saveDir}/{subID}.fif', overwrite=True)
		

