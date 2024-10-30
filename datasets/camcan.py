import mne_bids
import glob
import shutil
import os
import mne
from utils.EEGlab import read_raw_eeglab
from pathlib import Path
import pandas as pd


def load_covariates(path):
    
    """Load age and gender for CamCAN dataset.

    Args:
        path (str): path to covariates.

    Returns:
        DataFrame: Pandas dataframe containing age and gender for camcan dataset.
    """
    
    df = pd.read_csv(path, sep='\t', index_col=0)
    df = df[['age', 'gender_code']]
    df = df.rename(columns={'gender_code':'gender'})
    df.gender = df.gender - 1 # 0 for males and 1 for females
    df.index.name = None
    df['site'] = np.zeros([df.shape[0],1], dtype=int)
    
    return df   


def load_camcan_data(feature_path, covariates_path):
    
    """Load camcan dataset

    Args:
        feature_path (str): Path to the the feature csv file.
        covariates_path (str): path to the covariates tsv file.

    Returns:
        DataFrame: Pandas dataframe with camcan covariates and features.
    """
    
    camcan_covariates = load_covariates(covariates_path)
    camcan_features = pd.read_csv(feature_path, index_col=0)
    camcan_data = camcan_covariates.join(camcan_features, how='inner')
    
    return camcan_data


def mne_bids_CAMCAN(input_base_path,
                   output_path):
    

    if os.path.exists(output_path): shutil.rmtree(output_path); os.mkdir(output_path)

    raw_fpath = glob.glob(os.path.join(input_base_path, "*/*.fif"))

    for counter in range(len(raw_fpath)):

        
        subject_id =Path(raw_fpath[counter]).parts[-2].split("-")[1]
        
        raw = mne.io.read_raw_fif(raw_fpath[counter], verbose=False)
        raw.info["line_freq"] = 50

        bids_path = mne_bids.BIDSPath(
            task="rest",
            subject=subject_id,
            root=output_path)
        
        # As stated in bellow link, apparantly CHPI signal must be droped. We have to investigate more later if we need other recordings
        # https://mne.discourse.group/t/chpi-channels-not-recognized-by-mne-bids-write-raw-bids/5609
        raw = raw.pick(picks=["meg", "ecg", "eog"], verbose=False)

        mne_bids.write_raw_bids(raw=raw,
                           bids_path=bids_path,
                           overwrite=True, 
                           verbose=False)
        
    return None
    
    


if __name__ == "__main__":
    input_base_path = "/project/meganorm/Data/camcan/CamCAN/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003"
    output_path = "/project/meganorm/Data/BTNRH/CAMCAN/BIDS_data"
    mne_bids_CAMCAN(input_base_path=input_base_path, output_path=output_path)