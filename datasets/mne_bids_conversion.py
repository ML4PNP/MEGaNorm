import mne_bids
import glob
import shutil
import os
import mne
from pathlib import Path




def mne_bids_BTNRH(input_base_path,
                   output_path):
    

    if os.path.exists(output_path): shutil.rmtree(output_path); os.mkdir(output_path)

    raw_fpath = glob.glob(os.path.join(input_base_path, "*.fif"))

    for counter in range(len(raw_fpath)):

        subject_id = os.path.basename(raw_fpath[counter])[:7].split("-")[1]
        
        raw = mne.io.read_raw_fif(raw_fpath[counter], verbose=False)
        raw.info["line_freq"] = 60

        bids_path = mne_bids.BIDSPath(
            task="rest",
            subject=subject_id,
            root=output_path
        )

        # As stated in bellow link, apparantly CHPI signal must be droped. We have to investigate more later if we need other recordings
        # https://mne.discourse.group/t/chpi-channels-not-recognized-by-mne-bids-write-raw-bids/5609
        raw = raw.pick(picks=["meg"], verbose=False)
        
        mne_bids.write_raw_bids(raw=raw,
                           bids_path=bids_path,
                           overwrite=True,
                           verbose=False)
        
        
    return None



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

    input_base_path = "/project/meganorm/Data/BTNRH/ecr_fifs"
    output_path = "/project/meganorm/Data/BTNRH/BTNRH/BIDS_data"
    mne_bids_BTNRH(input_base_path=input_base_path, output_path=output_path)



    input_base_path = "/project/meganorm/Data/camcan/CamCAN/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003"
    output_path = "/project/meganorm/Data/BTNRH/CAMCAN/BIDS_data"
    mne_bids_CAMCAN(input_base_path=input_base_path, output_path=output_path)
