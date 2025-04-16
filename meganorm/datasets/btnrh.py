import mne_bids
import glob
import shutil
import os
import mne
import pandas as pd


def load_BTNRH_data(feature_path, covariates_path):
    """
    load BTNRH dataset
    """
    BTNRH_covariates = pd.read_excel(
        "/project/meganorm/Data/BTNRH/Rempe_Ott_PNAS_2022_Data.xlsx", index_col=0
    )
    BTNRH_features = pd.read_csv(feature_path, index_col=0)

    BTNRH_covariates = BTNRH_covariates.rename(columns={"Sex": "gender", "Age": "age"})
    BTNRH_covariates.gender = BTNRH_covariates.gender.replace({"M": 0, "F": 1})

    BTNRH_data = BTNRH_covariates.join(BTNRH_features, how="inner")

    # Assigning 1 as the site for the BTNRH dataset
    BTNRH_data["site"] = 1

    return BTNRH_data


def mne_bids_BTNRH(input_base_path, output_path):

    # make new data directory
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        os.mkdir(output_path)

    raw_fpath = glob.glob(os.path.join(input_base_path, "*.fif"))
    for counter in range(len(raw_fpath)):

        subject_id = os.path.basename(raw_fpath[counter])[:7].split("-")[1]
        raw = mne.io.read_raw_fif(
            raw_fpath[counter], verbose=False, allow_maxshield=False
        )
        raw = raw.pick(picks=["meg"], verbose=False)

        raw.info["line_freq"] = 60
        bids_path = mne_bids.BIDSPath(
            task="rest", subject=subject_id, root=output_path, datatype="meg"
        )

        mne_bids.write_raw_bids(
            raw=raw, bids_path=bids_path, overwrite=True, verbose=False, symlink=True
        )

    return None


if __name__ == "__main__":

    input_base_path = "/project/meganorm/Data/BTNRH/ecr_fifs"
    output_path = "/project/meganorm/Data/BTNRH/BIDS"
    mne_bids_BTNRH(input_base_path=input_base_path, output_path=output_path)
