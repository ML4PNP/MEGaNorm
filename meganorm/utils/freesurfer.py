import os
import shutil
import glob
import sys
import subprocess
import argparse
import pickle
import re
import numpy as np
import pandas as pd
from pathlib import Path


def get_freesurfer_home(freesurfer_path: str | None) -> str:
    """
    Resolve FreeSurfer installation path:
    - If freesurfer_path is provided, use it.
    - Else read $FREESURFER_HOME from the environment.
    """
    if freesurfer_path:
        return freesurfer_path
    fs_home = os.environ.get("FREESURFER_HOME")
    if not fs_home:
        raise EnvironmentError(
            "FREESURFER_HOME is not set and no freesurfer_path was provided. "
            "Set FREESURFER_HOME or pass freesurfer_path explicitly."
        )
    return fs_home

def find_bids_t1w_files(subjects_directory: str, subject_id: str):
    """
    Return a list of dicts for all T1w files for a BIDS subject.
    Handles no-session and multi-session.
    Each item:
      {
        "subject_id": "sub-01",
        "session": "ses-01" or None,
        "t1_path": "/.../sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz",
        "job_label": "sub-01_ses-01" or "sub-01"
      }
    """
    bids_root = Path(subjects_directory)
    subj_dir = bids_root / subject_id
    if not subj_dir.is_dir():
        return []

    patterns = [
        str(subj_dir / "anat" / f"{subject_id}_*T1w.nii.gz"),
        str(subj_dir / "anat" / f"{subject_id}_*T1w.nii"),
        str(subj_dir / "anat" / f"{subject_id}_*T1w.mgz"),
        str(subj_dir / "ses-*" / "anat" / f"{subject_id}_ses-*_T1w.nii.gz"),
        str(subj_dir / "ses-*" / "anat" / f"{subject_id}_ses-*_T1w.nii"),
        str(subj_dir / "ses-*" / "anat" / f"{subject_id}_ses-*_T1w.mgz"),
    ]

    t1_files = []
    for pat in patterns:
        t1_files.extend(glob.glob(pat))

    # De-duplicate preserving order
    seen = set()
    unique_t1_files = []
    for p in t1_files:
        if p not in seen:
            seen.add(p)
            unique_t1_files.append(p)

    results = []
    for t1 in unique_t1_files:
        t1_path = Path(t1)
        ses_match = re.search(r"(ses-[^/\\]+)", str(t1_path))
        session = ses_match.group(1) if ses_match else None
        job_label = f"{subject_id}_{session}" if session else subject_id
        results.append(
            {"subject_id": subject_id, "session": session, "t1_path": str(t1_path), "job_label": job_label}
        )
    return results

def prepare_mri_data(mri_directory):
    """This function is written to prepare the BTNRH MRI data for recon-all processing.

    Args:
        mri_directory (str): Directory to MRI data
    """
    subject_list = glob.glob(mri_directory + "/*.nii")
    subject_list = [os.path.basename(file).split(".")[0] for file in subject_list]

    for subject in subject_list:
        os.makedirs(os.path.join(mri_directory, subject, "anat"))
        shutil.move(
            os.path.join(mri_directory, subject + ".nii"),
            os.path.join(mri_directory, subject, "anat", subject + ".nii"),
        )


def create_slurm_script(
    t1_path,
    job_label,
    results_dir,              # DEFAULT: <BIDS_ROOT>/derivatives/freesurfer
    processing_directory,
    freesurfer_path,       
    nodes=1,
    ntasks=1,
    cpus_per_task=1,
    mem="16G",
    time="48:00:00",
    i_option=True,
):
    """
    Create a Slurm batch script for running recon-all with given parameters.
    BIDS-aware, and FreeSurfer path comes from freesurfer_path or $FREESURFER_HOME.
    """
    if not os.path.isfile(t1_path):
        raise FileNotFoundError(f"T1 path does not exist: {t1_path}")
    
    fs_home = get_freesurfer_home(freesurfer_path)

    # Paths & logs
    script_filename = f"{job_label}_recon_all_slurm.sh"
    log_path = os.path.join(processing_directory, "log")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    recon_all_command = (
        f"recon-all -s ${{SUBJECT_ID}} -i ${{VOLUME}} -all"
        if i_option else
        f"recon-all -s ${{SUBJECT_ID}} -all -no-isrunning"
    )
    bids_out_cmd = f"recon-all -s ${{SUBJECT_ID}} --bids-out || true"

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_label}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={log_path}/%x_%j.log
#SBATCH --error={log_path}/%x_%j.err

# FreeSurfer env (from resolved path or $FREESURFER_HOME)
export FREESURFER_HOME={fs_home}
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Output root (BIDS derivatives by default)
export SUBJECTS_DIR={results_dir}

# Input T1
VOLUME="{t1_path}"

# Keep BIDS subject id as-is (e.g., sub-01)
SUBJECT_ID="$1"

echo "Running recon-all for ${{SUBJECT_ID}} with input ${{VOLUME}}"
{recon_all_command}

# export BIDS-derivatives friendly outputs
{bids_out_cmd}

echo "Done: ${{SUBJECT_ID}}"
"""

    os.makedirs(processing_directory, exist_ok=True)
    script_path = os.path.join(processing_directory, script_filename)
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    return script_path


def run_parallel_reconall(
    subjects_directory,
    results_directory=None,        # DEFAULT: <BIDS_ROOT>/derivatives/freesurfer
    processing_directory=".",
    freesurfer_path=None,          # <— now optional; resolved from env if None
    file_postfix=".nii",
):
    """Runs FreeSurfer recon-all in parallel on a SLURM cluster for a BIDS dataset.
    - Submits one job per T1w file discovered:
      * sub-XX/anat/*T1w.*
      * sub-XX/ses-YY/anat/*T1w.*

    Args:
        subjects_directory (str): Path to data.
        results_directory (str): Path to save the results.
        processing_directory (str): Path to save the bash script.
        freesurfer_path (str): Path to freesurfer.
        file_postfix (str): file postfix for nifti files (could be different from one dataset to another).

    Returns:
        A list of submitted jobs.

    """

    bids_root = Path(subjects_directory)
    subject_ids = sorted(d.name for d in bids_root.glob("sub-*") if d.is_dir())
    if not subject_ids:
        raise RuntimeError(f"No BIDS subjects found in {subjects_directory}")

    if results_directory is None:
        results_directory = str(bids_root / "derivatives" / "freesurfer")
    os.makedirs(results_directory, exist_ok=True)

    submitted = []
    failed_job_submissions = []
    for subject_id in subject_ids:
        # discover all T1w files for subject (supports sessions)
        t1_entries = find_bids_t1w_files(subjects_directory, subject_id)

        if not t1_entries:
            # fallback to non-BIDS pattern if needed
            fallback = os.path.join(subjects_directory, subject_id, "anat", subject_id + file_postfix)
            if os.path.isfile(fallback):
                t1_entries = [{
                    "subject_id": subject_id,
                    "session": None,
                    "t1_path": fallback,
                    "job_label": subject_id,
                }]
            else:
                print(f"[WARN] No T1w for {subject_id}; skipping")
                failed_job_submissions.append(subject_id)
                continue

        for entry in t1_entries:
            script_file_path = create_slurm_script(
                t1_path=entry["t1_path"],
                job_label=entry["job_label"],
                results_dir=results_directory,
                processing_directory=processing_directory,
                freesurfer_path=freesurfer_path  # may be None -> resolved from env
            )

            # pass the BIDS subject id as-is
            command = ["sbatch", script_file_path, subject_id]
            print(f"Submitting job: {entry['job_label']} -> {' '.join(command)}")
            subprocess.run(command, check=False, capture_output=True, text=True)
            submitted.append(entry["job_label"])

    return submitted, failed_job_submissions


def check_log_for_success(results_directory, subject_ids):
    """Check the log file for the success message.

    Args:
        results_directory (str): Path for the freesurfer results.
        subject_ids (list): List of subject IDs.

    Returns:
        List: List of failed subject Ids.
    """
    failed_subjects = []

    for subject_id in subject_ids:
        log_path = os.path.join(
            results_directory, subject_id, "scripts", "recon-all.log"
        )
        if not os.path.exists(log_path):
            failed_subjects.append(subject_id)
        else:
            with open(log_path, "r") as f:
                log_content = f.read()
            if "finished without error" not in log_content:
                failed_subjects.append(subject_id)
    return failed_subjects


def rerun_failed_subs(
    failed_subjetcs,
    subjects_directory,
    results_directory,
    processing_directory,
    freesurfer_path,
    file_postfix=".nii",
):
    """Re-runs Freesurfer recon-all for failed subjects.

    Args:
        failed_subjetcs (list): List of failed subjects IDs.
        subjects_directory (str): Path to data.
        results_directory (str): Path to save the results.
        processing_directory (str): Path to save the bash script.
        freesurfer_path (str): Path to freesurfer.

    """

    for subject_id in failed_subjetcs:

        script_file_path = create_slurm_script(
            subjects_directory,
            subject_id,
            results_directory,
            processing_directory,
            freesurfer_path,
            i_option=False,
            file_postfix=file_postfix,
        )

        command = ["sbatch", script_file_path, subject_id]

        print(f"Submitting job for subject: {subject_id}")

        subprocess.run(command, capture_output=True, text=True)


def retrieve_freesurfer_eulernum(freesurfer_dir, subjects=None, save_path=None):
    """
    This function receives the freesurfer directory (including processed data 
    for several subjects) and retrieves the Euler number from the log files. If
    the log file does not exist, this function uses 'mris_euler_number' to recompute
    the Euler numbers (ENs). The function returns the ENs in a dataframe and 
    the list of missing subjects (that for which computing EN is failed). If 
    'save_path' is specified then the results will be saved in a pickle file.

    Basic usage::

        ENs, missing_subjects = retrieve_freesurfer_eulernum(freesurfer_dir)

    where the arguments are defined below.

    :param freesurfer_dir: absolute path to the Freesurfer directory.
    :param subjects: List of subject that we want to retrieve the ENs for. 
     If it is 'None' (the default), the list of the subjects will be automatically
     retreived from existing directories in the 'freesurfer_dir' (i.e. the ENs
     for all subjects will be retrieved).
    :param save_path: The path to save the results. If 'None' (default) the 
     results are not saves on the disk.


    :outputs: * ENs - A dataframe of retrieved ENs.
              * missing_subjects - The list of missing subjects.

    Developed by S.M. Kia

    """

    if subjects is None:
        subjects = [temp for temp in os.listdir(freesurfer_dir)
                    if os.path.isdir(os.path.join(freesurfer_dir, temp))]

    df = pd.DataFrame(index=subjects, columns=['lh_en', 'rh_en', 'avg_en'])
    missing_subjects = []

    for s, sub in enumerate(subjects):
        sub_dir = os.path.join(freesurfer_dir, sub)
        log_file = os.path.join(sub_dir, 'scripts', 'recon-all.log')

        if os.path.exists(sub_dir):
            if os.path.exists(log_file):
                with open(log_file) as f:
                    for line in f:
                        # find the part that refers to the EC
                        if re.search('orig.nofix lheno', line):
                            eno_line = line
                f.close()
                eno_l = eno_line.split()[3][0:-1]  # remove the trailing comma
                eno_r = eno_line.split()[6]
                euler = (float(eno_l) + float(eno_r)) / 2

                df.at[sub, 'lh_en'] = eno_l
                df.at[sub, 'rh_en'] = eno_r
                df.at[sub, 'avg_en'] = euler

                print('%d: Subject %s is successfully processed. EN = %f'
                      % (s, sub, df.at[sub, 'avg_en']))
            else:
                print('%d: Subject %s is missing log file, running QC ...' % (s, sub))
                try:
                    bashCommand = 'mris_euler_number ' + freesurfer_dir + \
                        sub + '/surf/lh.orig.nofix>' + 'temp_l.txt 2>&1'
                    res = subprocess.run(
                        bashCommand, stdout=subprocess.PIPE, shell=True)
                    file = open('temp_l.txt', mode='r', encoding='utf-8-sig')
                    lines = file.readlines()
                    file.close()
                    words = []
                    for line in lines:
                        line = line.strip()
                        words.append([item.strip()
                                     for item in line.split(' ')])
                    eno_l = np.float32(words[0][12])

                    bashCommand = 'mris_euler_number ' + freesurfer_dir + \
                        sub + '/surf/rh.orig.nofix>' + 'temp_r.txt 2>&1'
                    res = subprocess.run(
                        bashCommand, stdout=subprocess.PIPE, shell=True)
                    file = open('temp_r.txt', mode='r', encoding='utf-8-sig')
                    lines = file.readlines()
                    file.close()
                    words = []
                    for line in lines:
                        line = line.strip()
                        words.append([item.strip()
                                     for item in line.split(' ')])
                    eno_r = np.float32(words[0][12])

                    df.at[sub, 'lh_en'] = eno_l
                    df.at[sub, 'rh_en'] = eno_r
                    df.at[sub, 'avg_en'] = (eno_r + eno_l) / 2

                    print('%d: Subject %s is successfully processed. EN = %f'
                          % (s, sub, df.at[sub, 'avg_en']))

                except:
                    e = sys.exc_info()[0]
                    missing_subjects.append(sub)
                    print('%d: QC is failed for subject %s: %s.' % (s, sub, e))

        else:
            missing_subjects.append(sub)
            print('%d: Subject %s is missing.' % (s, sub))
        df = df.dropna()

        if save_path is not None:
            with open(save_path, 'wb') as file:
                pickle.dump({'ENs': df}, file)

    return df, missing_subjects



def freesurfer_QC(results_directory):
    """Performs Euler number based quality control on the results of Freesurfer.

    Args:
        results_directory (str): The path to the Freesurfer results directory.

    Returns:
        qc_passed_samples (list): List of passed QC subjects.
        qc_failed_samples (list): List of failed QC subjects.
        missing_samples (list): List of missing subjects.
    """

    euler_numbers, missing_samples = retrieve_freesurfer_eulernum(results_directory)

    euler_nums = euler_numbers["avg_en"].to_numpy(dtype=np.float32)

    qc_measure = np.sqrt(-(euler_nums)) - np.median(np.sqrt(-(euler_nums)))

    qc_passed_samples = list(euler_numbers.loc[qc_measure <= 5].index)
    qc_failed_samples = list(euler_numbers.loc[qc_measure > 5].index)

    return qc_passed_samples, qc_failed_samples, missing_samples


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run FreeSurfer recon-all in parallel on a Slurm cluster."
    )
    parser.add_argument(
        "subjects_directory",
        type=str,
        help="Path to the directory containing subject folders",
    )
    parser.add_argument(
        "results_directory", type=str, help="Path to the directory to save the results"
    )
    parser.add_argument("script_path", type=str, help="Path to save the Slurm script")
    parser.add_argument(
        "scripfreesurfer_patht_path", type=str, help="Path to freesurfer"
    )

    args = parser.parse_args()
    run_parallel_reconall(
        args.subjects_directory,
        args.results_directory,
        args.script_path,
        args.freesurfer_path,
    )
