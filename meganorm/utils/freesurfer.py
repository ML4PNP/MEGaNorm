import os
import shutil
import glob
import subprocess
import argparse
import numpy as np
from pcntoolkit.util.utils import retrieve_freesurfer_eulernum


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


def list_subject_ids(directory, save_path=None):
    """Retrieves all folders in the given directory as subject IDs, and
    store them in a text file.

    Args:
        directory (str): Path to data directory.
        save_path (str): If specified, path to the text file to save the subject IDs,
        e.g. "/home/subjects.txt". Defaults to None.

    Returns:
        list: List of subject Ids.
    """

    subject_ids = [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]

    if save_path is not None:
        with open(save_path, "w") as file:
            for subject_id in subject_ids:
                file.write(f"{subject_id}\n")

    return subject_ids


def create_slurm_script(
    subjects_directory,
    subject_id,
    results_dir,
    processing_directory,
    freesurfer_path,
    nodes=1,
    ntasks=1,
    cpus_per_task=1,
    mem="16G",
    time="48:00:00",
    i_option=True,
    file_postfix=".nii",
):
    """
    Create a Slurm batch script for running recon-all with given parameters.
    """

    # TODO: This line works for camcan. Check for other datasets.
    t1_volume_path = os.path.join(
        subjects_directory, subject_id, "anat", subject_id + file_postfix
    )

    script_filename = subject_id + "_recon_all_slurm.sh"

    log_path = os.path.join(processing_directory, "log")
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    if i_option:
        recon_all_command = f"recon-all -subject ${{SUBJECT_ID}} -i ${{VOLUME}} -all"
    else:
        recon_all_command = f"recon-all -subject ${{SUBJECT_ID}} -all -no-isrunning"

    script_content = f"""#!/bin/bash
        #SBATCH --job-name={subject_id}      # Job name
        #SBATCH --nodes={nodes}            # Run all processes on a single node
        #SBATCH --ntasks={ntasks}          # Number of processes
        #SBATCH --cpus-per-task={cpus_per_task}  # Number of CPU cores per process
        #SBATCH --mem={mem}                # Total memory limit
        #SBATCH --time={time}              # Time limit hrs:min:sec
        #SBATCH --output={log_path}/%x_%j.log          # Standard output log
        #SBATCH --error={log_path}/%x_%j.err            # Standard error log

        # Source the FreeSurfer setup script
        export FREESURFER_HOME={freesurfer_path}
        source $FREESURFER_HOME/SetUpFreeSurfer.sh
        
        # Set the SUBJECTS_DIR environment variable
        export SUBJECTS_DIR={results_dir}

        # Specify the result directory
        RESULTS_DIR={results_dir}
        
        VOLUME={t1_volume_path}

        # Specify the subject ID
        SUBJECT_ID=$1

        # Run the recon-all command
        {recon_all_command}
        """

    with open(os.path.join(processing_directory, script_filename), "w") as file:
        file.write(script_content)

    # Make the script executable
    os.chmod(os.path.join(processing_directory, script_filename), 0o755)

    return os.path.join(processing_directory, script_filename)


def run_parallel_reconall(
    subjects_directory,
    results_directory,
    processing_directory,
    freesurfer_path,
    file_postfix=".nii",
):
    """Runs Freesurfer recon-all in parallel on an Slurm cluster.

    Args:
        subjects_directory (str): Path to data.
        results_directory (str): Path to save the results.
        processing_directory (str): Path to save the bash script.
        freesurfer_path (str): Path to freesurfer.
        file_postfix (str): file postfix for nifti files (could be different from one dataset to another).

    Returns:
        A list of subject IDs.

    """

    subject_ids = list_subject_ids(subjects_directory)

    for subject_id in subject_ids:

        script_file_path = create_slurm_script(
            subjects_directory,
            subject_id,
            results_directory,
            processing_directory,
            freesurfer_path,
            file_postfix=file_postfix,
        )

        command = ["sbatch", script_file_path, subject_id]

        print(f"Submitting job for subject: {subject_id}")

        subprocess.run(command, capture_output=True, text=True)

    return subject_ids


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
