import os
import time
import shutil
import subprocess
from datetime import datetime
import pandas as pd


def progress_bar(current, total, bar_length=20):
    """
    Displays or updates a console progress bar.

    Parameters
    ----------
    current : int
        The current progress (must be between 0 and total).
    total : int
        The total steps for complete progress.
    bar_length : int, optional
        The character length of the progress bar. Default is 20.
    """
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * ">" + ">"
    padding = (bar_length - len(arrow)) * " "
    progress_percentage = round(fraction * 100, 1)

    print(f"\rProgress: [{'>' + arrow + padding}] {progress_percentage}%", end="")

    if current == total:
        print()  # Move to the next line when progress is complete.


def sbatchfile(
    mainParallel_path,
    bash_file_path,
    log_path=None,
    module="mne",
    time="1:00:00",
    memory="20GB",
    partition="normal",
    core=1,
    node=1,
    batch_file_name="batch_job",
    with_config=True,
):
    """
    Generates a batch script file for submission to a job scheduler (e.g., SLURM) for parallel execution.

    Parameters
    ----------
    mainParallel_path : str
        Path to the `mainParallel.py` script that will be executed in the batch job.
    bash_file_path : str
        Path where the generated batch job file will be saved.
    log_path : str, optional
        Path to the log file where output from the job will be saved. Default is None.
    module : str, optional
        The module to load in the batch job environment. Default is 'mne'.
    time : str, optional
        Maximum wall time for the job (format: HH:MM:SS). Default is '1:00:00'.
    memory : str, optional
        Amount of memory allocated for the job (e.g., '20GB'). Default is '20GB'.
    partition : str, optional
        The partition or queue to submit the job to. Default is 'normal'.
    core : int, optional
        Number of CPU cores to allocate for the job. Default is 1.
    node : int, optional
        Number of nodes to request for the job. Default is 1.
    batch_file_name : str, optional
        Name for the generated batch job file. Default is 'batch_job'.
    with_config : bool, optional
        Whether to include the configuration in the batch file. Default is True.

    Returns
    -------
    None
        This function generates a batch script file and saves it to the specified path.
    """
    sbatch_init = "#!/bin/bash\n"
    sbatch_nodes = "#SBATCH -N " + str(node) + "\n"
    sbatch_tasks = "#SBATCH -c " + str(core) + "\n"
    sbatch_partition = "#SBATCH -p " + partition + "\n"
    sbatch_time = "#SBATCH --time=" + time + "\n"
    sbatch_memory = "#SBATCH --mem=" + memory + "\n"
    sbatch_module = "source activate " + module + "\n"
    if log_path is not None:
        sbatch_log_out = "#SBATCH -o " + log_path + "/%x_%j.out" + "\n"
        sbatch_log_error = "#SBATCH -e " + log_path + "/%x_%j.err" + "\n"

    sbatch_input_1 = "source=$1\n"
    sbatch_input_2 = "target=$2\n"
    sbatch_input_3 = "subject=$3\n"
    sbatch_input_4 = "config=$4\n"

    if with_config:
        command = (
            "srun python "
            + mainParallel_path
            + " $source $target $subject --configs $config"
        )
    else:
        command = "srun python " + mainParallel_path + " $source $target $subject"

    bash_environment = [
        sbatch_init
        + sbatch_nodes
        + sbatch_tasks
        + sbatch_partition
        + sbatch_time
        + sbatch_memory
    ]

    if log_path is not None:
        bash_environment[0] += sbatch_log_out
        bash_environment[0] += sbatch_log_error

    bash_environment[0] += sbatch_module
    bash_environment[0] += sbatch_input_1
    bash_environment[0] += sbatch_input_2
    bash_environment[0] += sbatch_input_3
    if with_config:
        bash_environment[0] += sbatch_input_4
    bash_environment[0] += command

    job_path = os.path.join(bash_file_path, batch_file_name + ".sh")
    # writes bash file into processing dir
    with open(job_path, "w") as bash_file:
        bash_file.writelines(bash_environment)

    # changes permissoins for bash.sh file
    os.chmod(job_path, 0o770)

    return job_path


def submit_jobs(
    mainParallel_path,
    bash_file_path,
    subjects,
    temp_path,
    config_file=None,
    job_configs=None,
    progress=False,
):
    """
    Submits jobs for each subject to the SLURM cluster for parallel execution.

    Parameters
    ----------
    mainParallel_path : str
        Path to the `mainParallel.py` script that will be executed in the batch job.
    bash_file_path : str
        Path where the generated batch job file will be saved.
    subjects : dict
        A dictionary of subject names (keys) and their corresponding paths (values).
        Each subject will have a job submitted to the cluster.
    temp_path : str
        Path where temporary files will be stored.
    config_file : str, optional
        Path to a JSON configuration file. If provided, this will be passed to the batch job.
        Default is None.
    job_configs : dict, optional
        Dictionary containing job-specific configurations (e.g., memory, time, partition).
        Defaults to None, in which case default configurations will be used.
    progress : bool, optional
        Whether to show a progress bar during job submission. Default is False.

    Returns
    -------
    str
        The start time for the batch job submission, formatted as 'YYYY-MM-DDTHH:MM:SS'.
    """
    if not os.path.isdir(temp_path):
        os.makedirs(temp_path)

    if job_configs is None:
        job_configs = {
            "log_path": None,
            "module": "mne",
            "time": "1:00:00",
            "memory": "20GB",
            "partition": "normal",
            "core": 1,
            "node": 1,
            "batch_file_name": "batch_job",
        }

    batch_file = sbatchfile(
        mainParallel_path,
        bash_file_path,
        log_path=job_configs["log_path"],
        module=job_configs["module"],
        time=job_configs["time"],
        memory=job_configs["memory"],
        partition=job_configs["partition"],
        core=job_configs["core"],
        node=job_configs["node"],
        batch_file_name=job_configs["batch_file_name"],
        with_config=config_file is not None,
    )

    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    for s, subject in enumerate(subjects.keys()):
        # fname = os.path.join(subjects[subject], 'meg', subject + '_task-rest_meg.fif')
        fname = subjects[subject]
        # if os.path.exists(fname[0]):
        if config_file is None:
            subprocess.check_call(
                f"sbatch --job-name={subject} {batch_file} {fname} {temp_path} {subject}",
                shell=True,
            )
        else:
            subprocess.check_call(
                f"sbatch --job-name={subject} {batch_file} {fname} {temp_path} {subject} {config_file}",
                shell=True,
            )
        # else:
        #     print('File does not exist!')

        if progress:
            progress_bar(s, len(subjects))

    return start_time


def check_jobs_status(username, start_time, delay=20):
    """
    Checks the status of submitted jobs to the SLURM cluster.

    Parameters
    ----------
    username : str
        The SLURM username used to check the status of the jobs.
    start_time : str
        The start time for the batch job submission, formatted as 'YYYY-MM-DDTHH:MM:SS'.
        This is used to identify the specific set of jobs submitted in the `submit_jobs` function.
    delay : int, optional
        The delay, in seconds, between each check of job status. Default is 20 seconds.

    Returns
    -------
    list
        A list of names of jobs that have failed.
    """
    n = 1
    while n > 0:
        job_counts, failed_job_names = check_user_jobs(username, start_time)
        if job_counts:
            print(f"Status for user {username} from {start_time}: {job_counts}")
            if failed_job_names:
                print("Failed Jobs:", ", ".join(failed_job_names))
        else:
            print("No job data available.")
        n = job_counts["PENDING"] + job_counts["RUNNING"]
        time.sleep(delay)

    return failed_job_names


def check_user_jobs(username, start_time):
    """
    Utility function for counting the status of jobs submitted to the SLURM scheduler.

    Parameters
    ----------
    username : str
        The SLURM username used to check the status of the jobs.
    start_time : str
        The start time for the batch job submission, formatted as 'YYYY-MM-DDTHH:MM:SS'.
        This is used to filter the jobs that were submitted after the specified start time.

    Returns
    -------
    tuple
        A tuple containing:
        - status_counts : dict
            A dictionary with counts of jobs in various states (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED).
        - failed_jobs : list
            A list of job names that have failed.
    """
    try:
        # Format the current datetime to match Slurm's expected format
        end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # sacct command to get job states and names within the specified time frame
        cmd = [
            "sacct",
            "-n",
            "-X",
            "--parsable2",
            "--noheader",
            "-S",
            start_time,
            "-E",
            end_time,
            "-u",
            username,
            "--format=JobName,State",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("Failed to query jobs:", result.stderr)
            return

        # Initialize status counts and a list for failed job names
        status_counts = {
            "PENDING": 0,
            "RUNNING": 0,
            "COMPLETED": 0,
            "FAILED": 0,
            "CANCELLED": 0,
        }
        failed_jobs = []

        # Process each line to count statuses and collect names of failed jobs
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if line:
                parts = line.split("|")
                if len(parts) >= 2:
                    job_name, state = parts[0], parts[1]
                    if state == "PENDING":
                        status_counts["PENDING"] += 1
                    elif state == "RUNNING":
                        status_counts["RUNNING"] += 1
                    elif state == "COMPLETED":
                        status_counts["COMPLETED"] += 1
                    elif state == "FAILED":
                        status_counts["FAILED"] += 1
                        failed_jobs.append(job_name)
                    elif state == "CANCELLED":
                        status_counts["CANCELLED"] += 1

        return status_counts, failed_jobs

    except Exception as e:
        print("An error occurred while checking the job status:", str(e))
        return


def collect_results(target_dir, subjects, temp_path, file_name="features", clean=True):
    """
    Collects and merges the results of all jobs into a single file.

    Parameters
    ----------
    target_dir : str
        Path to the target directory where the merged results will be saved.
    subjects : dict
        A dictionary with subject names as keys and their corresponding file paths as values.
    temp_path : str
        Path to the temporary directory where individual subject result files are stored.
    file_name : str, optional
        The name of the file where the merged results will be saved. Default is 'features'.
    clean : bool, optional
        Whether to remove the temporary files after merging the results. Default is True.

    Returns
    -------
    None
        This function does not return anything but writes the merged results to a CSV file in the target directory.
    """

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    all_features = []
    for subject in subjects.keys():
        try:
            all_features.append(
                pd.read_csv(os.path.join(temp_path, subject + ".csv"), index_col=0)
            )
        except:
            continue
    features = pd.concat(all_features)
    features.to_csv(os.path.join(target_dir, file_name + ".csv"))
    if clean:
        shutil.rmtree(temp_path)


def auto_parallel_feature_extraction(
    mainParallel_path,
    features_dir,
    subjects,
    job_configs,
    config_file,
    username=None,
    auto_rerun=True,
    auto_collect=True,
    max_try=3,
):
    """
    Automatically submits, monitors, and reruns jobs for feature extraction on multiple subjects,
    and collects the results.

    Parameters
    ----------
    mainParallel_path : str
        Path to the `mainParallel.py` script that will be executed in parallel for each subject.
    features_dir : str
        Path to the directory where the feature extraction results and temporary files will be saved.
    subjects : dict
        A dictionary of subject names (keys) and their corresponding file paths (values).
    job_configs : dict
        Dictionary containing job configuration settings (e.g., memory, time, partition, etc.).
    config_file : str
        Path to a JSON configuration file containing additional settings for the feature extraction jobs.
    username : str, optional
        The SLURM username. If not provided, it will be fetched from the environment. Default is None.
    auto_rerun : bool, optional
        Whether to automatically rerun failed jobs. Default is True.
    auto_collect : bool, optional
        Whether to automatically collect and merge results after job completion. Default is True.
    max_try : int, optional
        The maximum number of retry attempts for failed jobs. Default is 3.

    Returns
    -------
    list
        A list of failed jobs after all attempts. If no jobs failed, the list will be empty.

    """
    features_temp_path = os.path.join(features_dir, "temp")

    if username is None:
        username = os.environ.get("USER")

    # Running Jobs
    start_time = submit_jobs(
        mainParallel_path,
        features_dir,
        subjects,
        features_temp_path,
        job_configs=job_configs,
        config_file=config_file,
    )
    # Checking jobs
    failed_jobs = check_jobs_status(username, start_time)

    falied_subjects = {failed_job: subjects[failed_job] for failed_job in failed_jobs}

    try_num = 0

    while len(failed_jobs) > 0 and auto_rerun and try_num < max_try:
        # Re-running Jobs
        start_time = submit_jobs(
            mainParallel_path,
            features_dir,
            falied_subjects,
            features_temp_path,
            job_configs=job_configs,
            config_file=config_file,
        )
        # Checking jobs
        failed_jobs = check_jobs_status(username, start_time)

        try_num += 1

    if auto_collect:
        collect_results(
            features_dir,
            subjects,
            features_temp_path,
            file_name="all_features",
            clean=False,
        )

    return failed_jobs
