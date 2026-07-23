import os
import shlex
import time
import shutil
import subprocess
from datetime import datetime
import pandas as pd
import json
import meganorm
import meganorm.utils.parallel
import meganorm.src.mainParallel
from meganorm.utils.IO import set_path, merge_datasets_with_glob
from meganorm.utils.IO import Config
from meganorm.utils.IO import merge_fidp_demo


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
    freesurfer_home=None,
    freesurfer_license=None,
    with_config=None,
    # with_source_localization = None,
    # with_empty_room_recording = None
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

    if freesurfer_home:
        sbatch_module = (
            "source activate "
            + module
            + "\n"
            + f"export FREESURFER_HOME={freesurfer_home}\n"
            + f"export FREESURFER_LICENSE={freesurfer_license}\n"
            +
            # "chmod +x $FREESURFER_HOME/SetUpFreeSurfer.sh\n" +
            "source $FREESURFER_HOME/SetUpFreeSurfer.sh\n"
        )
    else:
        sbatch_module = "source activate " + module + "\n"

    if log_path is not None:
        sbatch_log_out = "#SBATCH -o " + log_path + "/%x_%j.out" + "\n"
        sbatch_log_error = "#SBATCH -e " + log_path + "/%x_%j.err" + "\n"

    sbatch_input_1 = "source=$1\n"
    sbatch_input_2 = "target=$2\n"
    sbatch_input_3 = "subject=$3\n"
    sbatch_input_4 = "config=$4\n"
    sbatch_input_5 = "line_freq=$5\n"
    sbatch_input_6 = "surfaces_dir=$6\n"
    sbatch_input_7 = "empty_room_recording_path=$7\n"
    sbatch_input_8 = "event_record=$8\n"
    sbatch_input_9 = "event_of_interest=$9\n"
    sbatch_input_10 = "device_type=${10}\n"
    sbatch_input_11 = "pos_file=${11}\n"
    sbatch_input_12 = "trans_file=${12}\n"

    # if with_config:
    command = (
        "srun --cpus-per-task="
        + str(core)
        + " python "
        + mainParallel_path
        + " $source $target $subject $config"
    )
    # command = (
    #     "srun --cpus-per-task="
    #     + str(core)
    #     + " xvfb-run -a --server-args='-screen 0 1920x1080x24' python "
    #     + mainParallel_path
    #     + " $source $target $subject $config"
    # )

    command += f" --line_freq $line_freq"
    command += f" --surfaces_dir $surfaces_dir"
    command += " --empty_room_recording_path $empty_room_recording_path"
    command += " --event_record $event_record"
    command += " --event_of_interest $event_of_interest"
    command += " --device_type $device_type"
    command += " --pos_file $pos_file"
    command += " --trans_file $trans_file"

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
    bash_environment[0] += sbatch_input_4
    bash_environment[0] += sbatch_input_5
    bash_environment[0] += sbatch_input_6
    bash_environment[0] += sbatch_input_7
    bash_environment[0] += sbatch_input_8
    bash_environment[0] += sbatch_input_9
    bash_environment[0] += sbatch_input_10
    bash_environment[0] += sbatch_input_11
    bash_environment[0] += sbatch_input_12

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
    freesurfer_home=None,
    freesurfer_license=None,
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

    def add_command(new_arg, command):
        if new_arg:
            command += f" {shlex.quote(str(new_arg))}"
        else:
            command += " None"
        return command

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
        freesurfer_home=freesurfer_home,
        freesurfer_license=freesurfer_license,
        with_config=config_file is not None,
        # with_source_localization=surfaces_dir is not None,
        # with_empty_room_recording=empty_room_recording is not None
    )

    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    for s, subject in enumerate(subjects.keys()):

        rs_fname = subjects[subject]["rest_record"]
        er_fname = subjects[subject]["empty_room_record"]
        event_record = subjects[subject].get("event_record")
        event_of_interest = subjects[subject].get("event_of_interest")
        mri_surface = subjects[subject]["mri_surface"]
        line_freq = subjects[subject]["line_freq"]
        device = subjects[subject]["device"]
        trans_path = subjects[subject].get("trans_path")  
        pos_path = subjects[subject].get("pos_path")

        command = f"sbatch --job-name={shlex.quote(subject)} {batch_file} {shlex.quote(rs_fname)} {temp_path} {subject} {shlex.quote(str(config_file))}"

        command = add_command(line_freq, command)
        command = add_command(mri_surface, command)
        command = add_command(er_fname, command)
        command = add_command(event_record, command)
        command = add_command(event_of_interest, command)
        command = add_command(device, command)
        command = add_command(pos_path, command)
        command = add_command(trans_path, command)
        

        subprocess.check_call(command, shell=True)

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
    delay : int, optional
        The delay, in seconds, between each status check. Default is 20 seconds.

    Returns
    -------
    list
        A list of names of jobs that have failed.
    """
    failed_job_names = []

    while True:
        job_counts, failed_job_names, ok = check_user_jobs(username, start_time)

        if not ok:
            # The sacct query itself failed (nonzero return or exception).
            # Wait and retry rather than crashing or falsely concluding the
            # jobs are done.
            print("Job status query failed; retrying...")
            time.sleep(delay)
            continue

        print(f"Status for user {username} from {start_time}: {job_counts}")
        if failed_job_names:
            print("Failed Jobs:", ", ".join(failed_job_names))

        n = (
            job_counts["PENDING"] + job_counts["RUNNING"] - 1
        )  # TODO: this "-1" should be removed: solution use job-id instead of time

        if n <= 0:
            break

        time.sleep(delay)

    return failed_job_names


def check_user_jobs(username, start_time):
    """
    Count the status of jobs submitted to the SLURM scheduler.

    Parameters
    ----------
    username : str
        The SLURM username used to check the status of the jobs.
    start_time : str
        The start time for the batch job submission, formatted as 'YYYY-MM-DDTHH:MM:SS'.

    Returns
    -------
    tuple
        A 3-tuple ``(status_counts, failed_jobs, ok)``:
        - status_counts : dict
            Counts of jobs per state (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED).
        - failed_jobs : list
            Job names that have failed.
        - ok : bool
            True if the sacct query succeeded, False otherwise. When False the
            counts are all zero and failed_jobs is empty.
    """
    empty_counts = {
        "PENDING": 0,
        "RUNNING": 0,
        "COMPLETED": 0,
        "FAILED": 0,
        "CANCELLED": 0,
    }

    try:
        end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

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
            return empty_counts.copy(), [], False

        status_counts = empty_counts.copy()
        failed_jobs = []

        lines = result.stdout.strip().split("\n")
        for line in lines:
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            job_name, state = parts[0], parts[1]
            # State can carry a suffix, e.g. "CANCELLED by 12345"
            state = state.split()[0]
            if state in status_counts:
                status_counts[state] += 1
            if state == "FAILED":
                failed_jobs.append(job_name)

        return status_counts, failed_jobs, True

    except Exception as e:
        print("An error occurred while checking the job status:", str(e))
        return empty_counts.copy(), [], False


def collect_results(
    target_dir, subjects, temp_path, file_name="features", clean=True, append=True
):
    """
    Collect per-subject result files and merge them into a single CSV.

    If ``append`` is True and an existing ``file_name``.csv is present in
    ``target_dir``, the newly extracted subjects are added to it. Subjects
    present in both the existing file and the new results are updated with
    the new values (new rows win).

    Parameters
    ----------
    target_dir : str
        Directory where the merged results CSV is written.
    subjects : dict
        Subject names (keys) whose per-subject CSVs are read from ``temp_path``.
    temp_path : str
        Directory holding the per-subject ``<subject>.csv`` files.
    file_name : str, optional
        Base name of the merged output file. Default "features".
    clean : bool, optional
        Remove ``temp_path`` after merging. Default True.
    append : bool, optional
        Merge into an existing output file instead of overwriting it.
        Default True.
    """
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    out_path = os.path.join(target_dir, file_name + ".csv")

    new_features = []
    for subject in subjects.keys():
        try:
            df = pd.read_csv(os.path.join(temp_path, subject + ".csv"), index_col=0)
        except Exception:
            continue
        # tag each row with its subject so we can dedup on rerun
        df["subject"] = subject
        new_features.append(df)

    if not new_features:
        print("No new per-subject result files were found; nothing collected.")
        # still clean temp if asked
        if clean and os.path.isdir(temp_path):
            shutil.rmtree(temp_path)
        return

    features = pd.concat(new_features)

    if append and os.path.exists(out_path):
        existing = pd.read_csv(out_path, index_col=0)
        if "subject" not in existing.columns:
            # older file without the tag; treat its index as the subject id
            existing["subject"] = existing.index
        combined = pd.concat([existing, features])
        # new rows come last, so keep="last" lets reruns overwrite old values
        combined = combined.drop_duplicates(subset="subject", keep="last")
    else:
        combined = features

    combined.to_csv(out_path)

    if clean and os.path.isdir(temp_path):
        shutil.rmtree(temp_path)


def auto_parallel_feature_extraction(
    mainParallel_path,
    project_dir,
    datasets,
    job_configs,
    config_file_path,
    which_subjects=None,
    username=None,
    auto_rerun=True,
    auto_collect=True,
    freesurfer_home=None,
    freesurfer_license=None,
    max_try=3,
    combine_features_and_demographics=False,
):
    """
    Automatically submits, monitors, and reruns jobs for feature extraction on multiple subjects,
    and collects the results.

    Parameters
    ----------
    mainParallel_path : str
        Path to the `mainParallel.py` script that will be executed in parallel for each subject.
    project_dir : str
        Root project directory containing the Features directory where
        results, temporary files, and configuration are stored.
    datasets : dict
        Mapping of dataset names to dataset metadata (e.g., base
        directory, surfaces directory), used to locate subjects and
        merge them via glob patterns.
    job_configs : dict
        Dictionary containing job configuration settings (e.g., memory, time, partition, etc.).
    config_file_path : str
        Path to a JSON configuration file containing additional settings for the feature extraction jobs.
    which_subjects : list or None, optional
        If provided, restrict processing to these subject IDs only.
        Default is None.
    username : str, optional
        The SLURM username. If not provided, it will be fetched from the environment. Default is None.
    auto_rerun : bool, optional
        Whether to automatically rerun failed jobs. Default is True.
    auto_collect : bool, optional
        Whether to automatically collect and merge results after job completion. Default is True.
    freesurfer_home : str or None, optional
        Path to the FreeSurfer installation directory, passed to each
        submitted job. Default is None.
    freesurfer_license : str or None, optional
        Path to the FreeSurfer license file, passed to each submitted
        job. Default is None.
    max_try : int, optional
        The maximum number of retry attempts for failed jobs. Default is 3.

    Returns
    -------
    list
        A list of failed jobs after all attempts. If no jobs failed, the list will be empty.

    Notes
    -----
    - Subjects missing a resting-state recording, failing MRI QC (when
      source localization and MRI QC are enabled without a template),
      or not present in `which_subjects` are excluded before submission.
      Excluded subject lists are written as JSON files under
      `Features/excluded_participants`.
    - Runner parameters are persisted to
      `Features/Configurations/runner_params.json` before job
      submission.
    - If `auto_collect` is True, per-subject results are merged and
      combined with demographic data into
      `Features/all_features.csv`.
    """
    features_dir = os.path.join(project_dir, "Features")
    subjects = merge_datasets_with_glob(datasets)
    conf = meganorm.utils.IO.Config.load(path=config_file_path)

    all_qc_passed_samples = []
    all_qc_failed_samples = []
    all_missing_samples = []
    if (
        conf.apply_source_localization
        and conf.apply_mri_QC
        and not conf.apply_mri_template
    ):
        for keys, values in datasets.items():
            qc_passed_samples, qc_failed_samples, missing_samples = (
                meganorm.utils.freesurfer.freesurfer_QC(values["surfaces_dir"])
            )
            all_qc_passed_samples.extend(qc_passed_samples)
            all_missing_samples.extend(missing_samples)
            all_qc_failed_samples.extend(qc_failed_samples)

    with open(
        os.path.join(
            features_dir, "excluded_participants", "failed_mri_qc_participants.json"
        ),
        "w",
    ) as file:
        json.dump(all_qc_failed_samples, file, indent=4)
    with open(
        os.path.join(
            features_dir, "excluded_participants", "missing_mri_participants.json"
        ),
        "w",
    ) as file:
        json.dump(all_missing_samples, file, indent=4)

    missing_meg_participants = []
    subjects_temp = subjects.copy()

    for subj, meta in subjects.items():

        if not meta["rest_record"]:
            missing_meg_participants.append(subj)
            subjects_temp.pop(subj)
            continue
        if all_qc_passed_samples and subj not in all_qc_passed_samples:
            subjects_temp.pop(subj)
            continue
        if which_subjects and subj not in which_subjects:
            subjects_temp.pop(subj)
            continue

    subjects = subjects_temp.copy()

    with open(
        os.path.join(
            features_dir, "excluded_participants", "missing_meg_participants.json"
        ),
        "w",
    ) as file:
        json.dump(missing_meg_participants, file, indent=4)

    with open(
        os.path.join(features_dir, "Configurations", "runner_params.json"), "r"
    ) as file:
        runner_params = json.load(file)
        runner_params["subjects"] = subjects
    with open(
        os.path.join(features_dir, "Configurations", "runner_params.json"), "w"
    ) as file:
        json.dump(runner_params, file, indent=4)

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
        config_file=config_file_path,
        freesurfer_home=freesurfer_home,
        freesurfer_license=freesurfer_license,
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
            config_file=config_file_path,
            freesurfer_home=freesurfer_home,
            freesurfer_license=freesurfer_license,
        )
        # Checking jobs
        failed_jobs = check_jobs_status(username, start_time)
        falied_subjects = {
            failed_job: subjects[failed_job] for failed_job in failed_jobs
        }

        try_num += 1

    if auto_collect:
        collect_results(
            features_dir,
            subjects,
            features_temp_path,
            file_name="all_features",
            clean=False,
        )

    # Merge demographic data and extracted f-IDPS
    if combine_features_and_demographics:
        data_base_dirs = [values["base_dir"] for values in datasets.values()]
        dataset_names = list(datasets.keys())
        df = merge_fidp_demo(
            datasets_paths=data_base_dirs,
            features_dir=features_dir,
            dataset_names=dataset_names,
        )
        df.to_csv(os.path.join(features_dir, "all_features.csv"))

    return failed_jobs


def sbatch_feature_extraction_runner(
    project_dir,
    datasets,
    job_configs,
    config_file=None,
    time="48:00:00",
    mem="16GB",
    freesurfer_home=None,
    freesurfer_license=None,
    auto_rerun=True,
    auto_collect=True,
    max_try=5,
    which_subjects=None,
    combine_features_and_demographics=False,
):
    """
    Set up and generate a SLURM sbatch script that launches the full
    parallel feature-extraction pipeline as a single driver job.

    Creates the project's Features directory structure, saves the
    pipeline configuration (custom or default), serializes all runner
    parameters needed by `auto_parallel_feature_extraction` to a JSON
    file, and writes an sbatch script that runs the parallel driver
    when submitted to the scheduler.

    Parameters
    ----------
    project_dir : str
        Root project directory in which the Features directory and
        outputs will be created.
    datasets : dict
        Mapping of dataset names to dataset metadata (e.g., base
        directory, surfaces directory), used to locate subjects and
        anatomical data.
    job_configs : dict
        SLURM job configuration, including keys such as "partition",
        "module", and "slurm_username". Updated in place with the
        computed "log_path".
    config_file : Config or None, optional
        A `meganorm.utils.IO.Config` instance specifying pipeline
        settings. If None, a default `Config` is created and saved.
        Default is None.
    time : str, optional
        Maximum wall time for the sbatch driver job (format
        "HH:MM:SS"). Default is "48:00:00".
    mem : str, optional
        Memory allocation for the sbatch driver job (e.g., "16GB").
        Default is "16GB".
    freesurfer_home : str or None, optional
        Path to the FreeSurfer installation directory, passed through
        to per-subject jobs. Default is None.
    freesurfer_license : str or None, optional
        Path to the FreeSurfer license file, passed through to
        per-subject jobs. Default is None.
    auto_rerun : bool, optional
        Whether failed per-subject jobs should be automatically
        resubmitted. Default is True.
    auto_collect : bool, optional
        Whether results should be automatically collected and merged
        after job completion. Default is True.
    max_try : int, optional
        Maximum number of rerun attempts for failed jobs. Default is 5.
    which_subjects : list or None, optional
        Optional list restricting processing to specific subject IDs.
        Default is None.

    Returns
    -------
    None
        Writes `runner_params.json` and
        `feature_extraction_runner.sbatch` to the project's Features
        directory.
    """

    features_dir, features_log_path = set_path(project_dir)
    job_configs["log_path"] = features_log_path

    features_dir = os.path.join(project_dir, "Features")
    config_file_path = os.path.join(
        features_dir, "Configurations", "Configuration.json"
    )
    if config_file:
        config_file.save(save_path=config_file_path, overwrite=True)
    else:
        conf = Config()
        conf.save(save_path=config_file_path)

    params = {
        "mainParallel_path": os.path.abspath(meganorm.src.mainParallel.__file__),
        "project_dir": project_dir,
        "config_file_path": config_file_path,
        "job_configs": job_configs,
        "username": job_configs["slurm_username"],
        "freesurfer_home": freesurfer_home,
        "freesurfer_license": freesurfer_license,
        "auto_rerun": auto_rerun,
        "auto_collect": auto_collect,
        "max_try": max_try,
        "which_subjects": which_subjects,
        "datasets": datasets,
        "combine_features_and_demographics": combine_features_and_demographics,
    }

    features_dir = os.path.join(project_dir, "Features")
    save_path = os.path.join(features_dir, "Configurations", "runner_params.json")
    with open(save_path, "w") as f:
        json.dump(params, f, indent=4)

    sbatch_text = f"""#!/bin/bash
#SBATCH --job-name=feature_extraction_runner
#SBATCH --output=Features/feature_extraction_runner.out
#SBATCH --error=Features/feature_extraction_runner.err
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=1
#SBATCH --partition={job_configs["partition"]}

# Activate your environment
source activate {job_configs["module"]}

python {os.path.abspath(meganorm.utils.parallel.__file__)}
"""

    save_path = os.path.join(features_dir, "feature_extraction_runner.sbatch")
    with open(save_path, "w") as f:
        f.write(sbatch_text)

    print("Created run_driver.sbatch")


if __name__ == "__main__":

    with open("Features/Configurations/runner_params.json") as f:
        params = json.load(f)

    if params.get("mainParallel_path", None) is None:
        params["mainParallel_path"] = os.path.abspath(mainParallel.__file__)

    # Run
    auto_parallel_feature_extraction(**params)
