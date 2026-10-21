import os
import posixpath
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
from meganorm.src.normative_modeling import anova_group_level_effect
from meganorm.utils.IO import check_demographic_format, load_demographic_file


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
    modules=None,
    conda_env="meganorm",
    log_path=None,
    time="1:00:00",
    memory="20GB",
    partition="normal",
    core=1,
    node=1,
    batch_file_name="batch_job",
    freesurfer_home=None,
    freesurfer_license=None,
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
    modules : list of str, optional
        List of modules to load in the batch job environment. Default is None.
    conda_env : str, optional
        The conda environment to activate in the batch job environment. Default is 'meganorm'.
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

    environment_setup = []
    environment_setup.extend(
        f"module load {shlex.quote(module_name)}\n" for module_name in (modules or [])
    )
    environment_setup.append(f"source activate {shlex.quote(conda_env)}\n")

    if freesurfer_home:
        environment_setup.append(
            f"export FREESURFER_HOME={shlex.quote(freesurfer_home)}\n"
        )
        if freesurfer_license:
            environment_setup.append(
                f"export FREESURFER_LICENSE={shlex.quote(freesurfer_license)}\n"
            )
        environment_setup.append('source "$FREESURFER_HOME/SetUpFreeSurfer.sh"\n')

    sbatch_module = "".join(environment_setup)

    if log_path is not None:
        # These paths are consumed by the Linux SLURM host, not this machine.
        output_log = shlex.quote(posixpath.join(log_path, "%x_%j.out"))
        error_log = shlex.quote(posixpath.join(log_path, "%x_%j.err"))
        sbatch_log_out = f"#SBATCH -o {output_log}\n"
        sbatch_log_error = f"#SBATCH -e {error_log}\n"

    sbatch_input_1 = 'source="$1"\n'
    sbatch_input_2 = 'target="$2"\n'
    sbatch_input_3 = 'subject="$3"\n'
    sbatch_input_4 = 'config="$4"\n'
    sbatch_input_5 = 'line_freq="$5"\n'
    sbatch_input_6 = 'surfaces_dir="$6"\n'
    sbatch_input_7 = 'empty_room_recording_path="$7"\n'
    sbatch_input_8 = 'event_record="$8"\n'
    sbatch_input_9 = 'event_of_interest="$9"\n'
    sbatch_input_10 = 'device_type="${10}"\n'
    sbatch_input_11 = 'pos_file="${11}"\n'
    sbatch_input_12 = 'trans_file="${12}"\n'
    sbatch_input_13 = 'annotation_path="${13}"\n'
    sbatch_input_14 = 'layout_path="${14}"\n'
    sbatch_input_15 = 'demographic_path="${15}"\n'

    command = (
        "srun --cpus-per-task="
        + str(core)
        + " python "
        + shlex.quote(mainParallel_path)
        + ' "$source" "$target" "$subject" "$config"'
    )

    command += ' --line_freq "$line_freq"'
    command += ' --surfaces_dir "$surfaces_dir"'
    command += ' --empty_room_recording_path "$empty_room_recording_path"'
    command += ' --event_record "$event_record"'
    command += ' --event_of_interest "$event_of_interest"'
    command += ' --device_type "$device_type"'
    command += ' --pos_file "$pos_file"'
    command += ' --trans_file "$trans_file"'
    command += ' --annotation_path "$annotation_path"'
    command += ' --layout_path "$layout_path"'
    command += ' --demographic_path "$demographic_path"'

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
    bash_environment[0] += sbatch_input_13
    bash_environment[0] += sbatch_input_14
    bash_environment[0] += sbatch_input_15

    bash_environment[0] += command

    job_path = os.path.join(bash_file_path, batch_file_name + ".sh")
    # writes bash file into processing dir
    with open(job_path, "w", newline="\n") as bash_file:
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
    return_job_ids=False,
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
    return_job_ids : bool, optional
        Return the start time and a mapping of submitted SLURM job IDs to
        subject names. The default returns only the start time.

    Returns
    -------
    str or tuple[str, dict[str, str]]
        Submission start time, formatted as 'YYYY-MM-DDTHH:MM:SS', and
        optionally the submitted job IDs.
    """

    def add_argument(command, value):
        command.append("None" if value is None else str(value))

    if not os.path.isdir(temp_path):
        os.makedirs(temp_path)

    if job_configs is None:
        job_configs = {
            "log_path": None,
            "conda_env": "meganorm",
            "modules": None,
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
        log_path=job_configs.get("log_path"),
        conda_env=job_configs.get("conda_env", "meganorm"),
        modules=job_configs.get("modules"),
        time=job_configs["time"],
        memory=job_configs["memory"],
        partition=job_configs["partition"],
        core=job_configs["core"],
        node=job_configs["node"],
        batch_file_name=job_configs["batch_file_name"],
        freesurfer_home=freesurfer_home,
        freesurfer_license=freesurfer_license,
    )

    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    job_ids = {}

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
        annotation_path = subjects[subject].get("annotation_path")
        layout_path = subjects[subject].get("layout_path")
        demographic_path = subjects[subject].get("demographic_path")

        command = [
            "sbatch",
            f"--job-name={subject}",
            batch_file,
            str(rs_fname),
            str(temp_path),
            str(subject),
            str(config_file),
        ]

        for value in (
            line_freq,
            mri_surface,
            er_fname,
            event_record,
            event_of_interest,
            device,
            pos_path,
            trans_path,
            annotation_path,
            layout_path,
            demographic_path,
        ):
            add_argument(command, value)

        if return_job_ids:
            command.insert(1, "--parsable")
            output = subprocess.check_output(command, text=True)
            job_id = output.strip().split(";")[0]
            if not job_id.isdigit():
                raise RuntimeError(f"Unexpected sbatch job ID: {output.strip()!r}")
            job_ids[job_id] = subject
        else:
            subprocess.check_call(command)

        if progress:
            progress_bar(s + 1, len(subjects))

    return (start_time, job_ids) if return_job_ids else start_time


def check_jobs_status(username, start_time, delay=20, job_ids=None):
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
    job_ids : dict[str, str] or None, optional
        Submitted job IDs mapped to their subject names. When provided, only
        those jobs are monitored.

    Returns
    -------
    list
        A list of names of jobs that have failed.
    """
    failed_job_names = []

    while True:
        if job_ids is None:
            job_counts, failed_job_names, ok = check_user_jobs(username, start_time)
        else:
            job_counts, failed_job_names, ok = check_user_jobs(
                username, start_time, job_ids=job_ids
            )

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

        n = job_counts["PENDING"] + job_counts["RUNNING"]

        if n <= 0:
            break

        time.sleep(delay)

    return failed_job_names


def check_user_jobs(username, start_time, job_ids=None):
    """
    Count the status of jobs submitted to the SLURM scheduler.

    Parameters
    ----------
    username : str
        The SLURM username used to check the status of the jobs.
    start_time : str
        The start time for the batch job submission, formatted as 'YYYY-MM-DDTHH:MM:SS'.

    job_ids : dict[str, str] or None, optional
        Submitted job IDs mapped to their subject names. When provided, only
        those jobs are counted, and failures use the original subject names.

    Returns
    -------
    status_counts : dict
        Counts in the existing pending, running, completed, failed and
        cancelled categories. Nonterminal flags and unrecognized states
        remain pending or running until a terminal state is reported.
    failed_jobs : list of str
        Names of jobs that failed.
    ok : bool
        Whether the ``sacct`` query completed successfully.
    """
    empty_counts = {
        "PENDING": 0,
        "RUNNING": 0,
        "COMPLETED": 0,
        "FAILED": 0,
        "CANCELLED": 0,
    }
    if job_ids is not None and not job_ids:
        return empty_counts.copy(), [], True

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
            "--format=JobIDRaw,JobName,State",
        ]
        if job_ids is not None:
            cmd.append("--jobs=" + ",".join(job_ids))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("Failed to query jobs:", result.stderr)
            return empty_counts.copy(), [], False

        status_counts = empty_counts.copy()
        failed_jobs = []
        current_job_id = os.environ.get("SLURM_JOB_ID")
        seen_job_ids = set()

        lines = result.stdout.strip().split("\n")
        for line in lines:
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            job_id, job_name, state = parts[0], parts[1], parts[2]
            if job_ids is not None:
                if job_id not in job_ids:
                    continue
                job_name = job_ids[job_id]
                seen_job_ids.add(job_id)
            if current_job_id and job_id == current_job_id:
                continue
            # State can carry a suffix, e.g. "CANCELLED by 12345"
            state = state.split()[0].rstrip("+")
            failed_states = {
                "BOOT_FAIL",
                "CANCELLED",
                "DEADLINE",
                "FAILED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "PREEMPTED",
                "TIMEOUT",
            }
            if state in {
                "RUNNING",
                "COMPLETING",
                "CONFIGURING",
                "POWER_UP_NODE",
                "RESIZING",
                "SIGNALING",
                "STAGE_OUT",
                "STOPPED",
                "SUSPENDED",
                "UPDATE_DB",
            }:
                status_counts["RUNNING"] += 1
            elif state in status_counts:
                status_counts[state] += 1
            elif state in failed_states:
                status_counts["FAILED"] += 1
            else:
                # A state flag can hide the base state. Holds, launch flags,
                # and unknown future states do not establish termination.
                status_counts["PENDING"] += 1
            if state in failed_states:
                failed_jobs.append(job_name)

        if job_ids is not None:
            # Newly submitted jobs may not have reached SLURM accounting yet.
            status_counts["PENDING"] += len(set(job_ids) - seen_job_ids)
        return status_counts, failed_jobs, True

    except Exception as e:
        print("An error occurred while checking the job status:", str(e))
        return empty_counts.copy(), [], False


def collect_results(
    target_dir,
    subjects,
    temp_path,
    file_name="features",
    clean=True,
    append=True,
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

    def read_features(path):
        # Read the identifier as a column first: pandas can infer a numeric
        # index from an unnamed CSV column even when dtype={0: str} is given.
        df = pd.read_csv(path, dtype={0: str, "subject": str})
        index_name = df.columns[0]
        df = df.set_index(index_name)
        if index_name.startswith("Unnamed:"):
            df.index.name = None
        return df

    new_features = []
    for subject in subjects.keys():
        try:
            df = read_features(os.path.join(temp_path, subject + ".csv"))
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
        existing = read_features(out_path)
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
    - If `auto_collect` is True, per-subject results are merged and
      written to `Features/all_features.csv`. Demographics are joined
      only when `combine_features_and_demographics` is True.
    """
    features_dir = os.path.join(project_dir, "Features")
    subjects = merge_datasets_with_glob(datasets)
    conf = meganorm.utils.IO.Config.load(path=config_file_path)

    from meganorm.src.source_localization import produce_aparc_a2009s_aseg

    if conf.apply_mri_template:
        produce_aparc_a2009s_aseg(
            save_path=conf.freesurfer_template_path,
            freesurfer_home=freesurfer_home,
            freesurfer_license=freesurfer_license,
        )

    all_qc_passed_samples = []
    all_qc_failed_samples = []
    all_missing_samples = []
    apply_mri_qc = (
        conf.apply_source_localization
        and conf.apply_mri_QC
        and not conf.apply_mri_template
    )
    if apply_mri_qc:
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
        if apply_mri_qc and subj not in all_qc_passed_samples:
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

    features_temp_path = os.path.join(features_dir, "temp")

    if username is None:
        username = os.environ.get("USER")

    # Running Jobs
    start_time, job_ids = submit_jobs(
        mainParallel_path,
        features_dir,
        subjects,
        features_temp_path,
        job_configs=job_configs,
        config_file=config_file_path,
        freesurfer_home=freesurfer_home,
        freesurfer_license=freesurfer_license,
        return_job_ids=True,
    )

    # Checking jobs
    failed_jobs = check_jobs_status(username, start_time, job_ids=job_ids)

    falied_subjects = {failed_job: subjects[failed_job] for failed_job in failed_jobs}

    try_num = 0

    while len(failed_jobs) > 0 and auto_rerun and try_num < max_try:
        # Re-running Jobs
        start_time, job_ids = submit_jobs(
            mainParallel_path,
            features_dir,
            falied_subjects,
            features_temp_path,
            job_configs=job_configs,
            config_file=config_file_path,
            freesurfer_home=freesurfer_home,
            freesurfer_license=freesurfer_license,
            return_job_ids=True,
        )
        # Checking jobs
        failed_jobs = check_jobs_status(username, start_time, job_ids=job_ids)
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
        demographic_paths = [
            values.get("demographic_path")
            or os.path.join(values["base_dir"], "participants_bids.tsv")
            for values in datasets.values()
        ]
        dataset_names = list(datasets.keys())
        df = merge_fidp_demo(
            demographic_paths=demographic_paths,
            features_dir=features_dir,
            dataset_names=dataset_names,
            drop_columns=None,
        )
        df.to_csv(os.path.join(features_dir, "all_features.csv"))

        for batch_effect in ["site", "sex"]:
            anova_group_level_effect(
                df,
                batch_effect,
                save_tag="raw",
                save_output_path=os.path.join(
                    project_dir, "Features/Saved_outputs/Grouping_effects"
                ),
            )

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
        "conda_env", "modules", and "slurm_username". Updated in place
        with the computed "log_path".
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

    if (
        config_file or Config()
    ).apply_mri_template or combine_features_and_demographics:
        for dataset_name, values in datasets.items():
            demo_path = values.get("demographic_path") or os.path.join(
                values["base_dir"], "participants_bids.tsv"
            )
            if not os.path.exists(demo_path):
                raise FileNotFoundError(
                    f"The demographic file for '{dataset_name}' was not found at "
                    f"{demo_path}. Set 'demographic_path' for this dataset, or "
                    "create the file with 'make_demo_file_bids'."
                )
            check_demographic_format(load_demographic_file(demo_path))

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

    module_setup = "".join(
        f"module load {shlex.quote(module_name)}\n"
        for module_name in job_configs.get("modules") or []
    )
    conda_env = shlex.quote(job_configs.get("conda_env", "meganorm"))

    sbatch_text = f"""#!/bin/bash
#SBATCH --job-name=feature_extraction_runner
#SBATCH --output=Features/feature_extraction_runner.out
#SBATCH --error=Features/feature_extraction_runner.err
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=1
#SBATCH --partition={job_configs["partition"]}

# Activate your environment
{module_setup}source activate {conda_env}

python {os.path.abspath(meganorm.utils.parallel.__file__)}
"""

    save_path = os.path.join(features_dir, "feature_extraction_runner.sbatch")
    with open(save_path, "w") as f:
        f.write(sbatch_text)

    print("Created feature_extraction_runner.sbatch")


def _load_runner_params(path):
    """Load driver arguments, accepting files written by earlier releases."""
    with open(path) as f:
        params = json.load(f)
    params.pop("subjects", None)
    if params.get("mainParallel_path", None) is None:
        params["mainParallel_path"] = os.path.abspath(
            meganorm.src.mainParallel.__file__
        )
    return params


if __name__ == "__main__":
    params = _load_runner_params("Features/Configurations/runner_params.json")

    auto_parallel_feature_extraction(**params)
