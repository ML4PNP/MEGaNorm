import os
import time
import shutil
import subprocess
from datetime import datetime
import pandas as pd

def progress_bar(current, total, bar_length=20):
    
    """Displays or updates a console progress bar.

    Args:
        current (int): Current progress (must be between 0 and total).
        total (int): Total steps for complete progress.
        bar_length (int, optional): Character length of the bar. Defaults to 20.
    """    
    
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '>' + '>'
    padding = (bar_length - len(arrow)) * ' '
    progress_percentage = round(fraction * 100, 1)

    print(f"\rProgress: [{'>' + arrow + padding}] {progress_percentage}%", end='')

    if current == total:
        print()  # Move to the next line when progress is complete.


def sbatchfile(mainParallel_path,
                bash_file_path,
                log_path=None,
                module='mne',
                time='1:00:00',
                memory='20GB',
                partition='normal',
                core=1,
                node=1,
                batch_file_name='batch_job',
                with_config=True):
    """_summary_

    Args:
        mainParallel_path (str): Path to the mainParallel.py file.
        bash_file_path (str): Path to save the create batch job file.
        log_path (str, optional): _description_. Defaults to None.
        module (str, optional): _description_. Defaults to 'mne'.
        time (str, optional): _description_. Defaults to '1:00:00'.
        memory (str, optional): _description_. Defaults to '20GB'.
        partition (str, optional): _description_. Defaults to 'normal'.
        core (int, optional): _description_. Defaults to 1.
        node (int, optional): _description_. Defaults to 1.
        batch_file_name (str, optional): _description_. Defaults to 'batch_job'.
        with_config (boolean, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    
    sbatch_init = '#!/bin/bash\n'
    sbatch_nodes = '#SBATCH -N ' + str(node) + '\n'
    sbatch_tasks = '#SBATCH -c ' + str(core) + '\n'
    sbatch_partition = '#SBATCH -p ' + partition + '\n'
    sbatch_time = '#SBATCH --time=' + time + '\n'
    sbatch_memory = '#SBATCH --mem=' + memory + '\n'
    sbatch_module = 'source activate ' + module +'\n'
    if log_path is not None:
        sbatch_log_out = '#SBATCH -o ' + log_path + "/%x_%j.out" + '\n'
        sbatch_log_error =  '#SBATCH -e ' + log_path + "/%x_%j.err" + '\n'
    
    sbatch_input_1 = 'source=$1\n'
    sbatch_input_2 = 'target=$2\n'
    sbatch_input_3 = 'config=$3\n'
    
    if with_config:
        command = 'srun python ' + mainParallel_path + ' $source $target --configs $config'
    else:
        command = 'srun python ' + mainParallel_path + ' $source $target'

    bash_environment = [sbatch_init +
                        sbatch_nodes +
                        sbatch_tasks +
                        sbatch_partition +
                        sbatch_time +
                        sbatch_memory 
                        ]
    
    if log_path is not None:
        bash_environment[0] += sbatch_log_out
        bash_environment[0] += sbatch_log_error
    
    bash_environment[0] += sbatch_module
    bash_environment[0] += sbatch_input_1 
    bash_environment[0] += sbatch_input_2 
    if with_config:
        bash_environment[0] += sbatch_input_3
    bash_environment[0] += command

    job_path = os.path.join(bash_file_path, batch_file_name + '.sh')
    # writes bash file into processing dir
    with open(job_path, 'w') as bash_file:
        bash_file.writelines(bash_environment)

    # changes permissoins for bash.sh file
    os.chmod(job_path, 0o770)
    
    return job_path

def submit_jobs(mainParallel_path, bash_file_path, subjects, 
                temp_path, config_file=None, job_configs=None, progress=False):
    
    """Submits jobs for each subject to the Slurm cluster.

    Args:
        mainParallel_path (string): Path to the mainParallel.py.
        bash_file_path (string): Path to save the batch bash file.
        subjects (dict): A dictionary of subject names (key) and paths (values).
        temp_path (string): Path for saving temporary files.
        config_file (string): Path to the json config file. Defaults to None.
        job_configs (dictionary, optional): Dictionary of job configurations. Defaults to None.
        progress (bool, optional): Show the progress bar or not. Defaults to False.

    Returns:
        string: The start time for the batch job submission.
    """
    
    if not os.path.isdir(temp_path):
        os.makedirs(temp_path)
        
    if job_configs is None:
        job_configs = {'log_path':None, 'module':'mne', 'time':'1:00:00', 'memory':'20GB', 
                       'partition':'normal', 'core':1, 'node':1, 'batch_file_name':'batch_job'}
        
    
    batch_file = sbatchfile(mainParallel_path, bash_file_path, log_path=job_configs['log_path'], 
                            module=job_configs['module'], time=job_configs['time'], 
                            memory=job_configs['memory'], partition=job_configs['partition'], 
                            core=job_configs['core'], node=job_configs['node'],
                            batch_file_name=job_configs['batch_file_name'], with_config=config_file is not None)
    
    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    for s, subject in enumerate(subjects.keys()):
        #fname = os.path.join(subjects[subject], 'meg', subject + '_task-rest_meg.fif')
        fname = subjects[subject]
        if os.path.exists(fname):
            if config_file is None:
                subprocess.check_call(f"sbatch --job-name={subject} {batch_file} {fname} {temp_path}", 
                                  shell=True)
            else:
                subprocess.check_call(f"sbatch --job-name={subject} {batch_file} {fname} {temp_path} {config_file}", 
                                  shell=True)
        else:
            print('File does not exist!')
        
        if progress:
            progress_bar(s, len(subjects))
    
    return start_time


def check_jobs_status(username, start_time, delay=20):
    
    """ Checks the status of submitted jobs.

    Args:
        username (string): Slurm username.
        start_time (string): The start time for the batch job submission (see submit_jobs).
        delay (int, optional): The delay in seconds for checks. Defaults to 20.

    Returns:
        list: List of failed job names.
    """
    
    n = 1
    while n > 0: 
        job_counts, failed_job_names = check_user_jobs(username, start_time)
        if job_counts:
            print(f"Status for user {username} from {start_time}: {job_counts}")
            if failed_job_names:
                print("Failed Jobs:", ', '.join(failed_job_names))
        else:
            print("No job data available.")
        n = job_counts['PENDING'] + job_counts['RUNNING']
        time.sleep(delay)
    
    return failed_job_names
    


def check_user_jobs(username, start_time):
    
    """_ Utility function for counting the jobs with different stata.

    Args:
        username (string): Slurm username.
        start_time (string): The start time for the batch job submission (see submit_jobs).

    Returns:
        status_counts (dict): Dictionary of different job stata counts
        failed_jobs (list): list of failed jobs.
    """
    
    try:
        # Format the current datetime to match Slurm's expected format
        end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        
        # sacct command to get job states and names within the specified time frame
        cmd = ['sacct', '-n', '-X', '--parsable2', '--noheader',
               '-S', start_time, '-E', end_time, '-u', username,
               '--format=JobName,State']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Failed to query jobs:", result.stderr)
            return
        
        # Initialize status counts and a list for failed job names
        status_counts = {"PENDING": 0, "RUNNING": 0, "COMPLETED": 0, "FAILED": 0, "CANCELLED": 0}
        failed_jobs = []
        
        # Process each line to count statuses and collect names of failed jobs
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line:
                parts = line.split('|')
                if len(parts) >= 2:
                    job_name, state = parts[0], parts[1]
                    if state == 'PENDING':
                        status_counts["PENDING"] += 1
                    elif state == 'RUNNING':
                        status_counts["RUNNING"] += 1
                    elif state == 'COMPLETED':
                        status_counts["COMPLETED"] += 1
                    elif state == 'FAILED':
                        status_counts["FAILED"] += 1
                        failed_jobs.append(job_name)
                    elif state == 'CANCELLED':
                        status_counts["CANCELLED"] += 1

        return status_counts, failed_jobs

    except Exception as e:
        print("An error occurred while checking the job status:", str(e))
        return
    
    
def collect_results(target_dir, subjects, temp_path, file_name='features', clean=True):
    
    """Collects and merges the resulst of all jobs.

    Args:
        target_dir (str): Target directory path to save the collected results.
        subjects (dict): dict of subject names and paths.
        temp_path (str): Path to the temp directory.
        file_name (str, optional): The file name for the collected results. Defaults to 'features'.
        clean (bool, optional): Whether to clean the temporary files or not. Defaults to True.
    """
    
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    
    all_features = []
    for subject in subjects.keys():
        try:
            all_features.append(pd.read_csv(os.path.join(temp_path, subject + '.csv'), index_col=0))
        except: continue
    features = pd.concat(all_features)
    features.to_csv(os.path.join(target_dir, file_name + '.csv'))
    if clean:  
        shutil.rmtree(temp_path)
    
    
