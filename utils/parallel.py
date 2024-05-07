import os
import time
import subprocess
from datetime import datetime
import pandas as pd

def progress_bar(current, total, bar_length=20):
    """
    Displays or updates a console progress bar.

    Args:
    current (int): Current progress (must be between 0 and total).
    total (int): Total steps for complete progress.
    bar_length (int): Character length of the bar.
    """
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '>' + '>'
    padding = (bar_length - len(arrow)) * ' '
    progress_percentage = round(fraction * 100, 1)

    print(f"\rProgress: [{'>' + arrow + padding}] {progress_percentage}%", end='')

    if current == total:
        print()  # Move to the next line when progress is complete.


def submit_jobs(data_path, subjects, temp_path):
    
    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    for s, subject in enumerate(subjects):
        fname = os.path.join(data_path, subject, 'mf2pt2_' + subject + '_ses-rest_task-rest_megtransdef.fif')
        if os.path.isfile(fname):
            subprocess.check_call("sbatch --job-name=%s ./src/batch_job.sh %s %s" % (subject, fname, temp_path), 
                                  shell=True)
        else:
            print('File does not exist!')
    
        progress_bar(s, len(subjects))
    
    return start_time


def check_jobs_status(username, start_time, delay = 20):
    
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
    
    
def collect_results(target_dir, subjects, temp_path, file_name='features'):
    
    all_features = []
    for subject in subjects:
        all_features.append(pd.read_csv(os.path.join(temp_path, subject + '.csv'), index_col=0))
    features = pd.concat(all_features)
    features.to_csv(os.path.join(target_dir, file_name + '.csv'))
    