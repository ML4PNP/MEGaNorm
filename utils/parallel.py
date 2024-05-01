import subprocess
from datetime import datetime

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
    