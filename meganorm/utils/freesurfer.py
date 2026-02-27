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
import time
import json
from datetime import datetime


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

    results = []
    for t1 in t1_files:
        t1_path = Path(t1)
        ses_match = re.search(r"(ses-[^/\\]+)", str(t1_path))
        session = ses_match.group(1) if ses_match else None
        job_label = f"{subject_id}_{session}" if session else subject_id
        results.append(
            {"subject_id": subject_id, "session": session, "t1_path": str(t1_path), "job_label": job_label}
        )
        break # TODO: For now ignores multiple runs and only run it for one available run
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

echo "Done: ${{SUBJECT_ID}}"
"""

    os.makedirs(processing_directory, exist_ok=True)
    script_path = os.path.join(processing_directory, script_filename)
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    return script_path

def is_success(
    results_directory: str,
    subject_id: str,
    token: str = "finished without error",
    tail_lines_to_scan: int = 200,
    fresh_minutes: int = 30,
    stalled_hours: int = 24,
) -> bool:
    
    status, _ = classify_subject_status(
        results_directory,
        subject_id,
        success_token=token,
        tail_lines_to_scan=tail_lines_to_scan,
        fresh_minutes=fresh_minutes,
        stalled_hours=stalled_hours,
    )
    return status == "success"


def run_parallel_reconall(
    subjects_directory,
    results_directory=None,
    processing_directory=".",
    freesurfer_path=None,
    file_postfix=".nii",
    skip_completed: bool = True,
    skip_running: bool = True,                         # NEW: avoid double-submitting active jobs
    resubmit_statuses: tuple[str, ...] = ("failed", "missing", "stalled"),  
    success_token: str = "finished without error",
    tail_lines_to_scan: int = 200,
    fresh_minutes: int = 30,
    stalled_hours: int = 24,
):
    """
    Submit recon-all for BIDS subjects, using classify_subject_status() to decide
    whether to (re)submit. Saves submission manifests to processing_directory.
    """

    Path(processing_directory).mkdir(parents=True, exist_ok=True)

    bids_root = Path(subjects_directory)
    subject_ids = sorted(d.name for d in bids_root.glob("sub-*") if d.is_dir())
    if not subject_ids:
        raise RuntimeError(f"No BIDS subjects found in {subjects_directory}")

    if results_directory is None:
        results_directory = str(bids_root / "derivatives" / "freesurfer")
    os.makedirs(results_directory, exist_ok=True)

    submitted: list[str] = []
    failed_job_submissions: list[str] = []

    submitted_records: list[dict] = []
    failed_records: list[dict] = []

    for subject_id in subject_ids:
        # Decide what to do based on unified status
        status, _info = classify_subject_status(
            results_directory,
            subject_id,
            success_token=success_token,
            tail_lines_to_scan=tail_lines_to_scan,
            fresh_minutes=fresh_minutes,
            stalled_hours=stalled_hours,
        )

        if skip_completed and status == "success":
            print(f"[SKIP] {subject_id} already finished without error.")
            continue
        if skip_running and status == "running":
            print(f"[SKIP] {subject_id} currently running.")
            continue
        if status not in resubmit_statuses and status != "success":
            # e.g., you might choose not to resubmit 'idle' or other states
            print(f"[SKIP] {subject_id}: status '{status}' not in resubmit_statuses.")
            continue

        # Discover T1w inputs
        t1_entries = find_bids_t1w_files(subjects_directory, subject_id)
        if not t1_entries:
            print(f"[WARN] No T1w for {subject_id}; skipping")
            failed_job_submissions.append(subject_id)
            failed_records.append({
                "subject_id": subject_id,
                "reason": "no_T1w_found",
                "attempt_time": datetime.now().isoformat(timespec="seconds"),
            })
            continue

        # Avoid re-importing T1 on reruns (keeps mri/orig clean)
        subj_fs_dir = Path(results_directory) / subject_id
        i_opt = not subj_fs_dir.exists()

        for entry in t1_entries:
            script_file_path = create_slurm_script(
                t1_path=entry["t1_path"],
                job_label=entry["job_label"],
                results_dir=results_directory,
                processing_directory=processing_directory,
                freesurfer_path=freesurfer_path,
                i_option=i_opt,
            )

            cmd = ["sbatch", script_file_path, subject_id]
            print(f"Submitting job: {entry['job_label']} -> {' '.join(cmd)}")

            res = subprocess.run(cmd, check=False, capture_output=True, text=True)

            # Parse SLURM job ID
            job_id = None
            m = re.search(r"Submitted batch job\s+(\d+)", (res.stdout or ""))
            if m:
                job_id = m.group(1)

            submitted.append(entry["job_label"])
            submitted_records.append({
                "subject_id": subject_id,
                "job_label": entry["job_label"],
                "session": entry.get("session"),
                "t1_path": entry["t1_path"],
                "script_path": script_file_path,
                "sbatch_cmd": " ".join(cmd),
                "job_id": job_id,
                "sbatch_returncode": res.returncode,
                "sbatch_stdout": (res.stdout or "").strip(),
                "sbatch_stderr": (res.stderr or "").strip(),
                "submit_time": datetime.now().isoformat(timespec="seconds"),
                "pre_status": status,  # status before submitting
            })

    # ---- Persist manifests ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    proc_dir = Path(processing_directory)

    with open(proc_dir / f"submitted_jobs_{ts}.json", "w") as f:
        json.dump(submitted_records, f, indent=2)

    with open(proc_dir / f"failed_jobs_{ts}.json", "w") as f:
        if failed_records:
            json.dump(failed_records, f, indent=2)
        else:
            json.dump([{"subject_id": sid, "reason": "no_T1w_found"} for sid in failed_job_submissions], f, indent=2)
    
    return submitted, failed_job_submissions



def log_tail_lines(path: Path, n: int = 200) -> list[str]:
    """Return the last n lines of a log file efficiently."""
    if not path.exists():
        return []
    with path.open("rb") as f:
        try:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            block = 4096
            data = b""
            while end > 0 and data.count(b"\n") <= n:
                start = max(0, end - block)
                f.seek(start)
                data = f.read(end - start) + data
                end = start
        except OSError:
            f.seek(0)
            data = f.read()
    lines = data.splitlines()[-n:]
    return [l.decode(errors="replace") for l in lines]


def discover_subjects(results_directory: str, exclude_subjects: set[str]) -> list[str]:
    root = Path(results_directory)
    if not root.is_dir():
        return []
    def keep(name: str) -> bool:
        if name.startswith("."): return False
        if name in exclude_subjects: return False
        if name.startswith("fsaverage"): return False
        return True
    with_logs = [p.name for p in root.iterdir()
                 if p.is_dir() and keep(p.name) and (p / "scripts" / "recon-all.log").exists()]
    return sorted(with_logs) if with_logs else sorted([p.name for p in root.iterdir() if p.is_dir() and keep(p.name)])


def classify_subject_status(
    results_directory: str,
    subject_id: str,
    *,
    success_token: str = "finished without error",
    tail_lines_to_scan: int = 200,
    fresh_minutes: int = 30,
    stalled_hours: int = 24,
) -> tuple[str, dict]:
    """
    Classify a single subject by inspecting recon-all logs and IsRunning* locks.

    Returns:
        (status, info_dict)
        status ∈ {"success","running","stalled","failed","missing"}
    """
    scripts_dir = Path(results_directory) / subject_id / "scripts"
    log_path = scripts_dir / "recon-all.log"

    # Gather candidate IsRunning* lock files (ignore backups)
    isrunning_files = []
    for p in scripts_dir.glob("IsRunning*"):
        name = p.name
        if name.endswith((".bak", ".old", "~")):
            continue
        if p.exists():
            isrunning_files.append(p)

    # Running/stalled logic based on mtimes
    now = time.time()
    fresh_s = fresh_minutes * 60
    stalled_s = stalled_hours * 3600

    log_mtime = log_path.stat().st_mtime if log_path.exists() else 0.0
    ir_mtimes = [p.stat().st_mtime for p in isrunning_files] if isrunning_files else []
    newest_ir = max(ir_mtimes) if ir_mtimes else 0.0

    recently_touched = (
        (log_path.exists() and now - log_mtime <= fresh_s) or
        (ir_mtimes and now - newest_ir <= fresh_s)
    )

    # Build info dict (filled incrementally)
    info: dict[str, object] = {
        "subject_id": subject_id,
        "log_path": str(log_path),
        "has_log": log_path.exists(),
        "is_running_files": [p.name for p in isrunning_files],
        "last_mod_time": log_mtime if log_path.exists() else None,
        "tail_excerpt": [],
        "error_hints": [],
    }

    # No log at all
    if not log_path.exists():
        status = "missing"
        info["status"] = status
        return status, info

    # Read tail and check for success
    tail = log_tail_lines(log_path, n=tail_lines_to_scan)
    info["tail_excerpt"] = tail

    if any(success_token in line for line in tail):
        status = "success"
        info["status"] = status
        return status, info

    # Not successful yet → decide running/stalled/failed
    if recently_touched:
        status = "running"
    elif isrunning_files and (
        ((not log_path.exists()) or (now - log_mtime > stalled_s)) and (now - newest_ir > stalled_s)
    ):
        status = "stalled"
    else:
        # Try to extract error hints from tail
        error_patterns = [re.compile(p, re.IGNORECASE) for p in [
            r"\bERROR\b", r"Segmentation (fault|violation)", r"\bKilled\b",
            r"out of memory", r"No space left on device", r"Bus error",
            r"floating point exception", r"Abort", r"assertion.*failed",
        ]]
        hints: list[str] = []
        for line in tail:
            if any(p.search(line) for p in error_patterns):
                hints.append(line.strip())
        info["error_hints"] = hints[-10:]
        status = "failed"

    info["status"] = status
    return status, info


def check_log_for_success(
    results_directory: str,
    subject_ids: list[str] | None = None,
    *,
    processing_directory: str | None = None,   # where to save failed manifests
    write_manifests: bool = True,              # save JSON outputs
    success_token: str = "finished without error",
    consider_running_as_failure: bool = False,
    tail_lines_to_scan: int = 200,
    fresh_minutes: int = 30,
    stalled_hours: int = 24,
    return_details: bool = True,               # return only FAILED/MISSING/STALLED by default
):
    """
    Scan SUBJECTS_DIR for recon-all outcomes, print a summary, and
    (optionally) write failed/stalled/missing manifests to processing_directory.

    Returns:
        dict[str, dict] of FAILED/MISSING/STALLED subjects (default),
        or list[str] of subject IDs if return_details=False.
    """
    # Subject discovery (skip FS templates)
    default_exclude = {
        "fsaverage","fsaverage_sym","fsaverage5","fsaverage6",
        "lh.EC_average","rh.EC_average","bert"
    }
    subjects = subject_ids if subject_ids else discover_subjects(results_directory, default_exclude)

    counts = {"success": 0, "running": 0, "stalled": 0, "failed": 0, "missing": 0}
    failed_ids: list[str] = []
    failed_details: dict[str, dict] = {}

    for sid in subjects:
        status, info = classify_subject_status(
            results_directory,
            sid,
            success_token=success_token,
            tail_lines_to_scan=tail_lines_to_scan,
            fresh_minutes=fresh_minutes,
            stalled_hours=stalled_hours,
        )

        counts[status] += 1

        if status in ("failed", "missing", "stalled") or (status == "running" and consider_running_as_failure):
            failed_ids.append(sid)
            failed_details[sid] = info

    # Summary
    checked = sum(counts.values())
    print(f"[check_log_for_success] Checked {checked} subjects "
          f"(success={counts['success']}, running={counts['running']}, "
          f"stalled={counts['stalled']}, failed={counts['failed']}, missing={counts['missing']}).")

    # Persist manifests (FAILED/MISSING/STALLED only)
    if write_manifests and processing_directory:
        Path(processing_directory).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_json = Path(processing_directory) / f"failed_jobs_{ts}.json"

        with open(ts_json, "w") as f:
            json.dump(failed_details, f, indent=2, default=str)


    return failed_details if return_details else failed_ids


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
        subjects = [s for s in os.listdir(freesurfer_dir) if os.path.isdir(os.path.join(freesurfer_dir, s))]

    df = pd.DataFrame(index=subjects, columns=["lh_en","rh_en","avg_en"])
    missing_subjects = []

    for s, sub in enumerate(subjects):
        sub_dir = os.path.join(freesurfer_dir, sub)
        log_file = os.path.join(sub_dir, "scripts", "recon-all.log")

        if not os.path.isdir(sub_dir):
            missing_subjects.append(sub); print(f"{s}: Subject {sub} is missing.")
            continue

        if os.path.exists(log_file):
            eno_line = None
            with open(log_file) as f:
                for line in f:
                    if "orig.nofix lheno" in line:
                        eno_line = line
            if eno_line:
                parts = eno_line.split()
                try:
                    eno_l, eno_r = float(parts[3].rstrip(",")), float(parts[6])
                    df.at[sub,"lh_en"] = eno_l
                    df.at[sub,"rh_en"] = eno_r
                    df.at[sub,"avg_en"] = (eno_l + eno_r)/2.0
                    print(f"{s}: Subject {sub} EN = {df.at[sub,'avg_en']:.3f}")
                    continue
                except Exception:
                    pass  # fall through to recompute
            print(f"{s}: Subject {sub} missing EN line, recomputing ...")

        try:
            # recompute with mris_euler_number
            lh_cmd = ["mris_euler_number", os.path.join(sub_dir,"surf","lh.orig.nofix")]
            rh_cmd = ["mris_euler_number", os.path.join(sub_dir,"surf","rh.orig.nofix")]
            lh_out = subprocess.run(lh_cmd, capture_output=True, text=True, check=True).stdout.split()
            rh_out = subprocess.run(rh_cmd, capture_output=True, text=True, check=True).stdout.split()
            # typically the value is near the end; be defensive
            eno_l = float([t for t in lh_out if re.fullmatch(r"-?\d+(\.\d+)?", t)][-1])
            eno_r = float([t for t in rh_out if re.fullmatch(r"-?\d+(\.\d+)?", t)][-1])
            df.at[sub,"lh_en"] = eno_l
            df.at[sub,"rh_en"] = eno_r
            df.at[sub,"avg_en"] = (eno_l + eno_r)/2.0
            print(f"{s}: Subject {sub} EN = {df.at[sub,'avg_en']:.3f}")
        except Exception as e:
            missing_subjects.append(sub)
            print(f"{s}: QC failed for {sub}: {e}")

    df = df.dropna()
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump({"ENs": df}, f)

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

    parser = argparse.ArgumentParser(description="Run FreeSurfer recon-all in parallel on a Slurm cluster.")
    parser.add_argument("subjects_directory", type=str, help="BIDS root with sub-*")
    parser.add_argument("processing_directory", type=str, help="Directory to write SLURM scripts/logs")
    parser.add_argument("--results-directory", type=str, default=None,
                        help="SUBJECTS_DIR (default: <BIDS>/derivatives/freesurfer)")
    parser.add_argument("--freesurfer-path", type=str, default=None,
                        help="Path to FreeSurfer install (default: $FREESURFER_HOME)")
    parser.add_argument("--no-skip-completed", action="store_true", help="Do not skip already successful subjects")
    args = parser.parse_args()

    submitted, failed = run_parallel_reconall(
        subjects_directory=args.subjects_directory,
        results_directory=args.results_directory,
        processing_directory=args.processing_directory,
        freesurfer_path=args.freesurfer_path,
        skip_completed=not args.no_skip_completed,
    )
    print(f"Submitted jobs: {len(submitted)}; missing T1w: {len(failed)}")