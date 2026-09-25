import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from meganorm.utils import freesurfer as fs


def _write_t1(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"T1")
    return path


def _write_log(root: Path, subject: str, text: str, mtime=None) -> Path:
    path = root / subject / "scripts" / "recon-all.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _write_lock(root: Path, subject: str, mtime: float) -> Path:
    path = root / subject / "scripts" / "IsRunning.lh+rh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("locked")
    os.utime(path, (mtime, mtime))
    return path


@pytest.mark.unit
def test_get_freesurfer_home_prefers_explicit_path(monkeypatch):
    monkeypatch.setenv("FREESURFER_HOME", "/from/environment")

    assert fs.get_freesurfer_home("/explicit") == "/explicit"


@pytest.mark.unit
def test_get_freesurfer_home_uses_environment(monkeypatch):
    monkeypatch.setenv("FREESURFER_HOME", "/from/environment")

    assert fs.get_freesurfer_home(None) == "/from/environment"


@pytest.mark.unit
def test_get_freesurfer_home_requires_a_configured_path(monkeypatch):
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    with pytest.raises(EnvironmentError, match="FREESURFER_HOME is not set"):
        fs.get_freesurfer_home(None)


@pytest.mark.unit
def test_find_bids_t1w_files_discovers_sessions_runs_and_formats(tmp_path):
    expected = [
        _write_t1(tmp_path / "sub-01" / "anat" / "sub-01_T1w.nii.gz"),
        _write_t1(
            tmp_path / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_run-02_T1w.nii"
        ),
        _write_t1(tmp_path / "sub-01" / "ses-02" / "anat" / "sub-01_ses-02_T1w.mgz"),
    ]
    _write_t1(tmp_path / "sub-01" / "anat" / "sub-01_T2w.nii.gz")

    found = fs.find_bids_t1w_files(str(tmp_path), "sub-01")

    assert [item["t1_path"] for item in found] == sorted(map(str, expected))
    assert {(item["session"], item["run"], item["job_label"]) for item in found} == {
        (None, None, "sub-01"),
        ("ses-01", "run-02", "sub-01_ses-01_run-02"),
        ("ses-02", None, "sub-01_ses-02"),
    }


@pytest.mark.unit
def test_find_bids_t1w_files_returns_empty_for_unknown_subject(tmp_path):
    assert fs.find_bids_t1w_files(str(tmp_path), "sub-missing") == []


@pytest.mark.unit
def test_prepare_mri_data_moves_flat_nii_files_into_subject_anat(tmp_path):
    (tmp_path / "sub-01.nii").write_bytes(b"one")
    (tmp_path / "sub-02.nii").write_bytes(b"two")
    (tmp_path / "notes.txt").write_text("keep")

    fs.prepare_mri_data(str(tmp_path))

    assert (tmp_path / "sub-01" / "anat" / "sub-01.nii").read_bytes() == b"one"
    assert (tmp_path / "sub-02" / "anat" / "sub-02.nii").read_bytes() == b"two"
    assert (tmp_path / "notes.txt").read_text() == "keep"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("i_option", "expected_command"),
    [
        (True, "recon-all -s ${SUBJECT_ID} -i ${VOLUME} -all"),
        (False, "recon-all -s ${SUBJECT_ID} -all -no-isrunning"),
    ],
)
def test_create_slurm_script_writes_executable_job(
    tmp_path, i_option, expected_command
):
    t1 = _write_t1(tmp_path / "bids" / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    processing = tmp_path / "processing"
    results = tmp_path / "results"

    script = Path(
        fs.create_slurm_script(
            str(t1),
            "sub-01",
            str(results),
            str(processing),
            "/opt/freesurfer",
            nodes=2,
            ntasks=3,
            cpus_per_task=4,
            mem="24G",
            time="12:00:00",
            i_option=i_option,
        )
    )
    content = script.read_text()

    assert script == processing / "sub-01_recon_all_slurm.sh"
    assert os.access(script, os.X_OK)
    assert results.is_dir()
    assert (processing / "log").is_dir()
    assert "#SBATCH --nodes=2" in content
    assert "#SBATCH --ntasks=3" in content
    assert "#SBATCH --cpus-per-task=4" in content
    assert "#SBATCH --mem=24G" in content
    assert "#SBATCH --time=12:00:00" in content
    assert "export FREESURFER_HOME=/opt/freesurfer" in content
    assert f"export SUBJECTS_DIR={results}" in content
    assert f'VOLUME="{t1}"' in content
    assert expected_command in content


@pytest.mark.unit
def test_create_slurm_script_rejects_missing_t1(tmp_path):
    with pytest.raises(FileNotFoundError, match="T1 path does not exist"):
        fs.create_slurm_script(
            str(tmp_path / "missing.nii.gz"),
            "sub-01",
            str(tmp_path / "results"),
            str(tmp_path / "processing"),
            "/opt/freesurfer",
        )


@pytest.mark.unit
def test_log_tail_lines_returns_only_requested_lines_and_handles_bad_bytes(tmp_path):
    log = tmp_path / "recon-all.log"
    log.write_bytes(b"first\nsecond\ninvalid-\xff\nfourth\n")

    tail = fs.log_tail_lines(log, n=2)

    assert tail == ["invalid-\ufffd", "fourth"]
    assert fs.log_tail_lines(tmp_path / "missing.log", n=2) == []


@pytest.mark.unit
def test_discover_subjects_prefers_processed_subjects_and_excludes_templates(tmp_path):
    _write_log(tmp_path, "sub-02", "working")
    _write_log(tmp_path, "sub-01", "working")
    (tmp_path / "sub-unprocessed").mkdir()
    (tmp_path / "fsaverage").mkdir()
    (tmp_path / ".hidden").mkdir()

    subjects = fs.discover_subjects(str(tmp_path), {"sub-02"})

    assert subjects == ["sub-01"]


@pytest.mark.unit
def test_discover_subjects_falls_back_to_directories_when_no_logs_exist(tmp_path):
    (tmp_path / "sub-02").mkdir()
    (tmp_path / "sub-01").mkdir()
    (tmp_path / "fsaverage6").mkdir()

    assert fs.discover_subjects(str(tmp_path), set()) == ["sub-01", "sub-02"]
    assert fs.discover_subjects(str(tmp_path / "unknown"), set()) == []


@pytest.mark.unit
def test_classify_subject_status_recognizes_success(monkeypatch, tmp_path):
    monkeypatch.setattr(fs.time, "time", lambda: 10_000.0)
    _write_log(tmp_path, "sub-01", "step\nfinished without error\n", mtime=100.0)

    status, info = fs.classify_subject_status(str(tmp_path), "sub-01")

    assert status == "success"
    assert info["status"] == "success"
    assert info["tail_excerpt"][-1] == "finished without error"


@pytest.mark.unit
def test_classify_subject_status_recognizes_recent_log_as_running(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(fs.time, "time", lambda: 10_000.0)
    _write_log(tmp_path, "sub-01", "still working\n", mtime=9_900.0)

    status, _ = fs.classify_subject_status(str(tmp_path), "sub-01", fresh_minutes=5)

    assert status == "running"


@pytest.mark.unit
def test_classify_subject_status_recognizes_stale_lock_as_stalled(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(fs.time, "time", lambda: 100_000.0)
    _write_log(tmp_path, "sub-01", "still working\n", mtime=1_000.0)
    _write_lock(tmp_path, "sub-01", mtime=1_000.0)

    status, info = fs.classify_subject_status(
        str(tmp_path), "sub-01", fresh_minutes=5, stalled_hours=1
    )

    assert status == "stalled"
    assert info["is_running_files"] == ["IsRunning.lh+rh"]


@pytest.mark.unit
def test_classify_subject_status_reports_error_hints_for_failed_job(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(fs.time, "time", lambda: 100_000.0)
    _write_log(
        tmp_path,
        "sub-01",
        "setup\nERROR: No space left on device\ncleanup\n",
        mtime=1_000.0,
    )

    status, info = fs.classify_subject_status(str(tmp_path), "sub-01")

    assert status == "failed"
    assert info["error_hints"] == ["ERROR: No space left on device"]


@pytest.mark.unit
def test_classify_subject_status_reports_missing_without_log_or_lock(tmp_path):
    status, info = fs.classify_subject_status(str(tmp_path), "sub-01")

    assert status == "missing"
    assert info["has_log"] is False


@pytest.mark.unit
def test_classify_subject_status_uses_fresh_lock_when_log_has_not_started(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(fs.time, "time", lambda: 10_000.0)
    _write_lock(tmp_path, "sub-01", mtime=9_900.0)

    status, _ = fs.classify_subject_status(str(tmp_path), "sub-01", fresh_minutes=5)

    assert status == "running"


@pytest.mark.unit
def test_classify_subject_status_uses_stale_lock_when_log_is_absent(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(fs.time, "time", lambda: 100_000.0)
    _write_lock(tmp_path, "sub-01", mtime=1_000.0)

    status, _ = fs.classify_subject_status(
        str(tmp_path), "sub-01", fresh_minutes=5, stalled_hours=1
    )

    assert status == "stalled"


@pytest.mark.unit
def test_classify_subject_status_keeps_lock_running_until_stalled_threshold(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(fs.time, "time", lambda: 100_000.0)
    _write_lock(tmp_path, "sub-01", mtime=96_400.0)

    status, _ = fs.classify_subject_status(
        str(tmp_path), "sub-01", fresh_minutes=5, stalled_hours=2
    )

    assert status == "running"


@pytest.mark.unit
def test_is_success_delegates_to_status_classification(tmp_path):
    _write_log(tmp_path, "sub-01", "CUSTOM SUCCESS\n")

    assert fs.is_success(str(tmp_path), "sub-01", token="CUSTOM SUCCESS") is True
    assert fs.is_success(str(tmp_path), "sub-02") is False


@pytest.mark.unit
def test_check_log_for_success_returns_failures_and_writes_manifest(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(fs.time, "time", lambda: 100_000.0)
    results = tmp_path / "results"
    processing = tmp_path / "processing"
    _write_log(results, "sub-success", "finished without error\n", mtime=1_000.0)
    _write_log(results, "sub-failed", "ERROR: failed\n", mtime=1_000.0)
    _write_lock(results, "sub-running", mtime=99_900.0)

    details = fs.check_log_for_success(
        str(results),
        ["sub-success", "sub-failed", "sub-running", "sub-missing"],
        processing_directory=str(processing),
        consider_running_as_failure=True,
    )

    assert set(details) == {"sub-failed", "sub-running", "sub-missing"}
    assert "success=1, running=1" in capsys.readouterr().out
    manifests = list(processing.glob("failed_jobs_*.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text()) == details


@pytest.mark.unit
def test_check_log_for_success_can_return_ids_without_writing(tmp_path):
    _write_log(tmp_path, "sub-failed", "ERROR: failed\n", mtime=1.0)

    failed = fs.check_log_for_success(
        str(tmp_path),
        ["sub-failed", "sub-missing"],
        write_manifests=False,
        return_details=False,
    )

    assert failed == ["sub-failed", "sub-missing"]


@pytest.fixture
def submission_doubles(monkeypatch):
    script_calls = []
    sbatch_calls = []

    def create_script(**kwargs):
        script_calls.append(kwargs)
        path = Path(kwargs["processing_directory"]) / f"{kwargs['job_label']}.sh"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\n")
        return str(path)

    def submit(cmd, **kwargs):
        sbatch_calls.append((cmd, kwargs))
        return SimpleNamespace(
            returncode=0, stdout="Submitted batch job 12345\n", stderr=""
        )

    monkeypatch.setattr(fs, "create_slurm_script", create_script)
    monkeypatch.setattr(fs.subprocess, "run", submit)
    return script_calls, sbatch_calls


@pytest.mark.unit
def test_run_parallel_reconall_submits_first_t1_and_records_job(
    tmp_path, submission_doubles
):
    script_calls, sbatch_calls = submission_doubles
    bids = tmp_path / "bids"
    results = tmp_path / "results"
    processing = tmp_path / "processing"
    first = _write_t1(bids / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz")
    _write_t1(bids / "sub-01" / "ses-02" / "anat" / "sub-01_ses-02_T1w.nii.gz")

    submitted, failed = fs.run_parallel_reconall(
        str(bids),
        str(results),
        str(processing),
        freesurfer_path="/opt/freesurfer",
    )

    assert submitted == ["sub-01_ses-01"]
    assert failed == []
    assert script_calls[0]["t1_path"] == str(first)
    assert script_calls[0]["i_option"] is True
    assert sbatch_calls[0][0][-1] == "sub-01"
    records = json.loads(next(processing.glob("submitted_jobs_*.json")).read_text())
    assert records[0]["job_id"] == "12345"
    assert records[0]["fs_subject_id"] == "sub-01"
    assert json.loads(next(processing.glob("failed_jobs_*.json")).read_text()) == []


@pytest.mark.unit
def test_run_parallel_reconall_processes_each_selected_session(
    tmp_path, submission_doubles
):
    script_calls, sbatch_calls = submission_doubles
    bids = tmp_path / "bids"
    for session in ("ses-01", "ses-02", "ses-03"):
        _write_t1(bids / "sub-01" / session / "anat" / f"sub-01_{session}_T1w.nii.gz")

    submitted, failed = fs.run_parallel_reconall(
        str(bids),
        str(tmp_path / "results"),
        str(tmp_path / "processing"),
        selected_subjects="sub-01",
        selected_sessions=["ses-02", "ses-03"],
    )

    assert submitted == ["sub-01_ses-02", "sub-01_ses-03"]
    assert failed == []
    assert [call["job_label"] for call in script_calls] == [
        "sub-01_ses-02",
        "sub-01_ses-03",
    ]
    assert [call[0][-1] for call in sbatch_calls] == [
        "sub-01_ses-02",
        "sub-01_ses-03",
    ]


@pytest.mark.unit
def test_run_parallel_reconall_skips_completed_and_running_subjects(
    monkeypatch, tmp_path, submission_doubles
):
    script_calls, sbatch_calls = submission_doubles
    monkeypatch.setattr(fs.time, "time", lambda: 10_000.0)
    bids = tmp_path / "bids"
    results = tmp_path / "results"
    for subject in ("sub-complete", "sub-running"):
        _write_t1(bids / subject / "anat" / f"{subject}_T1w.nii.gz")
    _write_log(results, "sub-complete", "finished without error\n", mtime=1.0)
    _write_lock(results, "sub-running", mtime=9_900.0)

    submitted, failed = fs.run_parallel_reconall(
        str(bids), str(results), str(tmp_path / "processing")
    )

    assert (submitted, failed) == ([], [])
    assert script_calls == []
    assert sbatch_calls == []


@pytest.mark.unit
def test_run_parallel_reconall_records_missing_t1_and_failed_submission(
    monkeypatch, tmp_path, submission_doubles
):
    _, sbatch_calls = submission_doubles
    bids = tmp_path / "bids"
    processing = tmp_path / "processing"
    (bids / "sub-missing").mkdir(parents=True)
    _write_t1(bids / "sub-submit" / "anat" / "sub-submit_T1w.nii.gz")

    def failed_submit(cmd, **kwargs):
        sbatch_calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=1, stdout="", stderr="queue unavailable")

    monkeypatch.setattr(fs.subprocess, "run", failed_submit)

    submitted, failed = fs.run_parallel_reconall(
        str(bids),
        str(tmp_path / "results"),
        str(processing),
    )

    assert submitted == []
    assert failed == ["sub-missing", "sub-submit"]
    records = json.loads(next(processing.glob("failed_jobs_*.json")).read_text())
    assert {record["reason"] for record in records} == {
        "no_T1w_found",
        "sbatch_submission_failed",
    }
    submission = next(r for r in records if r["reason"] == "sbatch_submission_failed")
    assert submission["sbatch_stderr"] == "queue unavailable"


@pytest.mark.unit
def test_run_parallel_reconall_reports_no_matching_selected_session(
    tmp_path, submission_doubles
):
    bids = tmp_path / "bids"
    processing = tmp_path / "processing"
    _write_t1(bids / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz")

    submitted, failed = fs.run_parallel_reconall(
        str(bids),
        str(tmp_path / "results"),
        str(processing),
        selected_sessions="ses-99",
    )

    assert submitted == []
    assert failed == ["sub-01"]
    records = json.loads(next(processing.glob("failed_jobs_*.json")).read_text())
    assert records[0]["reason"] == "no_matching_session_found"
    assert records[0]["requested_sessions"] == ["ses-99"]


@pytest.mark.unit
def test_run_parallel_reconall_validates_subject_selection(tmp_path):
    bids = tmp_path / "bids"
    bids.mkdir()

    with pytest.raises(RuntimeError, match="No BIDS subjects found"):
        fs.run_parallel_reconall(str(bids), processing_directory=str(tmp_path / "p1"))

    (bids / "sub-01").mkdir()
    with pytest.raises(RuntimeError, match="selected subjects were not found"):
        fs.run_parallel_reconall(
            str(bids),
            processing_directory=str(tmp_path / "p2"),
            selected_subjects="sub-99",
        )


@pytest.mark.unit
def test_retrieve_freesurfer_eulernum_reads_values_from_log(tmp_path):
    _write_log(
        tmp_path,
        "sub-01",
        "earlier output\norig.nofix lheno = -10, rhno = -12\n",
    )

    values, missing = fs.retrieve_freesurfer_eulernum(str(tmp_path))

    assert missing == []
    assert values.loc["sub-01"].to_dict() == {
        "lh_en": -10.0,
        "rh_en": -12.0,
        "avg_en": -11.0,
    }


@pytest.mark.unit
def test_retrieve_freesurfer_eulernum_recomputes_missing_log_values(
    monkeypatch, tmp_path
):
    subject = tmp_path / "sub-01"
    (subject / "surf").mkdir(parents=True)
    commands = []

    def run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        value = "-20" if "lh.orig.nofix" in cmd[-1] else "-24"
        return SimpleNamespace(stdout=f"Euler number is {value}\n")

    monkeypatch.setattr(fs.subprocess, "run", run)

    values, missing = fs.retrieve_freesurfer_eulernum(
        str(tmp_path), subjects=["sub-01"]
    )

    assert missing == []
    assert values.loc["sub-01", "avg_en"] == -22.0
    assert [Path(call[0][-1]).name for call in commands] == [
        "lh.orig.nofix",
        "rh.orig.nofix",
    ]
    assert all(call[1]["check"] is True for call in commands)


@pytest.mark.unit
def test_retrieve_freesurfer_eulernum_tracks_missing_and_failed_subjects(
    monkeypatch, tmp_path
):
    (tmp_path / "sub-failed" / "surf").mkdir(parents=True)

    def fail(*args, **kwargs):
        raise fs.subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(fs.subprocess, "run", fail)

    values, missing = fs.retrieve_freesurfer_eulernum(
        str(tmp_path), subjects=["sub-absent", "sub-failed"]
    )

    assert values.empty
    assert missing == ["sub-absent", "sub-failed"]


@pytest.mark.unit
def test_retrieve_freesurfer_eulernum_saves_dataframe(tmp_path):
    _write_log(tmp_path, "sub-01", "orig.nofix lheno = -8, rhno = -10\n")
    save_path = tmp_path / "euler.pkl"

    values, _ = fs.retrieve_freesurfer_eulernum(str(tmp_path), save_path=str(save_path))

    with save_path.open("rb") as stream:
        saved = pickle.load(stream)
    pd.testing.assert_frame_equal(saved["ENs"], values)


@pytest.mark.unit
def test_freesurfer_qc_mad_flags_only_unusually_negative_euler(monkeypatch):
    values = pd.DataFrame(
        {
            "lh_en": [-10, -11, -9, -100, 40],
            "rh_en": [-9, -10, -8, -90, 50],
            "avg_en": [-9.5, -10.5, -8.5, -95, 45],
        },
        index=["typical-1", "typical-2", "typical-3", "poor", "high"],
    )
    monkeypatch.setattr(
        fs, "retrieve_freesurfer_eulernum", lambda *args, **kwargs: (values, [])
    )

    passed, failed, missing = fs.freesurfer_QC("unused", method="mad", threshold=3)

    assert failed == ["poor"]
    assert set(passed) == {"typical-1", "typical-2", "typical-3", "high"}
    assert missing == []


@pytest.mark.unit
def test_freesurfer_qc_mad_handles_zero_mad(monkeypatch):
    values = pd.DataFrame(
        {
            "lh_en": [-10, -10, -10],
            "rh_en": [-10, -10, -10],
            "avg_en": [-10, -10, -10],
        },
        index=["sub-01", "sub-02", "sub-03"],
    )
    monkeypatch.setattr(
        fs, "retrieve_freesurfer_eulernum", lambda *args, **kwargs: (values, [])
    )

    passed, failed, missing = fs.freesurfer_QC("unused", method="MAD")

    assert passed == ["sub-01", "sub-02", "sub-03"]
    assert failed == []
    assert missing == []


@pytest.mark.unit
def test_freesurfer_qc_absolute_method_and_incomplete_rows(monkeypatch):
    values = pd.DataFrame(
        {
            "lh_en": [-10, -11, -30, "bad"],
            "rh_en": [-9, -10, -29, -8],
            "avg_en": [-9.5, -10.5, -29.5, -8],
        },
        index=["sub-01", "sub-02", "outlier", "incomplete"],
    )
    monkeypatch.setattr(
        fs,
        "retrieve_freesurfer_eulernum",
        lambda *args, **kwargs: (values, ["not-computed"]),
    )

    passed, failed, missing = fs.freesurfer_QC("unused", method="ABS", threshold=5)

    assert passed == ["sub-01", "sub-02"]
    assert failed == ["outlier"]
    assert set(missing) == {"incomplete", "not-computed"}


@pytest.mark.unit
def test_freesurfer_qc_returns_missing_when_no_values_are_available(monkeypatch):
    monkeypatch.setattr(
        fs,
        "retrieve_freesurfer_eulernum",
        lambda *args, **kwargs: (pd.DataFrame(), ["sub-01"]),
    )

    assert fs.freesurfer_QC("unused") == ([], [], ["sub-01"])


@pytest.mark.unit
def test_freesurfer_qc_rejects_unknown_method(monkeypatch):
    values = pd.DataFrame(
        {"lh_en": [-10], "rh_en": [-9], "avg_en": [-9.5]}, index=["sub-01"]
    )
    monkeypatch.setattr(
        fs, "retrieve_freesurfer_eulernum", lambda *args, **kwargs: (values, [])
    )

    with pytest.raises(ValueError, match="method must be 'MAD' or 'ABS'"):
        fs.freesurfer_QC("unused", method="typo")
