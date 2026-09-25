import inspect
import os
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from meganorm.utils import parallel


def _subject(rest_record="/data/rest.fif"):
    return {
        "rest_record": rest_record,
        "empty_room_record": None,
        "event_record": "/data/events raw.fif",
        "event_of_interest": "stim A",
        "mri_surface": "/surfaces/sub 01",
        "line_freq": 50,
        "device": "FIF",
        "trans_path": None,
        "pos_path": "/data/head position.pos",
        "annotation_path": None,
        "layout_path": "/data/custom layout.json",
        "demographic_path": None,
    }


@pytest.mark.unit
def test_progress_bar_reports_intermediate_progress_without_newline(capsys):
    parallel.progress_bar(1, 4, bar_length=8)

    output = capsys.readouterr().out
    assert output.startswith("\rProgress: [")
    assert output.endswith("25.0%")
    assert not output.endswith("\n")


@pytest.mark.unit
def test_progress_bar_finishes_at_100_percent_with_newline(capsys):
    parallel.progress_bar(4, 4, bar_length=8)

    assert capsys.readouterr().out.endswith("100.0%\n")


@pytest.mark.unit
def test_sbatchfile_uses_modules_without_redundant_module_argument(tmp_path):
    assert "module" not in inspect.signature(parallel.sbatchfile).parameters

    script = Path(
        parallel.sbatchfile(
            "/project/mainParallel.py",
            str(tmp_path),
            modules=["python/3.12", "cuda/12"],
            conda_env="meganorm-dev",
            log_path="/project/log files",
            time="12:00:00",
            memory="32GB",
            partition="compute",
            core=8,
            node=2,
            batch_file_name="extract",
        )
    )
    content = script.read_text()

    assert script == tmp_path / "extract.sh"
    assert os.access(script, os.X_OK)
    assert "#SBATCH -N 2" in content
    assert "#SBATCH -c 8" in content
    assert "#SBATCH -p compute" in content
    assert "#SBATCH --time=12:00:00" in content
    assert "#SBATCH --mem=32GB" in content
    assert "#SBATCH -o '/project/log files/%x_%j.out'" in content
    assert "#SBATCH -e '/project/log files/%x_%j.err'" in content
    assert "module load python/3.12" in content
    assert "module load cuda/12" in content
    assert "source activate meganorm-dev" in content
    assert 'source="$1"' in content
    assert 'demographic_path="${15}"' in content


@pytest.mark.unit
def test_sbatchfile_composes_conda_and_freesurfer_setup(tmp_path):
    script = Path(
        parallel.sbatchfile(
            "/project/mainParallel.py",
            str(tmp_path),
            conda_env="meganorm-dev",
            freesurfer_home="/opt/freesurfer",
            freesurfer_license="/licenses/license.txt",
        )
    )
    content = script.read_text()

    assert "source activate meganorm-dev" in content
    assert "export FREESURFER_HOME=/opt/freesurfer" in content
    assert "export FREESURFER_LICENSE=/licenses/license.txt" in content
    assert 'source "$FREESURFER_HOME/SetUpFreeSurfer.sh"' in content


@pytest.mark.unit
def test_sbatchfile_preserves_spaced_arguments_when_executed(tmp_path):
    capture_path = tmp_path / "arguments.txt"
    setup_marker = tmp_path / "freesurfer-setup-ran"
    freesurfer_home = tmp_path / "Free Surfer"
    freesurfer_home.mkdir()
    (freesurfer_home / "SetUpFreeSurfer.sh").write_text('touch "$SETUP_MARKER"\n')
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n')
    fake_srun.chmod(0o755)
    script = parallel.sbatchfile(
        "/project/main parallel.py",
        str(tmp_path),
        batch_file_name="quoted",
        freesurfer_home=str(freesurfer_home),
    )
    arguments = [
        "rest recording.fif",
        "target directory",
        "sub 01",
        "config file.json",
        "50",
        "surface directory",
        "empty room.fif",
        "event recording.fif",
        "stim A",
        "FIF",
        "head position.pos",
        "transform file-trans.fif",
        "annotation file.annot",
        "layout file.json",
        "demographic file.tsv",
    ]
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["CAPTURE_PATH"] = str(capture_path)
    env["SETUP_MARKER"] = str(setup_marker)

    result = subprocess.run(
        ["bash", script, *arguments], capture_output=True, text=True, env=env
    )

    assert result.returncode == 0
    assert setup_marker.is_file()
    assert capture_path.read_text().splitlines() == [
        "--cpus-per-task=1",
        "python",
        "/project/main parallel.py",
        *arguments[:4],
        "--line_freq",
        arguments[4],
        "--surfaces_dir",
        arguments[5],
        "--empty_room_recording_path",
        arguments[6],
        "--event_record",
        arguments[7],
        "--event_of_interest",
        arguments[8],
        "--device_type",
        arguments[9],
        "--pos_file",
        arguments[10],
        "--trans_file",
        arguments[11],
        "--annotation_path",
        arguments[12],
        "--layout_path",
        arguments[13],
        "--demographic_path",
        arguments[14],
    ]


@pytest.mark.unit
def test_submit_jobs_uses_modules_conda_and_argument_list(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        parallel.subprocess,
        "check_call",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )
    processing = tmp_path / "processing files"
    processing.mkdir()
    temp = tmp_path / "temporary files"
    job_configs = {
        "log_path": None,
        "conda_env": "custom-env",
        "modules": ["python/3.12", "cuda/12"],
        "time": "2:00:00",
        "memory": "8GB",
        "partition": "short",
        "core": 2,
        "node": 1,
        "batch_file_name": "subject_job",
    }

    start_time = parallel.submit_jobs(
        "/project/main parallel.py",
        str(processing),
        {"sub 01": _subject("/data/rest recording.fif")},
        str(temp),
        config_file=None,
        job_configs=job_configs,
    )

    datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    assert temp.is_dir()
    assert commands == [
        (
            [
                "sbatch",
                "--job-name=sub 01",
                str(processing / "subject_job.sh"),
                "/data/rest recording.fif",
                str(temp),
                "sub 01",
                "None",
                "50",
                "/surfaces/sub 01",
                "None",
                "/data/events raw.fif",
                "stim A",
                "FIF",
                "/data/head position.pos",
                "None",
                "None",
                "/data/custom layout.json",
                "None",
            ],
            {},
        )
    ]
    script = (processing / "subject_job.sh").read_text()
    assert "module load python/3.12" in script
    assert "module load cuda/12" in script
    assert "source activate custom-env" in script


@pytest.mark.unit
def test_submit_jobs_progress_reaches_total(monkeypatch, tmp_path):
    updates = []
    monkeypatch.setattr(parallel.subprocess, "check_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        parallel,
        "progress_bar",
        lambda current, total: updates.append((current, total)),
    )

    parallel.submit_jobs(
        "/project/mainParallel.py",
        str(tmp_path),
        {"sub-01": _subject(), "sub-02": _subject()},
        str(tmp_path / "temp"),
        progress=True,
    )

    assert updates == [(1, 2), (2, 2)]
    content = (tmp_path / "batch_job.sh").read_text()
    assert "source activate meganorm" in content
    assert "module load" not in content


@pytest.mark.unit
def test_check_user_jobs_parses_supported_states_and_suffixes(monkeypatch):
    calls = []
    stdout = "\n".join(
        [
            "101|queued|PENDING",
            "102|active|RUNNING",
            "103|finished|COMPLETED",
            "104|broken|FAILED",
            "105|stopped|CANCELLED by 12345",
            "106|unknown|TIMEOUT",
            "malformed",
        ]
    )

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(parallel.subprocess, "run", run)

    counts, failed, ok = parallel.check_user_jobs("researcher", "2026-09-25T08:00:00")

    assert counts == {
        "PENDING": 1,
        "RUNNING": 1,
        "COMPLETED": 1,
        "FAILED": 1,
        "CANCELLED": 1,
    }
    assert failed == ["broken"]
    assert ok is True
    command, kwargs = calls[0]
    assert command[:6] == ["sacct", "-n", "-X", "--parsable2", "--noheader", "-S"]
    assert command[6] == "2026-09-25T08:00:00"
    assert command[-3:] == ["-u", "researcher", "--format=JobIDRaw,JobName,State"]
    assert kwargs == {"capture_output": True, "text": True}


@pytest.mark.unit
def test_check_user_jobs_excludes_current_slurm_driver(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "700")
    monkeypatch.setattr(
        parallel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="700|feature-runner|RUNNING\n701|sub-01|RUNNING",
            stderr="",
        ),
    )

    counts, failed, ok = parallel.check_user_jobs("researcher", "start")

    assert counts["RUNNING"] == 1
    assert sum(counts.values()) == 1
    assert failed == []
    assert ok is True


@pytest.mark.unit
def test_check_user_jobs_reports_scheduler_query_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        parallel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="accounting unavailable"
        ),
    )

    counts, failed, ok = parallel.check_user_jobs("researcher", "start")

    assert all(value == 0 for value in counts.values())
    assert failed == []
    assert ok is False
    assert "accounting unavailable" in capsys.readouterr().out


@pytest.mark.unit
def test_check_user_jobs_handles_command_exception(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise OSError("sacct not installed")

    monkeypatch.setattr(parallel.subprocess, "run", fail)

    counts, failed, ok = parallel.check_user_jobs("researcher", "start")

    assert all(value == 0 for value in counts.values())
    assert failed == []
    assert ok is False
    assert "sacct not installed" in capsys.readouterr().out


@pytest.mark.unit
def test_check_jobs_status_retries_query_and_waits_for_last_active_job(monkeypatch):
    zero = {
        "PENDING": 0,
        "RUNNING": 0,
        "COMPLETED": 0,
        "FAILED": 0,
        "CANCELLED": 0,
    }
    responses = iter(
        [
            (zero.copy(), [], False),
            ({**zero, "RUNNING": 1}, [], True),
            ({**zero, "COMPLETED": 1, "FAILED": 1}, ["sub-failed"], True),
        ]
    )
    calls = []
    sleeps = []

    def check(username, start_time):
        calls.append((username, start_time))
        return next(responses)

    monkeypatch.setattr(parallel, "check_user_jobs", check)
    monkeypatch.setattr(parallel.time, "sleep", sleeps.append)

    failed = parallel.check_jobs_status("researcher", "start", delay=7)

    assert failed == ["sub-failed"]
    assert calls == [("researcher", "start")] * 3
    assert sleeps == [7, 7]


def _write_subject_result(directory, subject, value):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"power": [value]}, index=[subject]).to_csv(
        directory / f"{subject}.csv"
    )


@pytest.mark.unit
def test_collect_results_merges_available_subject_files(tmp_path):
    target = tmp_path / "features"
    temp = tmp_path / "temp"
    _write_subject_result(temp, "sub-01", 1.5)
    _write_subject_result(temp, "sub-02", 2.5)

    parallel.collect_results(
        str(target),
        {"sub-01": {}, "sub-02": {}, "sub-missing": {}},
        str(temp),
        clean=False,
    )

    collected = pd.read_csv(target / "features.csv", index_col=0)
    assert collected.set_index("subject")["power"].to_dict() == {
        "sub-01": 1.5,
        "sub-02": 2.5,
    }
    assert temp.is_dir()


@pytest.mark.unit
def test_collect_results_append_updates_subjects_and_migrates_legacy_output(tmp_path):
    target = tmp_path / "features"
    target.mkdir()
    pd.DataFrame({"power": [1.0, 2.0]}, index=["sub-old", "sub-01"]).to_csv(
        target / "features.csv"
    )
    temp = tmp_path / "temp"
    _write_subject_result(temp, "sub-01", 9.0)
    _write_subject_result(temp, "sub-02", 3.0)

    parallel.collect_results(
        str(target), {"sub-01": {}, "sub-02": {}}, str(temp), clean=False, append=True
    )

    collected = pd.read_csv(target / "features.csv", index_col=0)
    assert collected.set_index("subject")["power"].to_dict() == {
        "sub-old": 1.0,
        "sub-01": 9.0,
        "sub-02": 3.0,
    }
    assert collected["subject"].is_unique


@pytest.mark.unit
def test_collect_results_overwrites_existing_output_when_append_is_false(tmp_path):
    target = tmp_path / "features"
    target.mkdir()
    pd.DataFrame({"power": [1.0], "subject": ["sub-old"]}, index=["sub-old"]).to_csv(
        target / "features.csv"
    )
    temp = tmp_path / "temp"
    _write_subject_result(temp, "sub-new", 4.0)

    parallel.collect_results(
        str(target), {"sub-new": {}}, str(temp), clean=False, append=False
    )

    collected = pd.read_csv(target / "features.csv", index_col=0)
    assert collected["subject"].tolist() == ["sub-new"]
    assert collected["power"].tolist() == [4.0]


@pytest.mark.unit
def test_collect_results_cleans_temp_when_no_results_exist(tmp_path, capsys):
    target = tmp_path / "features"
    temp = tmp_path / "temp"
    temp.mkdir()

    result = parallel.collect_results(
        str(target), {"sub-missing": {}}, str(temp), clean=True
    )

    assert result is None
    assert not temp.exists()
    assert not (target / "features.csv").exists()
    assert "No new per-subject result files were found" in capsys.readouterr().out


@pytest.mark.unit
def test_collect_results_cleans_temp_after_success(tmp_path):
    target = tmp_path / "features"
    temp = tmp_path / "temp"
    _write_subject_result(temp, "sub-01", 1.0)

    parallel.collect_results(str(target), {"sub-01": {}}, str(temp), clean=True)

    assert (target / "features.csv").is_file()
    assert not temp.exists()


@pytest.mark.unit
def test_sbatch_feature_extraction_runner_uses_modules_and_conda_env(tmp_path):
    class FakeConfig:
        apply_mri_template = False

        def save(self, save_path, overwrite=False):
            Path(save_path).write_text("{}")

    job_configs = {
        "conda_env": "meganorm-dev",
        "modules": ["python/3.12", "cuda/12"],
        "partition": "compute",
        "slurm_username": "researcher",
    }

    parallel.sbatch_feature_extraction_runner(
        str(tmp_path), datasets={}, job_configs=job_configs, config_file=FakeConfig()
    )

    script = (tmp_path / "Features" / "feature_extraction_runner.sbatch").read_text()
    assert "module load python/3.12" in script
    assert "module load cuda/12" in script
    assert "source activate meganorm-dev" in script
