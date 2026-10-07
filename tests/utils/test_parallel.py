import inspect
import json
import ntpath
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
            "106|timed-out|TIMEOUT",
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
        "FAILED": 2,
        "CANCELLED": 1,
    }
    assert failed == ["broken", "stopped", "timed-out"]
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


@pytest.mark.unit
def test_sbatchfile_keeps_linux_log_paths_on_windows_host(monkeypatch, tmp_path):
    # Simulate the host join only at the script-generation boundary; the output
    # script is still a real file on this test runner's filesystem.
    host_join = os.path.join

    def windows_join(first, *parts):
        if first == "/project/log files":
            return ntpath.join(first, *parts)
        return host_join(first, *parts)

    monkeypatch.setattr(
        parallel,
        "os",
        SimpleNamespace(path=SimpleNamespace(join=windows_join), chmod=os.chmod),
    )
    script = Path(
        parallel.sbatchfile(
            "/project/mainParallel.py", str(tmp_path), log_path="/project/log files"
        )
    )
    content = script.read_text()
    assert "#SBATCH -o '/project/log files/%x_%j.out'" in content
    assert "#SBATCH -e '/project/log files/%x_%j.err'" in content


@pytest.mark.unit
def test_sbatchfile_writes_linux_line_endings_on_windows_host(monkeypatch, tmp_path):
    host_open = open

    def windows_open(path, mode, **kwargs):
        kwargs.setdefault("newline", "\r\n")
        return host_open(path, mode, **kwargs)

    monkeypatch.setattr(parallel, "open", windows_open, raising=False)
    script = Path(parallel.sbatchfile("/project/mainParallel.py", str(tmp_path)))
    assert b"\r\n" not in script.read_bytes()
    assert script.read_bytes().startswith(b"#!/bin/bash\n")


@pytest.mark.unit
def test_submit_jobs_can_return_exact_job_ids(monkeypatch, tmp_path):
    commands = []

    def check_output(command, **kwargs):
        commands.append((command, kwargs))
        return "123;cluster\n"

    monkeypatch.setattr(parallel.subprocess, "check_output", check_output)
    start_time, job_ids = parallel.submit_jobs(
        "/project/mainParallel.py",
        str(tmp_path),
        {"sub-01": _subject()},
        str(tmp_path / "temp"),
        return_job_ids=True,
    )

    datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    assert job_ids == {"123": "sub-01"}
    assert commands[0][0][:3] == ["sbatch", "--parsable", "--job-name=sub-01"]
    assert commands[0][1] == {"text": True}


@pytest.mark.unit
def test_check_user_jobs_filters_exact_ids_and_uses_submitted_subject_names(
    monkeypatch,
):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="123|truncated-name|FAILED\n999|unrelated|FAILED\n",
            stderr="",
        )

    monkeypatch.setattr(parallel.subprocess, "run", run)
    counts, failed, ok = parallel.check_user_jobs(
        "researcher", "start", job_ids={"123": "sub-very-long-name"}
    )

    assert counts["FAILED"] == 1
    assert failed == ["sub-very-long-name"]
    assert ok is True
    assert "--jobs=123" in commands[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "state", ["TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED"]
)
def test_check_user_jobs_returns_terminal_failures_for_retry(monkeypatch, state):
    monkeypatch.setattr(
        parallel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=f"123|sub-01|{state}\n", stderr=""
        ),
    )

    counts, failed, ok = parallel.check_user_jobs("researcher", "start")

    assert failed == ["sub-01"]
    assert counts["FAILED"] + counts["CANCELLED"] == 1
    assert ok is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("state", "counted_state"),
    [
        ("STAGE_OUT", "RUNNING"),
        ("STOPPED", "RUNNING"),
        ("POWER_UP_NODE", "RUNNING"),
        ("SIGNALING", "RUNNING"),
        ("UPDATE_DB", "RUNNING"),
        ("EXPEDITING", "PENDING"),
        ("RESV_DEL_HOLD", "PENDING"),
        ("SPECIAL_EXIT", "PENDING"),
        ("LAUNCH_FAILED", "PENDING"),
        ("RECONFIG_FAIL", "PENDING"),
        ("REVOKED", "PENDING"),
        ("FUTURE_STATE", "PENDING"),
    ],
)
def test_check_user_jobs_monitors_nonterminal_and_unknown_states(
    monkeypatch, state, counted_state
):
    monkeypatch.setattr(
        parallel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=f"123|sub-01|{state}\n", stderr=""
        ),
    )

    counts, failed, ok = parallel.check_user_jobs(
        "researcher", "start", job_ids={"123": "sub-01"}
    )

    assert counts[counted_state] == 1
    assert sum(counts.values()) == 1
    assert failed == []
    assert ok is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "state", ["SPECIAL_EXIT", "RESV_DEL_HOLD", "LAUNCH_FAILED", "FUTURE_STATE"]
)
def test_check_jobs_status_waits_for_terminal_after_held_or_unknown_state(
    monkeypatch, state
):
    states = iter([state, "COMPLETED"])
    sleeps = []
    monkeypatch.setattr(
        parallel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=f"123|sub-01|{next(states)}\n", stderr=""
        ),
    )
    monkeypatch.setattr(parallel.time, "sleep", sleeps.append)

    assert (
        parallel.check_jobs_status(
            "researcher", "start", delay=1, job_ids={"123": "sub-01"}
        )
        == []
    )
    assert sleeps == [1]


@pytest.mark.unit
def test_check_jobs_status_waits_until_completing_job_is_finished(monkeypatch):
    states = iter(["COMPLETING", "COMPLETED"])
    sleeps = []
    monkeypatch.setattr(
        parallel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=f"123|sub-01|{next(states)}\n", stderr=""
        ),
    )
    monkeypatch.setattr(parallel.time, "sleep", sleeps.append)

    assert parallel.check_jobs_status("researcher", "start", delay=1) == []
    assert sleeps == [1]


@pytest.mark.unit
def test_check_jobs_status_waits_for_submitted_job_to_appear_in_accounting(monkeypatch):
    outputs = iter(["", "123|sub-01|COMPLETED\n"])
    sleeps = []
    monkeypatch.setattr(
        parallel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=next(outputs), stderr=""
        ),
    )
    monkeypatch.setattr(parallel.time, "sleep", sleeps.append)

    assert (
        parallel.check_jobs_status(
            "researcher", "start", delay=1, job_ids={"123": "sub-01"}
        )
        == []
    )
    assert sleeps == [1]


@pytest.fixture
def isolated_feature_driver(monkeypatch, tmp_path):
    parallel.set_path(str(tmp_path))
    config_path = tmp_path / "config.json"
    parallel.Config().save(config_path)
    params = {
        "mainParallel_path": "/project/mainParallel.py",
        "project_dir": str(tmp_path),
        "datasets": {"demo": {"surfaces_dir": "/surfaces"}},
        "job_configs": {},
        "config_file_path": str(config_path),
        "auto_collect": False,
        "auto_rerun": False,
    }
    params_path = tmp_path / "Features" / "Configurations" / "runner_params.json"
    params_path.write_text(json.dumps(params))
    submissions = []
    queries = []

    def submit(*args, **kwargs):
        subjects = args[2]
        submissions.append(subjects)
        return "start", {str(i): subject for i, subject in enumerate(subjects, 1)}

    def check(*args, **kwargs):
        queries.append(kwargs)
        return []

    monkeypatch.setattr(
        parallel, "merge_datasets_with_glob", lambda datasets: {"sub-01": _subject()}
    )
    monkeypatch.setattr(parallel, "submit_jobs", submit)
    monkeypatch.setattr(parallel, "check_jobs_status", check)
    return params, params_path, submissions, queries


@pytest.mark.unit
def test_auto_parallel_driver_can_run_again_from_persisted_parameters(
    isolated_feature_driver,
):
    params, params_path, submissions, queries = isolated_feature_driver

    assert parallel.auto_parallel_feature_extraction(**params) == []
    assert (
        parallel.auto_parallel_feature_extraction(**json.loads(params_path.read_text()))
        == []
    )
    assert len(submissions) == 2
    assert "subjects" not in json.loads(params_path.read_text())
    assert queries == [{"job_ids": {"1": "sub-01"}}] * 2


@pytest.mark.unit
def test_load_runner_params_accepts_legacy_runtime_subjects_and_missing_script(
    tmp_path,
):
    params_path = tmp_path / "runner_params.json"
    params_path.write_text(
        json.dumps({"project_dir": "/project", "subjects": {"sub-01": {}}})
    )

    params = parallel._load_runner_params(params_path)

    assert "subjects" not in params
    assert params["mainParallel_path"] == os.path.abspath(
        parallel.meganorm.src.mainParallel.__file__
    )


@pytest.mark.unit
def test_auto_parallel_excludes_subject_when_no_subject_passes_mri_qc(
    monkeypatch,
    isolated_feature_driver,
):
    params, _, submissions, _ = isolated_feature_driver
    parallel.Config(apply_source_localization=True, apply_mri_QC=True).save(
        params["config_file_path"], overwrite=True
    )
    monkeypatch.setattr(
        parallel.meganorm.utils.freesurfer,
        "freesurfer_QC",
        lambda path: ([], ["sub-01"], []),
    )

    assert parallel.auto_parallel_feature_extraction(**params) == []
    assert submissions == [{}]


@pytest.mark.unit
@pytest.mark.parametrize("index_name", [None, "participant_id"])
def test_collect_results_preserves_leading_zero_ids_across_append(tmp_path, index_name):
    temp = tmp_path / "temp"
    temp.mkdir()
    target = tmp_path / "features"
    for subject, value in [("001", 1.0), ("010", 2.0)]:
        pd.DataFrame(
            {"feature": [value]}, index=pd.Index([subject], name=index_name)
        ).to_csv(temp / f"{subject}.csv")
    subjects = {"001": {}, "010": {}}
    parallel.collect_results(str(target), subjects, str(temp), clean=False)
    pd.DataFrame({"feature": [3.0]}, index=pd.Index(["001"], name=index_name)).to_csv(
        temp / "001.csv"
    )
    parallel.collect_results(str(target), {"001": {}}, str(temp), clean=False)

    result = pd.read_csv(target / "features.csv", dtype={0: str, "subject": str})
    assert result.iloc[:, 0].tolist() == ["010", "001"]
    assert result["subject"].tolist() == ["010", "001"]
    assert result["feature"].tolist() == [2.0, 3.0]
