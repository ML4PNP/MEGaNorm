"""Live progress must remain observational, isolated, and interruption-safe."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.unit


def read_events(path):
    return [
        json.loads(line)
        for line in (path / "processing_events.jsonl").read_text().splitlines()
    ]


def test_live_start_and_monotonic_duration(tmp_path, monkeypatch):
    from meganorm.utils import erp_logging as log

    tick = iter([10.0, 11.0, 14.0, 16.0])
    monkeypatch.setattr(log.time, "perf_counter", lambda: next(tick))
    with log.ERPProgress(tmp_path, "001", recording="input.fif"):
        with log.stage("filter", l_freq=1) as summary:
            events = read_events(tmp_path)
            assert events[-1]["status"] == "started"
            assert "filter" in (tmp_path / "processing.log").read_text()
            summary["channels"] = 4
    events = read_events(tmp_path)
    assert [(e["stage"], e["status"]) for e in events] == [
        ("run", "started"),
        ("filter", "started"),
        ("filter", "completed"),
        ("run", "completed"),
    ]
    assert events[2]["duration_seconds"] == 3
    assert events[-1]["duration_seconds"] == 6
    assert events[2]["details"]["channels"] == 4
    assert {e["subject"] for e in events} == {"001"}
    assert all(e["timestamp"].endswith("+00:00") for e in events)
    assert events[1]["span_id"] == events[2]["span_id"]


@pytest.mark.parametrize("exception", [ValueError("bad data"), KeyboardInterrupt()])
def test_failure_logs_active_stage_and_preserves_exception(tmp_path, exception):
    from meganorm.utils.erp_logging import ERPProgress, stage

    with pytest.raises(type(exception)) as caught:
        with ERPProgress(tmp_path, "sub-1"):
            with stage("ica_fit"):
                raise exception
    assert caught.value is exception
    events = read_events(tmp_path)
    assert events[-2]["stage"] == "ica_fit"
    assert events[-2]["status"] == (
        "interrupted" if isinstance(exception, KeyboardInterrupt) else "failed"
    )
    assert events[-1]["details"]["error_type"] == type(exception).__name__
    assert "Traceback" in (tmp_path / "processing.log").read_text()


def test_logging_sink_failure_does_not_break_work(tmp_path, caplog):
    from meganorm.utils.erp_logging import ERPProgress, stage

    (tmp_path / "processing.log").mkdir()
    with ERPProgress(tmp_path, "sub-1"):
        with stage("load"):
            pass
    assert read_events(tmp_path)[-1]["status"] == "completed"
    assert len([r for r in caplog.records if "processing.log" in r.message]) == 1


def test_participant_context_isolated_and_restored(tmp_path):
    from meganorm.utils.erp_logging import ERPProgress, stage, skipped

    def worker(subject):
        folder = tmp_path / subject
        folder.mkdir()
        with ERPProgress(folder, subject):
            with stage("work"):
                skipped("ica", reason="disabled")
        return read_events(folder)

    with ThreadPoolExecutor(2) as pool:
        a, b = list(pool.map(worker, ["001", "002"]))
    assert {e["subject"] for e in a} == {"001"}
    assert {e["subject"] for e in b} == {"002"}
    assert len({e["run_id"] for e in a + b}) == 2
    with stage("outside") as summary:
        summary["no_log"] = True
    assert len(read_events(tmp_path / "001")) == len(a)
