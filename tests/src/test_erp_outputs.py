"""ERP metadata parity, saved averages, and convenient result access."""

import json
from dataclasses import replace
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from meganorm.API import Config, Dataset
from meganorm.src import erp
from meganorm.utils.erp import make_demo_recording, ERPEventSpec

pytestmark = pytest.mark.unit


@pytest.fixture
def recording(tmp_path):
    raw, events = make_demo_recording()
    raw.set_montage(None)
    raw.set_annotations(
        mne.Annotations(
            (events[:, 0] - raw.first_samp) / raw.info["sfreq"],
            np.zeros(len(events)),
            ["standard" if e == 1 else "target" for e in events[:, 2]],
        )
    )
    root = tmp_path / "input"
    folder = root / "sub-41"
    folder.mkdir(parents=True)
    path = folder / "sub-41_task-oddball_eeg.fif"
    raw.save(path, fmt="double", overwrite=True, verbose=False)
    (folder / "sub-41_task-oddball_channels.tsv").write_text(
        "name\ttype\tunits\nCz\tEEG\tµV\nPz\tEEG\tµV\n", encoding="cp1252"
    )
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        apply_ica=False,
        cutoffFreqLow=None,
        cutoffFreqHigh=None,
        rereference_method=None,
        eeg_var_threshold=None,
        erp={
            "min_trials": 1,
            "components": {
                "P300": {
                    "conditions": ["standard", "target", "missing"],
                    "channels": ["Cz", "Pz"],
                    "window": [0.25, 0.5],
                    "polarity": "pos",
                    "metrics": ["peak_amplitude", "peak_latency", "mean_amplitude"],
                }
            },
        },
    )
    spec = ERPEventSpec(
        mode="annotations",
        annotation_map={
            "standard": "standard",
            "target": "target",
            "missing": "absent",
        },
    )
    ds = Dataset(
        name="demo",
        root=root,
        task="oddball",
        extension=".fif",
        erp_events=spec,
        erp_montage={"template": "auto"},
    )
    return path, config, ds


def test_dataset_extraction_prepares_metadata_and_matches_worker(recording, tmp_path):
    path, config, ds = recording
    with pytest.warns(UserWarning):
        single = ds.extract_erp("sub-41", config=config)
    # Disable reports for this scientific parity check.
    config.erp.save_report = False
    with pytest.warns(UserWarning):
        cohort = erp.process_erp_recording(
            path,
            "sub-41",
            config=config,
            event_spec=ds.erp_events,
            montage_settings=ds.erp_montage,
            line_freq=ds.line_freq,
            output_root=tmp_path / "out",
        )
    pd.testing.assert_frame_equal(single.features, cohort.features, check_names=False)
    pd.testing.assert_frame_equal(single.qc, cohort.qc)
    assert single.preprocessing_qc["montage"] == cohort.preprocessing_qc["montage"]
    assert single.preprocessing_qc["montage"]["source"] == "template"
    assert single.evokeds["target"].info["line_freq"] == 50
    assert single.components.keys() == config.erp.components.keys()
    with pytest.raises(
        (TypeError, ValueError), match="string|identifier|Participant ID"
    ):
        ds.extract_erp(41, config=config)
    with pytest.raises(KeyError):
        ds.extract_erp("41", config=config)


def test_saved_evokeds_roundtrip_without_reprocessing(recording, tmp_path, monkeypatch):
    path, config, ds = recording
    result = ds.extract_erp("sub-41", config=config)
    destination = tmp_path / "saved"
    result.save(destination, report=False)
    from meganorm.API import load_erp_result

    monkeypatch.setattr(
        erp, "run_erp_pipeline", lambda *a, **k: pytest.fail("must not reprocess")
    )
    loaded = load_erp_result(destination)
    assert loaded.epochs is None
    assert loaded.evokeds["missing"] is None
    pd.testing.assert_frame_equal(loaded.features, result.features, check_names=False)
    pd.testing.assert_frame_equal(loaded.qc, result.qc)
    assert loaded.components == result.components
    for key in ["standard", "target"]:
        a, b = loaded.evokeds[key], result.evokeds[key]
        np.testing.assert_allclose(a.data, b.data, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(a.times, b.times, atol=1e-8)
        assert a.ch_names == b.ch_names and a.nave == b.nave
        np.testing.assert_allclose(a.baseline, b.baseline, atol=1e-8)
    assert (destination / "evoked-ave.fif").is_file()
    with pytest.raises(FileExistsError):
        result.save(destination, report=False)


def test_saved_result_validates_identity(recording, tmp_path):
    from meganorm.API import load_erp_result

    _, config, ds = recording
    result = ds.extract_erp("sub-41", config=config)
    result.save(tmp_path / "saved", report=False)
    path = tmp_path / "saved" / "erp_result.json"
    data = json.loads(path.read_text())
    data["subject"] = "someone_else"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity|subject"):
        load_erp_result(path.parent)


def test_all_absent_conditions_can_be_saved_and_reloaded(tmp_path):
    from meganorm.API import load_erp_result

    components = {
        "P": erp.ERPComponent(
            conditions=("target",), channels=("Pz",), window=(0.2, 0.5), polarity="pos"
        )
    }
    features, qc = erp.extract_erp_idps("001", {"target": None}, components)
    result = erp.ERPResult(
        features, qc, None, {"target": None}, {}, components=components
    )
    result.save(tmp_path / "saved", report=False)
    loaded = load_erp_result(tmp_path / "saved")
    assert loaded.evokeds == {"target": None}
    assert not (tmp_path / "saved" / "evoked-ave.fif").exists()
    assert loaded.features.index.tolist() == ["001"]


def test_report_failure_preserves_scientific_outputs(recording, tmp_path, monkeypatch):
    import meganorm.plots.erp_report as report_module

    path, config, ds = recording
    monkeypatch.setattr(
        report_module,
        "write_erp_report",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )
    result = erp.process_erp_recording(
        path,
        "sub-41",
        config=config,
        event_spec=ds.erp_events,
        montage_settings=ds.erp_montage,
        output_root=tmp_path / "out",
    )
    directory = tmp_path / "out" / "participants" / "sub-41"
    status = json.loads((directory / "report_outputs.json").read_text())
    assert status["report"]["status"] == "failed"
    assert "plot failed" in status["report"]["error"]
    assert json.loads((directory / "complete.json").read_text())["status"] == "partial"
    assert (directory / "evoked-ave.fif").is_file()
    assert result.features.notna().any().any()


def test_cohort_saved_result_and_index(recording, tmp_path):
    from meganorm.API import Pipeline
    import matplotlib.pyplot as plt

    _, config, ds = recording
    result = Pipeline(config=config, output_dir=tmp_path / "cohort").run(ds)
    saved = result.load_erp("sub-41")
    pd.testing.assert_frame_equal(
        saved.features, result.data[list(saved.features.columns)], check_names=False
    )
    fig, axes = result.plot_erp("sub-41", channels=("Cz", "Pz"))
    assert len(axes) == 2
    plt.close(fig)
    index = result.paths["report"].read_text()
    assert "participants/sub-41/report.html" in index
    assert (result.output_dir / "participants/sub-41/report.html").is_file()
    with pytest.raises(KeyError):
        result.load_erp("41")


def test_fif_failure_does_not_mark_complete(recording, tmp_path, monkeypatch):
    path, config, ds = recording

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(mne, "write_evokeds", fail)
    with pytest.raises(OSError, match="disk full"):
        erp.process_erp_recording(
            path,
            "sub-41",
            config=config,
            event_spec=ds.erp_events,
            montage_settings=ds.erp_montage,
            output_root=tmp_path / "out",
        )
    assert not (tmp_path / "out/participants/sub-41/complete.json").exists()


def test_outputs_disabled(recording, tmp_path):
    path, config, ds = recording
    config.erp.save_evokeds = config.erp.save_report = False
    erp.process_erp_recording(
        path,
        "sub-41",
        config=config,
        event_spec=ds.erp_events,
        montage_settings=ds.erp_montage,
        output_root=tmp_path / "out",
    )
    folder = tmp_path / "out/participants/sub-41"
    assert (folder / "complete.json").exists()
    assert not (folder / "evoked-ave.fif").exists()
    assert not (folder / "report.html").exists()


def test_index_escapes_labels_and_encodes_links(tmp_path):
    from meganorm.utils.erp_index import write_erp_index

    subject = "sub <41>"
    folder = tmp_path / "participants" / subject
    folder.mkdir(parents=True)
    (folder / "report.html").write_text("report")
    table = pd.DataFrame(
        [{"dataset": "<script>", "status": "failed", "error_message": "<bad>"}],
        index=[subject],
    )
    page = write_erp_index(tmp_path, table).read_text()
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "participants/sub%20%3C41%3E/report.html" in page


def _progress_events(directory):
    return [
        json.loads(line)
        for line in (directory / "processing_events.jsonl").read_text().splitlines()
    ]


def test_worker_progress_is_detailed_and_numerically_observational(recording, tmp_path):
    path, config, ds = recording
    config.erp.save_report = False
    expected = ds.extract_erp("sub-41", config=config)
    actual = erp.process_erp_recording(
        path,
        "sub-41",
        config=config,
        event_spec=ds.erp_events,
        montage_settings=ds.erp_montage,
        line_freq=ds.line_freq,
        output_root=tmp_path / "out",
    )
    pd.testing.assert_frame_equal(actual.features, expected.features, check_names=False)
    pd.testing.assert_frame_equal(actual.qc, expected.qc)
    assert actual.preprocessing_qc == expected.preprocessing_qc
    for condition, evoked in expected.evokeds.items():
        if evoked is not None:
            np.testing.assert_array_equal(actual.evokeds[condition].data, evoked.data)
    directory = tmp_path / "out/participants/sub-41"
    events = _progress_events(directory)
    completed = {e["stage"]: e for e in events if e["status"] == "completed"}
    assert {
        "load_recording",
        "prepare_metadata",
        "resolve_events",
        "preprocessing",
        "epoching",
        "epoch_cleaning",
        "average_conditions",
        "extract_idps",
        "save_tables",
        "save_evokeds",
        "publish_completion",
        "run",
    } <= completed.keys()
    assert completed["load_recording"]["details"]["n_channels"] == 2
    assert completed["prepare_metadata"]["details"]["montage"]["source"] == "template"
    assert completed["resolve_events"]["details"]["condition_counts"] == {
        "standard": 20,
        "target": 20,
        "missing": 0,
    }
    skips = {e["stage"] for e in events if e["status"] == "skipped"}
    assert {"reference", "analysis_filter", "ica", "resample", "report"} <= skips
    summary = next(e["details"] for e in events if e["stage"] == "erp_summary")
    assert summary["conditions"]["missing"]["n_trials"] == 0
    assert summary["idps"]["valid"] == int(actual.features.notna().sum().sum())
    assert events[-1]["details"]["processing_outcome"] == "partial"
    assert "Outcome: partial" in (directory / "processing.log").read_text()
    assert all(e["duration_seconds"] >= 0 for e in events if "duration_seconds" in e)


def test_live_load_start_and_interrupt(recording, tmp_path, monkeypatch):
    path, config, ds = recording
    directory = tmp_path / "out/participants/sub-41"
    interrupt = KeyboardInterrupt()

    def reader(*args, **kwargs):
        assert _progress_events(directory)[-1]["stage"] == "load_recording"
        assert _progress_events(directory)[-1]["status"] == "started"
        raise interrupt

    monkeypatch.setattr(mne.io, "read_raw", reader)
    with pytest.raises(KeyboardInterrupt) as caught:
        erp.process_erp_recording(
            path,
            "sub-41",
            config=config,
            event_spec=ds.erp_events,
            output_root=tmp_path / "out",
        )
    assert caught.value is interrupt
    assert _progress_events(directory)[-1]["status"] == "interrupted"
    assert not (directory / "complete.json").exists()


def test_optional_report_error_logged_without_failing_run(
    recording, tmp_path, monkeypatch
):
    import meganorm.plots.erp_report as report_module

    path, config, ds = recording

    def fail(*args, **kwargs):
        raise RuntimeError("figure error")

    monkeypatch.setattr(report_module, "write_erp_report", fail)
    erp.process_erp_recording(
        path,
        "sub-41",
        config=config,
        event_spec=ds.erp_events,
        montage_settings=ds.erp_montage,
        output_root=tmp_path / "out",
    )
    events = _progress_events(tmp_path / "out/participants/sub-41")
    failed = [e for e in events if e["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["stage"] == "report"
    assert failed[0]["details"]["error"] == "figure error"
    assert events[-1]["status"] == "completed"
    assert events[-1]["details"]["artifacts"]["report"]["status"] == "failed"
