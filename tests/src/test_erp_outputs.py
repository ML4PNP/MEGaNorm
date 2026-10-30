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


def test_reorganized_output_and_report_imports_remain_compatible():
    from meganorm.utils import erp_outputs, erp_bootstrap_outputs, erp_index
    from meganorm.plots import erp_report

    assert erp_bootstrap_outputs.save_bootstrap is erp_outputs.save_bootstrap
    assert erp_bootstrap_outputs.load_bootstrap is erp_outputs.load_bootstrap
    assert erp_bootstrap_outputs.validate_bootstrap is erp_outputs.validate_bootstrap
    assert erp_index.write_erp_index is erp_report.write_erp_index


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


@pytest.fixture
def bootstrap_result(recording):
    from types import SimpleNamespace

    _, config, ds = recording
    result = ds.extract_erp("sub-41", config=config)
    rows, waveforms = [], {}
    for row in result.qc.to_dict("records"):
        original = result.features.loc["sub-41", row["idp"]]
        valid = int(np.isfinite(original)) * 4
        rows.append(
            dict(
                subject="sub-41",
                condition=row["condition"],
                idp=row["idp"],
                original=original,
                median=original,
                p05=original,
                p95=original,
                n_valid=valid,
                n_repeats=4,
                n_trials=row["n_trials"],
                status="ok" if valid else "missing_condition",
                n_boundary_peaks=0,
            )
        )
    for condition, evoked in result.evokeds.items():
        waveforms[condition] = (
            None
            if evoked is None
            else dict(
                ch_names=tuple(evoked.ch_names),
                times=evoked.times.copy(),
                median=evoked.data.copy(),
                p05=evoked.data.copy(),
                p95=evoked.data.copy(),
                n_valid=np.full(evoked.data.shape, 4, dtype=int),
                n_trials=int(evoked.nave),
            )
        )
    result.bootstrap = SimpleNamespace(
        idps=pd.DataFrame(rows),
        waveforms=waveforms,
        provenance=dict(
            schema_version=1,
            subject="sub-41",
            n_repeats=4,
            random_state=0,
            min_trials=1,
            quantiles=[5, 50, 95],
            quantile_method="linear",
            method="within_condition",
            seed_scheme="sha256",
        ),
    )
    return result


def test_bootstrap_storage_roundtrip_and_stale_cleanup(bootstrap_result, tmp_path):
    from meganorm.API import load_erp_result

    folder = tmp_path / "saved"
    bootstrap_result.save(folder, report=False)
    descriptor = json.loads((folder / "erp_result.json").read_text())
    assert descriptor["bootstrap_file"] == "bootstrap.json"
    with np.load(folder / "bootstrap_waveforms.npz", allow_pickle=False) as archive:
        assert archive.files
        assert all(archive[key].dtype.kind in "fiu" for key in archive.files)
    loaded = load_erp_result(folder)
    pd.testing.assert_frame_equal(
        loaded.bootstrap.idps, bootstrap_result.bootstrap.idps
    )
    assert loaded.bootstrap.provenance == bootstrap_result.bootstrap.provenance
    np.testing.assert_array_equal(
        loaded.bootstrap.waveforms["target"]["median"],
        bootstrap_result.bootstrap.waveforms["target"]["median"],
    )
    bootstrap_result.bootstrap = None
    bootstrap_result.save(folder, report=False, overwrite=True)
    assert load_erp_result(folder).bootstrap is None
    assert not list(folder.glob("bootstrap*"))


def test_bootstrap_saved_without_evokeds_and_failures_propagate(
    bootstrap_result, tmp_path, monkeypatch
):
    from meganorm.utils.erp_outputs import save_erp_artifacts

    save_erp_artifacts(
        bootstrap_result, tmp_path, save_evokeds=False, save_report=False
    )
    assert (tmp_path / "bootstrap.json").is_file()
    assert not (tmp_path / "evoked-ave.fif").exists()

    def fail(*a, **k):
        raise OSError("bootstrap disk full")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="bootstrap disk full"):
        save_erp_artifacts(
            bootstrap_result, tmp_path, save_evokeds=False, save_report=False
        )


@pytest.mark.parametrize(
    "damage",
    ["subject", "original", "counts", "quantiles", "times", "channels", "seed"],
)
def test_bootstrap_rejects_corruption(bootstrap_result, tmp_path, damage):
    from meganorm.utils.erp_outputs import save_erp_artifacts

    summary = bootstrap_result.bootstrap
    if damage == "subject":
        summary.idps.loc[0, "subject"] = "other"
    elif damage == "original":
        summary.idps.loc[0, "original"] += 1
    elif damage == "counts":
        summary.idps.loc[0, "n_valid"] = 5
    elif damage == "quantiles":
        summary.waveforms["target"]["p05"] += 1
    elif damage == "times":
        summary.waveforms["target"]["times"] += 1
    elif damage == "channels":
        summary.waveforms["target"]["ch_names"] = ("wrong", "Pz")
    else:
        summary.provenance["random_state"] = -1
    with pytest.raises(ValueError, match="bootstrap|Bootstrap"):
        save_erp_artifacts(
            bootstrap_result, tmp_path, save_evokeds=False, save_report=False
        )


def test_bootstrap_load_rejects_saved_corruption_and_ignores_stale(
    bootstrap_result, tmp_path
):
    from meganorm.API import load_erp_result

    folder = tmp_path / "saved"
    bootstrap_result.save(folder, report=False)
    path = folder / "bootstrap_summary.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "n_valid"] = 100
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="bootstrap"):
        load_erp_result(folder)
    descriptor_path = folder / "erp_result.json"
    descriptor = json.loads(descriptor_path.read_text())
    descriptor.pop("bootstrap_file")
    descriptor_path.write_text(json.dumps(descriptor))
    assert load_erp_result(folder).bootstrap is None


def test_bootstrap_one_trial_waveform_unavailable_is_valid(bootstrap_result, tmp_path):
    from meganorm.utils.erp_outputs import save_erp_artifacts

    summary = bootstrap_result.bootstrap
    condition = "target"
    bootstrap_result.evokeds[condition].nave = 1
    selected = summary.idps["condition"].eq(condition)
    summary.idps.loc[selected, ["median", "p05", "p95"]] = np.nan
    summary.idps.loc[selected, "n_valid"] = 0
    summary.idps.loc[selected, "n_trials"] = 1
    summary.idps.loc[selected, "status"] = "insufficient_trials"
    bootstrap_result.qc.loc[
        bootstrap_result.qc["condition"].eq(condition), "n_trials"
    ] = 1
    summary.waveforms[condition] = None
    save_erp_artifacts(
        bootstrap_result, tmp_path, save_evokeds=False, save_report=False
    )
    assert (
        json.loads((tmp_path / "bootstrap.json").read_text())["conditions"][condition]
        is None
    )


def test_real_bootstrap_saved_roundtrip(recording, tmp_path):
    from meganorm.API import load_erp_result

    _, config, ds = recording
    config.erp.bootstrap = True
    config.erp.bootstrap_n_repeats = 5
    result = ds.extract_erp("sub-41", config=config)
    assert result.bootstrap is not None
    folder = tmp_path / "real-bootstrap"
    result.save(folder, report=False)
    loaded = load_erp_result(folder)
    pd.testing.assert_frame_equal(result.bootstrap.idps, loaded.bootstrap.idps)
    for condition, wave in result.bootstrap.waveforms.items():
        restored = loaded.bootstrap.waveforms[condition]
        if wave is None:
            assert restored is None
        else:
            assert restored["ch_names"] == wave["ch_names"]
            for key in ("times", "median", "p05", "p95", "n_valid"):
                np.testing.assert_array_equal(restored[key], wave[key])


def test_bootstrap_roundtrip_noninteger_sampling_rate_preserves_bands(tmp_path):
    from meganorm.API import load_erp_result
    from meganorm.src.erp_bootstrap import bootstrap_erp
    from matplotlib.collections import PolyCollection
    import matplotlib.pyplot as plt

    epochs = mne.EpochsArray(
        np.random.default_rng(5).normal(1e-6, 2e-6, (3, 1, 512)),
        mne.create_info(["Cz"], 512.3, "eeg"),
        tmin=-0.2,
        events=np.array([[0, 0, 1], [1000, 0, 1], [2000, 0, 1]]),
        event_id={"target": 1},
        verbose=False,
    )
    components = {"P": erp.ERPComponent(("target",), ("Cz",), (0.1, 0.3), "pos")}
    evokeds = erp.compute_evokeds(epochs)
    features, qc = erp.extract_erp_idps("s", evokeds, components)
    result = erp.ERPResult(
        features,
        qc,
        epochs,
        evokeds,
        {},
        components,
        bootstrap_erp(epochs, "s", components, n_repeats=3, random_state=0),
    )
    folder = result.save(tmp_path / "fractional", report=False)
    loaded = load_erp_result(folder)
    fig, axes = loaded.plot()
    try:
        assert any(isinstance(c, PolyCollection) for c in axes[0].collections)
        assert not any("mismatch" in t.get_text() for t in axes[0].texts)
    finally:
        plt.close(fig)
