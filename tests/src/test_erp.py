"""Synthetic numerical and integration checks; no dataset downloads required."""

import io
import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import mne
import numpy as np
import pandas as pd
from meganorm.src.erp import (
    ERPComponent,
    compute_evokeds,
    epoch_erp,
    extract_erp_idps,
    preprocess_erp,
    run_erp_pipeline,
    extract_recording,
)
from meganorm.utils.erp import make_demo_recording
import json
import pytest
from pydantic import ValidationError
from meganorm.utils.IO import Config
from meganorm.utils.erp import ERPEventSpec


class TestERP(unittest.TestCase):

    def test_demo_fixture_reproducible(self):
        raw, events = make_demo_recording()
        other, other_events = make_demo_recording()
        np.testing.assert_array_equal(raw.get_data(), other.get_data())
        np.testing.assert_array_equal(events, other_events)
        self.assertEqual(raw.first_samp, 500)
        self.assertEqual(np.sum(events[:, 2] == 1), 20)
        self.assertEqual(np.sum(events[:, 2] == 2), 20)

    def test_file_wrapper_matches_raw_pipeline(self):
        raw, events, config = self.make_raw()
        expected = run_erp_pipeline(
            raw,
            "s",
            self.components,
            events=events,
            event_id={"target": 1, "standard": 2},
            config=config,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo_raw.fif"
            raw.save(path, fmt="double", overwrite=True, verbose=False)
            actual = extract_recording(
                path,
                "s",
                self.components,
                events=events,
                event_id={"target": 1, "standard": 2},
                config=config,
            )
        pd.testing.assert_frame_equal(actual.features, expected.features)
        pd.testing.assert_frame_equal(actual.qc, expected.qc)

    def test_plot_units_windows_and_missing_channel(self):
        import matplotlib.pyplot as plt
        from meganorm.plots.plot_erp import plot_erp

        before = self.evoked.data.copy()
        fig, axes = plot_erp(
            {"target": self.evoked, "missing": None},
            components=self.components,
            channels=("Cz", "Pz", "absent"),
        )
        try:
            self.assertEqual(len(axes), 3)
            self.assertEqual(axes[0].get_ylabel(), "Amplitude (µV)")
            np.testing.assert_allclose(axes[0].lines[0].get_ydata(), before[0] * 1e6)
            self.assertEqual(len(axes[0].patches), 1)
            self.assertEqual(len(axes[1].patches), 2)
            self.assertEqual(axes[2].texts[0].get_text(), "No usable data")
            np.testing.assert_array_equal(self.evoked.data, before)
        finally:
            plt.close(fig)

    def setUp(self):
        self.info = mne.create_info(["Cz", "Pz"], 1000, "eeg")
        self.times = np.arange(-200, 801) / 1000
        signal = np.zeros((2, len(self.times)))
        signal[:, 300] = [-4e-6, -2e-6]  # 100 ms negative component
        signal[:, 500] = [3e-6, 6e-6]  # 300 ms positive component
        self.evoked = mne.EvokedArray(signal, self.info, tmin=-0.2, nave=20)
        self.components = {
            "N100": ERPComponent(("target",), ("Cz", "Pz"), (0.08, 0.12), "neg"),
            "late_positive": ERPComponent(
                ("target",),
                ("Pz",),
                (0.25, 0.35),
                "pos",
                ("peak_amplitude", "peak_latency", "mean_amplitude"),
            ),
        }

    def test_known_multiple_components_and_csv(self):
        before = self.evoked.data.copy()
        features, qc = extract_erp_idps(
            "sub-01", {"target": self.evoked}, self.components
        )
        self.assertEqual(features.shape, (1, 7))
        self.assertAlmostEqual(features.iloc[0, 0], -4)
        self.assertAlmostEqual(features.iloc[0, 1], 100)
        self.assertAlmostEqual(features.iloc[0, 4], 6)
        self.assertAlmostEqual(features.iloc[0, 5], 300)
        self.assertAlmostEqual(features.iloc[0, 6], 6 / 101)
        self.assertTrue((qc.status == "ok").all())
        np.testing.assert_array_equal(before, self.evoked.data)
        restored = pd.read_csv(io.StringIO(features.to_csv()), index_col=0)
        pd.testing.assert_frame_equal(restored, features)

    def test_missing_conditions_channels_trials_and_bad_channels(self):
        spec = ERPComponent(
            ("target", "missing"), ("Cz", "absent"), (0.08, 0.12), "neg"
        )
        bad = self.evoked.copy()
        bad.info["bads"] = ["Cz"]
        features, qc = extract_erp_idps("s", {"target": bad}, {"custom": spec})
        self.assertEqual(features.shape, (1, 8))
        self.assertTrue(features.isna().all().all())
        self.assertEqual(
            set(qc.status), {"bad_channel", "missing_channel", "missing_condition"}
        )
        features, qc = extract_erp_idps(
            "s", {"target": self.evoked}, self.components, 21
        )
        self.assertTrue(features.isna().all().all())
        self.assertEqual(set(qc.status), {"insufficient_trials"})

    def test_polarity_ties_mean_and_units(self):
        evoked = self.evoked.copy()
        evoked.data[:, 501] = evoked.data[:, 500]
        spec = ERPComponent(("target",), ("Pz",), (0.25, 0.35), "abs")
        features, _ = extract_erp_idps("s", {"target": evoked}, {"peak": spec})
        self.assertAlmostEqual(features.iloc[0, 1], 300)
        wrong = replace(
            spec,
            polarity="neg",
            metrics=("peak_amplitude", "peak_latency", "mean_amplitude"),
        )
        features, qc = extract_erp_idps("s", {"target": evoked}, {"peak": wrong})
        self.assertTrue(features.iloc[0, :2].isna().all())
        self.assertTrue(np.isfinite(features.iloc[0, 2]))
        self.assertEqual(
            qc.status.tolist(), ["no_peak_with_requested_polarity"] * 2 + ["ok"]
        )
        mag = mne.EvokedArray(
            np.full((1, 1101), 2e-13),
            mne.create_info(["M1"], 1000, "mag"),
            tmin=-0.2,
            nave=10,
        )
        spec = ERPComponent(("target",), ("M1",), (0.1, 0.2), "pos", sensor_type="mag")
        features, _ = extract_erp_idps("s", {"target": mag}, {"field": spec})
        self.assertIn("amplitude_T", features.columns[0])
        self.assertAlmostEqual(features.iloc[0, 0] / 1e-13, 2)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ERPComponent(("target",), ("Pz",), (0.2, 0.1), "pos")
        with self.assertRaises(ValueError):
            Config(
                analysis_type="erp", which_sensor="eeg", erp=dict(baseline=(-0.3, 0))
            )
        with self.assertRaises(ValueError):
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                cutoffFreqLow=35,
                cutoffFreqHigh=10,
            )
        with self.assertRaises(ValueError):
            extract_erp_idps(
                "s",
                {"target": self.evoked},
                {"outside": replace(self.components["N100"], window=(0.7, 0.9))},
            )
        with self.assertRaises(ValueError):
            extract_erp_idps(
                "s",
                {"target": self.evoked},
                {"wrong": replace(self.components["N100"], sensor_type="mag")},
            )

    def make_raw(self):
        data = np.full((2, 6000), 2e-06)
        for sample, amplitude in [(1000, 5e-06), (2500, 7e-06), (4000, 5e-06)]:
            data[:, sample + 300] += amplitude
            data[:, sample + 100] -= 3e-06
        raw = mne.io.RawArray(data, self.info, first_samp=500, verbose=False)
        events = np.array([[1500, 0, 1], [3000, 0, 2], [4500, 0, 1]])
        config = Config(
            analysis_type="erp",
            which_sensor="eeg",
            erp=dict(min_trials=1),
            apply_ica=False,
            rereference_method=None,
            cutoffFreqLow=None,
            cutoffFreqHigh=None,
        )
        return (raw, events, config)

    def test_raw_pipeline_baseline_conditions_and_no_mutation(self):
        raw, events, config = self.make_raw()
        original = raw.get_data().copy()
        spec = ERPComponent(
            ("target", "standard", "missing"), ("Pz",), (0.25, 0.35), "pos"
        )
        result = run_erp_pipeline(
            raw,
            "s",
            {"late": spec},
            events=events,
            event_id={"target": 1, "standard": 2, "missing": 3},
            config=config,
        )
        np.testing.assert_allclose(result.features.iloc[0, :4], [5, 300, 7, 300])
        self.assertTrue(result.features.iloc[0, 4:].isna().all())
        self.assertEqual(result.evokeds["target"].nave, 2)
        np.testing.assert_allclose(
            result.epochs.get_data()[:, :, :201].mean(axis=2), 0, atol=1e-20
        )
        np.testing.assert_array_equal(raw.get_data(), original)
        np.testing.assert_array_equal(result.epochs.events, events)

    def test_joint_resampling_and_annotations(self):
        raw, events, config = self.make_raw()
        raw.set_annotations(mne.Annotations([1, 2.5, 4], [0, 0, 0], ["A", "B", "A"]))
        result = run_erp_pipeline(
            raw,
            "s",
            self.components,
            annotation_map={"target": "A", "standard": "B"},
            config=Config.model_validate(
                {**config.model_dump(), **dict(resampling_rate=500)}
            ),
        )
        np.testing.assert_array_equal(result.epochs.events[:, 0], [750, 1500, 2250])
        self.assertEqual(result.epochs.info["sfreq"], 500)
        self.assertAlmostEqual(result.features.iloc[0, 1], 100)
        self.assertAlmostEqual(result.features.iloc[0, 5], 300)

    def test_rejection_and_all_rejected_qc(self):
        raw, events, config = self.make_raw()
        epochs = epoch_erp(
            raw,
            events,
            {"target": 1, "standard": 2},
            Config.model_validate(
                {**config.model_dump(), **dict(eeg_var_threshold=1e-06)}
            ),
        )
        self.assertEqual(len(epochs), 0)
        features, qc = extract_erp_idps("s", compute_evokeds(epochs), self.components)
        self.assertTrue(features.isna().all().all())
        raw.set_annotations(mne.Annotations([0.95], [0.1], ["BAD_test"]))
        epochs = epoch_erp(raw, events, {"target": 1, "standard": 2}, config)
        self.assertEqual(len(epochs), 2)
        with self.assertRaises(ValueError):
            epoch_erp(
                raw, np.array([[1500, 0, 1], [1500, 0, 2]]), {"target": 1}, config
            )

    def test_real_automatic_ica_with_synthetic_eog(self):
        rng = np.random.default_rng(42)
        times = np.arange(6000) / 200
        blink = sum(
            (np.exp(-(((times - center) / 0.12) ** 2)) for center in range(2, 29, 3))
        )
        sources = np.array(
            [blink, np.sin(2 * np.pi * 10 * times), rng.laplace(size=len(times))]
        )
        mixture = (
            np.array([[4, 1, 0.3], [2, -1, 0.7], [1, 0.5, -0.5]]) @ sources * 1e-05
        )
        raw = mne.io.RawArray(
            np.vstack([mixture, blink * 0.0001]),
            mne.create_info(["Fz", "Cz", "Pz", "EOG"], 200, ["eeg"] * 3 + ["eog"]),
            verbose=False,
        )
        cleaned, qc = preprocess_erp(
            raw,
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                rereference_method=None,
                cutoffFreqLow=0.5,
                cutoffFreqHigh=35,
                ica_n_component=3,
                auto_ica_corr_thr=0.5,
            ),
        )
        self.assertTrue(qc["ica_excluded"])
        self.assertEqual(qc["ica_method"], "physiological_correlation")
        self.assertTrue(np.isfinite(cleaned.get_data()).all())
        before = abs(np.corrcoef(raw.get_data()[0], blink)[0, 1])
        after = abs(np.corrcoef(cleaned.get_data()[0], blink)[0, 1])
        self.assertLess(after, before / 2)

    def test_ica_uses_shared_correlation_and_separate_fit_copy(self):
        raw, _, config = self.make_raw()
        raw._data[1] += np.sin(np.arange(6000) / 20) * 1e-06
        eog = mne.io.RawArray(
            np.sin(np.arange(6000)[None, :] / 10),
            mne.create_info(["EOG"], 1000, "eog"),
            first_samp=500,
            verbose=False,
        )
        raw.add_channels([eog])
        with (
            patch("mne.preprocessing.ICA") as factory,
            patch(
                "meganorm.src.preprocess.find_ica_component", return_value=([0], [0.9])
            ) as correlation,
        ):
            factory.return_value.n_components_ = 2
            cleaned, qc = preprocess_erp(
                raw,
                Config.model_validate(
                    {**config.model_dump(), **dict(apply_ica=True, cutoffFreqHigh=35)}
                ),
            )
            fitted = factory.return_value.fit.call_args.args[0]
            self.assertIsNot(fitted, cleaned)
            self.assertEqual(qc["ica_excluded"], [0])
            self.assertTrue(qc["ica_applied"])
            correlation.assert_called_once()
            factory.return_value.apply.assert_called_once()


def settings():
    return Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(
            components={
                "P300": dict(
                    conditions=["target"],
                    channels=["Pz"],
                    window=[0.25, 0.5],
                    polarity="pos",
                )
            }
        ),
    )


def test_rest_and_erp_roundtrip():
    assert Config.model_validate_json("{}").analysis_type == "rest"
    config = Config(analysis_type="erp", which_sensor="eeg", erp=settings().erp)
    assert Config.model_validate_json(config.model_dump_json()) == config
    assert config.erp.components["P300"].conditions == ("target",)
    assert config.ica_h_freq == config.cutoffFreqHigh


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(analysis_type="task"),
        dict(analysis_type="erp"),
        dict(erp={}),
        dict(analysis_type="erp", which_sensor="meg"),
        dict(analysis_type="erp", which_sensor="eeg", apply_source_localization=True),
    ],
)
def test_invalid_mode(kwargs):
    if kwargs.get("analysis_type") == "erp":
        kwargs.setdefault("erp", settings().erp)
    with pytest.raises(ValidationError):
        Config(**{**kwargs})


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(erp=dict(tmin=1)),
        dict(erp=dict(baseline=(-1, 0))),
        dict(erp=dict(min_trials=True)),
        dict(erp=dict(tmax=float("nan"))),
        dict(ica_n_component=0),
        dict(ica_n_component=1.5),
        dict(reject={"mag": 1}),
        dict(cutoffFreqHigh=60, resampling_rate=100),
        dict(cutoffFreqLow=40),
        dict(rereference_method=[]),
        dict(unknown=True),
    ],
)
def test_invalid_preprocessing(kwargs):
    with pytest.raises(ValidationError):
        Config(
            **{"analysis_type": "erp", "which_sensor": "eeg", "erp": dict(), **kwargs}
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mode="annotations"),
        dict(mode="stim", event_id={"target": 1}),
        dict(mode="stim", stim_channel="STI", event_id={"target": True}),
        dict(mode="stim", stim_channel="STI", event_id={"target": 1, "other": 1}),
        dict(mode="annotations", annotation_map={"target": "A"}, initial_event=True),
    ],
)
def test_invalid_events(kwargs):
    with pytest.raises(ValidationError):
        ERPEventSpec(**kwargs)


def test_events_components_and_dataset(tmp_path):
    from meganorm.API import Dataset

    spec = ERPEventSpec(mode="annotations", annotation_map={"target": "Stimulus/S 2"})
    settings().erp.validate_events(spec)
    with pytest.raises(ValueError):
        settings().erp.validate_events(
            ERPEventSpec(mode="annotations", annotation_map={"other": "A"})
        )
    dataset = Dataset(
        name="test", root=tmp_path, task="oddball", extension=".fif", erp_events=spec
    )
    restored = Dataset.from_dict("test", json.loads(json.dumps(dataset._settings())))
    assert restored.erp_events == spec
    folder = tmp_path / "sub-01"
    folder.mkdir()
    (folder / "sub-01_oddball.fif").touch()
    assert dataset.discover().iloc[0].erp_events == spec.model_dump(mode="json")
    with pytest.raises(ValidationError):
        Config(
            analysis_type="erp",
            which_sensor="eeg",
            erp=dict(
                components={
                    "P300": dict(
                        conditions=["target"],
                        channels=["Pz"],
                        window=[0.5, 1],
                        polarity="pos",
                    )
                }
            ),
        )


def test_local_guard_and_mutation(tmp_path):
    from meganorm.API import Pipeline

    erp = Config(analysis_type="erp", which_sensor="eeg", erp=settings().erp)
    Pipeline(config=erp, output_dir=tmp_path / "new")
    from meganorm.utils.erp import require_rest_runner

    config = Config()
    config.analysis_type = "task"
    with pytest.raises(ValueError):
        require_rest_runner(config)
    assert not (tmp_path / "new").exists()


def test_cli_and_cluster_guard_before_side_effects(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from meganorm.src import mainParallel as cli
    from meganorm.utils import parallel

    config = Config(analysis_type="erp", which_sensor="eeg", erp=settings().erp)
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    out = tmp_path / "outputs"
    monkeypatch.setattr(
        cli,
        "main_argparser",
        lambda args: SimpleNamespace(configs=str(path), save_dir=str(out)),
    )
    with pytest.raises(ValueError, match="erp_events"):
        cli.main([])
    jobs = {}
    with pytest.raises(NotImplementedError):
        parallel.sbatch_feature_extraction_runner(
            str(out), {}, jobs, config_file=config
        )
    with pytest.raises(NotImplementedError):
        parallel.auto_parallel_feature_extraction(
            "worker.py", str(out), {}, jobs, str(path)
        )
    assert jobs == {}
    assert not out.exists()


def test_pipeline_rechecks_mutated_config(tmp_path):
    from meganorm.API import Pipeline

    pipeline = Pipeline(config=Config(), output_dir=tmp_path / "outputs")
    pipeline.config.analysis_type = "erp"
    with pytest.raises(ValueError):
        pipeline.run([])
    assert not (tmp_path / "outputs").exists()


def test_nondefault_stim_and_preprocessing_roundtrip():
    spec = ERPEventSpec(
        mode="stim",
        event_id={"target": 0},
        stim_channel="STI 014",
        consecutive=False,
        shortest_event=1,
        min_duration=0.01,
        initial_event=True,
    )
    assert ERPEventSpec.model_validate_json(spec.model_dump_json()) == spec
    config = settings()
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(
            tmin=-0.1,
            tmax=0.7,
            baseline=None,
            min_trials=4,
            components=config.erp.components,
        ),
        cutoffFreqLow=None,
        cutoffFreqHigh=20,
        apply_ica=False,
        rereference_method=("Cz",),
        ica_n_component=0.95,
        eeg_var_threshold=None,
        resampling_rate=100,
    )
    restored = Config.model_validate_json(config.model_dump_json())
    assert restored == config
    restored.erp.validate_events(spec)


@pytest.mark.parametrize(
    "kwargs", [dict(cutoffFreqLow=0), dict(auto_ica_corr_thr=0), dict(ica_l_freq=40)]
)
def test_preprocessing_rejects_values_invalid_for_runtime(kwargs):
    with pytest.raises(ValidationError):
        Config(
            **{"analysis_type": "erp", "which_sensor": "eeg", "erp": dict(), **kwargs}
        )


def test_erp_default_task_supports_standalone_preprocessing():
    config = Config(analysis_type="erp", which_sensor="eeg")
    assert config.erp.components == {}


if __name__ == "__main__":
    unittest.main()


def _stage3_raw():
    raw = mne.io.RawArray(
        np.zeros((2, 600)),
        mne.create_info(["Pz", "STI"], 100.0, ["eeg", "stim"]),
        first_samp=500,
        verbose=False,
    )
    raw._data[1, 100:103] = 2
    raw._data[1, 300:303] = 7
    return raw


def test_event_resolver_annotations_crop_exact_and_missing():
    from meganorm.utils.erp import resolve_erp_events

    raw = _stage3_raw()
    raw.set_annotations(
        mne.Annotations([1.0, 3.0, 4.0], [0, 0, 0], ["A", "A/sub", "B"])
    )
    raw.crop(tmin=0.5, tmax=4.5)
    before = raw.get_data().copy()
    spec = ERPEventSpec(
        mode="annotations",
        annotation_map={"target": "A", "target/sub": "A/sub", "absent": "C"},
    )
    events, ids = resolve_erp_events(raw, spec)
    np.testing.assert_array_equal(events[:, 0], [600, 800])
    np.testing.assert_array_equal(events[:, 2], [ids["target"], ids["target/sub"]])
    assert "absent" in ids
    np.testing.assert_array_equal(raw.get_data(), before)


def test_event_resolver_stim_and_empty():
    from meganorm.utils.erp import resolve_erp_events

    raw = _stage3_raw()
    spec = ERPEventSpec(
        mode="stim", stim_channel="STI", event_id={"target": 2, "missing": 9}
    )
    events, ids = resolve_erp_events(raw, spec)
    np.testing.assert_array_equal(events, [[600, 0, 2]])
    assert ids == {"target": 2, "missing": 9}
    with pytest.raises(ValueError, match="No requested"):
        resolve_erp_events(
            raw, ERPEventSpec(mode="stim", stim_channel="STI", event_id={"missing": 9})
        )


def test_event_resolver_duplicates_and_missing_channel():
    from meganorm.utils.erp import resolve_erp_events

    raw = _stage3_raw()
    raw.set_annotations(mne.Annotations([1, 1], [0, 0], ["A", "B"]))
    with pytest.raises(ValueError, match="unique"):
        resolve_erp_events(
            raw, ERPEventSpec(mode="annotations", annotation_map={"a": "A", "b": "B"})
        )
    with pytest.raises(ValueError, match="channel"):
        resolve_erp_events(
            raw, ERPEventSpec(mode="stim", stim_channel="absent", event_id={"a": 1})
        )


def test_pipeline_event_spec_and_exact_conditions():
    raw = _stage3_raw()
    raw._data[0, 100:110] = 2e-06
    raw._data[0, 300:310] = 8e-06
    component = ERPComponent(
        ("target", "target/sub", "absent"),
        ("Pz",),
        (0, 0.09),
        "pos",
        ("mean_amplitude",),
    )
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(tmin=0, tmax=0.1, baseline=None, min_trials=1),
        cutoffFreqLow=None,
        cutoffFreqHigh=None,
        rereference_method=None,
        apply_ica=False,
        eeg_var_threshold=None,
    )
    result = run_erp_pipeline(
        raw,
        "s",
        {"P": component},
        event_spec=ERPEventSpec(
            mode="stim",
            stim_channel="STI",
            event_id={"target": 2, "target/sub": 7, "absent": 9},
        ),
        config=config,
    )
    np.testing.assert_allclose(result.features.iloc[0, :2], [2, 8])
    assert np.isnan(result.features.iloc[0, 2])
    with pytest.raises(ValueError, match="one|OR"):
        run_erp_pipeline(
            raw,
            "s",
            {"P": component},
            event_spec=ERPEventSpec(
                mode="stim", stim_channel="STI", event_id={"target": 2}
            ),
            events=np.array([[600, 0, 2]]),
            event_id={"target": 2},
            config=config,
        )


def test_boundary_peak_is_flagged_but_kept():
    evoked = mne.EvokedArray(
        np.array([[3.0, 1.0, 3.0]]) * 1e-6,
        mne.create_info(["Pz"], 100.0, "eeg"),
        tmin=0,
        nave=10,
        verbose=False,
    )
    component = ERPComponent(
        ("a",),
        ("Pz",),
        (0, 0.02),
        "pos",
        ("peak_amplitude", "peak_latency", "mean_amplitude"),
    )
    before = evoked.data.copy()
    features, qc = extract_erp_idps("s", {"a": evoked}, {"P": component})
    np.testing.assert_allclose(features.iloc[0], [3, 0, 7 / 3])
    assert qc.boundary_peak.tolist() == [True, True, False]
    assert qc.status.tolist() == ["ok"] * 3
    np.testing.assert_array_equal(evoked.data, before)


def test_resampling_collision_fails_before_preprocessing():
    raw = _stage3_raw()
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(tmin=0, tmax=0.1, baseline=None),
        cutoffFreqLow=None,
        cutoffFreqHigh=None,
        apply_ica=False,
        resampling_rate=10,
    )
    component = ERPComponent(("a",), ("Pz",), (0, 0.1), "pos")
    with patch(
        "meganorm.src.erp.preprocess_erp",
        side_effect=AssertionError("must fail before cleaning"),
    ):
        with pytest.raises(ValueError, match="resampl"):
            run_erp_pipeline(
                raw,
                "s",
                {"P": component},
                events=np.array([[600, 0, 0], [601, 0, 0]]),
                event_id={"a": 0},
                config=config,
            )


def test_drop_counts_partition_edge_annotation_and_amplitude():
    raw = _stage3_raw()
    raw._data[0, 300] = 0.0002
    raw.set_annotations(mne.Annotations([1.9], [0.3], ["BAD_manual"]))
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(tmin=-0.1, tmax=0.1, baseline=None, min_trials=1),
        cutoffFreqLow=None,
        cutoffFreqHigh=None,
        rereference_method=None,
        apply_ica=False,
        eeg_var_threshold=0.0001,
    )
    component = ERPComponent(("a", "missing"), ("Pz",), (0, 0.1), "pos")
    result = run_erp_pipeline(
        raw,
        "s",
        {"P": component},
        events=np.array(
            [[500, 0, 0], [600, 0, 0], [700, 0, 0], [800, 0, 0], [1099, 0, 0]]
        ),
        event_id={"a": 0, "missing": 3},
        config=config,
    )
    row = result.qc.iloc[0]
    assert (row.n_events, row.n_trials, row.n_rejected) == (5, 1, 4)
    assert (
        row.n_edge_dropped,
        row.n_annotation_rejected,
        row.n_amplitude_rejected,
        row.n_other_dropped,
    ) == (2, 1, 1, 0)
    assert result.qc.iloc[-1].n_events == 0


def test_stim_zero_requires_explicit_events_and_initial_event_options():
    from meganorm.utils.erp import resolve_erp_events

    raw = _stage3_raw()
    with pytest.raises(ValueError, match="zero.*idle"):
        resolve_erp_events(
            raw, ERPEventSpec(mode="stim", stim_channel="STI", event_id={"a": 0})
        )
    raw._data[1, :3] = 2
    spec = ERPEventSpec(
        mode="stim", stim_channel="STI", event_id={"a": 2}, initial_event=True
    )
    events, _ = resolve_erp_events(raw, spec)
    np.testing.assert_array_equal(events[:, 0], [500, 600])
    spec.initial_event = False
    events, _ = resolve_erp_events(raw, spec)
    np.testing.assert_array_equal(events[:, 0], [600])
    spec.min_duration = 0.05
    with pytest.raises(ValueError, match="No requested"):
        resolve_erp_events(raw, spec)


def test_stim_consecutive_and_shortest_event():
    from meganorm.utils.erp import resolve_erp_events

    raw = _stage3_raw()
    raw._data[1, :] = 0
    raw._data[1, 100:103] = 7
    raw._data[1, 103:106] = 2
    spec = ERPEventSpec(
        mode="stim", stim_channel="STI", event_id={"a": 2, "b": 7}, consecutive=True
    )
    events, _ = resolve_erp_events(raw, spec)
    np.testing.assert_array_equal(events[:, 0], [600, 603])
    spec.consecutive = "increasing"
    events, _ = resolve_erp_events(raw, spec)
    np.testing.assert_array_equal(events[:, 0], [600])
    raw._data[1, :] = 0
    raw._data[1, 100] = 7
    raw._data[1, 101:104] = 2
    spec.consecutive = True
    with pytest.raises(ValueError):
        resolve_erp_events(raw, spec)
    spec.shortest_event = 1
    assert resolve_erp_events(raw, spec)[0][0, 0] == 600


def test_no_requested_events_fails_before_cleaning():
    raw = _stage3_raw()
    component = ERPComponent(("a",), ("Pz",), (0, 0.1), "pos")
    with patch(
        "meganorm.src.erp.preprocess_erp",
        side_effect=AssertionError("unexpected cleaning"),
    ):
        with pytest.raises(ValueError, match="No requested"):
            run_erp_pipeline(
                raw,
                "s",
                {"P": component},
                events=np.array([[600, 0, 8]]),
                event_id={"a": 0},
            )


def test_crop_resample_absolute_samples_and_input_preservation():
    raw = _stage3_raw().crop(tmin=0.5, tmax=4.5)
    before = raw.get_data().copy()
    spec = ERPEventSpec(mode="stim", stim_channel="STI", event_id={"a": 2, "b": 7})
    component = ERPComponent(("a", "b"), ("Pz",), (0, 0.1), "abs")
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(tmin=0, tmax=0.1, baseline=None, min_trials=1),
        cutoffFreqLow=None,
        cutoffFreqHigh=None,
        rereference_method=None,
        apply_ica=False,
        eeg_var_threshold=None,
        resampling_rate=50,
    )
    result = run_erp_pipeline(
        raw, "s", {"P": component}, event_spec=spec, config=config
    )
    np.testing.assert_array_equal(result.epochs.events[:, 0], [300, 400])
    assert raw.first_samp == 550 and raw.info["sfreq"] == 100
    np.testing.assert_array_equal(raw.get_data(), before)


@pytest.mark.parametrize(
    "signal,status",
    [
        ([0, 0, 0], "no_peak_with_requested_polarity"),
        ([np.nan, 1, 2], "nonfinite_data"),
    ],
)
def test_unavailable_peak_schema_and_boundary_flag(signal, status):
    evoked = mne.EvokedArray(
        np.array([signal]) * 1e-6,
        mne.create_info(["Pz"], 100, "eeg"),
        tmin=0,
        nave=3,
        verbose=False,
    )
    component = ERPComponent(("a",), ("Pz",), (0, 0.02), "abs")
    features, qc = extract_erp_idps("s", {"a": evoked}, {"P": component})
    missing, missing_qc = extract_erp_idps("s", {"a": None}, {"P": component})
    assert features.columns.tolist() == missing.columns.tolist()
    assert features.isna().all().all()
    assert qc.status.tolist() == [status, status]
    assert not qc.boundary_peak.any()
    assert not missing_qc.boundary_peak.any()


def test_window_uses_only_inclusive_sample_times():
    evoked = mne.EvokedArray(
        np.array([[9, 1, 3, 9]]) * 1e-6,
        mne.create_info(["Pz"], 100, "eeg"),
        tmin=0,
        nave=3,
        verbose=False,
    )
    component = ERPComponent(
        ("a",),
        ("Pz",),
        (0.005, 0.025),
        "pos",
        ("peak_amplitude", "peak_latency", "mean_amplitude"),
    )
    features, qc = extract_erp_idps("s", {"a": evoked}, {"P": component})
    np.testing.assert_allclose(features.iloc[0], [3, 20, 2])
    assert qc.boundary_peak.tolist() == [True, True, False]
    with pytest.raises(ValueError, match="no samples"):
        extract_erp_idps(
            "s", {"a": evoked}, {"P": replace(component, window=(0.001, 0.002))}
        )


def test_ambiguous_annotation_channel_label_is_not_misclassified():
    raw = _stage3_raw()
    raw.rename_channels({"Pz": "BAD_sensor"})
    raw.set_annotations(mne.Annotations([0.9], [0.3], ["BAD_sensor"]))
    component = ERPComponent(("a",), ("BAD_sensor",), (0, 0.1), "pos")
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(tmin=0, tmax=0.1, baseline=None, min_trials=1),
        cutoffFreqLow=None,
        cutoffFreqHigh=None,
        rereference_method=None,
        apply_ica=False,
        eeg_var_threshold=None,
    )
    result = run_erp_pipeline(
        raw,
        "s",
        {"P": component},
        events=np.array([[600, 0, 2], [800, 0, 2]]),
        event_id={"a": 2},
        config=config,
    )
    row = result.qc.iloc[0]
    assert row.n_other_dropped == 1
    assert row.n_amplitude_rejected == 0
    assert row.n_annotation_rejected == 0
    assert row.n_trials + row.n_rejected == row.n_events


def test_flatness_rejection_counts_as_amplitude():
    raw = _stage3_raw()
    raw._data[0, 300:310] = np.arange(10) * 1e-06
    component = ERPComponent(("a",), ("Pz",), (0, 0.1), "pos")
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(tmin=0, tmax=0.1, baseline=None, min_trials=1),
        cutoffFreqLow=None,
        cutoffFreqHigh=None,
        rereference_method=None,
        apply_ica=False,
        eeg_var_threshold=None,
        eeg_flat_threshold=1e-07,
    )
    result = run_erp_pipeline(
        raw,
        "s",
        {"P": component},
        events=np.array([[600, 0, 2], [800, 0, 2]]),
        event_id={"a": 2},
        config=config,
    )
    row = result.qc.iloc[0]
    assert (
        row.n_events,
        row.n_trials,
        row.n_amplitude_rejected,
        row.n_other_dropped,
    ) == (2, 1, 1, 0)


def test_erp_worker_failed_idps_and_atomic_write_failure(tmp_path, monkeypatch):
    from meganorm.src.erp import process_erp_recording
    from meganorm.utils import erp as utils

    raw = _stage3_raw()
    path = tmp_path / "input_raw.fif"
    raw.save(path, overwrite=True, verbose=False)
    definition = settings()
    definition = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(
            tmin=0,
            tmax=0.6,
            baseline=None,
            min_trials=1,
            components=definition.erp.components,
        ),
        cutoffFreqLow=None,
        cutoffFreqHigh=None,
        rereference_method=None,
        apply_ica=False,
        eeg_var_threshold=None,
    )
    spec = ERPEventSpec(mode="stim", stim_channel="STI", event_id={"target": 2})
    result = process_erp_recording(
        path, "001", config=definition, event_spec=spec, output_root=tmp_path / "first"
    )
    directory = tmp_path / "first" / "participants" / "001"
    assert result.features.isna().all().all()
    assert json.loads((directory / "complete.json").read_text())["status"] == "failed"
    assert (directory / "qc.csv").exists()
    with pytest.raises(FileExistsError):
        process_erp_recording(
            path,
            "001",
            config=definition,
            event_spec=spec,
            output_root=tmp_path / "first",
        )
    original = utils.write_erp_table

    def fail_qc(path, table, **kwargs):
        if path.name == "qc.csv":
            raise OSError("disk error")
        original(path, table, **kwargs)

    monkeypatch.setattr(utils, "write_erp_table", fail_qc)
    with pytest.raises(OSError, match="disk error"):
        process_erp_recording(
            path,
            "001",
            config=definition,
            event_spec=spec,
            output_root=tmp_path / "second",
        )
    assert not (tmp_path / "second" / "participants" / "001" / "complete.json").exists()
    with pytest.raises(ValueError, match="Participant ID"):
        process_erp_recording(
            path,
            "../escape",
            config=definition,
            event_spec=spec,
            output_root=tmp_path / "third",
        )
    assert not (tmp_path / "third").exists()


def test_atomic_json_failure_keeps_previous_file(tmp_path):
    from meganorm.utils.erp import write_erp_json

    path = tmp_path / "state.json"
    path.write_text('{"old": true}')
    with pytest.raises(ValueError):
        write_erp_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]


def test_erp_json_supports_numpy_qc_scalars(tmp_path):
    from meganorm.utils.erp import write_erp_json

    path = tmp_path / "qc.json"
    write_erp_json(
        path,
        {
            "ica_excluded": [np.int64(1)],
            "value": np.float64(0.5),
            "samples": np.array([1, 2]),
        },
    )
    assert json.loads(path.read_text()) == {
        "ica_excluded": [1],
        "value": 0.5,
        "samples": [1, 2],
    }


def test_erp_worker_sidecar_types_and_montage(tmp_path):
    from meganorm.src.erp import process_erp_recording

    raw = _stage3_raw()
    raw.set_channel_types({"Pz": "misc"})
    raw._data[0, 100:111] = np.arange(11) * 1e-06
    path = tmp_path / "s_task_raw.fif"
    raw.save(path, overwrite=True, verbose=False)
    (tmp_path / "s_task_channels.tsv").write_text(
        "name\ttype\tstatus\nPz\tEEG\tgood\nSTI\tSTIM\tgood\n"
    )
    (tmp_path / "s_montage.csv").write_text("Channel,X,Y,Z\nPz,0,-0.05,0.08\n")
    definition = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(
            tmin=0,
            tmax=0.1,
            baseline=None,
            min_trials=1,
            components={
                "P": dict(
                    conditions=["a"], channels=["Pz"], window=[0, 0.1], polarity="pos"
                )
            },
        ),
        cutoffFreqLow=None,
        cutoffFreqHigh=None,
        rereference_method=None,
        apply_ica=False,
        eeg_var_threshold=None,
    )
    result = process_erp_recording(
        path,
        "s",
        config=definition,
        event_spec=ERPEventSpec(mode="stim", stim_channel="STI", event_id={"a": 2}),
        output_root=tmp_path / "out",
    )
    assert np.isfinite(result.features.to_numpy()).all()
    assert result.epochs.get_montage() is not None
    assert mne.io.read_raw_fif(path, verbose=False).get_channel_types()[0] == "misc"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(ica_mode="unknown"),
        dict(auto_ica_corr_thr=0),
        dict(ica_label_threshold=1.1),
        dict(auto_ica_corr_thr=float("nan")),
        dict(apply_ica=False, ica_mode="physiological"),
        dict(ica_n_component=True),
        dict(ica_n_component=1),
    ],
)
def test_ica_settings_reject_invalid_choices(kwargs):
    with pytest.raises(ValueError):
        Config(**{**dict(analysis_type="erp", which_sensor="eeg", **kwargs)})


def test_ica_settings_roundtrip_with_distinct_thresholds():
    definition = settings()
    definition = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(components=definition.erp.components),
        ica_mode="physiological",
        auto_ica_corr_thr=0.4,
        ica_label_threshold=0.9,
    )
    restored = Config.model_validate_json(definition.model_dump_json())
    assert restored == definition
    assert restored.auto_ica_corr_thr == 0.4
    assert restored.ica_label_threshold == 0.9


def ica_raw(aux="good", montage=False):
    rng = np.random.default_rng(12)
    raw = mne.io.RawArray(
        rng.normal(size=(4, 2000)) * 1e-6,
        mne.create_info(["Fz", "Cz", "Pz", "Oz"], 200, "eeg"),
        verbose=False,
    )
    if aux is not None:
        signal = np.sin(np.arange(2000) / 10)[None, :] * 1e-4
        if aux == "flat":
            signal[:] = 0
        elif aux == "nonfinite":
            signal[0, 10] = np.nan
        raw.add_channels(
            [
                mne.io.RawArray(
                    signal, mne.create_info(["EOG"], 200, "eog"), verbose=False
                )
            ]
        )
    if montage:
        raw.set_montage("standard_1020")
    return raw


@pytest.mark.parametrize("kwargs", [dict(ica_mode="none"), dict(apply_ica=False)])
def test_ica_disabled_is_explicit_in_qc(kwargs):
    cleaned, qc = preprocess_erp(
        ica_raw(aux="flat"),
        Config(
            **{"analysis_type": "erp", "which_sensor": "eeg", "erp": dict(), **kwargs}
        ),
    )
    assert qc["ica_mode_effective"] == "none"
    assert not qc["ica_applied"]
    assert qc["bad_channel_detection"] == "not_performed"
    assert np.isfinite(cleaned.get_data()).all()


@pytest.mark.parametrize("aux", ["flat", "nonfinite", None])
def test_physiological_requires_usable_auxiliary(aux):
    with pytest.raises(ValueError, match="usable EOG/ECG"):
        preprocess_erp(
            ica_raw(aux),
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                ica_mode="physiological",
            ),
        )


@pytest.mark.parametrize("aux", ["flat", "nonfinite", None])
def test_auto_cannot_silently_skip_ica_without_locations(aux):
    with pytest.raises(ValueError, match="sensor locations"):
        preprocess_erp(
            ica_raw(aux), Config(analysis_type="erp", which_sensor="eeg", erp=dict())
        )


def test_ica_rank_guard_and_reference_consistency():
    raw = ica_raw()
    with pytest.raises(ValueError, match="rank"):
        preprocess_erp(
            raw,
            Config(
                analysis_type="erp", which_sensor="eeg", erp=dict(), ica_n_component=4
            ),
        )
    raw._data[:4] = raw._data[0]
    with pytest.raises(ValueError, match="rank"):
        preprocess_erp(
            raw,
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                rereference_method=None,
            ),
        )


@pytest.mark.parametrize(
    "kwargs", [dict(rereference_method=None), dict(ica_l_freq=0.5)]
)
def test_iclabel_requires_reference_and_fit_highpass(kwargs):
    with pytest.raises(ValueError, match="ICLabel"):
        preprocess_erp(
            ica_raw(montage=True),
            Config(
                **{
                    "analysis_type": "erp",
                    "which_sensor": "eeg",
                    "erp": dict(),
                    "ica_mode": "iclabel",
                    **kwargs,
                }
            ),
        )


def test_auto_records_bad_auxiliary_and_label_decisions():
    raw = ica_raw(aux="flat", montage=True)
    before = raw.get_data().copy()
    with patch(
        "mne_icalabel.label_components",
        return_value=dict(
            labels=["eye blink", "brain", "muscle artifact"],
            y_pred_proba=np.array([0.85, 0.99, 0.65]),
        ),
    ):
        cleaned, qc = preprocess_erp(
            raw,
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                auto_ica_corr_thr=0.6,
                ica_label_threshold=0.8,
            ),
        )
    assert qc["ica_mode_effective"] == "iclabel"
    assert qc["ica_excluded"] == [0]
    assert qc["ica_rank"] == 3
    assert qc["ica_n_components"] == 3
    assert qc["ica_auxiliary"]["EOG"] == "flat"
    assert qc["ica_thresholds"] == dict(correlation=0.6, label_probability=0.8)
    assert len(qc["ica_component_scores"]) == 3
    assert qc["ica_warnings"]
    np.testing.assert_array_equal(raw.get_data(), before)
    np.testing.assert_allclose(cleaned.get_data()[:4].mean(axis=0), 0, atol=1e-18)
    json.dumps(qc, allow_nan=False)


def test_physiological_threshold_is_independent_and_recorded():
    with patch(
        "meganorm.src.preprocess.find_ica_component", return_value=([1], [-0.7])
    ) as correlation:
        _, qc = preprocess_erp(
            ica_raw(),
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                ica_mode="physiological",
                auto_ica_corr_thr=0.6,
                ica_label_threshold=0.99,
            ),
        )
    assert correlation.call_args.args[3] == 0.6
    assert qc["ica_excluded"] == [1]
    assert qc["ica_mode_effective"] == "physiological"
    assert qc["ica_auxiliary"] == {"EOG": "usable"}
    assert qc["ica_component_scores"][0]["correlation"] == -0.7
    json.dumps(qc, allow_nan=False)


@pytest.mark.parametrize("aux", ["nonfinite", "marked_bad"])
def test_auto_falls_back_from_unusable_auxiliary(aux):
    raw = ica_raw(aux, montage=True)
    if aux == "marked_bad":
        raw.info["bads"] = ["EOG"]
    with patch(
        "mne_icalabel.label_components",
        return_value=dict(labels=["brain"] * 3, y_pred_proba=np.array([0.9] * 3)),
    ):
        _, qc = preprocess_erp(
            raw,
            Config(
                analysis_type="erp", which_sensor="eeg", erp=dict(), cutoffFreqLow=1
            ),
        )
    assert qc["ica_mode_effective"] == "iclabel"
    assert qc["ica_auxiliary"]["EOG"] == aux
    assert qc["ica_applied"] and qc["ica_excluded"] == []


def test_physiological_omits_bad_annotation_samples():
    raw = ica_raw("nonfinite")
    raw.set_annotations(mne.Annotations([0], [0.2], ["BAD_aux"]))
    with patch(
        "meganorm.src.preprocess.find_ica_component", return_value=([], [])
    ) as correlation:
        _, qc = preprocess_erp(
            raw,
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                ica_mode="physiological",
                cutoffFreqLow=1,
            ),
        )
    assert qc["ica_auxiliary"]["EOG"] == "usable"
    assert correlation.call_args.args[1].n_times == 1960
    assert len(correlation.call_args.args[2]) == 1960
    assert np.isfinite(correlation.call_args.args[2]).all()


def test_iclabel_checks_every_usable_sensor_location():
    raw = ica_raw(montage=True)
    raw.info["chs"][0]["loc"][:3] = np.nan
    with pytest.raises(ValueError, match="sensor locations"):
        preprocess_erp(
            raw,
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                ica_mode="iclabel",
                cutoffFreqLow=1,
            ),
        )


def test_real_iclabel_explicit_mode_with_auxiliary():
    _, qc = preprocess_erp(
        ica_raw(montage=True),
        Config(
            analysis_type="erp",
            which_sensor="eeg",
            erp=dict(),
            ica_mode="iclabel",
            cutoffFreqLow=1,
        ),
    )
    assert qc["ica_mode_effective"] == "iclabel"
    assert qc["ica_selection_reason"] == "explicit strategy"
    assert qc["ica_applied"]
    assert len(qc["ica_component_scores"]) == 3
    assert all((0 <= row["probability"] <= 1 for row in qc["ica_component_scores"]))
    json.dumps(qc, allow_nan=False)


@pytest.mark.parametrize("active, expected_rank", [(False, 4), (True, 2)])
def test_ica_rank_matches_applied_projectors(active, expected_rank):
    raw = ica_raw()
    projectors = [
        mne.Projection(
            data=dict(
                col_names=raw.ch_names[:4],
                row_names=None,
                nrow=1,
                ncol=4,
                data=np.eye(4)[i : i + 1],
            ),
            desc=f"EEG SSP {i}",
            kind=1,
            active=False,
        )
        for i in range(2)
    ]
    raw.add_proj(projectors)
    if active:
        raw.apply_proj()
    _, qc = preprocess_erp(
        raw,
        Config(
            analysis_type="erp",
            which_sensor="eeg",
            erp=dict(),
            rereference_method=None,
            cutoffFreqLow=1,
            ica_mode="physiological",
        ),
    )
    assert qc["ica_rank"] == expected_rank
    assert qc["ica_n_components"] == expected_rank
    assert [p["active"] for p in raw.info["projs"]] == [active, active]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(autoreject_cv=1),
        dict(autoreject_cv=True),
        dict(autoreject_n_interpolates=[]),
        dict(autoreject_n_interpolates=[-1]),
        dict(autoreject_n_interpolates=[1, 1]),
        dict(autoreject_consensus_percs=[0]),
        dict(autoreject_consensus_percs=[1.1]),
        dict(autoreject_consensus_percs=[float("nan")]),
    ],
)
def test_autoreject_settings_invalid(kwargs):
    with pytest.raises(ValueError):
        Config(
            **{
                **dict(
                    analysis_type="erp",
                    which_sensor="eeg",
                    bad_segment_removal_method="autoreject",
                    **kwargs,
                )
            }
        )


def test_autoreject_settings_runtime_roundtrip():
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=settings().erp,
        bad_segment_removal_method="autoreject",
        autoreject_cv=3,
        autoreject_n_interpolates=[1],
        autoreject_consensus_percs=[0.5, 1.0],
    )
    assert Config.model_validate_json(config.model_dump_json()) == config
    assert config.bad_segment_removal_method == "autoreject"
    assert config.autoreject_cv == 3
    assert config.autoreject_n_interpolates == [1]
    assert config.autoreject_consensus_percs == [0.5, 1.0]
    with pytest.raises(ValueError, match="rejection_mode"):
        Config(analysis_type="erp", which_sensor="eeg", rejection_mode="threshold")


def ar_epochs(n_epochs=12, n_channels=6, montage=True):
    names = ["Fz", "Cz", "Pz", "Oz", "C3", "C4"][:n_channels]
    rng = np.random.default_rng(21)
    data = rng.normal(size=(n_epochs, n_channels, 101)) * 1e-6
    events = np.column_stack(
        [
            500 + np.arange(n_epochs) * 200,
            np.zeros(n_epochs, int),
            1 + np.arange(n_epochs) % 2,
        ]
    )
    epochs = mne.EpochsArray(
        data,
        mne.create_info(names, 100, "eeg"),
        events=events,
        event_id={"standard": 1, "target": 2},
        tmin=-0.2,
        baseline=(-0.2, 0),
        metadata=pd.DataFrame({"trial": np.arange(n_epochs)}),
        verbose=False,
    )
    if montage:
        epochs.set_montage("standard_1020")
    return epochs


@pytest.mark.parametrize(
    "case,match",
    [
        ("few_trials", "epochs"),
        ("few_channels", "channels"),
        ("no_montage", "locations"),
        ("nonfinite", "finite"),
        ("duplicate_locations", "locations"),
        ("bad_grid", "n_interpolate"),
    ],
)
def test_autoreject_prerequisites(case, match):
    from meganorm.src.erp import reject_erp_epochs

    epochs = ar_epochs(
        n_epochs=3 if case == "few_trials" else 12,
        n_channels=3 if case == "few_channels" else 6,
        montage=case != "no_montage",
    )
    if case == "nonfinite":
        epochs._data[0, 0, 0] = np.nan
    if case == "duplicate_locations":
        epochs.info["chs"][0]["loc"][:3] = epochs.info["chs"][1]["loc"][:3]
    options = dict(autoreject_n_interpolates=[6]) if case == "bad_grid" else None
    with pytest.raises(ValueError, match=match):
        reject_erp_epochs(
            epochs,
            Config(
                **{
                    "analysis_type": "erp",
                    "which_sensor": "eeg",
                    "erp": dict(),
                    "bad_segment_removal_method": "autoreject",
                    **(options or {}),
                }
            ),
        )


def test_threshold_helper_preserves_epochs_and_metadata():
    from meganorm.src.erp import reject_erp_epochs

    epochs = ar_epochs()
    epochs._data[2, 0, 50] = 0.0002
    before = epochs.get_data().copy()
    cleaned, qc = reject_erp_epochs(
        epochs, Config(analysis_type="erp", which_sensor="eeg", erp=dict())
    )
    assert len(cleaned) == 11 and cleaned.drop_log[2] == ("Fz",)
    assert cleaned.metadata["trial"].tolist() == [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert cleaned.tmin == -0.2 and cleaned.baseline == epochs.baseline
    np.testing.assert_array_equal(epochs.get_data(), before)
    assert qc["mode"] == "threshold"


def test_real_autoreject_epochs_reproducible_and_aligned():
    from meganorm.src.erp import reject_erp_epochs

    epochs = ar_epochs()
    epochs._data[0, 0, 40:50] += 0.0002
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(),
        bad_segment_removal_method="autoreject",
        autoreject_cv=3,
        autoreject_n_interpolates=[1],
        autoreject_consensus_percs=[0.5, 1.0],
    )
    before = epochs.get_data().copy()
    cleaned, qc = reject_erp_epochs(epochs, config)
    again, qc_again = reject_erp_epochs(epochs, config)
    np.testing.assert_allclose(cleaned.get_data(), again.get_data())
    assert qc == qc_again
    np.testing.assert_array_equal(epochs.get_data(), before)
    np.testing.assert_array_equal(cleaned.times, epochs.times)
    np.testing.assert_array_equal(cleaned.events, epochs.events[cleaned.selection])
    assert cleaned.metadata["trial"].tolist() == cleaned.selection.tolist()
    assert cleaned.baseline == epochs.baseline
    assert qc["n_input_epochs"] == 12 and qc["n_retained_epochs"] == len(cleaned)
    assert qc["trials"][0]["interpolated_channels"] or not qc["trials"][0]["retained"]
    json.dumps(qc, allow_nan=False)


def controlled_ar(monkeypatch, all_bad=False):
    """Control fitted decisions; retain AutoReject's real MNE transformation."""
    from autoreject import AutoReject, RejectLog

    def fit(self, epochs):
        self.picks_ = np.asarray(self.picks)
        self.n_interpolate_ = {"eeg": 1}
        self.consensus_ = {"eeg": 0.5}
        self.loss_ = {"eeg": np.zeros((1, 1, 3))}
        self.threshes_ = {epochs.ch_names[p]: 1e-4 for p in self.picks_}
        self.dots = None
        return self

    def log(self, epochs):
        labels = np.full((len(epochs), len(epochs.ch_names)), np.nan)
        labels[:, self.picks_] = 0
        bad = np.zeros(len(epochs), bool)
        bad[0] = True
        if all_bad:
            bad[:] = True
        labels[0, self.picks_] = 1
        if len(epochs) > 1:
            labels[1, self.picks_[0]] = 2
        return RejectLog(bad_epochs=bad, labels=labels, ch_names=epochs.ch_names)

    monkeypatch.setattr(AutoReject, "fit", fit)
    monkeypatch.setattr(AutoReject, "get_reject_log", log)


def ar_pipeline_raw():
    rng = np.random.default_rng(20)
    raw = mne.io.RawArray(
        rng.normal(size=(6, 5000)) * 1e-6,
        mne.create_info(["Fz", "Cz", "Pz", "Oz", "C3", "C4"], 100, "eeg"),
        verbose=False,
    )
    raw.set_montage("standard_1020")
    events = np.column_stack(
        [np.arange(9) * 500, np.zeros(9, int), 1 + np.arange(9) % 2]
    )
    raw.set_annotations(mne.Annotations([4.8], [0.5], ["BAD_motion"]))
    return raw, events


@pytest.mark.parametrize("all_bad", [False, True])
def test_autoreject_pipeline_trial_accounting_and_all_rejected(monkeypatch, all_bad):
    controlled_ar(monkeypatch, all_bad)
    raw, events = ar_pipeline_raw()
    components = {
        "response": ERPComponent(
            ("standard", "target"), ("Fz",), (0.1, 0.4), "pos", ("mean_amplitude",)
        )
    }
    result = run_erp_pipeline(
        raw,
        "s",
        components,
        events=events,
        event_id={"standard": 1, "target": 2},
        config=Config(
            analysis_type="erp",
            which_sensor="eeg",
            erp=dict(min_trials=1),
            ica_mode="none",
            rereference_method=None,
            cutoffFreqLow=None,
            cutoffFreqHigh=None,
            bad_segment_removal_method="autoreject",
        ),
    )
    report = result.preprocessing_qc["epoch_rejection"]
    assert report["autoreject_rejected_event_indices"] == (
        list(range(2, 9)) if all_bad else [2]
    )
    assert result.epochs.tmin == -0.2
    assert report["n_interpolated_trials"] == (0 if all_bad else 1)
    assert result.qc["n_autoreject_rejected"].sum() == (7 if all_bad else 1)
    assert result.qc["n_annotation_rejected"].sum() == 1
    assert result.qc["n_edge_dropped"].sum() == 1
    assert result.qc["n_amplitude_rejected"].sum() == 0
    columns = [
        "n_edge_dropped",
        "n_annotation_rejected",
        "n_amplitude_rejected",
        "n_autoreject_rejected",
        "n_other_dropped",
    ]
    np.testing.assert_array_equal(result.qc[columns].sum(axis=1), result.qc.n_rejected)
    np.testing.assert_array_equal(
        result.qc.n_trials + result.qc.n_rejected, result.qc.n_events
    )
    if all_bad:
        assert result.features.isna().all().all()
        assert result.qc.status.eq("no_retained_trials").all()
    else:
        np.testing.assert_array_equal(result.epochs.selection, np.arange(3, 9))
        assert report["trials"][1]["interpolated_channels"] == ["Fz"]
    json.dumps(result.preprocessing_qc, allow_nan=False)


def test_autoreject_flat_precheck_bad_channel_and_copy(monkeypatch):
    from meganorm.src.erp import reject_erp_epochs

    controlled_ar(monkeypatch)
    epochs = ar_epochs()
    epochs._data[3, 0, :] = 0
    epochs.info["bads"] = ["C4"]
    epochs._data[4, 1, 40] = 0.0002
    before = epochs.get_data().copy()
    clean, qc = reject_erp_epochs(
        epochs,
        Config(
            analysis_type="erp",
            which_sensor="eeg",
            erp=dict(),
            bad_segment_removal_method="autoreject",
            eeg_flat_threshold=1e-12,
        ),
    )
    assert qc["n_prefit_flat_rejected"] == 1
    assert 3 not in clean.selection and 4 in clean.selection
    assert "C4" not in qc["fitted_channels"] and clean.info["bads"] == ["C4"]
    np.testing.assert_array_equal(epochs.get_data(), before)
    assert epochs.drop_log[3] == ()


def test_autoreject_failures_do_not_fallback(monkeypatch):
    from meganorm.src.erp import reject_erp_epochs
    from autoreject import AutoReject

    def broken(self, epochs):
        raise RuntimeError("backend failure")

    monkeypatch.setattr(AutoReject, "fit", broken)
    with pytest.raises(RuntimeError, match="backend failure"):
        reject_erp_epochs(
            ar_epochs(),
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                bad_segment_removal_method="autoreject",
            ),
        )


def test_autoreject_rejects_infeasible_and_nonfinite_fit(monkeypatch):
    from meganorm.src.erp import reject_erp_epochs
    from autoreject import AutoReject

    with pytest.raises(ValueError, match="combination"):
        reject_erp_epochs(
            ar_epochs(),
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                bad_segment_removal_method="autoreject",
                autoreject_n_interpolates=[4],
                autoreject_consensus_percs=[0.2],
            ),
        )

    def no_solution(self, epochs):
        self.loss_ = {"eeg": np.full((1, 1, 3), np.inf)}
        return self

    monkeypatch.setattr(AutoReject, "fit", no_solution)
    with pytest.raises(ValueError, match="finite cross-validation"):
        reject_erp_epochs(
            ar_epochs(),
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                bad_segment_removal_method="autoreject",
            ),
        )


def test_autoreject_local_api_persists_rejection_settings_and_qc(tmp_path):
    from meganorm.API import Dataset, Pipeline

    raw, events = ar_pipeline_raw()
    raw.set_annotations(
        mne.annotations_from_events(
            events[1:],
            100,
            event_desc={1: "standard", 2: "target"},
            first_samp=raw.first_samp,
        )
    )
    folder = tmp_path / "input" / "001"
    folder.mkdir(parents=True)
    raw.save(folder / "001_oddball_raw.fif", verbose=False)
    definition = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(
            min_trials=1,
            components={
                "response": dict(
                    conditions=["standard", "target"],
                    channels=["Fz"],
                    window=[0.1, 0.4],
                    polarity="pos",
                    metrics=["mean_amplitude"],
                )
            },
        ),
        bad_segment_removal_method="autoreject",
        autoreject_cv=3,
        autoreject_n_interpolates=[1],
        autoreject_consensus_percs=[0.5, 1.0],
        ica_mode="none",
        rereference_method=None,
        cutoffFreqLow=None,
        cutoffFreqHigh=None,
    )
    dataset = Dataset(
        name="demo",
        root=tmp_path / "input",
        task="oddball",
        extension=".fif",
        erp_events=ERPEventSpec(
            mode="annotations",
            annotation_map={"standard": "standard", "target": "target"},
        ),
    )
    result = Pipeline(
        config=definition,
        output_dir=tmp_path / "output",
        on_error="raise",
        progress=False,
    ).run(dataset)
    report = json.loads(
        (result.output_dir / "participants" / "001" / "preprocessing.json").read_text()
    )
    assert report["epoch_rejection"]["mode"] == "autoreject"
    assert report["epoch_rejection"]["cv"] == 3
    assert report["epoch_rejection"]["n_input_epochs"] == 8
    assert len(result.feature_names) == 2
    assert result.summary["collected"] == 1
    qc = pd.read_csv(result.paths["qc"])
    assert "n_autoreject_rejected" in qc and "n_interpolated_trials" in qc


def test_independent_ica_fit_preserves_analysis_samples_and_input():
    raw = ica_raw()
    before = raw.get_data().copy()
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(),
        ica_h_freq=80,
        ica_resample_sfreq=180,
        cutoffFreqHigh=30,
        resampling_rate=100,
        auto_ica_corr_thr=1,
    )
    cleaned, qc = preprocess_erp(raw, config)
    expected, _ = preprocess_erp(
        raw,
        Config(
            analysis_type="erp",
            which_sensor="eeg",
            erp=dict(),
            apply_ica=False,
            cutoffFreqHigh=30,
        ),
    )
    assert cleaned.info["sfreq"] == raw.info["sfreq"]
    np.testing.assert_allclose(cleaned.get_data(), expected.get_data(), atol=1e-18)
    np.testing.assert_array_equal(raw.get_data(), before)
    assert qc["ica_fit_filter"] == {"l_freq": 1, "h_freq": 80}
    assert qc["ica_fit_sfreq"] == 180
    assert qc["ica_n_components"] == 3


def test_average_reference_caps_overestimated_numerical_rank():
    with patch("mne.compute_rank", return_value={"eeg": 4}):
        _, qc = preprocess_erp(
            ica_raw(), Config(analysis_type="erp", which_sensor="eeg", erp=dict())
        )
        assert qc["ica_rank"] == 3
        assert qc["ica_rank_estimated"] == 4
        with pytest.raises(ValueError, match="rank"):
            preprocess_erp(
                ica_raw(),
                Config(
                    analysis_type="erp",
                    which_sensor="eeg",
                    erp=dict(),
                    ica_n_component=4,
                ),
            )


def test_average_reference_rank_bound_excludes_bad_channels():
    raw = ica_raw()
    raw.info["bads"] = ["Oz"]
    with patch("mne.compute_rank", return_value={"eeg": 3}):
        _, qc = preprocess_erp(
            raw, Config(analysis_type="erp", which_sensor="eeg", erp=dict())
        )
    assert qc["ica_rank"] == 2
    assert qc["ica_n_components"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(ica_h_freq=-1),
        dict(ica_resample_sfreq=float("nan")),
        dict(ica_l_freq=80, ica_h_freq=70),
        dict(ica_h_freq=80, ica_resample_sfreq=160),
    ],
)
def test_independent_ica_fit_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Config(**{**dict(analysis_type="erp", which_sensor="eeg", **kwargs)})


def test_independent_ica_fit_settings_roundtrip_and_nyquist():
    settings = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(),
        ica_h_freq=80,
        ica_resample_sfreq=180,
        cutoffFreqHigh=30,
        resampling_rate=100,
    )
    config = Config.model_validate_json(settings.model_dump_json())
    assert config.ica_h_freq == 80
    settings.validate_sampling_rate(200)
    with pytest.raises(ValueError, match="Nyquist"):
        settings.validate_sampling_rate(160)
    with pytest.raises(ValueError, match="Nyquist"):
        preprocess_erp(
            ica_raw(),
            Config(analysis_type="erp", which_sensor="eeg", erp=dict(), ica_h_freq=100),
        )


def test_ica_fit_resampling_does_not_move_analysis_events():
    raw = ica_raw()
    raw.set_annotations(mne.Annotations([1.125, 3.375, 6.125], [0, 0, 0], ["A"] * 3))
    components = {"P300": ERPComponent(("target",), ("Pz",), (0.25, 0.6), "pos")}
    common = dict(
        erp=dict(min_trials=1),
        cutoffFreqLow=1,
        cutoffFreqHigh=30,
        resampling_rate=100,
        eeg_var_threshold=None,
    )
    result = run_erp_pipeline(
        raw,
        "sub-01",
        components,
        annotation_map={"target": "A"},
        config=Config(
            **{
                "analysis_type": "erp",
                "which_sensor": "eeg",
                "erp": dict(),
                **common,
                "ica_h_freq": 80,
                "ica_resample_sfreq": 180,
                "auto_ica_corr_thr": 1,
            }
        ),
    )
    expected = run_erp_pipeline(
        raw,
        "sub-01",
        components,
        annotation_map={"target": "A"},
        config=Config(
            **{
                "analysis_type": "erp",
                "which_sensor": "eeg",
                "erp": dict(),
                **common,
                "apply_ica": False,
            }
        ),
    )
    np.testing.assert_array_equal(result.epochs.events, expected.epochs.events)
    np.testing.assert_allclose(
        result.epochs.get_data(), expected.epochs.get_data(), atol=1e-18
    )


def test_ica_fit_defaults_inherit_existing_filter_and_sample_rate():
    raw = ica_raw()
    _, qc = preprocess_erp(
        raw,
        Config(
            analysis_type="erp",
            which_sensor="eeg",
            erp=dict(),
            cutoffFreqLow=1,
            cutoffFreqHigh=30,
        ),
    )
    assert qc["ica_fit_filter"] == dict(l_freq=1, h_freq=30)
    assert qc["ica_fit_sfreq"] == raw.info["sfreq"]


def test_ica_fit_resampling_cannot_silently_disable_auxiliary_rejection():
    raw = ica_raw()
    raw._data[-1, :40] = np.nan
    raw.set_annotations(mne.Annotations([0], [0.2], ["BAD_aux"]))
    with pytest.raises(ValueError, match="nonfinite.*resampling"):
        preprocess_erp(
            raw,
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                cutoffFreqLow=1,
                ica_mode="physiological",
                ica_resample_sfreq=100,
            ),
        )


def test_iclabel_uses_independent_fit_copy():
    raw = ica_raw(montage=True).resample(500, verbose=False)

    def classify(fit_data, ica, method):
        assert method == "iclabel"
        assert fit_data.info["sfreq"] == 250
        assert fit_data.info["highpass"] == 1
        assert fit_data.info["lowpass"] == 100
        return dict(labels=["brain"] * 3, y_pred_proba=np.ones(3))

    with patch("mne_icalabel.label_components", side_effect=classify):
        cleaned, qc = preprocess_erp(
            raw,
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                erp=dict(),
                ica_mode="iclabel",
                cutoffFreqLow=1,
                cutoffFreqHigh=30,
                ica_h_freq=100,
                ica_resample_sfreq=250,
            ),
        )
    assert cleaned.info["sfreq"] == 500
    assert cleaned.info["lowpass"] == 30
    assert qc["ica_warnings"] == []


def test_template_montage_fallback_and_exclusions():
    from meganorm.utils.erp import prepare_erp_montage

    names = ["Pz", "Cz", "TP9 LEFT EAR", "TP10 RIGHT EAR", "O9"]
    raw = mne.io.RawArray(
        np.zeros((5, 100)), mne.create_info(names, 250, "eeg"), verbose=False
    )
    messages = []
    qc = prepare_erp_montage(
        raw,
        {},
        dict(template="auto", exclude_channels=["O9"]),
        required_channels=["Pz"],
        notify=messages.append,
    )
    assert qc["source"] == "template" and qc["excluded_channels"] == ["O9"]
    assert raw.info["bads"] == ["O9"] and raw.ch_names == names
    assert np.isfinite(raw.info["chs"][2]["loc"][:3]).all()
    assert messages and "template" in messages[0]


def test_template_requires_explicit_exclusion_and_protects_idps():
    from meganorm.utils.erp import prepare_erp_montage

    raw = mne.io.RawArray(
        np.zeros((2, 100)),
        mne.create_info(["Pz", "Unknown"], 250, "eeg"),
        verbose=False,
    )
    with pytest.raises(ValueError, match="Unknown"):
        prepare_erp_montage(raw.copy(), {}, dict(template="auto"))
    with pytest.raises(ValueError, match="requested"):
        prepare_erp_montage(
            raw.copy(),
            {},
            dict(template="auto", exclude_channels=["Pz", "Unknown"]),
            required_channels=["Pz"],
        )


def test_template_preserves_embedded_and_rejects_partial_mixing():
    from meganorm.utils.erp import prepare_erp_montage

    raw = ica_raw(montage=True)
    before = raw.info["chs"][0]["loc"].copy()
    qc = prepare_erp_montage(raw, {}, dict(template="auto"))
    assert qc["source"] == "embedded"
    np.testing.assert_array_equal(before, raw.info["chs"][0]["loc"])
    raw.info["chs"][0]["loc"][:3] = np.nan
    with pytest.raises(ValueError, match="incomplete"):
        prepare_erp_montage(raw, {}, dict(template="auto"))


def test_embedded_montage_requires_head_frame():
    from meganorm.utils.erp import prepare_erp_montage
    from mne._fiff.constants import FIFF

    raw = ica_raw(montage=True)
    raw.info["chs"][0]["coord_frame"] = FIFF.FIFFV_COORD_MRI
    with pytest.raises(ValueError, match="head"):
        prepare_erp_montage(raw, {}, dict(template="auto"))


def test_montage_sidecar_prefers_matching_task(tmp_path):
    from meganorm.utils.erp import erp_metadata_paths

    exact = tmp_path / "sub-01_task-oddball_montage.csv"
    exact.touch()
    (tmp_path / "sub-01_task-rest_montage.csv").touch()
    assert (
        erp_metadata_paths(tmp_path / "sub-01_task-oddball_eeg.vhdr")["montage"]
        == exact
    )


def test_dataset_montage_settings_roundtrip(tmp_path):
    from meganorm.API import Dataset

    ds = Dataset(
        name="x",
        root=tmp_path,
        task="oddball",
        extension=".vhdr",
        erp_montage=dict(template="auto", exclude_channels=["O9", "O10"]),
    )
    assert Dataset.from_dict("x", ds._settings()).erp_montage == ds.erp_montage


def test_bids_head_montage_units(tmp_path):
    from meganorm.utils.erp import prepare_erp_montage, erp_metadata_paths

    (tmp_path / "sub-01_electrodes.tsv").write_text("name\tx\ty\tz\nPz\t0\t-50\t80\n")
    (tmp_path / "sub-01_coordsystem.json").write_text(
        json.dumps(dict(EEGCoordinateSystem="CapTrak", EEGCoordinateUnits="mm"))
    )
    raw = mne.io.RawArray(
        np.zeros((1, 100)), mne.create_info(["Pz"], 250, "eeg"), verbose=False
    )
    qc = prepare_erp_montage(
        raw, erp_metadata_paths(tmp_path / "sub-01_task-oddball_eeg.vhdr"), None
    )
    assert qc["source"] == "bids"
    np.testing.assert_allclose(raw.info["chs"][0]["loc"][:3], [0, -0.05, 0.08])


@pytest.mark.parametrize("frame,units", [("Other", "m"), ("CapTrak", "unknown")])
def test_bids_montage_rejects_unknown_frame_or_units(tmp_path, frame, units):
    from meganorm.utils.erp import prepare_erp_montage

    electrodes = tmp_path / "electrodes.tsv"
    coords = tmp_path / "coordsystem.json"
    electrodes.write_text("name\tx\ty\tz\nPz\t0\t-.05\t.08\n")
    coords.write_text(
        json.dumps(dict(EEGCoordinateSystem=frame, EEGCoordinateUnits=units))
    )
    with pytest.raises(ValueError, match="coordinate"):
        prepare_erp_montage(
            ica_raw(),
            dict(electrodes=electrodes, coordsystem=coords),
            dict(template="auto"),
        )


def test_montage_csv_origin_not_assumed_measured(tmp_path):
    from meganorm.utils.erp import prepare_erp_montage

    raw = ica_raw(montage=True)
    positions = raw.get_montage().get_positions()["ch_pos"]
    csv = tmp_path / "montage.csv"
    pd.DataFrame(
        [[n, *p] for n, p in positions.items()], columns=["Channel", "X", "Y", "Z"]
    ).to_csv(csv, index=False)
    raw.set_montage(None)
    qc = prepare_erp_montage(raw, dict(montage=csv), dict(template="auto"))
    assert qc["source"] == "sidecar" and qc["kind"] == "unknown"
    with pytest.warns(UserWarning, match="template"):
        qc = prepare_erp_montage(raw, {}, dict(supplied_kind="template"))
    assert qc["kind"] == "template"


def test_montage_local_api_records_fallback_without_modifying_inputs(tmp_path):
    from meganorm.API import Dataset, Pipeline, ERPMontageSettings

    raw, events = make_demo_recording()
    raw.set_montage(None)
    raw.set_annotations(
        mne.annotations_from_events(
            events,
            raw.info["sfreq"],
            event_desc={1: "standard", 2: "target"},
            first_samp=raw.first_samp,
        )
    )
    folder = tmp_path / "input" / "001"
    folder.mkdir(parents=True)
    path = folder / "001_oddball_raw.fif"
    raw.save(path, verbose=False)
    before = path.read_bytes()
    ds = Dataset(
        name="demo",
        root=tmp_path / "input",
        task="oddball",
        extension=".fif",
        erp_montage=ERPMontageSettings(template="auto"),
        erp_events=ERPEventSpec(
            mode="annotations",
            annotation_map={"standard": "standard", "target": "target"},
        ),
    )
    definition = Config(
        analysis_type="erp",
        which_sensor="eeg",
        erp=dict(
            components={
                "P": dict(
                    conditions=["target"],
                    channels=["Pz"],
                    window=[0.25, 0.5],
                    polarity="pos",
                )
            }
        ),
        apply_ica=False,
        cutoffFreqLow=None,
        cutoffFreqHigh=None,
    )
    with pytest.warns(UserWarning, match="template"):
        result = Pipeline(
            config=definition,
            output_dir=tmp_path / "out",
            on_error="raise",
            progress=False,
        ).run(ds)
    directory = result.output_dir / "participants" / "001"
    qc = json.loads((directory / "preprocessing.json").read_text())
    assert qc["montage"]["source"] == "template"
    assert (
        json.loads((directory / "provenance.json").read_text())["montage"]
        == qc["montage"]
    )
    assert "template" in (directory / "processing.log").read_text()
    assert (
        json.loads((result.output_dir / "datasets.json").read_text())["demo"][
            "erp_montage"
        ]["template"]
        == "auto"
    )
    assert path.read_bytes() == before and list(folder.iterdir()) == [path]


@pytest.mark.parametrize("ambiguous", [False, True])
def test_embedded_positions_take_priority_over_unused_bids_errors(tmp_path, ambiguous):
    from meganorm.utils.erp import erp_metadata_paths, prepare_erp_montage

    path = tmp_path / "sub-01_task-oddball_eeg.vhdr"
    if ambiguous:
        (tmp_path / "sub-01_space-A_electrodes.tsv").touch()
        (tmp_path / "sub-01_space-B_electrodes.tsv").touch()
    else:
        (tmp_path / "sub-01_electrodes.tsv").touch()
    metadata = erp_metadata_paths(path)
    assert prepare_erp_montage(ica_raw(montage=True), metadata)["source"] == "embedded"
    with pytest.raises(ValueError, match="coordinate|Ambiguous"):
        prepare_erp_montage(ica_raw(), metadata, dict(template="auto"))


def test_unified_config_defaults_and_serialization():
    config = Config(analysis_type="erp", which_sensor="eeg", erp={})
    assert config.cutoffFreqLow == 0.1
    assert config.cutoffFreqHigh == 35.0
    assert config.resampling_rate is None
    assert config.ica_method == "infomax"
    assert config.auto_ica_corr_thr == 0.8
    assert config.bad_segment_removal_method == "fixed_thr"
    assert config.eeg_var_threshold == 0.00015
    assert config.eeg_flat_threshold is None
    assert set(type(config.erp).model_fields) == {
        "tmin",
        "tmax",
        "baseline",
        "min_trials",
        "components",
        "save_evokeds",
        "save_report",
        "save_figures",
    }
    assert Config.model_validate_json(config.model_dump_json()) == config


def test_rest_config_defaults_regression():
    config = Config()
    assert (config.cutoffFreqLow, config.cutoffFreqHigh, config.resampling_rate) == (
        1,
        40,
        300,
    )
    assert config.ica_method == "fastica"
    assert config.auto_ica_corr_thr == 0.5
    assert config.bad_segment_removal_method == "autoreject"
    assert (config.eeg_var_threshold, config.eeg_flat_threshold) == (4e-05, 1e-06)
    assert config.autoreject_cv == "auto"
    assert config.autoreject_n_interpolates == [1, 4, 8, 16, 32]
    assert config.autoreject_consensus_percs == list(np.linspace(0, 1, 11))


@pytest.mark.parametrize(
    "name", ["cutoffFreqHigh", "resampling_rate", "ica_n_component"]
)
@pytest.mark.parametrize("value", [100, 100.0, "100", True])
def test_rest_shared_integer_fields_keep_original_coercion(name, value):
    overrides = {name: value, "digital_filter": False}
    if name == "resampling_rate" and value is True:
        # Integer coercion remains valid, but the unchanged REST spectral
        # constraints prohibit a 1 Hz sample rate.
        with pytest.raises(ValueError, match="Nyquist"):
            Config(**overrides)
        return
    config = Config(**overrides)
    assert type(getattr(config, name)) is int
    assert type(config.model_dump()[name]) is int
    assert (
        type(Config.model_validate_json(config.model_dump_json()).model_dump()[name])
        is int
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resampling_rate": 300.5},
        {"cutoffFreqHigh": 40.5},
        {"ica_n_component": 0.95},
        {"resampling_rate": None},
        {"cutoffFreqHigh": None},
        {"rereference_method": ("Cz",)},
        {"eeg_var_threshold": None},
        {"eeg_flat_threshold": None},
    ],
)
def test_rest_shared_fields_keep_original_rejections(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ica_threshold": 0.8},
        {"ica_n_components": 3},
        {"erp": {"preprocessing": {"l_freq": 1}}},
        {"erp": {"rejection_mode": "threshold"}},
        {"notch_filter": True},
        {"drop_noisy_flat_channel": True},
    ],
)
def test_unified_config_rejects_removed_or_unsupported_settings(kwargs):
    with pytest.raises(ValueError):
        Config(analysis_type="erp", which_sensor="eeg", **kwargs)


def test_components_default_to_config_for_raw_and_file(tmp_path):
    raw, events = make_demo_recording()
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        apply_ica=False,
        cutoffFreqLow=None,
        cutoffFreqHigh=None,
        erp=settings().erp,
    )
    explicit = {"P300": ERPComponent(**config.erp.components["P300"].model_dump())}
    expected = run_erp_pipeline(
        raw,
        "s",
        explicit,
        events=events,
        event_id={"standard": 1, "target": 2},
        config=config,
    )
    actual = run_erp_pipeline(
        raw, "s", events=events, event_id={"standard": 1, "target": 2}, config=config
    )
    pd.testing.assert_frame_equal(actual.features, expected.features)
    path = tmp_path / "raw.fif"
    raw.save(path, verbose=False)
    loaded = extract_recording(
        path, "s", events=events, event_id={"standard": 1, "target": 2}, config=config
    )
    np.testing.assert_allclose(loaded.features, expected.features, rtol=1e-6)
    pd.testing.assert_frame_equal(actual.qc, expected.qc)


def test_standalone_revalidates_mutated_canonical_config():
    config = Config(analysis_type="erp", which_sensor="eeg", apply_ica=False)
    config.erp.baseline = (-1, 0)
    with pytest.raises(ValueError, match="Baseline"):
        preprocess_erp(ica_raw(), config)
    with pytest.raises(ValueError, match="analysis_type"):
        preprocess_erp(ica_raw(), Config())


def test_erp_rejects_unsupported_ica_elbow_detection():
    with pytest.raises(ValueError, match="apply_ica_elbow_detection"):
        Config(analysis_type="erp", which_sensor="eeg", apply_ica_elbow_detection=True)
    assert Config(apply_ica_elbow_detection=True).apply_ica_elbow_detection


@pytest.mark.parametrize("fit_sfreq", [None, 100])
def test_physiological_ica_annotation_flag_uses_same_samples_for_fit_and_scoring(
    fit_sfreq,
):
    raw = ica_raw()
    raw._data[-1] = 0
    raw._data[-1, 200:600] = np.sin(np.arange(400) / 10) * 1e-4
    raw.set_annotations(mne.Annotations([1], [2], ["BAD_blink"]))
    with pytest.raises(ValueError, match="usable EOG/ECG"):
        preprocess_erp(
            raw,
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                ica_mode="physiological",
                ica_resample_sfreq=fit_sfreq,
            ),
        )
    with (
        patch("mne.preprocessing.ICA") as factory,
        patch(
            "meganorm.src.preprocess.find_ica_component", return_value=([], [])
        ) as correlation,
    ):
        factory.return_value.n_components_ = 3
        _, qc = preprocess_erp(
            raw,
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                ica_mode="physiological",
                ica_resample_sfreq=fit_sfreq,
                ica_if_reject_by_annotation=False,
            ),
        )
        assert (
            factory.return_value.fit.call_args.kwargs["reject_by_annotation"] is False
        )
        samples = raw.n_times if fit_sfreq is None else raw.n_times // 2
        assert correlation.call_args.args[1].n_times == samples
        assert len(correlation.call_args.args[2]) == samples
        assert np.ptp(correlation.call_args.args[2]) > 0
        assert qc["ica_auxiliary"]["EOG"] == "usable"


def test_disabled_rejection_preserves_amplitudes_and_reports_none():
    from meganorm.src.erp import reject_erp_epochs

    epochs = ar_epochs()
    epochs._data[2, 0, 50] = 200e-6
    config = Config(
        analysis_type="erp",
        which_sensor="eeg",
        bad_segment_removal_method=None,
        eeg_var_threshold=1e-6,
        eeg_flat_threshold=1e-12,
    )
    cleaned, qc = reject_erp_epochs(epochs, config)
    np.testing.assert_array_equal(cleaned.get_data(), epochs.get_data())
    assert qc["mode"] == "none"
    assert qc["n_retained_epochs"] == len(epochs)


def test_iclabel_alternative_ica_method_records_training_warning():
    with (
        patch("mne.preprocessing.ICA") as factory,
        patch(
            "mne_icalabel.label_components",
            return_value={"labels": [], "y_pred_proba": []},
        ),
    ):
        factory.return_value.n_components_ = 3
        _, qc = preprocess_erp(
            ica_raw(aux=None, montage=True),
            Config(
                analysis_type="erp",
                which_sensor="eeg",
                ica_mode="iclabel",
                ica_method="fastica",
            ),
        )
        assert factory.call_args.kwargs["method"] == "fastica"
        assert any("extended Infomax" in warning for warning in qc["ica_warnings"])
