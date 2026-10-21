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
    ERPConfig,
    compute_evokeds,
    epoch_erp,
    extract_erp_idps,
    preprocess_erp,
    run_erp_pipeline,
    extract_recording,
)

from meganorm.utils.erp import make_demo_recording


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
            ERPConfig(baseline=(-0.3, 0))
        with self.assertRaises(ValueError):
            ERPConfig(l_freq=35, h_freq=10)
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
        data = np.full((2, 6000), 2e-6)
        for sample, amplitude in [(1000, 5e-6), (2500, 7e-6), (4000, 5e-6)]:
            data[:, sample + 300] += amplitude
            data[:, sample + 100] -= 3e-6
        raw = mne.io.RawArray(data, self.info, first_samp=500, verbose=False)
        events = np.array([[1500, 0, 1], [3000, 0, 2], [4500, 0, 1]])
        config = ERPConfig(
            apply_ica=False, reference=None, l_freq=None, h_freq=None, min_trials=1
        )
        return raw, events, config

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
            config=replace(config, resample_sfreq=500),
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
            replace(config, reject={"eeg": 1e-6}),
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
            np.exp(-(((times - center) / 0.12) ** 2)) for center in range(2, 29, 3)
        )
        sources = np.array(
            [blink, np.sin(2 * np.pi * 10 * times), rng.laplace(size=len(times))]
        )
        mixture = (
            np.array([[4, 1, 0.3], [2, -1, 0.7], [1, 0.5, -0.5]]) @ sources * 10e-6
        )
        raw = mne.io.RawArray(
            np.vstack([mixture, blink * 100e-6]),
            mne.create_info(["Fz", "Cz", "Pz", "EOG"], 200, ["eeg"] * 3 + ["eog"]),
            verbose=False,
        )
        cleaned, qc = preprocess_erp(
            raw,
            ERPConfig(
                reference=None,
                l_freq=0.5,
                h_freq=35,
                ica_n_components=3,
                ica_threshold=0.5,
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
            cleaned, qc = preprocess_erp(
                raw, replace(config, apply_ica=True, h_freq=35)
            )
            fitted = factory.return_value.fit.call_args.args[0]
            self.assertIsNot(fitted, cleaned)
            self.assertEqual(qc["ica_excluded"], [0])
            self.assertTrue(qc["ica_applied"])
            correlation.assert_called_once()
            factory.return_value.apply.assert_called_once()


if __name__ == "__main__":
    unittest.main()
