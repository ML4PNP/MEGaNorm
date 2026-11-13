"""Numerical reference contracts on synthetic recordings."""

import mne
import numpy as np
import pytest

from meganorm.src.erp import preprocess_erp
from meganorm.utils.IO import Config
from meganorm.utils.eeg import make_standard_1020_montage


def recording():
    return mne.io.RawArray(
        np.random.default_rng(43).normal(size=(6, 2000)) * 1e-6,
        mne.create_info(
            ["Fz", "Cz", "Pz", "Oz", "LEFT", "RIGHT"], 250, ["eeg"] * 4 + ["misc"] * 2
        ),
        verbose=False,
    )


def config(**kwargs):
    return Config(
        analysis_type="erp",
        which_sensor="eeg",
        apply_ica=False,
        digital_filter=False,
        **kwargs,
    )


def test_explicit_reference_subtracts_mean_and_excludes_auxiliaries():
    raw = recording()
    original = raw.get_data()
    result, qc = preprocess_erp(raw, config(rereference_method=["LEFT", "RIGHT"]))
    np.testing.assert_allclose(result.get_data(), original[:4] - original[4:].mean(0))
    assert result.ch_names == raw.ch_names[:4]
    assert qc["reference"]["resolved_channels"] == ["LEFT", "RIGHT"]
    np.testing.assert_array_equal(raw.get_data(), original)


def test_default_average_reference_unchanged():
    raw = recording()
    result, _ = preprocess_erp(raw, config())
    original = raw.get_data()
    np.testing.assert_allclose(
        result.get_data()[:4], original[:4] - original[:4].mean(0)
    )
    np.testing.assert_array_equal(result.get_data()[4:], original[4:])


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "bad", "flat", "nonfinite", "eog"]
)
def test_reference_validation(failure):
    raw = recording()
    names = ["LEFT", "RIGHT"]
    if failure == "missing":
        names[1] = "MISSING"
    elif failure == "duplicate":
        names[1] = "LEFT"
    elif failure == "bad":
        raw.info["bads"] = ["LEFT"]
    elif failure == "flat":
        raw._data[4] = 0
    elif failure == "nonfinite":
        raw._data[4, 10] = np.nan
    elif failure == "eog":
        raw.set_channel_types({"LEFT": "eog"}, on_unit_change="ignore")
    with pytest.raises(ValueError, match="(?i)reference|unique|duplicate"):
        preprocess_erp(raw, config(rereference_method=names))


@pytest.mark.parametrize("active", [False, True])
def test_reference_projection_policy(active):
    raw = recording().set_eeg_reference("average", projection=True)
    if active:
        raw.apply_proj()
        with pytest.raises(ValueError, match="unprojected"):
            preprocess_erp(raw, config(rereference_method=["LEFT", "RIGHT"]))
    else:
        expected = raw.get_data()[:4] - raw.get_data()[4:].mean(0)
        result, _ = preprocess_erp(raw, config(rereference_method=["LEFT", "RIGHT"]))
        np.testing.assert_allclose(result.get_data(), expected)
        assert not result.info["projs"]


def test_final_reference_survives_ica_epochs_and_save(tmp_path, monkeypatch):
    from meganorm.src.erp import run_erp_pipeline, ERPComponent, epoch_erp
    from meganorm.utils.erp_outputs import load_erp_result
    from meganorm.plots.erp_report import write_erp_report

    raw = recording()
    raw.set_montage(make_standard_1020_montage())

    # Use real ICA fit/application with a deterministic external classifier decision.
    def classify(fit_data, ica, method):
        np.testing.assert_allclose(
            fit_data.get_data(picks="eeg").mean(0), 0, atol=1e-18
        )
        return {"labels": ["eye blink", "brain", "brain"], "y_pred_proba": np.ones(3)}

    monkeypatch.setattr("mne_icalabel.label_components", classify)
    settings = dict(
        analysis_type="erp",
        which_sensor="eeg",
        ica_mode="iclabel",
        cutoffFreqLow=1.0,
        cutoffFreqHigh=30.0,
        ica_h_freq=100.0,
        ica_n_component=3,
        bad_segment_removal_method=None,
        erp=dict(tmin=-0.1, tmax=0.2, baseline=None, min_trials=1),
    )
    mastoid = Config(**settings, rereference_method=["LEFT", "RIGHT"])
    average = Config(**settings)
    components = {
        "P": ERPComponent(
            ("target",), ("Cz",), (0.05, 0.15), "pos", ("mean_amplitude",)
        )
    }
    events = np.array([[300, 0, 1], [600, 0, 1], [900, 0, 1], [1200, 0, 1]])
    explicit_result = run_erp_pipeline(
        raw, "001", components, events=events, event_id={"target": 1}, config=mastoid
    )
    average_result = run_erp_pipeline(
        raw, "001", components, events=events, event_id={"target": 1}, config=average
    )
    # Independently filter and epoch the expected common-mode offset.
    offset = raw.get_data()[:4].mean(0) - raw.get_data()[4:].mean(0)
    expected = mne.io.RawArray(
        np.tile(offset, (4, 1)),
        mne.create_info(raw.ch_names[:4], 250, "eeg"),
        verbose=False,
    )
    expected.filter(1.0, 30.0, verbose=False)
    expected_epochs = epoch_erp(expected, events, {"target": 1}, mastoid)
    np.testing.assert_allclose(
        explicit_result.epochs.get_data() - average_result.epochs.get_data(picks="eeg"),
        expected_epochs.get_data(),
        atol=1e-12,
    )
    assert explicit_result.preprocessing_qc["ica_excluded"] == [0]
    assert explicit_result.preprocessing_qc["ica_rank"] == 3
    explicit_result.save(tmp_path / "result", report=False)
    loaded = load_erp_result(tmp_path / "result")
    assert (
        loaded.preprocessing_qc["reference"]
        == explicit_result.preprocessing_qc["reference"]
    )
    np.testing.assert_allclose(
        loaded.evokeds["target"].data,
        explicit_result.evokeds["target"].data,
        atol=1e-12,
    )
    write_erp_report(loaded, tmp_path / "report.html")
    assert "channel_mean" in (tmp_path / "report.html").read_text()


def test_reference_eeg_channels_need_no_montage_positions():
    from meganorm.src.erp import _prepare_erp_raw

    raw = recording()
    raw.set_channel_types({"LEFT": "eeg", "RIGHT": "eeg"}, on_unit_change="ignore")
    settings = config(rereference_method=["LEFT", "RIGHT"])
    _prepare_erp_raw(
        raw,
        {"channels": None, "electrodes": None, "coordsystem": None, "montage": None},
        {},
        settings,
        notify=lambda message: None,
        montage_settings={
            "template": (
                "colin27_1020"
                if "colin27_1020" in mne.channels.get_builtin_montages()
                else "standard_1020"
            )
        },
    )
    assert raw.get_channel_types()[-2:] == ["misc", "misc"]
    assert not raw.info["bads"]
    result, _ = preprocess_erp(raw, settings)
    assert result.ch_names == ["Fz", "Cz", "Pz", "Oz"]
