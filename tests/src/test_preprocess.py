from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import mne
import numpy as np
import pytest
import meganorm.src.preprocess as preprocess_module

from meganorm.src.preprocess import (
    _annotate_dropped_epochs,
    _chpi_usable,
    _complete,
    _detect_bad_channels_ransac,
    _find_tsss,
    _is_tsss,
    _validate_gedai_params,
    annotate_noisy_raw,
    annotate_nonfinite,
    auto_reject_segmentation,
    extract_rs_blocks,
    fix_physiological_channel_types,
    sss_records,
    tsss_params,
)

pytestmark = pytest.mark.unit


def make_raw(*, sfreq=10.0, duration=2.0, names=None, types=None):
    names = names or ["EEG001", "EEG002"]
    types = types or ["eeg"] * len(names)
    info = mne.create_info(names, sfreq, types)
    return mne.io.RawArray(
        np.ones((len(names), int(sfreq * duration))), info, verbose=False
    )


def make_epochs(names, types, *, n_epochs=3, sfreq=10.0):
    info = mne.create_info(names, sfreq, types)
    data = np.zeros((n_epochs, len(names), 5))
    return mne.EpochsArray(data, info, verbose=False)


def install_fake_autoreject(monkeypatch, bad_epochs):
    captured = {}

    class FakeRejectLog:
        def __init__(self, n_epochs, n_channels):
            self.bad_epochs = np.asarray(bad_epochs, dtype=bool)
            self.labels = np.zeros((n_epochs, n_channels))
            if n_epochs:
                self.labels[0, 0] = 2

        def plot(self, orientation, show):
            assert orientation == "horizontal"
            assert show is False
            return preprocess_module.plt.figure()

    class FakeAutoReject:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, epochs):
            self.consensus_ = {"eeg": 0.5}
            self.n_interpolate_ = {"eeg": 1}
            return self

        def transform(self, epochs, return_log):
            assert return_log is True
            mask = np.asarray(bad_epochs, dtype=bool)
            assert len(mask) == len(epochs)
            log = FakeRejectLog(len(epochs), len(epochs.ch_names))
            return epochs[~mask], log

    monkeypatch.setattr(preprocess_module, "AutoReject", FakeAutoReject)
    return captured


def test_detect_bad_channels_ransac_dispatches_by_type_and_forwards_parameters(
    monkeypatch,
):
    names = (
        [f"MAG{i}" for i in range(3)]
        + [f"GRAD{i}" for i in range(3)]
        + [f"EEG{i}" for i in range(3)]
    )
    types = ["mag"] * 3 + ["grad"] * 3 + ["eeg"] * 3
    epochs = make_epochs(names, types)
    epochs.info["bads"] = ["MAG2"]
    instances = []

    class FakeRansac:
        def __init__(self, picks, **kwargs):
            self.picks = np.asarray(picks)
            self.kwargs = kwargs
            instances.append(self)

        def fit(self, fitted_epochs):
            self.bad_chs_ = [fitted_epochs.ch_names[self.picks[0]]]
            self.bad_log = np.zeros((len(fitted_epochs), len(self.picks)))
            return self

    monkeypatch.setattr(preprocess_module, "Ransac", FakeRansac)

    bads, logs = _detect_bad_channels_ransac(
        epochs,
        n_resample=17,
        min_channels=0.4,
        min_corr=0.8,
        unbroken_time=0.3,
        n_jobs=2,
        random_state=9,
        min_good_channels=2,
    )

    assert bads == ["MAG0", "GRAD0", "EEG0"]
    assert [item.picks.tolist() for item in instances] == [[0, 1], [3, 4, 5], [6, 7, 8]]
    assert instances[0].kwargs == {
        "n_resample": 17,
        "min_channels": 0.4,
        "min_corr": 0.8,
        "unbroken_time": 0.3,
        "n_jobs": 2,
        "random_state": 9,
        "verbose": False,
    }
    assert logs["mag"][1] == ["MAG0", "MAG1"]
    assert logs["grad"][1] == ["GRAD0", "GRAD1", "GRAD2"]
    assert logs["eeg"][1] == ["EEG0", "EEG1", "EEG2"]


def test_detect_bad_channels_ransac_skips_types_with_too_few_good_channels(
    monkeypatch,
):
    epochs = make_epochs(["EEG0", "EEG1", "EEG2"], ["eeg"] * 3)

    class UnexpectedRansac:
        def __init__(self, **kwargs):
            raise AssertionError("RANSAC must not run with too few channels")

    monkeypatch.setattr(preprocess_module, "Ransac", UnexpectedRansac)

    bads, logs = _detect_bad_channels_ransac(epochs, min_good_channels=4)

    assert bads == []
    assert logs == {}


def test_auto_reject_with_precomputed_events_preserves_raw_and_writes_outputs(
    monkeypatch, tmp_path, caplog
):
    caplog.set_level("INFO", logger=preprocess_module.__name__)
    raw = make_raw(duration=2.0)
    events = np.array([[0, 0, 1], [5, 0, 1], [10, 0, 1], [15, 0, 1]])
    captured = install_fake_autoreject(monkeypatch, [False, True, False, False])
    original_samples = raw.n_times

    cleaned, reject_log = auto_reject_segmentation(
        raw,
        sampling_rate=10.0,
        subject="sub-01",
        project_dir=tmp_path,
        tmax=0,
        segments_length=0.5,
        segment_events=events,
        n_interpolates=np.array([1, 2]),
        consensus_percs=np.array([0.5, 0.8]),
        thresh_method="random_search",
        random_state=7,
    )

    assert raw.n_times == original_samples
    assert len(cleaned) == 3
    assert reject_log.bad_epochs.tolist() == [False, True, False, False]
    assert captured["cv"] == 4
    np.testing.assert_array_equal(captured["n_interpolate"], [1, 2])
    np.testing.assert_array_equal(captured["consensus"], [0.5, 0.8])
    assert captured["thresh_method"] == "random_search"
    assert captured["random_state"] == 7
    assert raw.annotations.description.tolist() == ["BAD_autoreject"]
    assert raw.annotations.onset[0] == pytest.approx(0.5)
    assert "Interpolated : 1" in caplog.text
    assert (
        tmp_path
        / "Saved_outputs"
        / "auto_reject_plot"
        / "sub-01_autoreject_res_plot.png"
    ).is_file()


def test_auto_reject_maps_log_to_epochs_surviving_existing_annotations(
    monkeypatch, tmp_path
):
    raw = make_raw(duration=2.0)
    raw.set_annotations(mne.Annotations([0.5], [0.5], ["BAD_existing"]))
    events = np.array([[0, 0, 1], [5, 0, 1], [10, 0, 1], [15, 0, 1]])
    install_fake_autoreject(monkeypatch, [False, True, False])

    auto_reject_segmentation(
        raw,
        10.0,
        "sub-01",
        tmp_path,
        tmax=0,
        segments_length=0.5,
        segment_events=events,
    )

    assert raw.annotations.description.tolist() == ["BAD_existing", "BAD_autoreject"]
    assert raw.annotations.onset[1] == pytest.approx(1.0)


def test_auto_reject_can_leave_rejected_epochs_unannotated(monkeypatch, tmp_path):
    raw = make_raw(duration=1.5)
    events = np.array([[0, 0, 1], [5, 0, 1], [10, 0, 1]])
    install_fake_autoreject(monkeypatch, [False, True, False])

    auto_reject_segmentation(
        raw,
        10.0,
        "sub-01",
        tmp_path,
        tmax=0,
        segments_length=0.5,
        segment_events=events,
        annotate_bad_epochs=False,
    )

    assert len(raw.annotations) == 0


def test_auto_reject_caps_automatic_cv_at_ten(monkeypatch, tmp_path):
    raw = make_raw(duration=6.0)
    events = np.array([[i * 5, 0, 1] for i in range(12)])
    captured = install_fake_autoreject(monkeypatch, [False] * 12)

    auto_reject_segmentation(
        raw,
        10.0,
        "sub-01",
        tmp_path,
        tmax=0,
        segments_length=0.5,
        segment_events=events,
    )

    assert captured["cv"] == 10


def test_auto_reject_preserves_explicit_cv(monkeypatch, tmp_path):
    raw = make_raw(duration=2.0)
    events = np.array([[0, 0, 1], [5, 0, 1], [10, 0, 1]])
    captured = install_fake_autoreject(monkeypatch, [False] * 3)

    auto_reject_segmentation(
        raw,
        10.0,
        "sub-01",
        tmp_path,
        tmax=0,
        segments_length=0.5,
        segment_events=events,
        cv=3,
    )

    assert captured["cv"] == 3


def test_auto_reject_crops_only_when_generating_events(monkeypatch, tmp_path):
    raw = make_raw(duration=5.0)
    captured = install_fake_autoreject(monkeypatch, [False] * 8)

    cleaned, _ = auto_reject_segmentation(
        raw,
        10.0,
        "sub-01",
        tmp_path,
        tmin=0.5,
        tmax=-0.5,
        segments_length=0.5,
        cv=3,
    )

    assert raw.first_samp == 5
    assert raw.n_times == 40
    assert len(cleaned) == 8
    assert captured["cv"] == 3


@pytest.mark.parametrize(
    ("sampling_rate", "segments_length", "overlap", "message"),
    [
        (0.0, 0.5, 0.0, "sampling_rate"),
        (np.nan, 0.5, 0.0, "sampling_rate"),
        (10.0, 0.0, 0.0, "segments_length"),
        (10.0, np.nan, 0.0, "segments_length"),
        (10.0, 0.5, 0.5, "overlap"),
        (10.0, 0.5, np.nan, "overlap"),
    ],
)
def test_auto_reject_validates_segmentation_parameters(
    tmp_path, sampling_rate, segments_length, overlap, message
):
    with pytest.raises(ValueError, match=message):
        auto_reject_segmentation(
            make_raw(),
            sampling_rate,
            "sub-01",
            tmp_path,
            tmin=0,
            tmax=-0.1,
            segments_length=segments_length,
            overlap=overlap,
        )


@pytest.mark.parametrize("n_events", [1, 2])
def test_auto_reject_requires_at_least_three_epochs(tmp_path, n_events):
    raw = make_raw(duration=2.0)
    events = np.array([[i * 5, 0, 1] for i in range(n_events)])

    with pytest.raises(ValueError, match="need at least 3"):
        auto_reject_segmentation(
            raw,
            10.0,
            "sub-01",
            tmp_path,
            tmax=0,
            segments_length=0.5,
            segment_events=events,
        )


def test_auto_reject_reports_empty_event_selection_clearly(tmp_path):
    with pytest.raises(ValueError, match="No epochs"):
        auto_reject_segmentation(
            make_raw(),
            10.0,
            "sub-01",
            tmp_path,
            tmax=0,
            segments_length=0.5,
            segment_events=np.empty((0, 3), dtype=int),
        )


def test_auto_reject_rejects_nonnegative_crop_tmax(tmp_path):
    with pytest.raises(ValueError, match="tmax"):
        auto_reject_segmentation(
            make_raw(duration=3.0),
            10.0,
            "sub-01",
            tmp_path,
            tmin=0,
            tmax=0,
            segments_length=0.5,
        )


def test_auto_reject_raises_when_all_epochs_are_rejected(monkeypatch, tmp_path):
    raw = make_raw(duration=1.5)
    events = np.array([[0, 0, 1], [5, 0, 1], [10, 0, 1]])
    install_fake_autoreject(monkeypatch, [True, True, True])

    with pytest.raises(ValueError, match="All epochs"):
        auto_reject_segmentation(
            raw,
            10.0,
            "sub-01",
            tmp_path,
            tmax=0,
            segments_length=0.5,
            segment_events=events,
        )


def test_annotate_noisy_raw_returns_empty_annotations_without_thresholds():
    annotations = annotate_noisy_raw(make_raw())

    assert len(annotations) == 0


def test_annotate_noisy_raw_detects_peak_and_flat_windows():
    raw = make_raw(duration=1.0)
    raw._data[0] = 0.0
    raw._data[1] = 1.0
    raw._data[0, 2] = 5.0

    annotations = annotate_noisy_raw(
        raw,
        reject={"eeg": 4.0},
        flat={"eeg": 0.1},
        window=0.5,
        step=0.5,
    )

    assert annotations.description.tolist() == ["BAD_peak", "BAD_flat"]
    np.testing.assert_allclose(annotations.onset, [0.0, 0.5])
    np.testing.assert_allclose(annotations.duration, [0.5, 0.5])


def test_annotate_noisy_raw_excludes_channels_marked_bad():
    raw = make_raw(duration=1.0)
    raw._data[:] = 0.0
    raw._data[0, 2] = 5.0
    raw.info["bads"] = ["EEG001"]

    annotations = annotate_noisy_raw(raw, reject={"eeg": 4.0}, window=0.5, step=0.5)

    assert len(annotations) == 0


def test_annotate_noisy_raw_supports_non_data_channel_types():
    raw = make_raw(duration=1.0, names=["EOG001"], types=["eog"])
    raw._data[:] = 0.0
    raw._data[0, 2] = 5.0

    annotations = annotate_noisy_raw(raw, reject={"eog": 4.0}, window=0.5, step=0.5)

    assert annotations.description.tolist() == ["BAD_peak"]
    assert annotations.onset[0] == pytest.approx(0.0)


def test_annotate_noisy_raw_offsets_onsets_for_absolute_measurement_time():
    info = mne.create_info(["EEG001"], 10.0, ["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 20)), info, first_samp=100, verbose=False)
    raw.set_meas_date(datetime(2024, 1, 1, tzinfo=timezone.utc))
    raw._data[0, 2] = 5.0

    annotations = annotate_noisy_raw(raw, reject={"eeg": 4.0}, window=0.5, step=0.5)
    raw.set_annotations(annotations)

    assert raw.annotations.description.tolist() == ["BAD_peak"]
    assert raw.annotations.onset[0] == pytest.approx(raw.first_time)


def test_annotate_noisy_raw_duration_matches_samples_examined():
    raw = make_raw(duration=0.5)
    raw._data[:] = 0.0
    raw._data[0, 0] = 5.0

    annotations = annotate_noisy_raw(raw, reject={"eeg": 4.0}, window=0.25, step=0.2)

    assert annotations.duration[0] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("window", "step", "message"),
    [(0.0, 0.5, "window"), (0.5, 0.0, "step")],
)
def test_annotate_noisy_raw_rejects_nonpositive_window_parameters(
    window, step, message
):
    with pytest.raises(ValueError, match=message):
        annotate_noisy_raw(make_raw(), reject={"eeg": 1.0}, window=window, step=step)


def test_annotate_dropped_epochs_appends_annotations_at_absolute_event_samples():
    info = mne.create_info(["EEG001"], 10.0, ["eeg"])
    raw = mne.io.RawArray(np.ones((1, 20)), info, first_samp=100, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.2],
            duration=[0.1],
            description=["BAD_existing"],
        )
    )

    returned = _annotate_dropped_epochs(raw, [105, 110], 0.4)

    assert returned is raw
    assert raw.annotations.description.tolist() == [
        "BAD_existing",
        "BAD_dropped_epoch",
        "BAD_dropped_epoch",
    ]
    np.testing.assert_allclose(raw.annotations.onset, [10.2, 10.5, 11.0])
    np.testing.assert_allclose(raw.annotations.duration, [0.1, 0.4, 0.4])


def test_annotate_dropped_epochs_is_noop_for_empty_selection():
    raw = make_raw()

    returned = _annotate_dropped_epochs(raw, [], 0.5)

    assert returned is raw
    assert len(raw.annotations) == 0


def test_extract_rs_blocks_uses_next_event_as_exclusive_boundary():
    raw = make_raw(sfreq=10.0, duration=2.0, names=["EEG001"], types=["eeg"])
    raw._data[0] = np.arange(raw.n_times)
    events = np.array([[2, 0, 7], [7, 0, 9]])

    rs_raw, seg_events = extract_rs_blocks(
        raw,
        events,
        rs_id=7,
        sampling_rate=10.0,
        segments_length=0.5,
        overlap=0.0,
        seg_event_id=42,
    )

    assert rs_raw.n_times == 5
    np.testing.assert_array_equal(rs_raw.get_data()[0], np.arange(2, 7))
    np.testing.assert_array_equal(seg_events, [[2, 0, 42]])


def test_extract_rs_blocks_concatenates_retained_blocks_and_builds_overlap_events():
    raw = make_raw(sfreq=10.0, duration=3.0, names=["EEG001"], types=["eeg"])
    events = np.array([[0, 0, 7], [10, 0, 9], [15, 0, 7], [25, 0, 9]])

    rs_raw, seg_events = extract_rs_blocks(
        raw,
        events,
        rs_id=7,
        sampling_rate=10.0,
        segments_length=0.6,
        overlap=0.2,
        seg_event_id=3,
    )

    assert rs_raw.n_times == 20
    np.testing.assert_array_equal(
        seg_events,
        [[0, 0, 3], [4, 0, 3], [10, 0, 3], [14, 0, 3]],
    )


def test_extract_rs_blocks_rejects_missing_or_too_short_blocks():
    raw = make_raw()
    events = np.array([[0, 0, 2], [5, 0, 7], [8, 0, 2]])

    with pytest.raises(ValueError, match="No RS blocks"):
        extract_rs_blocks(raw, events, 7, 10.0, 0.5, 0.0)


def test_extract_rs_blocks_rejects_invalid_overlap_without_looping():
    raw = make_raw()
    events = np.array([[0, 0, 7], [10, 0, 2]])

    with pytest.raises(ValueError, match="overlap"):
        extract_rs_blocks(raw, events, 7, 10.0, 0.5, 0.5)


@pytest.mark.parametrize(
    ("segments_length", "overlap", "message"),
    [(np.nan, 0.0, "segments_length"), (0.5, np.nan, "overlap")],
)
def test_extract_rs_blocks_rejects_nonfinite_timing_parameters(
    segments_length, overlap, message
):
    raw = make_raw()
    events = np.array([[0, 0, 7], [10, 0, 2]])

    with pytest.raises(ValueError, match=message):
        extract_rs_blocks(raw, events, 7, 10.0, segments_length, overlap)


@pytest.mark.parametrize(
    "method, level, duration, multiplier",
    [
        ("broadband", 0, 2.0, None),
        ("spectral", 2, None, None),
        ("spectral", "auto", None, None),
        ("both", 2, 2.0, 1.5),
    ],
)
def test_validate_gedai_params_accepts_supported_combinations(
    method, level, duration, multiplier
):
    assert _validate_gedai_params(method, level, duration, multiplier) is None


@pytest.mark.parametrize(
    "method, level, duration, multiplier, message",
    [
        ("broadband", 1, 2.0, None, "wavelet_level=0"),
        ("broadband", 0, None, None, "gedai_duration"),
        ("spectral", 0, None, None, "wavelet_level > 0"),
        ("both", 2, 2.0, None, "preliminary_broadband"),
    ],
)
def test_validate_gedai_params_rejects_incompatible_combinations(
    method, level, duration, multiplier, message
):
    with pytest.raises(ValueError, match=message):
        _validate_gedai_params(method, level, duration, multiplier)


def test_validate_gedai_params_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        _validate_gedai_params("unknown", 0, 2.0, 1.0)


def test_annotate_nonfinite_returns_clean_raw_unchanged():
    raw = make_raw()
    original = raw.get_data().copy()

    returned, intervals = annotate_nonfinite(raw, verbose=False)

    assert returned is raw
    assert intervals == []
    np.testing.assert_array_equal(raw.get_data(), original)
    assert len(raw.annotations) == 0


def test_annotate_nonfinite_groups_intervals_and_zero_fills_all_nonfinite_values():
    raw = make_raw()
    raw._data[0, 3:5] = np.nan
    raw._data[1, 8] = np.inf

    returned, intervals = annotate_nonfinite(raw, verbose=False)

    assert returned is raw
    assert intervals == [(0.3, 0.4, 2), (0.8, 0.8, 1)]
    assert raw.annotations.description.tolist() == ["BAD_nan", "BAD_nan"]
    assert np.isfinite(raw.get_data()).all()
    np.testing.assert_array_equal(raw.get_data()[0, 3:5], 0.0)
    np.testing.assert_array_equal(raw.get_data()[1, 3:5], 1.0)
    assert raw.get_data()[1, 8] == 0.0
    assert raw.get_data()[0, 8] == 1.0


@pytest.mark.parametrize("sample", [0, 19])
def test_annotate_nonfinite_handles_recording_boundaries(sample):
    raw = make_raw()
    raw._data[0, sample] = np.nan

    _, intervals = annotate_nonfinite(raw, verbose=False)

    expected_time = sample / raw.info["sfreq"]
    assert intervals == [(expected_time, expected_time, 1)]


def test_annotate_nonfinite_padding_changes_annotation_not_repaired_samples():
    raw = make_raw()
    raw._data[0, 5] = np.nan

    _, intervals = annotate_nonfinite(raw, pad=0.2, verbose=False)

    assert intervals == [(0.5, 0.5, 1)]
    assert raw.annotations.onset[0] == pytest.approx(0.3)
    assert raw.annotations.duration[0] == pytest.approx(0.5)
    assert raw.get_data()[0, 4] == 1.0
    assert raw.get_data()[0, 5] == 0.0
    assert raw.get_data()[0, 6] == 1.0


@pytest.mark.parametrize(
    ("sample", "expected_onset"),
    [(0, 0.0), (19, 1.7)],
)
def test_annotate_nonfinite_clips_padding_to_recording_boundaries(
    sample, expected_onset
):
    raw = make_raw()
    raw._data[0, sample] = np.nan

    annotate_nonfinite(raw, pad=0.2, verbose=False)

    assert raw.annotations.onset[0] == pytest.approx(expected_onset)
    assert raw.annotations.duration[0] == pytest.approx(0.3)


def test_annotate_nonfinite_padding_does_not_make_short_corruption_fatal():
    raw = make_raw()
    raw._data[0, 5] = np.nan

    _, intervals = annotate_nonfinite(
        raw,
        pad=0.5,
        remove_nonfinite_segment_threshold=1.0,
        verbose=False,
    )

    assert intervals == [(0.5, 0.5, 1)]


def test_annotate_nonfinite_rejects_long_corrupted_interval():
    raw = make_raw(duration=3.0)
    raw._data[0, 5:16] = np.nan

    with pytest.raises(ValueError, match="file likely compromised"):
        annotate_nonfinite(raw, remove_nonfinite_segment_threshold=1.0, verbose=False)


def test_annotate_nonfinite_respects_detection_channel_selection():
    raw = make_raw()
    raw._data[1, 5] = np.nan

    _, intervals = annotate_nonfinite(raw, picks=["EEG001"], verbose=False)

    assert intervals == []


@pytest.mark.parametrize("picks", [[0], np.array([0, 1])])
def test_annotate_nonfinite_accepts_array_like_integer_picks(picks):
    raw = make_raw()
    raw._data[0, 5] = np.nan

    _, intervals = annotate_nonfinite(raw, picks=picks, verbose=False)

    assert intervals == [(0.5, 0.5, 1)]


def test_annotate_nonfinite_requires_bad_annotation_description():
    with pytest.raises(ValueError, match="BAD"):
        annotate_nonfinite(make_raw(), description="nan", verbose=False)


def make_proc_history():
    return [
        {"creator": "unrelated"},
        {
            "creator": "MaxFilter",
            "date": 20,
            "max_info": {
                "sss_info": {
                    "origin": [0.0, 0.0, 0.04],
                    "frame": 4,
                    "in_order": 8,
                    "out_order": 3,
                    "nfree": 64,
                },
                "max_st": {"buflen": 10.0, "subspcorr": 0.98},
                "sss_cal": {"cal_chans": []},
                "sss_ctc": {"decoupler": []},
            },
        },
    ]


def test_sss_records_extracts_only_sss_history():
    records = sss_records({"proc_history": make_proc_history()})

    assert records == [
        {
            "idx": 1,
            "creator": "MaxFilter",
            "date": 20,
            "origin": [0.0, 0.0, 0.04],
            "frame": 4,
            "int_order": 8,
            "ext_order": 3,
            "nfree": 64,
            "st_duration": 10.0,
            "st_correlation": 0.98,
            "has_cal": True,
            "has_ctc": True,
        }
    ]


def test_sss_record_classifiers_distinguish_complete_tsss():
    record = sss_records({"proc_history": make_proc_history()})[0]

    assert _complete(record)
    assert _is_tsss(record)
    assert not _complete({**record, "origin": None})
    assert not _is_tsss({**record, "st_duration": 0.0})
    assert not _is_tsss({**record, "st_correlation": None})


def test_find_tsss_selects_earliest_complete_temporal_record():
    records = [
        {**sss_records({"proc_history": make_proc_history()})[0], "idx": 0, "date": 20},
        {**sss_records({"proc_history": make_proc_history()})[0], "idx": 1, "date": 10},
    ]

    assert _find_tsss(records)["idx"] == 1


def test_find_tsss_uses_first_record_when_timestamp_is_missing():
    base = sss_records({"proc_history": make_proc_history()})[0]
    records = [{**base, "idx": 2, "date": None}, {**base, "idx": 3, "date": 10}]

    assert _find_tsss(records)["idx"] == 2


def test_find_tsss_returns_none_without_complete_temporal_record():
    base = sss_records({"proc_history": make_proc_history()})[0]
    assert _find_tsss([{**base, "origin": None}]) is None
    assert _find_tsss([{**base, "st_duration": None}]) is None


def test_tsss_params_reconstructs_maxwell_filter_arguments():
    result = tsss_params({"proc_history": make_proc_history()})

    np.testing.assert_array_equal(result.pop("origin"), [0.0, 0.0, 0.04])
    assert result == {
        "int_order": 8,
        "ext_order": 3,
        "coord_frame": "head",
        "st_duration": 10.0,
        "st_correlation": 0.98,
    }


def test_tsss_params_rejects_missing_record_and_unknown_frame():
    with pytest.raises(ValueError, match="No complete tSSS"):
        tsss_params({})

    history = make_proc_history()
    history[1]["max_info"]["sss_info"]["frame"] = 99
    with pytest.raises(ValueError, match="Unknown coord frame"):
        tsss_params({"proc_history": history})


def test_fix_physiological_channel_types_retypes_only_ctf_auxiliary_channels():
    raw = make_raw(
        names=["EEG001", "EOGV", "ECG001", "EKG_aux", "MISC"],
        types=["eeg", "eeg", "eeg", "misc", "misc"],
    )

    returned = fix_physiological_channel_types(raw, device="CTF")

    assert returned is raw
    assert raw.get_channel_types() == ["eeg", "eog", "ecg", "ecg", "misc"]


def test_fix_physiological_channel_types_leaves_non_ctf_data_unchanged():
    raw = make_raw(names=["EOG001", "ECG001"], types=["eeg", "eeg"])

    fix_physiological_channel_types(raw, device="MEGIN")

    assert raw.get_channel_types() == ["eeg", "eeg"]


def test_chpi_usable_detects_ctf_hlc_channels():
    assert _chpi_usable(SimpleNamespace(ch_names=["MEG001", "HLC0011"]), "CTF")
    assert not _chpi_usable(SimpleNamespace(ch_names=["MEG001"]), "CTF")


def test_chpi_usable_megin_handles_available_missing_and_malformed_info(monkeypatch):
    raw = SimpleNamespace(info={})
    monkeypatch.setattr(
        mne.chpi, "get_chpi_info", lambda info, on_missing: ([293.0], None, None)
    )
    assert _chpi_usable(raw, "MEGIN")

    monkeypatch.setattr(
        mne.chpi, "get_chpi_info", lambda info, on_missing: ([], None, None)
    )
    assert not _chpi_usable(raw, "MEGIN")

    def malformed(*args, **kwargs):
        raise ValueError("bad HPI metadata")

    monkeypatch.setattr(mne.chpi, "get_chpi_info", malformed)
    assert not _chpi_usable(raw, "MEGIN")


def test_chpi_usable_bti_delegates_to_kit_extractor(monkeypatch):
    raw = object()
    monkeypatch.setattr(mne.chpi, "extract_chpi_locs_kit", lambda data, verbose: [])
    assert _chpi_usable(raw, "BTI")

    def unavailable(*args, **kwargs):
        raise RuntimeError("no cHPI")

    monkeypatch.setattr(mne.chpi, "extract_chpi_locs_kit", unavailable)
    assert not _chpi_usable(raw, "BTI")
    assert not _chpi_usable(raw, "UNKNOWN")
