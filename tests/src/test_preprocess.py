from copy import deepcopy
from types import SimpleNamespace

import mne
import numpy as np
import pytest

from meganorm.src.preprocess import (
    _chpi_usable,
    _complete,
    _find_tsss,
    _is_tsss,
    _validate_gedai_params,
    annotate_nonfinite,
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
