"""Metadata behavior contracts for legacy REST and strict ERP policies."""

import mne
import numpy as np
import pytest

from meganorm.src.preprocess import prepare_eeg_data

pytestmark = pytest.mark.unit


def raw_eeg():
    raw = mne.io.RawArray(
        np.arange(60, dtype=float).reshape(3, 20) * 1e-6,
        mne.create_info(["Cz", "Pz", "EOG"], 100, ["eeg", "eeg", "misc"]),
        first_samp=123,
        verbose=False,
    )
    raw.set_annotations(mne.Annotations([0.01], [0.02], ["test"]))
    return raw


def sidecars(tmp_path, text=None):
    path = tmp_path / "sub-01_task-rest_eeg.vhdr"
    if text is not None:
        (tmp_path / "sub-01_task-rest_channels.tsv").write_text(text)
    return str(path)


def test_rest_types_ignore_bad_status_and_preserve_signal(tmp_path):
    path = sidecars(
        tmp_path, "name\ttype\tstatus\nCz\tEEG\tbad\nPz\tEEG\tgood\nEOG\tEOG\tgood\n"
    )
    raw = raw_eeg()
    original = raw.copy()
    raw.info["bads"] = ["Pz"]
    assert prepare_eeg_data(raw, path) is raw
    assert raw.get_channel_types() == ["eeg", "eeg", "eog"]
    assert raw.info["bads"] == ["Pz"]
    np.testing.assert_array_equal(raw.get_data(), original.get_data())
    assert raw.first_samp == original.first_samp
    assert raw.annotations == original.annotations


def test_rest_duplicate_names_use_last_type(tmp_path):
    path = sidecars(tmp_path, "name\ttype\nEOG\tMISC\nEOG\tEOG\n")
    raw = prepare_eeg_data(raw_eeg(), path)
    assert raw.get_channel_types()[-1] == "eog"


def test_rest_non_utf8_still_raises(tmp_path):
    path = sidecars(tmp_path)
    (tmp_path / "sub-01_task-rest_channels.tsv").write_bytes(
        b"name\ttype\tunits\nCz\tEEG\t\xb5V\n"
    )
    with pytest.raises(UnicodeDecodeError):
        prepare_eeg_data(raw_eeg(), path)


@pytest.mark.parametrize(
    "table", ["wrong\ttype\nCz\tEEG\n", "name\ttype\nabsent\tEEG\n"]
)
def test_rest_bad_channel_table_errors_propagate(tmp_path, table):
    with pytest.raises((KeyError, ValueError)):
        prepare_eeg_data(raw_eeg(), sidecars(tmp_path, table))


@pytest.mark.parametrize("content", [None, "wrong\n1\n", "Channel,X,Y,Z\nCz,0,0,0.1\n"])
def test_rest_missing_invalid_or_incomplete_montage_warns(tmp_path, content, caplog):
    if content is not None:
        (tmp_path / "cap_montage.csv").write_text(content)
    raw = raw_eeg()
    before = raw.get_data().copy()
    assert prepare_eeg_data(raw, sidecars(tmp_path)) is raw
    assert "Could not set EEG montage" in caplog.text
    np.testing.assert_array_equal(raw.get_data(), before)


def test_rest_embedded_montage_is_preserved(tmp_path):
    raw = raw_eeg()
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
    before = raw.get_montage().get_positions()["ch_pos"]
    (tmp_path / "cap_montage.csv").write_text("invalid")
    prepare_eeg_data(raw, sidecars(tmp_path))
    after = raw.get_montage().get_positions()["ch_pos"]
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])


def test_rest_uses_first_glob_match_not_recording_specific_priority(
    tmp_path, monkeypatch
):
    import meganorm.src.preprocess as prep

    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    first.write_text("name\ttype\nEOG\tEOG\n")
    second.write_text("name\ttype\nEOG\tMISC\n")
    montage = tmp_path / "cap.csv"
    montage.write_text("Channel,X,Y,Z\nCz,0,0,0.1\nPz,0,-0.05,0.08\n")
    monkeypatch.setattr(
        prep.glob,
        "glob",
        lambda pattern, **kw: (
            [str(first), str(second)]
            if pattern.endswith("channels.tsv")
            else [str(montage)]
        ),
    )
    raw = prepare_eeg_data(raw_eeg(), sidecars(tmp_path))
    assert raw.get_channel_types()[-1] == "eog"
    np.testing.assert_allclose(raw.info["chs"][1]["loc"][:3], [0, -0.05, 0.08])


def test_shared_channel_policy_preserves_input_and_merges_bad_status(tmp_path):
    from meganorm.utils.eeg import read_channel_table, apply_channel_table

    path = tmp_path / "channels.tsv"
    original = (
        b"name\ttype\tunits\tstatus\nCz\tEEG\t\xb5V\tbad\nEOG\tEOG\t\xb5V\tgood\n"
    )
    path.write_bytes(original)
    with pytest.warns(UserWarning, match="Windows-1252"):
        table = read_channel_table(path, legacy=False)
    raw = raw_eeg()
    raw.info["bads"] = ["Pz"]
    apply_channel_table(raw, table, strict=True, apply_status=True)
    assert raw.info["bads"] == ["Pz", "Cz"]
    assert raw.get_channel_types()[-1] == "eog"
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "text",
    [
        "name\ttype\nCz\tEEG\nCz\tEEG\n",
        "name\ttype\nunknown\tEEG\n",
        "wrong\ttype\nCz\tEEG\n",
    ],
)
def test_shared_strict_channel_validation(tmp_path, text):
    from meganorm.utils.eeg import read_channel_table, apply_channel_table

    path = tmp_path / "channels.tsv"
    path.write_text(text, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="Channel sidecar"):
        apply_channel_table(
            raw_eeg(), read_channel_table(path), strict=True, apply_status=True
        )


def test_shared_csv_policy_keeps_legacy_duplicate_semantics(tmp_path):
    from meganorm.utils.eeg import read_montage_positions

    path = tmp_path / "cap.csv"
    path.write_text("Channel,X,Y,Z\nCz,0,0,0.1\nCz,0,0,0.2\n")
    assert read_montage_positions(path, strict=False)["Cz"] == [0, 0, 0.2]
    with pytest.raises(ValueError, match="unique"):
        read_montage_positions(path, strict=True)
