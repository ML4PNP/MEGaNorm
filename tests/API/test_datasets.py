from pathlib import Path
import os
import pytest


def make_dataset(tmp_path, **kwargs):
    from meganorm.API.datasets import Dataset

    return Dataset(
        name="cohort", root=tmp_path, task="rest", extension=".fif", **kwargs
    )


@pytest.mark.unit
def test_discovery_keeps_all_recordings_in_order_and_normalizes_paths(tmp_path):
    root = tmp_path / "data with spaces"
    for name in ["b_rest.fif", "a_rest.fif"]:
        path = root / "001" / "ses" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    dataset = make_dataset(root, demographics="participants.tsv", device="megin")
    manifest = dataset.discover()
    assert manifest.participant_id.tolist() == ["001"]
    assert [p.name for p in manifest.recording_paths.iloc[0]] == [
        "a_rest.fif",
        "b_rest.fif",
    ]
    assert manifest.device.iloc[0] == "MEGIN"
    assert dataset.demographics == root / "participants.tsv"
    assert not (root / "Features").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"line_freq": 0},
        {"line_freq": True},
        {"device": "unknown"},
        {"options": {"unknown": 1}},
        {"options": {"task": "override"}},
    ],
)
def test_dataset_rejects_invalid_settings(tmp_path, kwargs):
    with pytest.raises(ValueError):
        make_dataset(tmp_path, **kwargs)


@pytest.mark.unit
def test_empty_discovery_and_separator_path_fail(tmp_path):
    with pytest.raises(ValueError, match="rest"):
        make_dataset(tmp_path).discover()
    with pytest.raises(ValueError, match=r"\*"):
        make_dataset(tmp_path / "bad*path")


@pytest.mark.unit
def test_legacy_dictionary_uses_first_named_id_column(tmp_path):
    from meganorm.API.datasets import Dataset

    (tmp_path / "participants_bids.tsv").write_text("subject\tage\n001\t24\n")
    dataset = Dataset.from_dict(
        "cohort", {"base_dir": str(tmp_path), "task": "rest", "ending": ".fif"}
    )
    assert dataset.participant_id == "subject"
    assert dataset.demographics == tmp_path / "participants_bids.tsv"


@pytest.mark.unit
@pytest.mark.parametrize(
    "key",
    [
        "empty_room_path",
        "surfaces_dir",
        "event_file_path",
        "trans_path",
        "pos_path",
        "annotation_path",
        "layout_path",
    ],
)
def test_advanced_paths_resolve_against_dataset_root(tmp_path, key):
    dataset = make_dataset(tmp_path, options={key: "aux"})
    assert dataset.options[key] == tmp_path / "aux"


@pytest.mark.unit
@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows forbids '*' in filenames; path validation is tested separately.",
)
def test_discovery_rejects_literal_separator_in_recording_filename(tmp_path):
    path = tmp_path / "001" / "rest*record.fif"
    path.parent.mkdir()
    path.touch()
    with pytest.raises(ValueError, match=r"\*"):
        make_dataset(tmp_path).discover()


@pytest.mark.unit
@pytest.mark.parametrize("component", ["root", "participant"])
def test_literal_glob_components_cannot_reassign_recordings(tmp_path, component):
    if component == "root":
        root = tmp_path / "cohort[1]"
        sibling = tmp_path / "cohort1"
        for folder in [root / "001", sibling / "001"]:
            folder.mkdir(parents=True)
            (folder / "rest.fif").touch()
    else:
        root = tmp_path
        for name in ["sub[1]", "sub1"]:
            folder = root / name
            folder.mkdir()
            (folder / "rest.fif").touch()
    with pytest.raises(ValueError, match="glob"):
        make_dataset(root).discover()


@pytest.mark.unit
@pytest.mark.parametrize("character", ["*", "?", "[", "]"])
def test_dataset_rejects_glob_paths_without_creating_illegal_files(tmp_path, character):
    with pytest.raises(ValueError, match="glob"):
        make_dataset(tmp_path / f"cohort{character}")
