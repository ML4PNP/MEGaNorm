import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from meganorm.utils import IO as io
from meganorm.utils.IO import (
    Config,
    check_demographic_format,
    clean_nan_columns,
    find_failed_meg_subjects,
    find_other_meg_session,
    find_other_mri_session,
    infer_device,
    load_demographic_file,
    load_recording,
    make_demo_file_bids,
    merge_datasets_with_glob,
    merge_fidp_demo,
    select_session_path,
    set_path,
)


LOGGER = logging.getLogger(__name__)


def serialized_paths(value):
    return [Path(path) for path in value.split("*") if path]


class FakeInfo(dict):
    """Minimal MNE Info boundary double with the required unlock context."""

    def _unlock(self):
        return nullcontext()


class FakeRaw:
    def __init__(self, dig=None):
        self.info = FakeInfo(dig=dig)
        self.montages = []

    def set_montage(self, montage, on_missing=None):
        self.montages.append((montage, on_missing))


@pytest.fixture
def recording_readers(monkeypatch):
    calls = []

    def install(name):
        def reader(*args, **kwargs):
            result = SimpleNamespace(reader=name, args=args, kwargs=kwargs)
            calls.append(result)
            return result

        monkeypatch.setattr(io.mne.io, name, reader)

    for name in (
        "read_raw",
        "read_raw_bti",
        "read_raw_ctf",
        "read_raw_fif",
        "read_raw_artemis123",
    ):
        install(name)

    return calls


@pytest.mark.unit
def test_config_save_load_round_trip_preserves_custom_values(tmp_path):
    path = tmp_path / "config.json"
    config = Config(
        which_sensor="eeg",
        apply_source_localization=True,
        SL_conductivity=(0.3, 0.006, 0.3),
        source_space_spacing="oct5",
        source_space_spacing_number=5,
        random_state=17,
    )

    config.save(path)

    assert Config.load(path) == config


@pytest.mark.unit
def test_config_save_protects_existing_file_unless_overwrite_is_enabled(tmp_path):
    path = tmp_path / "config.json"
    Config(random_state=17).save(path)

    with pytest.raises(FileExistsError):
        Config(random_state=23).save(path)

    assert Config.load(path).random_state == 17

    Config(random_state=23).save(path, overwrite=True)
    assert Config.load(path).random_state == 23


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "which_sensor": "eeg",
                "apply_source_localization": True,
                "SL_conductivity": (0.3,),
            },
            "three layers conductivity",
        ),
        (
            {
                "source_space_spacing": "oct5",
                "source_space_spacing_number": 4,
            },
            "should match",
        ),
        (
            {
                "source_space_spacing": "all",
                "source_space_spacing_number": 4,
            },
            "must be None",
        ),
        (
            {
                "beamformer_pick_ori": "vector",
                "beamformer_weight_norm": "unit-noise-gain",
            },
            "unit-noise-gain-invariant",
        ),
        (
            {"apply_mri_template": True},
            "template MRI",
        ),
        (
            {
                "apply_mri_template": True,
                "freesurfer_template_path": "/templates/fsaverage",
                "apply_mri_QC": True,
            },
            "MRI QC",
        ),
        (
            {"gedai_method": "broadband", "gedai_wavelet_level": "auto"},
            "wavelet_level=0",
        ),
        (
            {"gedai_method": "spectral", "gedai_wavelet_level": 0},
            "wavelet_level > 0",
        ),
        (
            {
                "gedai_method": "both",
                "gedai_preliminary_broadband_noise_multiplier": 0,
            },
            "preliminary_broadband_noise_multiplier",
        ),
    ],
)
def test_config_rejects_incompatible_cross_field_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        Config(**overrides)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {
            "which_sensor": "eeg",
            "apply_source_localization": True,
            "SL_conductivity": (0.3, 0.006, 0.3),
        },
        {"source_space_spacing": "all", "source_space_spacing_number": None},
        {
            "beamformer_pick_ori": "vector",
            "beamformer_weight_norm": "unit-noise-gain-invariant",
        },
        {"gedai_method": "broadband", "gedai_wavelet_level": 0},
        {"gedai_method": "spectral", "gedai_wavelet_level": "auto"},
    ],
)
def test_config_accepts_compatible_cross_field_settings(overrides):
    Config(**overrides)


@pytest.mark.unit
def test_select_session_path_selects_from_serialized_paths_and_allows_missing_input():
    serialized = "/data/session-1.fif*/data/session-2.fif"

    assert select_session_path(serialized) == "/data/session-1.fif"
    assert select_session_path(serialized, index=1) == "/data/session-2.fif"
    assert select_session_path(None) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("subject.ds", "CTF"),
        ("subject.FIF", "MEGIN"),
        ("subject.Bin", "ARTEMIS123"),
    ],
)
def test_infer_device_treats_meg_file_extensions_case_insensitively(path, expected):
    assert infer_device(path, None, "meg", LOGGER) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "device_type", "which_sensor", "expected"),
    [
        ("subject.edf", None, "eeg", "EDF"),
        ("subject.unknown", "ctf", "meg", "CTF"),
        ("/data/4D/subject", None, "meg", "BTI"),
    ],
)
def test_infer_device_uses_sensor_override_and_4d_path(
    path, device_type, which_sensor, expected
):
    assert infer_device(path, device_type, which_sensor, LOGGER) == expected


@pytest.mark.unit
def test_infer_device_rejects_unsupported_meg_recording():
    with pytest.raises(ValueError, match="not supported"):
        infer_device("subject.unknown", None, "meg", LOGGER)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("device", "path", "empty_path", "expected_reader"),
    [
        ("BTI", "/data/subject", "/empty/subject", "read_raw_bti"),
        ("ARTEMIS123", "/data/subject.bin", "/empty/room.bin", "read_raw_artemis123"),
        ("MEGIN", "/data/subject.fif", "/empty/room.fif", "read_raw"),
    ],
)
def test_load_recording_loads_empty_room_for_ssp_without_source_localization(
    device, path, empty_path, expected_reader, recording_readers
):
    configs = Config(
        apply_source_localization=False,
        apply_empty_room_recording=True,
        apply_environmental_noise_ssp_with_eroom=True,
    )

    _, empty_room = load_recording(
        device,
        path,
        empty_path,
        configs,
        LOGGER,
    )

    assert empty_room.reader == expected_reader


@pytest.mark.unit
def test_load_recording_respects_disabled_empty_room_master_switch(
    recording_readers,
):
    configs = Config(
        apply_source_localization=True,
        apply_empty_room_recording=False,
        apply_environmental_noise_ssp_with_eroom=True,
    )

    data, empty_room = load_recording(
        "CTF",
        "/data/subject.ds",
        "/empty/room.ds",
        configs,
        LOGGER,
    )

    assert data.reader == "read_raw_ctf"
    assert empty_room is None
    assert len(recording_readers) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source_localization", "empty_room_ssp", "master_switch", "expected_loaded"),
    [
        (True, False, True, True),
        (False, True, True, True),
        (True, True, False, False),
        (False, False, True, False),
    ],
)
def test_load_recording_empty_room_policy_truth_table(
    source_localization,
    empty_room_ssp,
    master_switch,
    expected_loaded,
    recording_readers,
):
    configs = Config(
        apply_source_localization=source_localization,
        apply_empty_room_recording=master_switch,
        apply_environmental_noise_ssp_with_eroom=empty_room_ssp,
    )

    _, empty_room = load_recording(
        "MEGIN",
        "/data/subject.fif",
        "/empty/room.fif",
        configs,
        LOGGER,
    )

    assert (empty_room is not None) is expected_loaded
    assert len(recording_readers) == (2 if expected_loaded else 1)


@pytest.mark.unit
def test_load_recording_uses_fif_reader_for_converted_artemis_empty_room(
    recording_readers,
):
    configs = Config(
        apply_source_localization=False,
        apply_empty_room_recording=True,
        apply_environmental_noise_ssp_with_eroom=True,
    )

    data, empty_room = load_recording(
        "ARTEMIS123",
        "/data/subject.fif",
        "/empty/room.FIF",
        configs,
        LOGGER,
    )

    assert data.reader == "read_raw_fif"
    assert empty_room.reader == "read_raw_fif"
    assert [call.args[0] for call in recording_readers] == [
        "/data/subject.fif",
        "/empty/room.FIF",
    ]


@pytest.mark.unit
def test_load_recording_adds_ctf_pos_points_when_dig_is_missing(monkeypatch):
    raw = FakeRaw(dig=None)
    added_dig = [{"kind": "extra"}]
    monkeypatch.setattr(io.mne.io, "read_raw_ctf", lambda *args, **kwargs: raw)
    monkeypatch.setattr(io, "_read_pos", lambda fname: added_dig)
    monkeypatch.setattr(
        io.mne._fiff._digitization,
        "_format_dig_points",
        lambda digs: digs,
    )

    data, _ = load_recording(
        "CTF",
        "/data/subject.ds",
        None,
        Config(),
        LOGGER,
        pos_file="/data/headshape.pos",
    )

    assert data.info["dig"] == added_dig


@pytest.mark.unit
@pytest.mark.parametrize(
    ("device", "path", "expected_reader"),
    [
        ("CTF", "/data/subject.ds", "read_raw_ctf"),
        ("MEGIN", "/data/subject.fif", "read_raw"),
        ("EEG", "/data/subject.edf", "read_raw"),
        ("ARTEMIS123", "/data/subject.bin", "read_raw_artemis123"),
        ("ARTEMIS123", "/data/subject.FIF", "read_raw_fif"),
    ],
)
def test_load_recording_dispatches_main_recording_to_device_reader(
    device, path, expected_reader, recording_readers
):
    data, empty_room = load_recording(
        device,
        path,
        None,
        Config(apply_source_localization=False),
        LOGGER,
    )

    assert data.reader == expected_reader
    assert data.kwargs["preload"] is True
    assert empty_room is None


@pytest.mark.unit
def test_load_recording_builds_bti_paths_and_uses_available_headshape(
    tmp_path, recording_readers
):
    recording_dir = tmp_path / "subject"
    recording_dir.mkdir()
    (recording_dir / "hs_file").touch()

    data, _ = load_recording(
        "BTI",
        str(recording_dir),
        None,
        Config(apply_source_localization=False),
        LOGGER,
    )

    assert data.reader == "read_raw_bti"
    assert data.kwargs == {
        "pdf_fname": str(recording_dir / "c,rfDC"),
        "config_fname": str(recording_dir / "config"),
        "head_shape_fname": str(recording_dir / "hs_file"),
        "preload": True,
        "convert": True,
    }


@pytest.mark.unit
def test_load_recording_passes_artemis_pos_to_native_reader(recording_readers):
    data, _ = load_recording(
        "ARTEMIS123",
        "/data/subject.bin",
        None,
        Config(
            apply_source_localization=True,
            apply_empty_room_recording=False,
        ),
        LOGGER,
        pos_file="/data/headshape.pos",
    )

    assert data.reader == "read_raw_artemis123"
    assert data.kwargs == {
        "preload": True,
        "pos_fname": "/data/headshape.pos",
        "add_head_trans": True,
    }


@pytest.mark.unit
def test_load_recording_adds_headshape_to_converted_artemis_fif(
    monkeypatch, recording_readers
):
    augmented = object()

    def add_headshape(data, path, pos_file, logger):
        assert data.reader == "read_raw_fif"
        assert path == "/data/subject.fif"
        assert pos_file == "/data/headshape.pos"
        assert logger is LOGGER
        return augmented

    monkeypatch.setattr(io, "_add_artemis_headshape", add_headshape)

    data, _ = load_recording(
        "ARTEMIS123",
        "/data/subject.fif",
        None,
        Config(
            apply_source_localization=True,
            apply_empty_room_recording=False,
        ),
        LOGGER,
        pos_file="/data/headshape.pos",
    )

    assert data is augmented


@pytest.mark.unit
def test_load_recording_adds_ctf_fif_montage_case_insensitively(monkeypatch):
    raw = FakeRaw(dig=[])
    montage = object()
    monkeypatch.setattr(io.mne.io, "read_raw_ctf", lambda *args, **kwargs: raw)
    monkeypatch.setattr(io.mne.channels, "read_dig_fif", lambda path: montage)

    data, _ = load_recording(
        "CTF",
        "/data/subject.ds",
        None,
        Config(),
        LOGGER,
        pos_file="/data/headshape.FIF",
    )

    assert data.montages == [(montage, "warn")]


@pytest.mark.unit
def test_resolve_artemis_pos_file_prefers_explicit_path():
    assert (
        io._resolve_artemis_pos_file(
            "/data/subject.bin", "/chosen/headshape.pos", LOGGER
        )
        == "/chosen/headshape.pos"
    )


@pytest.mark.unit
def test_resolve_artemis_pos_file_searches_parent_tree_deterministically(tmp_path):
    recording_dir = tmp_path / "dataset" / "recordings"
    recording_dir.mkdir(parents=True)
    headshape_dir = tmp_path / "dataset" / "Headshape"
    headshape_dir.mkdir()
    second = headshape_dir / "z-subject.pos"
    first = headshape_dir / "a-subject.pos"
    second.touch()
    first.touch()

    result = io._resolve_artemis_pos_file(
        recording_dir / "subject.bin",
        None,
        LOGGER,
    )

    assert result == str(first)


@pytest.mark.unit
def test_resolve_artemis_pos_file_raises_when_no_headshape_exists(tmp_path):
    recording_dir = tmp_path / "dataset" / "recordings"
    recording_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No .pos file found"):
        io._resolve_artemis_pos_file(
            recording_dir / "subject.bin",
            None,
            LOGGER,
        )


@pytest.mark.unit
def test_add_artemis_headshape_keeps_existing_extra_points(monkeypatch):
    raw = FakeRaw(dig=[{"kind": io.FIFF.FIFFV_POINT_EXTRA}])

    def unexpected_resolution(*args, **kwargs):
        raise AssertionError("A new headshape should not be resolved")

    monkeypatch.setattr(io, "_resolve_artemis_pos_file", unexpected_resolution)

    assert io._add_artemis_headshape(raw, "/data/subject.fif", None, LOGGER) is raw


@pytest.mark.unit
def test_add_artemis_headshape_attaches_fif_montage(monkeypatch):
    raw = FakeRaw(dig=[])
    montage = object()
    monkeypatch.setattr(
        io,
        "_resolve_artemis_pos_file",
        lambda *args: "/data/headshape.fif",
    )
    monkeypatch.setattr(io.mne.channels, "read_dig_fif", lambda path: montage)

    result = io._add_artemis_headshape(raw, "/data/subject.fif", None, LOGGER)

    assert result is raw
    assert raw.montages == [(montage, "warn")]


@pytest.mark.unit
def test_add_artemis_headshape_attaches_pos_points(monkeypatch):
    fiducial = {"kind": io.FIFF.FIFFV_POINT_CARDINAL}
    raw = FakeRaw(dig=[fiducial])
    points = [{"kind": io.FIFF.FIFFV_POINT_EXTRA}]
    formatted_inputs = []
    monkeypatch.setattr(
        io,
        "_resolve_artemis_pos_file",
        lambda *args: "/data/headshape.pos",
    )
    monkeypatch.setattr(io, "_read_pos", lambda fname: points)
    monkeypatch.setattr(
        io.mne._fiff._digitization,
        "_format_dig_points",
        lambda digs: formatted_inputs.append(digs) or digs,
    )

    result = io._add_artemis_headshape(raw, "/data/subject.fif", None, LOGGER)

    assert result is raw
    assert raw.info["dig"] == [fiducial, *points]
    assert formatted_inputs == [[fiducial, *points]]


@pytest.mark.unit
@pytest.mark.parametrize(("suffix", "separator"), [(".csv", ","), (".tsv", "\t")])
def test_load_demographic_file_preserves_text_subject_ids(
    tmp_path, suffix, separator
):
    path = tmp_path / f"participants{suffix}"
    path.write_text(
        separator.join(["participant_id", "age", "sex", "eyes"])
        + "\n"
        + separator.join(["001", "20", "Female", "open"])
        + "\n"
        + separator.join(["010", "30", "Male", "closed"])
        + "\n"
    )

    result = load_demographic_file(path)

    assert result.index.tolist() == ["001", "010"]
    assert result["age"].tolist() == [20, 30]


@pytest.mark.unit
def test_load_demographic_file_rejects_unsupported_extension(tmp_path):
    with pytest.raises(ValueError, match="Unsupported demographic file type"):
        load_demographic_file(tmp_path / "participants.json")


@pytest.mark.unit
@pytest.mark.parametrize("suffix", [".xlsx", ".xls"])
def test_load_demographic_file_dispatches_excel_formats(monkeypatch, tmp_path, suffix):
    calls = []

    def read_excel(path, index_col):
        calls.append((path, index_col))
        return pd.DataFrame({"age": [20]}, index=[1])

    monkeypatch.setattr(io.pd, "read_excel", read_excel)
    path = tmp_path / f"participants{suffix}"

    result = load_demographic_file(path)

    assert calls == [(path, 0)]
    assert result.index.tolist() == ["1"]
    assert result["age"].tolist() == [20]


@pytest.mark.unit
def test_load_demographic_file_keeps_numeric_columns_when_index_is_disabled(tmp_path):
    path = tmp_path / "participants.csv"
    path.write_text(
        "age,participant_id,sex,eyes\n"
        "20,sub-001,Female,open\n"
        "30,sub-010,Male,closed\n"
    )

    result = load_demographic_file(path, index_col=False)

    assert result.index.tolist() == ["0", "1"]
    assert result["age"].tolist() == [20, 30]
    assert pd.api.types.is_numeric_dtype(result["age"])


@pytest.mark.unit
def test_merge_fidp_demo_preserves_subject_ids_and_assigns_missing_sites(tmp_path):
    first_demo = tmp_path / "first.tsv"
    first_demo.write_text(
        "participant_id\tage\tsex\teyes\n"
        "001\t20\tFemale\topen\n"
    )
    second_demo = tmp_path / "second.tsv"
    second_demo.write_text(
        "participant_id\tage\tsex\teyes\tsite\n"
        "010\t30\tMale\tclosed\texisting-site\n"
    )
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    (features_dir / "all_features.csv").write_text(
        "participant_id,alpha_power\n"
        "001,1.5\n"
        "010,2.5\n"
        "999,9.0\n"
    )

    result = merge_fidp_demo(
        [first_demo, second_demo],
        features_dir,
        ["first-site", "second-site"],
    )

    assert result.index.tolist() == ["001", "010"]
    assert result.loc["001", "site"] == "first-site"
    assert result.loc["010", "site"] == "existing-site"
    assert result["alpha_power"].tolist() == [1.5, 2.5]
    assert "eyes" not in result


@pytest.mark.unit
def test_merge_fidp_demo_rejects_missing_demographic_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="demographic file.*dataset-a"):
        merge_fidp_demo(
            [tmp_path / "missing.tsv"],
            tmp_path,
            ["dataset-a"],
        )


@pytest.mark.unit
def test_merge_fidp_demo_rejects_missing_feature_file(tmp_path):
    demographic = tmp_path / "participants.tsv"
    demographic.write_text(
        "participant_id\tage\tsex\teyes\nsub-01\t20\tFemale\topen\n"
    )

    with pytest.raises(FileNotFoundError, match="all_features.csv"):
        merge_fidp_demo([demographic], tmp_path, ["dataset-a"])


@pytest.mark.unit
def test_merge_datasets_with_glob_collects_optional_subject_files(tmp_path):
    base = tmp_path / "meg"
    rest_dir = base / "sub-01" / "session"
    rest_dir.mkdir(parents=True)
    rest_1 = rest_dir / "run-1_rest_raw.fif"
    rest_2 = rest_dir / "run-2_rest_raw.fif"
    rest_1.touch()
    rest_2.touch()
    (base / "sub-02").mkdir()

    empty = tmp_path / "empty" / "sub-01"
    empty.mkdir(parents=True)
    empty_file = empty / "empty_raw.fif"
    empty_file.touch()

    events = tmp_path / "events" / "sub-01"
    events.mkdir(parents=True)
    event_file = events / "rest_events.tsv"
    event_file.touch()

    transforms = tmp_path / "transforms" / "sub-01"
    transforms.mkdir(parents=True)
    trans_file = transforms / "sub-01-trans.fif"
    trans_file.touch()

    positions = tmp_path / "positions" / "sub-01"
    positions.mkdir(parents=True)
    pos_file = positions / "headshape.pos"
    pos_file.touch()

    surfaces = tmp_path / "surfaces"
    (surfaces / "sub-01").mkdir(parents=True)

    result = merge_datasets_with_glob(
        {
            "dataset-a": {
                "base_dir": str(base),
                "task": "rest",
                "ending": "raw.fif",
                "device_type": "MEGIN",
                "line_freq": 60,
                "empty_room_task": "empty",
                "empty_room_path": str(tmp_path / "empty"),
                "empty_room_ending": "raw.fif",
                "surfaces_dir": str(surfaces),
                "event_file_path": str(tmp_path / "events"),
                "event_file_task": "rest",
                "event_file_ending": "events.tsv",
                "event_of_interest": 7,
                "trans_path": str(tmp_path / "transforms"),
                "pos_path": str(tmp_path / "positions"),
                "pos_file_ending": ".pos",
                "layout_path": "/layouts/meg.json",
            }
        }
    )

    assert list(result) == ["sub-01"]
    subject = result["sub-01"]
    assert serialized_paths(subject["rest_record"]) == [rest_1, rest_2]
    assert serialized_paths(subject["empty_room_record"]) == [empty_file]
    assert serialized_paths(subject["event_record"]) == [event_file]
    assert serialized_paths(subject["trans_path"]) == [trans_file]
    assert serialized_paths(subject["pos_path"]) == [pos_file]
    assert subject["mri_surface"] == str(surfaces)
    assert subject["device"] == "MEGIN"
    assert subject["line_freq"] == "60"
    assert subject["event_of_interest"] == "7"


@pytest.mark.unit
def test_merge_datasets_with_glob_uses_documented_defaults(tmp_path):
    base = tmp_path / "meg"
    subject_dir = base / "sub-01"
    subject_dir.mkdir(parents=True)
    recording = subject_dir / "task-rest_raw.fif"
    recording.touch()

    result = merge_datasets_with_glob(
        {
            "dataset-a": {
                "base_dir": str(base),
                "task": "rest",
                "ending": "raw.fif",
            }
        }
    )["sub-01"]

    assert serialized_paths(result["rest_record"]) == [recording]
    assert result["line_freq"] == "50"
    assert result["device"] is None
    assert result["demographic_path"] == str(base / "participants_bids.tsv")
    assert result["empty_room_record"] is None


@pytest.mark.unit
@pytest.mark.parametrize(("suffix", "separator"), [(".csv", ","), (".tsv", "\t")])
def test_make_demo_file_bids_preserves_ids_maps_values_and_removes_duplicates(
    tmp_path, suffix, separator
):
    source = tmp_path / f"source{suffix}"
    rows = [
        ["participant", "years", "gender", "group"],
        ["001", "20", "F", "control"],
        ["001", "21", "F", "control"],
        ["010", "30", "M", ""],
    ]
    source.write_text("\n".join(separator.join(row) for row in rows) + "\n")

    make_demo_file_bids(
        str(source),
        str(tmp_path),
        0,
        1,
        {"col_name": "sex", "col_id": 2, "mapping": {"F": "Female", "M": "Male"}},
        {"col_name": "eyes", "single_value": "open"},
        {"col_name": "diagnosis", "col_id": 3},
    )

    result = pd.read_csv(
        tmp_path / "participants_bids.tsv",
        sep="\t",
        dtype={"participant_id": str},
        keep_default_na=False,
    )
    assert result.to_dict("list") == {
        "participant_id": ["001", "010"],
        "age": [20, 30],
        "sex": ["Female", "Male"],
        "eyes": ["open", "open"],
        "diagnosis": ["control", "nan"],
    }


@pytest.mark.unit
def test_make_demo_file_bids_rejects_single_value_together_with_mapping(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("participant,age,code\nsub-01,20,1\n")

    with pytest.raises(ValueError, match="can not be both defined"):
        make_demo_file_bids(
            str(source),
            str(tmp_path),
            0,
            1,
            {
                "col_name": "group",
                "single_value": 0,
                "mapping": {1: "case"},
            },
        )


@pytest.mark.unit
def test_make_demo_file_bids_dispatches_xlsx_input(monkeypatch, tmp_path):
    source = tmp_path / "source.xlsx"
    monkeypatch.setattr(
        io.pd,
        "read_excel",
        lambda path: pd.DataFrame(
            {
                "participant": ["001"],
                "age": [20],
                "sex": ["F"],
            }
        ),
    )

    make_demo_file_bids(
        str(source),
        str(tmp_path),
        0,
        1,
        {"col_name": "sex", "col_id": 2, "mapping": {"F": "Female"}},
    )

    result = pd.read_csv(
        tmp_path / "participants_bids.tsv",
        sep="\t",
        dtype={"participant_id": str},
    )
    assert result.to_dict("records") == [
        {"participant_id": "001", "age": 20, "sex": "Female"}
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("column", "message"),
    [
        ({"col_id": 2}, "col_name"),
        ({"col_name": "sex"}, "either 'col_id' or 'single_value'"),
    ],
)
def test_make_demo_file_bids_rejects_incomplete_column_specification(
    tmp_path, column, message
):
    source = tmp_path / "source.csv"
    source.write_text("participant,age,sex\nsub-01,20,F\n")

    with pytest.raises(ValueError, match=message):
        make_demo_file_bids(str(source), str(tmp_path), 0, 1, column)


@pytest.mark.unit
def test_check_demographic_format_accepts_ids_in_column_or_index():
    values = {
        "age": [20.0, 30.0],
        "sex": ["Female", "Male"],
        "eyes": ["open", "closed"],
    }
    with_column = pd.DataFrame({"participant_id": ["sub-01", "sub-02"], **values})
    with_index = pd.DataFrame(values, index=["sub-01", "sub-02"])

    assert check_demographic_format(with_column) is None
    assert check_demographic_format(with_index) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dataframe", "message"),
    [
        (
            pd.DataFrame(
                {
                    "participant_id": ["sub-01"],
                    "age": [20],
                    "sex": ["Female"],
                }
            ),
            "missing the 'eyes' column",
        ),
        (
            pd.DataFrame(
                {
                    "participant_id": [1],
                    "age": [20],
                    "sex": ["Female"],
                    "eyes": ["open"],
                }
            ),
            "IDs must be strings",
        ),
        (
            pd.DataFrame(
                {
                    "participant_id": ["sub-01"],
                    "age": ["twenty"],
                    "sex": ["Female"],
                    "eyes": ["open"],
                }
            ),
            "age.*integers or floats",
        ),
        (
            pd.DataFrame(
                {
                    "participant_id": ["sub-01"],
                    "age": [20],
                    "sex": ["unknown"],
                    "eyes": ["open"],
                }
            ),
            "only.*Male.*Female.*allowed",
        ),
    ],
)
def test_check_demographic_format_rejects_malformed_tables(dataframe, message):
    with pytest.raises(ValueError, match=message):
        check_demographic_format(dataframe)


@pytest.mark.unit
def test_clean_nan_columns_drops_above_threshold_and_imputes_numeric_columns():
    original = pd.DataFrame(
        {
            "kept_numeric": [1.0, np.nan, 3.0, 20.0],
            "dropped_numeric": [np.nan, np.nan, 3.0, np.nan],
            "kept_text": ["a", None, "c", "d"],
        }
    )

    result = clean_nan_columns(original, nan_threshold=1)

    assert result["kept_numeric"].tolist() == [1.0, 3.0, 3.0, 20.0]
    assert "dropped_numeric" not in result
    assert pd.isna(result.loc[1, "kept_text"])
    assert pd.isna(original.loc[1, "kept_numeric"])


@pytest.mark.unit
def test_set_path_creates_feature_and_normative_output_directories(tmp_path):
    features_dir, log_dir = set_path(tmp_path)

    assert features_dir == str(tmp_path / "Features")
    assert log_dir == str(tmp_path / "Features" / "log_slurm_jobs")
    expected_directories = [
        "Features/temp",
        "Features/excluded_participants",
        "Features/Configurations",
        "Features/MRI_templates",
        "Features/Saved_outputs/Epochs",
        "Features/Saved_outputs/PSDs",
        "Features/Saved_outputs/Preprocessed_data",
        "Features/Saved_outputs/coregistration_QC",
        "Features/Saved_outputs/auto_reject_plot",
        "Features/Saved_outputs/Covariance_figures",
        "Features/Saved_outputs/BEM_figures",
        "Features/Saved_outputs/transformation_FIF_file",
        "Features/Saved_outputs/Grouping_effects",
        "Normative_models",
    ]
    assert all((tmp_path / relative).is_dir() for relative in expected_directories)


@pytest.mark.unit
def test_find_other_mri_session_uses_sorted_one_based_session_index(monkeypatch):
    monkeypatch.setattr(
        io.glob,
        "glob",
        lambda pattern, recursive: [
            "/mri/sub-01/ses-02/T1w.nii.gz",
            "/mri/sub-01/ses-01/T1w.nii.gz",
        ],
    )

    result = find_other_mri_session(
        "/mri",
        ["sub-01"],
        "T1w.nii.gz",
        which_session=2,
    )

    assert result == {"sub-01": "/mri/sub-01/ses-02/T1w.nii.gz"}


@pytest.mark.unit
def test_find_other_mri_session_omits_subject_without_requested_session(tmp_path):
    session = tmp_path / "sub-01" / "ses-01"
    session.mkdir(parents=True)
    (session / "T1w.nii.gz").touch()

    result = find_other_mri_session(
        tmp_path,
        ["sub-01", "sub-02"],
        "T1w.nii.gz",
        which_session=2,
    )

    assert result == {}


@pytest.mark.unit
def test_find_other_meg_session_uses_sorted_matching_task_session(monkeypatch):
    patterns = []

    def fake_glob(pattern, recursive):
        patterns.append((pattern, recursive))
        return [
            "/meg/sub-01/ses-02/task-rest_raw.fif",
            "/meg/sub-01/ses-01/task-rest_raw.fif",
        ]

    monkeypatch.setattr(io.glob, "glob", fake_glob)

    result = find_other_meg_session(
        "/meg",
        ["sub-01"],
        "raw.fif",
        "rest",
        which_session=2,
    )

    assert result == {"sub-01": "/meg/sub-01/ses-02/task-rest_raw.fif"}
    assert patterns == [("/meg/sub-01/**/*rest*raw.fif", True)]


@pytest.mark.unit
def test_find_other_meg_session_omits_subject_without_requested_session(tmp_path):
    session = tmp_path / "sub-01" / "ses-01"
    session.mkdir(parents=True)
    (session / "task-rest_raw.fif").touch()

    result = find_other_meg_session(
        tmp_path,
        ["sub-01"],
        "raw.fif",
        "rest",
        which_session=2,
    )

    assert result == {}


@pytest.mark.unit
def test_find_failed_meg_subjects_reads_error_logs_case_insensitively(tmp_path):
    (tmp_path / "sub-01_job.err").write_text("Runtime ERROR: processing failed")
    (tmp_path / "sub-02_job.err").write_text("completed with a warning")
    (tmp_path / "sub-03_job.out").write_text("error in an unrelated stdout log")
    (tmp_path / "archive_err").mkdir()

    result = find_failed_meg_subjects(tmp_path)

    assert result == {"sub-01"}
