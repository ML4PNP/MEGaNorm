import json
from pathlib import Path

import pytest

from meganorm.layouts import layouts

EEG_LOBE_GROUPS = {
    "frontal_lh_",
    "frontal_rh_",
    "central_lh_",
    "central_rh_",
    "temporal_lh_",
    "temporal_rh_",
    "parietal_lh_",
    "parietal_rh_",
    "occipital_lh_",
    "occipital_rh_",
}

FIF_LOBE_GROUPS = {
    "MAG_frontal_lh_",
    "MAG_frontal_rh_",
    "MAG_temporal_lh_",
    "MAG_temporal_rh_",
    "MAG_parietal_lh_",
    "MAG_parietal_rh_",
    "MAG_occipital_lh_",
    "MAG_occipital_rh_",
    "GRAD1_frontal_lh_",
    "GRAD1_frontal_rh_",
    "GRAD1_temporal_lh_",
    "GRAD1_temporal_rh_",
    "GRAD1_parietal_lh_",
    "GRAD1_parietal_rh_",
    "GRAD1_occipital_lh_",
    "GRAD1_occipital_rh_",
    "GRAD2_frontal_lh_",
    "GRAD2_frontal_rh_",
    "GRAD2_temporal_lh_",
    "GRAD2_temporal_rh_",
    "GRAD2_parietal_lh_",
    "GRAD2_parietal_rh_",
    "GRAD2_occipital_lh_",
    "GRAD2_occipital_rh_",
}

PACKAGED_LAYOUTS = {
    "BDF": {"BDF_EEG_ALL": {"EEG_ALL"}, "BDF_EEG_LOBE": EEG_LOBE_GROUPS},
    "DS": {"DS_MAG_ALL": {"GRAD_ALL"}},
    "EDF": {"EDF_EEG_ALL": {"EEG_ALL"}, "EDF_EEG_LOBE": EEG_LOBE_GROUPS},
    "FIF": {
        "FIF_MAG_ALL": {"MAG_ALL"},
        "FIF_MEG_LOBE": FIF_LOBE_GROUPS,
        "FIF_GRAD_ALL": {"GRAD_ALL"},
    },
    "GES": {"GES_EEG_ALL": {"EEG_ALL"}},
    "SET": {"SET_EEG_ALL": {"EEG_ALL"}, "SET_EEG_LOBE": EEG_LOBE_GROUPS},
    "VHDR": {"VHDR_EEG_ALL": {"EEG_ALL"}, "VHDR_EEG_LOBE": EEG_LOBE_GROUPS},
}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("modality", "output_format"),
    [
        ("MEG", "FIF"),
        ("MEG", "DS"),
        ("EEG", "SET"),
        ("EEG", "VHDR"),
    ],
)
def test_create_layouts_matches_packaged_definition(
    monkeypatch, modality, output_format
):
    generated = {}
    expected_path = Path(layouts.get_relative_path(output_format))
    expected = json.loads(expected_path.read_text())

    monkeypatch.setattr(
        layouts,
        "save_sensor_layouts",
        lambda layout, filename: generated.setdefault(filename, layout),
    )

    returned_path = layouts.create_layouts(modality, output_format)

    assert returned_path == str(expected_path)
    assert generated[output_format] == expected


@pytest.mark.unit
def test_get_relative_path_resolves_packaged_json():
    path = Path(layouts.get_relative_path("FIF"))

    assert path.name == "FIF.json"
    assert path.parent == Path(layouts.__file__).parent
    assert path.is_file()


@pytest.mark.unit
def test_save_sensor_layouts_writes_readable_json(monkeypatch, tmp_path, capsys):
    path = tmp_path / "custom.json"
    expected = {"CUSTOM_EEG_ALL": {"EEG_ALL": ["C3", "C4"]}}
    monkeypatch.setattr(layouts, "get_relative_path", lambda filename: str(path))

    layouts.save_sensor_layouts(expected, "custom")

    assert json.loads(path.read_text()) == expected
    assert path.read_text().startswith("{\n    ")
    assert capsys.readouterr().out.strip() == f"Sensor layouts saved to {path}"


@pytest.mark.unit
def test_add_specific_layout_creates_new_layout_file(monkeypatch, tmp_path):
    path = tmp_path / "custom.json"
    layout_data = {"left": ["C3"], "right": ["C4"]}
    monkeypatch.setattr(layouts, "get_relative_path", lambda filename: str(path))

    layouts.add_specific_layout("custom", "CUSTOM_EEG_LOBE", layout_data)

    assert json.loads(path.read_text()) == {"CUSTOM_EEG_LOBE": layout_data}


@pytest.mark.unit
def test_add_specific_layout_updates_one_entry_without_losing_others(
    monkeypatch, tmp_path
):
    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps(
            {
                "CUSTOM_EEG_ALL": {"EEG_ALL": ["C3", "C4"]},
                "CUSTOM_EEG_LOBE": {"old": ["C3"]},
            }
        )
    )
    replacement = {"left": ["C3"], "right": ["C4"]}
    monkeypatch.setattr(layouts, "get_relative_path", lambda filename: str(path))

    layouts.add_specific_layout("custom", "CUSTOM_EEG_LOBE", replacement)

    assert json.loads(path.read_text()) == {
        "CUSTOM_EEG_ALL": {"EEG_ALL": ["C3", "C4"]},
        "CUSTOM_EEG_LOBE": replacement,
    }


@pytest.mark.unit
def test_load_specific_layout_returns_requested_entry(monkeypatch, tmp_path):
    path = tmp_path / "custom.json"
    expected = {"left": ["C3"], "right": ["C4"]}
    path.write_text(json.dumps({"CUSTOM_EEG_LOBE": expected}))
    monkeypatch.setattr(layouts, "get_relative_path", lambda filename: str(path))

    assert layouts.load_specific_layout("custom", "CUSTOM_EEG_LOBE") == expected


@pytest.mark.unit
def test_load_specific_layout_reports_unknown_layout(monkeypatch, tmp_path, capsys):
    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"CUSTOM_EEG_ALL": {"EEG_ALL": ["Cz"]}}))
    monkeypatch.setattr(layouts, "get_relative_path", lambda filename: str(path))

    result = layouts.load_specific_layout("custom", "UNKNOWN")

    assert result is None
    assert capsys.readouterr().out.strip() == f"Layout 'UNKNOWN' not found in {path}"


@pytest.mark.unit
def test_load_specific_layout_reports_missing_file(monkeypatch, tmp_path, capsys):
    path = tmp_path / "missing.json"
    monkeypatch.setattr(layouts, "get_relative_path", lambda filename: str(path))

    result = layouts.load_specific_layout("missing", "UNKNOWN")

    assert result is None
    assert capsys.readouterr().out.strip() == f"File '{path}' not found"


@pytest.mark.unit
def test_load_specific_layout_exposes_malformed_json(monkeypatch, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid JSON")
    monkeypatch.setattr(layouts, "get_relative_path", lambda filename: str(path))

    with pytest.raises(json.JSONDecodeError):
        layouts.load_specific_layout("broken", "ANY")


@pytest.mark.unit
@pytest.mark.parametrize(("output_format", "expected_schema"), PACKAGED_LAYOUTS.items())
def test_packaged_layout_resource_has_valid_channel_groups(
    output_format, expected_schema
):
    path = Path(layouts.get_relative_path(output_format))
    data = json.loads(path.read_text())

    assert set(data) == set(expected_schema)
    all_channel_sets = {}
    for layout_name, channel_groups in data.items():
        assert channel_groups
        assert set(channel_groups) == expected_schema[layout_name]
        for group_name, channels in channel_groups.items():
            assert isinstance(channels, list)
            assert channels
            assert all(isinstance(channel, str) and channel for channel in channels)
            assert len(channels) == len(set(channels))
            if group_name.endswith("_ALL"):
                all_channel_sets[group_name] = set(channels)

    for layout_name, channel_groups in data.items():
        if layout_name.endswith("_ALL"):
            continue
        for group_name, channels in channel_groups.items():
            if group_name.startswith("MAG_"):
                all_group = "MAG_ALL"
            elif group_name.startswith("GRAD"):
                all_group = "GRAD_ALL"
            else:
                all_group = "EEG_ALL"
            assert set(channels) <= all_channel_sets[all_group]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("modality", "output_format"),
    [("EEG", "FIF"), ("MEG", "VHDR"), ("EEG", "UNKNOWN")],
)
def test_create_layouts_reports_unsupported_combination(
    modality, output_format, capsys
):
    result = layouts.create_layouts(modality, output_format)

    assert result is None
    assert capsys.readouterr().out.strip() == (
        f"No predefined layout available for modality '{modality}' "
        f"and output_format '{output_format}'."
    )
