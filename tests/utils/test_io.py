import logging

import numpy as np
import pandas as pd
import pytest

from meganorm.utils.IO import (
    Config,
    check_demographic_format,
    clean_nan_columns,
    infer_device,
    load_demographic_file,
    select_session_path,
    set_path,
)


LOGGER = logging.getLogger(__name__)


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
