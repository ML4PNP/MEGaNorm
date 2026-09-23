import logging
from types import SimpleNamespace

import numpy as np
import pytest
from mne.io.constants import FIFF

from meganorm.src.source_localization import (
    build_template_index,
    capture_mne_log,
    check_digitization_points,
    check_tsss,
    nearest_template_dir,
    numpy_to_mne_epoch,
    regularized_cov_condition,
)


def make_tsss_record():
    return {
        "creator": "MaxFilter",
        "max_info": {
            "sss_info": {"in_order": 8, "out_order": 3},
            "max_st": {"buflen": 10.0, "subspcorr": 0.98},
        },
    }


@pytest.mark.unit
def test_check_tsss_finds_temporal_sss_after_unrelated_history():
    meg_data = SimpleNamespace(
        info={"proc_history": [{"creator": "unrelated"}, make_tsss_record()]}
    )

    assert check_tsss(meg_data)


@pytest.mark.unit
def test_check_tsss_rejects_spatial_sss_without_temporal_parameters():
    meg_data = SimpleNamespace(
        info={
            "proc_history": [
                {
                    "creator": "MaxFilter",
                    "max_info": {
                        "sss_info": {"in_order": 8, "out_order": 3},
                    },
                }
            ]
        }
    )

    assert not check_tsss(meg_data)


@pytest.mark.unit
@pytest.mark.parametrize(
    "max_st",
    [
        {"buflen": 0.0, "subspcorr": 0.98},
        {"buflen": 10.0},
        {"buflen": 10.0, "subspcorr": None},
    ],
)
def test_check_tsss_rejects_incomplete_temporal_parameters(max_st):
    record = make_tsss_record()
    record["max_info"]["max_st"] = max_st

    assert not check_tsss(SimpleNamespace(info={"proc_history": [record]}))


@pytest.mark.unit
def test_check_tsss_returns_false_without_processing_history():
    assert not check_tsss(SimpleNamespace(info={}))


@pytest.mark.unit
def test_capture_mne_log_supports_filename_without_parent_directory(
    tmp_path, monkeypatch
):
    mne_logger = logging.getLogger("mne")
    previous_level = mne_logger.level
    monkeypatch.chdir(tmp_path)

    with capture_mne_log("mne.log", level=logging.INFO) as log_path:
        mne_logger.info("captured source-localization message")

    assert "captured source-localization message" in (tmp_path / log_path).read_text()
    assert mne_logger.level == previous_level


@pytest.mark.unit
def test_capture_mne_log_restores_logger_after_exception(tmp_path):
    mne_logger = logging.getLogger("mne")
    previous_level = mne_logger.level
    previous_handlers = tuple(mne_logger.handlers)

    with pytest.raises(RuntimeError, match="stop processing"):
        with capture_mne_log(tmp_path / "logs" / "mne.log", level=logging.INFO):
            raise RuntimeError("stop processing")

    assert mne_logger.level == previous_level
    assert tuple(mne_logger.handlers) == previous_handlers


@pytest.mark.integration
def test_numpy_to_mne_epoch_preserves_source_time_courses_and_metadata():
    source_time_courses = np.arange(24.0).reshape(2, 3, 4)

    epochs = numpy_to_mne_epoch(
        source_time_courses,
        labels=["frontal-lh", "frontal-rh", "occipital-lh"],
        ch_name="misc",
        sampling_rate=200.0,
    )

    np.testing.assert_array_equal(epochs.get_data(), source_time_courses)
    assert epochs.ch_names == ["frontal-lh", "frontal-rh", "occipital-lh"]
    assert epochs.get_channel_types() == ["misc", "misc", "misc"]
    assert epochs.info["sfreq"] == pytest.approx(200.0)


@pytest.mark.unit
def test_regularized_cov_condition_applies_auto_scaled_shrinkage():
    data = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])

    covariance, condition_before, condition_after = regularized_cov_condition(
        data, shrinkage=0.25
    )

    np.testing.assert_allclose(
        covariance,
        [[4.0 / 3.0, -0.5], [-0.5, 4.0 / 3.0]],
    )
    assert condition_before == pytest.approx(3.0)
    assert condition_after == pytest.approx(2.2)


@pytest.mark.unit
def test_regularized_cov_condition_stabilizes_singular_covariance():
    data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    _, condition_before, condition_after = regularized_cov_condition(
        data, shrinkage=0.1
    )

    assert np.isfinite(condition_after)
    assert condition_after < condition_before


@pytest.mark.unit
def test_check_digitization_points_counts_each_point_kind():
    raw = SimpleNamespace(
        info={
            "dig": [
                {"kind": FIFF.FIFFV_POINT_EXTRA},
                {"kind": FIFF.FIFFV_POINT_EXTRA},
                {"kind": FIFF.FIFFV_POINT_CARDINAL},
                {"kind": FIFF.FIFFV_POINT_HPI},
                {"kind": FIFF.FIFFV_POINT_EEG},
                {"kind": FIFF.FIFFV_POINT_EEG},
            ]
        }
    )

    counts = check_digitization_points(raw, logging.getLogger("test.dig"))

    assert counts == (2, 1, 1, 2)


@pytest.mark.unit
def test_check_digitization_points_warns_when_information_is_absent(caplog):
    raw = SimpleNamespace(info={"dig": None})

    with caplog.at_level(logging.WARNING, logger="test.missing-dig"):
        counts = check_digitization_points(raw, logging.getLogger("test.missing-dig"))

    assert counts == (0, 0, 0, 0)
    assert "No dig info at all" in caplog.text


@pytest.mark.unit
def test_build_template_index_parses_months_and_years(tmp_path):
    for name in ["ANTS6-0Months3T", "ANTS1-5Years3T", "ANTS2-0Year3T"]:
        (tmp_path / name).mkdir()
    (tmp_path / "ANTS-invalid").mkdir()
    (tmp_path / "fsaverage").mkdir()
    (tmp_path / "ANTS9-0Months3T").write_text("not a template directory")

    index = build_template_index(tmp_path)

    assert index == {
        "ANTS6-0Months3T": 6.0,
        "ANTS1-5Years3T": 18.0,
        "ANTS2-0Year3T": 24.0,
    }


@pytest.mark.unit
def test_nearest_template_dir_selects_closest_available_age(tmp_path):
    for name in ["ANTS6-0Months3T", "ANTS1-5Years3T", "ANTS2-0Years3T"]:
        (tmp_path / name).mkdir()

    name, subjects_dir = nearest_template_dir(17.0, tmp_path)

    assert name == "ANTS1-5Years3T"
    assert subjects_dir == str(tmp_path)


@pytest.mark.unit
def test_nearest_template_dir_rejects_directory_without_templates(tmp_path):
    with pytest.raises(FileNotFoundError, match="No ANTS templates"):
        nearest_template_dir(12.0, tmp_path)
