"""Reject processing settings that cannot be used by the configured workflow."""

import pytest
from pydantic import ValidationError
from meganorm.API import Config


@pytest.mark.parametrize(
    "settings",
    [
        {"segments_overlap": -1},
        {"segments_overlap": 5},
        {"cutoffFreqLow": -1},
        {"cutoffFreqLow": 40},
        {"resampling_rate": 80},
        {"psd_parametrization_freq_range_low": 40},
        {"psd_n_overlap": 2},
        {"psd_n_fft": 1},
        {
            "which_sensor": "eeg",
            "bad_segment_removal_method": "fixed_thr",
            "eeg_flat_threshold": 40e-6,
        },
        {"psd_parametrization_peak_width_limits": (12, 1)},
        {"irasa_hset": (1.05, 2.0, 0.0)},
        {"psd_parametrization_method": "irasa"},
        {
            "psd_parametrization_method": "irasa",
            "cutoffFreqLow": 2,
            "cutoffFreqHigh": 80,
        },
        {"feature_categories": {"unknown_feature": True}},
        {"feature_categories": {"Offset": True}},
        {
            "which_sensor": "eeg",
            "apply_source_localization": True,
            "SL_conductivity": (),
        },
        {
            "which_sensor": "eeg",
            "apply_source_localization": True,
            "SL_conductivity": (0.3, 0.006),
        },
        {
            "which_sensor": "eeg",
            "apply_source_localization": True,
            "SL_conductivity": (0.3, 0.006, 0.3, 0.3),
        },
    ],
)
def test_invalid_processing_relationships_fail_when_config_is_created(settings):
    with pytest.raises(ValidationError):
        Config(**settings)


def test_filter_relationship_does_not_apply_when_bandpass_is_disabled():
    config = Config(digital_filter=False, cutoffFreqLow=80, cutoffFreqHigh=40)
    assert not config.digital_filter


def test_valid_custom_settings_roundtrip_as_json(tmp_path):
    settings = Config().feature_categories.copy()
    settings["Peak_Center"] = True
    config = Config(segments_length=8, segments_overlap=4, feature_categories=settings)
    path = tmp_path / "config.json"
    config.save(path)
    loaded = Config.load(path)
    assert loaded.segments_length == 8
    assert loaded.segments_overlap == 4
    assert loaded.feature_categories["Peak_Center"] is True


def test_irasa_config_retains_bandwidth_needed_for_resampling():
    config = Config(psd_parametrization_method="irasa", cutoffFreqHigh=80)
    assert config.cutoffFreqHigh == 80
