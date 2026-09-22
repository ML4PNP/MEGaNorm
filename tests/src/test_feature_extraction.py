import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from meganorm.src.featureExtraction import (
    FOOOFDecomposer,
    PYRASADecomposer,
    add_feature,
    abs_canonical_power,
    abs_individual_power,
    band_power_ratio,
    compute_hemispheric_asymmetry,
    create_feature_container,
    rel_canonical_power,
    rel_individual_power,
    summarizeFeatures,
)

pytestmark = pytest.mark.unit


def test_abs_canonical_power_integrates_only_inclusive_band(simple_psd, freqs):
    # Integral of y=x from 2 through 4 Hz is (4**2 - 2**2) / 2 = 6.
    assert abs_canonical_power(simple_psd, freqs, 2.0, 4.0) == pytest.approx(6.0)


def test_rel_canonical_power_is_band_fraction(simple_psd, freqs):
    # Band integral is 6 and total integral from 0 through 20 Hz is 200.
    assert rel_canonical_power(simple_psd, freqs, 2.0, 4.0) == pytest.approx(0.03)


def test_rel_canonical_power_returns_nan_for_zero_total(zero_psd, freqs):
    assert np.isnan(rel_canonical_power(zero_psd, freqs, 8.0, 12.0))


def test_band_power_ratio_returns_raw_ratio(freqs):
    psd = np.where(freqs <= 4.0, 2.0, 4.0)

    # 0--4 Hz has area 8; 5--9 Hz has area 16. The raw ratio is 0.5.
    result = band_power_ratio(psd, freqs, 0.0, 4.0, 5.0, 9.0)

    assert result == pytest.approx(0.5)


def test_band_power_ratio_returns_nan_for_zero_denominator(freqs):
    psd = np.where(freqs <= 4.0, 2.0, 0.0)

    assert np.isnan(band_power_ratio(psd, freqs, 0.0, 4.0, 5.0, 9.0))


def test_abs_individual_power_uses_highest_power_peak(
    synthetic_alpha_psd, freqs, individualized_band_ranges
):
    peaks = [(8.0, 2.0, 1.0), (10.0, 7.0, 1.5)]

    result = abs_individual_power(
        synthetic_alpha_psd,
        freqs,
        peaks,
        individualized_band_ranges,
        "Alpha",
    )

    # Dominant 10-Hz peak selects 8--12 Hz: trapz([3, 1, 8, 1, 1]) = 12.
    assert result == pytest.approx(12.0)


def test_individual_power_respects_asymmetric_offsets(freqs):
    psd = np.ones_like(freqs)
    ranges = {"Alpha": (-1.0, 3.0)}
    peaks = [(10.0, 5.0, 1.0)]

    assert abs_individual_power(psd, freqs, peaks, ranges, "Alpha") == 4.0


@pytest.mark.parametrize(
    "peaks, ranges", [([], {"Alpha": (-2, 2)}), ([(10, 5, 1)], {})]
)
def test_abs_individual_power_returns_nan_for_missing_definition(
    synthetic_alpha_psd, freqs, peaks, ranges
):
    assert np.isnan(
        abs_individual_power(synthetic_alpha_psd, freqs, peaks, ranges, "Alpha")
    )


def test_rel_individual_power_is_fraction_of_total(
    synthetic_alpha_psd, freqs, individualized_band_ranges
):
    peaks = [(8.0, 2.0, 1.0), (10.0, 7.0, 1.5)]

    result = rel_individual_power(
        synthetic_alpha_psd,
        freqs,
        peaks,
        individualized_band_ranges,
        "Alpha",
    )

    # Selected-band area is 12; total area is 29 for this fixture.
    assert result == pytest.approx(12.0 / 29.0)


def test_rel_individual_power_returns_nan_for_zero_total(
    zero_psd, freqs, individualized_band_ranges
):
    peaks = [(10.0, 7.0, 1.5)]

    assert np.isnan(
        rel_individual_power(
            zero_psd, freqs, peaks, individualized_band_ranges, "Alpha"
        )
    )


def test_create_feature_container_encodes_enabled_schema(freq_bands):
    categories = {
        "Offset": True,
        "Adjusted_Canonical_Absolute_Power": True,
        "Adjusted_Canonical_Relative_Power": True,
        "Adjusted_Band_Ratio": True,
        "Disabled_Feature": False,
        "Hemispheric_Asymmetry_index": True,
    }
    ratios = [SimpleNamespace(numerator="Theta", denominator="Alpha")]

    result = create_feature_container(
        categories, freq_bands, ["MEG002", "MEG001"], ratios
    )

    assert result.columns.tolist() == ["MEG002", "MEG001"]
    assert result.index.tolist() == [
        "Offset__",
        "Adjusted_Canonical_Absolute_Power__Broadband",
        "Adjusted_Canonical_Absolute_Power__Theta",
        "Adjusted_Canonical_Absolute_Power__Alpha",
        "Adjusted_Canonical_Absolute_Power__Beta",
        "Adjusted_Canonical_Relative_Power__Theta",
        "Adjusted_Canonical_Relative_Power__Alpha",
        "Adjusted_Canonical_Relative_Power__Beta",
        "Adjusted_Band_Ratio__Theta_over_Alpha",
    ]


def test_create_feature_container_omits_ratio_with_unknown_band(freq_bands):
    categories = {"Adjusted_Band_Ratio": True}
    ratios = [SimpleNamespace(numerator="Delta", denominator="Alpha")]

    result = create_feature_container(categories, freq_bands, ["MEG001"], ratios)

    assert result.empty


def test_add_feature_assigns_requested_feature_channel_cell():
    container = pd.DataFrame(
        index=["Peak_Power__Alpha"], columns=["MEG001", "MEG002"], dtype=float
    )

    result = add_feature(container, 3.25, "Peak_Power", "MEG002", "Alpha")

    assert result.at["Peak_Power__Alpha", "MEG002"] == 3.25
    assert pd.isna(result.at["Peak_Power__Alpha", "MEG001"])


def test_summarize_features_all_drops_only_all_nan_rows_and_preserves_index():
    features = pd.DataFrame(
        {"MEG001": [1.0, np.nan, 5.0], "MEG002": [3.0, np.nan, np.nan]},
        index=["feature_a", "empty", "feature_b"],
    )
    original = features.copy(deep=True)

    result = summarizeFeatures(features, "FIF", "all", {"meg": True})

    assert result.index.tolist() == ["feature_a", "feature_b"]
    assert result["all"].tolist() == [2.0, 5.0]
    pd.testing.assert_frame_equal(features, original)


def test_summarize_features_uses_custom_lobe_layout(tmp_path):
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "FIF_MEG_LOBE": {
                    "frontal": ["MEG001", "MEG002"],
                    "posterior": ["MEG003", "MEG004"],
                }
            }
        ),
        encoding="utf-8",
    )
    features = pd.DataFrame(
        [[1.0, 3.0, 10.0, 14.0]],
        index=["feature"],
        columns=["MEG001", "MEG002", "MEG003", "MEG004"],
    )

    result = summarizeFeatures(
        features, "FIF", "lobe", {"meg": True}, layout_path=str(layout_path)
    )

    assert result.loc["feature"].to_dict() == {"frontal": 2.0, "posterior": 12.0}


def test_compute_hemispheric_asymmetry_adds_left_minus_right_and_keeps_inputs():
    left = "Adjusted_Canonical_Absolute_Power__Alpha__parcel_lh_value"
    right = "Adjusted_Canonical_Absolute_Power__Alpha__parcel_rh_value"
    unrelated = "Peak_Power__Alpha__parcel_lh_value"
    original = pd.DataFrame(
        {left: [5.0], right: [3.0], unrelated: [9.0]}, index=["sub-01"]
    )

    result = compute_hemispheric_asymmetry(original)

    expected_name = (
        "Hemispheric_Asymmetry__Adjusted_Canonical_Absolute_Power"
        "__Alpha__parcel_lh_vs_rh_value"
    )
    assert result.loc["sub-01", expected_name] == 2.0
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert len(result.columns) == len(original.columns) + 1


def test_compute_hemispheric_asymmetry_honors_selected_base_features():
    first_left = "Feature_A__Alpha__parcel_lh_value"
    first_right = "Feature_A__Alpha__parcel_rh_value"
    second_left = "Feature_B__Alpha__parcel_lh_value"
    second_right = "Feature_B__Alpha__parcel_rh_value"
    data = pd.DataFrame(
        {first_left: [4.0], first_right: [1.0], second_left: [8.0], second_right: [2.0]}
    )

    result = compute_hemispheric_asymmetry(data, base_features=["Feature_B"])

    asymmetry_columns = [
        col for col in result if col.startswith("Hemispheric_Asymmetry")
    ]
    assert asymmetry_columns == [
        "Hemispheric_Asymmetry__Feature_B__Alpha__parcel_lh_vs_rh_value"
    ]
    assert result[asymmetry_columns[0]].item() == 6.0


class FakeFooofFit:
    def __init__(self):
        self._ap_fit = np.array([0.0, 1.0, 2.0])
        self.r_squared_ = 0.91

    def get_params(self, name):
        if name == "aperiodic_params":
            return np.array([2.0, 4.0, 6.0])
        if name == "peak_params":
            return np.array(
                [
                    [8.0, 1.0, 2.0],
                    [10.0, 5.0, 1.5],
                    [14.0, 9.0, 3.0],
                    [np.nan, 7.0, 1.0],
                ]
            )
        raise KeyError(name)


class FakeFooofGroup:
    def __init__(self):
        self.fit = FakeFooofFit()

    def get_fooof(self, ind):
        assert ind == 0
        return self.fit


@pytest.mark.parametrize(
    "mode, expected", [("fixed", [2.0, 4.0]), ("knee", [2.0, 6.0])]
)
def test_fooof_decomposer_reorders_aperiodic_parameters(mode, expected):
    decomposer = FOOOFDecomposer(FakeFooofGroup(), mode=mode, ch_num=0)

    assert decomposer.get_aperiodic_params() == expected


def test_fooof_decomposer_rejects_unknown_mode():
    decomposer = FOOOFDecomposer(FakeFooofGroup(), mode="invalid", ch_num=0)

    with pytest.raises(ValueError, match="Unknown aperiodic_mode"):
        decomposer.get_aperiodic_params()


def test_fooof_decomposer_returns_periodic_spectrum_and_fit_quality():
    decomposer = FOOOFDecomposer(FakeFooofGroup(), mode="fixed", ch_num=0)

    periodic = decomposer.get_periodic_spectrum(np.array([[11.0, 102.0, 1003.0]]))

    np.testing.assert_allclose(periodic, [10.0, 92.0, 903.0])
    assert decomposer.get_r_squared() == 0.91


def test_fooof_decomposer_filters_band_and_selects_strongest_valid_peak():
    decomposer = FOOOFDecomposer(FakeFooofGroup(), mode="fixed", ch_num=0)

    dominant, peaks = decomposer.get_peak_params(7.0, 12.0)

    np.testing.assert_allclose(dominant, [10.0, 5.0, 1.5])
    assert len(peaks) == 2


def test_fooof_decomposer_returns_no_peak_when_band_is_empty():
    decomposer = FOOOFDecomposer(FakeFooofGroup(), mode="fixed", ch_num=0)

    assert decomposer.get_peak_params(30.0, 40.0) == (None, None)


class FakePeriodic:
    def __init__(self, peaks=None, error=None):
        self._peaks = peaks
        self._error = error

    def get_data(self):
        return np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    def get_peaks(self, **kwargs):
        if self._error:
            raise self._error
        return self._peaks.copy()


def make_pyrasa_decomposer(mode="fixed", peaks=None, periodic_error=None):
    if peaks is None:
        peaks = pd.DataFrame(
            {
                "ch_name": ["MEG001", "MEG002", "MEG002", "MEG002"],
                "cf": [9.0, 7.5, 8.0, 10.0],
                "pw": [10.0, 20.0, 2.0, 7.0],
                "bw": [1.0, 1.0, 1.5, 2.0],
            }
        )
    model = SimpleNamespace(periodic=FakePeriodic(peaks, periodic_error))
    aperiodic = SimpleNamespace(
        aperiodic_params=pd.DataFrame(
            {
                "ch_name": ["MEG001", "MEG002"],
                "Offset": [1.0, 2.0],
                "Exponent": [3.0, 4.0],
                "Exponent_1": [5.0, 6.0],
                "Exponent_2": [7.0, 8.0],
                "Knee Frequency (Hz)": [9.0, 10.0],
            }
        ),
        gof=pd.DataFrame({"ch_name": ["MEG001", "MEG002"], "R2": [0.8, 0.95]}),
    )
    return PYRASADecomposer(model, mode, "MEG002", 1, aperiodic)


@pytest.mark.parametrize(
    "mode, expected", [("fixed", [2.0, 4.0]), ("knee", [2.0, 6.0, 8.0, 10.0])]
)
def test_pyrasa_decomposer_selects_channel_aperiodic_parameters(mode, expected):
    assert make_pyrasa_decomposer(mode).get_aperiodic_params() == expected


def test_pyrasa_decomposer_selects_channel_periodic_spectrum_and_fit_quality():
    decomposer = make_pyrasa_decomposer()

    np.testing.assert_allclose(decomposer.get_periodic_spectrum(), [4.0, 5.0, 6.0])
    assert decomposer.get_r_squared() == 0.95


def test_pyrasa_decomposer_preserves_frequency_axis_for_single_channel():
    model = SimpleNamespace(
        periodic=SimpleNamespace(get_data=lambda: np.array([[[1.0, 2.0, 3.0]]]))
    )
    decomposer = PYRASADecomposer(
        model=model,
        mode="fixed",
        ch_name="MEG001",
        ch_num=0,
        aperiodic=None,
    )

    np.testing.assert_allclose(decomposer.get_periodic_spectrum(), [1.0, 2.0, 3.0])


def test_pyrasa_decomposer_selects_strongest_peak_for_channel():
    dominant, peaks = make_pyrasa_decomposer().get_peak_params(8.0, 12.0)

    assert dominant == (10.0, 7.0, 2.0)
    assert peaks == [(8.0, 2.0, 1.5), (10.0, 7.0, 2.0)]


def test_pyrasa_decomposer_returns_no_peak_for_empty_channel_selection():
    peaks = pd.DataFrame({"ch_name": ["MEG001"], "cf": [9.0], "pw": [2.0], "bw": [1.0]})

    assert make_pyrasa_decomposer(peaks=peaks).get_peak_params(7.0, 12.0) == (
        None,
        None,
    )


def test_pyrasa_decomposer_handles_peak_detection_failure():
    decomposer = make_pyrasa_decomposer(periodic_error=ValueError("no stable fit"))

    assert decomposer.get_peak_params(7.0, 12.0) == (None, None)
