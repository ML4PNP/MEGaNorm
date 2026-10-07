import json
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest
from pyrasa.irasa_mne.mne_objs import AperiodicEpochsSpectrum

from meganorm.src.featureExtraction import (
    SpecParamDecomposer,
    PYRASADecomposer,
    _average_aperiodic,
    add_feature,
    abs_canonical_power,
    abs_individual_power,
    average_peaks_across_epochs,
    band_power_ratio,
    compute_hemispheric_asymmetry,
    create_feature_container,
    rel_canonical_power,
    rel_individual_power,
    summarizeFeatures,
)

pytestmark = pytest.mark.unit

def test_abs_canonical_power_integrates_only_inclusive_band(simple_psd, freqs):
    # Integral of y=x from 2 through 4 Hz is 6; returned on natural-log scale.
    assert abs_canonical_power(simple_psd, freqs, 2.0, 4.0) == pytest.approx(np.log(6.0))


def test_rel_canonical_power_is_band_fraction(simple_psd, freqs):
    # Band integral is 6 and total integral from 0 through 20 Hz is 200.
    assert rel_canonical_power(simple_psd, freqs, 2.0, 4.0) == pytest.approx(0.03)


def test_rel_canonical_power_returns_nan_for_zero_total(zero_psd, freqs):
    assert np.isnan(rel_canonical_power(zero_psd, freqs, 8.0, 12.0))


def test_band_power_ratio_returns_log_ratio(freqs):
    psd = np.where(freqs <= 4.0, 2.0, 4.0)

    # 0--4 Hz has area 8; 5--9 Hz has area 16. ln(8/16) = ln(0.5).
    result = band_power_ratio(psd, freqs, 0.0, 4.0, 5.0, 9.0)

    assert result == pytest.approx(np.log(0.5))


def test_abs_canonical_power_returns_nan_for_zero_power(zero_psd, freqs):
    assert np.isnan(abs_canonical_power(zero_psd, freqs, 8.0, 12.0))


def test_abs_individual_power_returns_nan_for_zero_power(
    zero_psd, freqs, individualized_band_ranges
):
    peaks = [(10.0, 7.0, 1.5)]
    assert np.isnan(
        abs_individual_power(zero_psd, freqs, peaks, individualized_band_ranges, "Alpha")
    )


def test_band_power_ratio_returns_nan_for_zero_numerator(freqs):
    psd = np.where(freqs <= 4.0, 0.0, 4.0)
    assert np.isnan(band_power_ratio(psd, freqs, 0.0, 4.0, 5.0, 9.0))


def test_band_power_ratio_returns_nan_for_zero_denominator(freqs):
    psd = np.where(freqs <= 4.0, 2.0, 0.0)

    assert np.isnan(band_power_ratio(psd, freqs, 0.0, 4.0, 5.0, 9.0))


def test_abs_individual_power_uses_highest_power_peak(
    synthetic_alpha_psd, freqs, individualized_band_ranges
):
    peaks = [(8.0, 2.0, 1.0), (10.0, 7.0, 1.5)]

    result = abs_individual_power(
        synthetic_alpha_psd, freqs, peaks, individualized_band_ranges, "Alpha"
    )

    # Dominant 10-Hz peak selects 8--12 Hz: trapz([3, 1, 8, 1, 1]) = 12 -> ln(12).
    assert result == pytest.approx(np.log(12.0))


def test_individual_power_respects_asymmetric_offsets(freqs):
    psd = np.ones_like(freqs)
    ranges = {"Alpha": (-1.0, 3.0)}
    peaks = [(10.0, 5.0, 1.0)]

    assert abs_individual_power(psd, freqs, peaks, ranges, "Alpha") == pytest.approx(np.log(4.0))


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

class FakeSpecParamData:
    def get_data(self, component, space):
        assert component == "peak"
        assert space == "linear"
        return np.array([10.0, 92.0, 903.0])


class FakeSpecParamFit:
    def __init__(self):
        self.data = FakeSpecParamData()

        periodic_params = self.get_params("periodic")
        n_peaks = np.sum(~np.isnan(periodic_params).any(axis=1))

        self.results = SimpleNamespace(
            n_peaks=n_peaks,
            metrics=SimpleNamespace(results={"gof_rsquared": 0.91}),
        )

    def get_params(self, name):
        if name == "aperiodic":
            return np.array([2.0, 4.0, 6.0])
        if name == "periodic":
            return np.array(
                [
                    [8.0, 1.0, 2.0],
                    [10.0, 5.0, 1.5],
                    [14.0, 9.0, 3.0],
                    [np.nan, 7.0, 1.0],
                ]
            )
        raise KeyError(name)


class FakeSpecParamGroup:
    def __init__(self):
        self.fit = FakeSpecParamFit()

    def get_model(self, ind):
        assert ind == 0
        return self.fit


@pytest.mark.parametrize(
    "mode, expected", [("fixed", [2.0, 4.0]), ("knee", [2.0, 6.0])]
)
def test_specparam_decomposer_reorders_aperiodic_parameters(mode, expected):
    decomposer = SpecParamDecomposer(FakeSpecParamGroup(), mode=mode, ch_num=0)

    assert decomposer.get_aperiodic_params() == expected


def test_specparam_decomposer_rejects_unknown_mode():
    decomposer = SpecParamDecomposer(FakeSpecParamGroup(), mode="invalid", ch_num=0)

    with pytest.raises(ValueError, match="Unknown aperiodic_mode"):
        decomposer.get_aperiodic_params()


def test_specparam_decomposer_returns_periodic_spectrum_and_fit_quality():
    decomposer = SpecParamDecomposer(FakeSpecParamGroup(), mode="fixed", ch_num=0)

    periodic = decomposer.get_periodic_spectrum()

    np.testing.assert_allclose(periodic, [10.0, 92.0, 903.0])
    assert decomposer.get_r_squared() == 0.91


def test_specparam_decomposer_filters_band_and_selects_strongest_valid_peak():
    decomposer = SpecParamDecomposer(FakeSpecParamGroup(), mode="fixed", ch_num=0)

    dominant, peaks = decomposer.get_peak_params(7.0, 12.0)

    np.testing.assert_allclose(dominant, [10.0, 5.0, 1.5])
    assert len(peaks) == 2


def test_specparam_decomposer_returns_no_peak_when_band_is_empty():
    decomposer = SpecParamDecomposer(FakeSpecParamGroup(), mode="fixed", ch_num=0)

    assert decomposer.get_peak_params(30.0, 40.0) == (None, None)


class FakePeriodic:
    """Stand-in for the per-epoch periodic spectrum: (n_epochs=1, n_channels=2, n_freqs=3)."""

    def get_data(self):
        return np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])


def make_pyrasa_decomposer(mode="fixed", band_peaks_avg=None):
    if band_peaks_avg is None:
        # Epoch-averaged peaks per band, as produced by average_peaks_across_epochs
        band_peaks_avg = {
            "Alpha": pd.DataFrame(
                {"cf": [9.0, 10.0], "pw": [10.0, 7.0], "bw": [1.0, 2.0]},
                index=pd.Index(["MEG001", "MEG002"], name="ch_name"),
            )
        }
    model = SimpleNamespace(periodic=FakePeriodic())
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
    return PYRASADecomposer(model, mode, "MEG002", 1, aperiodic, band_peaks_avg)


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
        band_peaks_avg={},
    )

    np.testing.assert_allclose(decomposer.get_periodic_spectrum(), [1.0, 2.0, 3.0])


def test_pyrasa_decomposer_averages_periodic_spectrum_over_epochs():
    # Two epochs, one channel: mean of [1, 2, 3] and [3, 4, 5]
    model = SimpleNamespace(
        periodic=SimpleNamespace(
            get_data=lambda: np.array([[[1.0, 2.0, 3.0]], [[3.0, 4.0, 5.0]]])
        )
    )
    decomposer = PYRASADecomposer(model, "fixed", "MEG001", 0, None, {})

    np.testing.assert_allclose(decomposer.get_periodic_spectrum(), [2.0, 3.0, 4.0])


def test_pyrasa_decomposer_returns_epoch_averaged_peak_for_its_channel():
    dominant, peaks = make_pyrasa_decomposer().get_peak_params(
        8.0, 12.0, band_name="Alpha"
    )

    assert dominant == (10.0, 7.0, 2.0)
    assert peaks == [(10.0, 7.0, 2.0)]


def test_pyrasa_decomposer_returns_no_peak_when_channel_has_none():
    band_peaks_avg = {
        "Alpha": pd.DataFrame(
            {"cf": [9.0], "pw": [2.0], "bw": [1.0]},
            index=pd.Index(["MEG001"], name="ch_name"),
        )
    }
    decomposer = make_pyrasa_decomposer(band_peaks_avg=band_peaks_avg)

    assert decomposer.get_peak_params(8.0, 12.0, band_name="Alpha") == (None, None)


@pytest.mark.parametrize("band_peaks_avg", [{}, {"Alpha": None}])
def test_pyrasa_decomposer_returns_no_peak_when_band_unavailable(band_peaks_avg):
    # {} = band never computed; None = peak detection failed in every epoch
    decomposer = make_pyrasa_decomposer(band_peaks_avg=band_peaks_avg)

    assert decomposer.get_peak_params(8.0, 12.0, band_name="Alpha") == (None, None)

class FakeEpoch:
    """One epoch of a PeriodicEpochsSpectrum; only get_peaks is needed."""

    def __init__(self, peaks=None, error=None):
        self._peaks = peaks
        self._error = error
        self.get_peaks_kwargs = None

    def get_peaks(self, **kwargs):
        self.get_peaks_kwargs = kwargs
        if self._error:
            raise self._error
        return self._peaks.copy()


class FakeEpochsPeriodic:
    """Indexable stand-in for PeriodicEpochsSpectrum: periodic[i] -> one epoch."""

    def __init__(self, epochs, ch_names=("MEG001", "MEG002")):
        self._epochs = epochs
        self.ch_names = list(ch_names)

    def __len__(self):
        return len(self._epochs)

    def __getitem__(self, idx):
        return self._epochs[idx]


def peaks_table(rows):
    """Build a get_peaks()-style DataFrame from (ch_name, cf, bw, pw) rows."""
    return pd.DataFrame(rows, columns=["ch_name", "cf", "bw", "pw"])


def test_average_peaks_keeps_strongest_per_epoch_then_averages_across_epochs():
    epoch_0 = peaks_table(
        [
            ("MEG001", 9.0, 1.0, 2.0),   # weaker peak, ignored
            ("MEG001", 11.0, 2.0, 6.0),  # strongest in epoch 0
            ("MEG002", 10.0, 1.0, 4.0),
        ]
    )
    epoch_1 = peaks_table(
        [
            ("MEG001", 10.0, 1.0, 8.0),  # strongest in epoch 1
        ]
    )
    periodic = FakeEpochsPeriodic([FakeEpoch(epoch_0), FakeEpoch(epoch_1)])

    result = average_peaks_across_epochs(periodic, 8.0, 12.0)

    # MEG001: strongest per epoch is (11, 6, 2) and (10, 8, 1); pw averaged in ln space
    assert result.loc["MEG001"].to_dict() == pytest.approx(
        {"cf": 10.5, "pw": (np.log(6.0) + np.log(8.0)) / 2, "bw": 1.5}
    )
    # MEG002: only epoch 0 has a peak
    assert result.loc["MEG002"].to_dict() == pytest.approx(
        {"cf": 10.0, "pw": np.log(4.0), "bw": 1.0}
    )


def test_average_peaks_ignores_out_of_band_and_nan_peaks():
    epoch_0 = peaks_table(
        [
            ("MEG001", 10.0, 1.0, 3.0),
            ("MEG001", 20.0, 1.0, 100.0),        # outside band, must not win
            ("MEG002", np.nan, np.nan, np.nan),  # no peak for this channel
        ]
    )
    periodic = FakeEpochsPeriodic([FakeEpoch(epoch_0)])

    result = average_peaks_across_epochs(periodic, 8.0, 12.0)

    # Every channel is reported; MEG002 had no valid peak, so it's all NaN
    assert result.index.tolist() == ["MEG001", "MEG002"]
    assert result.loc["MEG001"].to_dict() == pytest.approx(
        {"cf": 10.0, "pw": np.log(3.0), "bw": 1.0}
    )
    assert result.loc["MEG002"].isna().all()


def test_average_peaks_passes_padded_band_to_get_peaks():
    epoch = FakeEpoch(peaks_table([("MEG001", 10.0, 1.0, 3.0)]))

    average_peaks_across_epochs(FakeEpochsPeriodic([epoch]), 8.0, 12.0)

    assert epoch.get_peaks_kwargs["cut_spectrum"] == (7.0, 13.0)


def test_average_peaks_skips_failed_epochs():
    good = FakeEpoch(peaks_table([("MEG001", 10.0, 1.0, 3.0)]))
    bad = FakeEpoch(error=ValueError("no stable fit"))
    periodic = FakeEpochsPeriodic([bad, good])

    result = average_peaks_across_epochs(periodic, 8.0, 12.0)

    assert result.loc["MEG001"].to_dict() == pytest.approx(
        {"cf": 10.0, "pw": np.log(3.0), "bw": 1.0}
    )


def test_average_peaks_returns_none_when_every_epoch_fails():
    periodic = FakeEpochsPeriodic(
        [FakeEpoch(error=ValueError("no stable fit")) for _ in range(2)]
    )

    assert average_peaks_across_epochs(periodic, 8.0, 12.0) is None

def test_average_aperiodic_collapses_epochs_into_single_epoch():
    info = mne.create_info(["MEG001", "MEG002"], 100.0, "mag")
    # Two epochs, 2 channels, 4 freqs: epoch 1 = 1, epoch 2 = 3
    data = np.stack([np.ones((2, 4)), 3 * np.ones((2, 4))])
    aperiodic = AperiodicEpochsSpectrum(
        data,
        info,
        freqs=np.array([1.0, 2.0, 3.0, 4.0]),
        events=np.array([[0, 0, 1], [1, 0, 1]]),
        event_id={"1": 1},
    )

    result = _average_aperiodic(aperiodic)

    assert isinstance(result, AperiodicEpochsSpectrum)
    assert result.get_data().shape == (1, 2, 4)
    np.testing.assert_allclose(result.get_data(), 2.0)
    assert result.ch_names == ["MEG001", "MEG002"]
    np.testing.assert_array_equal(result.freqs, [1.0, 2.0, 3.0, 4.0])