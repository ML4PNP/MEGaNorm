import numpy as np
import pytest
from fooof import FOOOFGroup

from meganorm.src.featureExtraction import feature_extract

pytestmark = pytest.mark.integration


def test_feature_extract_consumes_fitted_fooof_group_and_recovers_alpha_peak():
    freqs = np.arange(2.0, 25.5, 0.5)
    background = 1.0 / freqs
    alpha_peak = 0.8 * np.exp(-0.5 * ((freqs - 10.0) / 0.8) ** 2)
    psds = (background + alpha_peak)[None, :]
    models = FOOOFGroup(
        aperiodic_mode="fixed",
        max_n_peaks=3,
        peak_width_limits=(1.0, 6.0),
        verbose=False,
    )
    models.fit(freqs, psds, [2.0, 25.0])
    categories = {
        "Offset": True,
        "Exponent": True,
        "Knee_Frequency": False,
        "Peak_Center": True,
        "Peak_Power": True,
        "Peak_Width": True,
        "Adjusted_Canonical_Absolute_Power": True,
        "Adjusted_Canonical_Relative_Power": True,
        "OriginalPSD_Canonical_Absolute_Power": True,
        "OriginalPSD_Canonical_Relative_Power": True,
        "Adjusted_Individualized_Absolute_Power": False,
        "Adjusted_Individualized_Relative_Power": False,
        "OriginalPSD_Individualized_Absolute_Power": False,
        "OriginalPSD_Individualized_Relative_Power": False,
        "Adjusted_Band_Ratio": False,
        "OriginalPSD_Band_Ratio": False,
        "Hemispheric_Asymmetry_index": False,
    }

    result, aperiodic = feature_extract(
        subject_id="sub-01",
        spectral_models=models,
        psds=psds,
        feature_categories=categories,
        freqs=freqs,
        freq_bands={"Broadband": (2.0, 25.0), "Alpha": (8.0, 12.0)},
        channel_names=["MEG001"],
        individualized_band_ranges={"Alpha": (-2.0, 2.0)},
        device="FIF",
        which_layout=None,
        which_sensor={"meg": True, "eeg": False},
        aperiodic_mode="fixed",
        min_r_squared=0.0,
        power_band_ratios_list=[],
        freq_range_low=2,
        freq_range_high=25,
    )

    assert result.index.tolist() == ["sub-01"]
    assert result.loc["sub-01", "Peak_Center__Alpha__MEG001"] == pytest.approx(
        10.0, abs=0.5
    )
    assert np.isfinite(
        result.loc["sub-01", "OriginalPSD_Canonical_Absolute_Power__Alpha__MEG001"]
    )
    assert aperiodic is None
