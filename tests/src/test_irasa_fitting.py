"""Regression coverage for conditional IRASA scaling at MEG power levels."""

import logging
import mne
import numpy as np
import pandas as pd
import pytest
from pyrasa.irasa_mne.mne_objs import AperiodicEpochsSpectrum
from pyrasa.utils.types import AperiodicFit
from meganorm.src import featureExtraction as fe


def spectrum(powers):
    freqs = np.arange(3, 40.5, 0.5)
    return AperiodicEpochsSpectrum(
        np.asarray(powers)[None, :, :],
        mne.create_info([f"MEG{i}" for i in range(len(powers))], 1000, "mag"),
        freqs=freqs,
        events=np.array([[0, 0, 1]]),
        event_id={"1": 1},
    )


@pytest.mark.parametrize("mode", ["fixed", "knee"])
def test_tiny_meg_power_recovers_fit_and_original_spectrum_units(mode, caplog):
    freqs = np.arange(3, 40.5, 0.5)
    power = 1e-26 / (10 + freqs**2)
    source = spectrum([power])
    with caplog.at_level(logging.INFO, logger=fe.__name__):
        fit = fe._fit_aperiodic_with_retry(source, fit_func=mode, fit_bounds=(4, 39))
    assert fit.gof.R2.item() > 0.9
    model = fit.model
    expected = 1e-26 / (10 + model["Frequency (Hz)"].to_numpy() ** 2)
    assert np.median(model.aperiodic_model.to_numpy() / expected) == pytest.approx(
        1, rel=0.15
    )
    params = fit.aperiodic_params.iloc[0]
    x = model["Frequency (Hz)"].to_numpy()
    if mode == "fixed":
        reconstructed = 10**params.Offset / x**params.Exponent
    else:
        reconstructed = 10**params.Offset / (
            x**params.Exponent_1 * (params.Knee + x**params.Exponent_2)
        )
    np.testing.assert_allclose(reconstructed, model.aperiodic_model, rtol=1e-8, atol=0)
    assert "scale=True" in caplog.text
    np.testing.assert_array_equal(source.get_data()[0, 0], power)


@pytest.mark.parametrize("bad_score", [-800.0, np.nan, np.inf])
@pytest.mark.parametrize("ordinary_score", [0.0, 0.5])
def test_retry_preserves_other_channels_and_ordinary_low_quality_fit(
    monkeypatch, bad_score, ordinary_score
):
    freqs = np.arange(3, 40.5, 0.5)
    source = spectrum([np.full(len(freqs), 0.5)] * 3)

    def fit(self, *, scale, **kwargs):
        names = self.ch_names
        scores = {"MEG0": 0.99, "MEG1": bad_score, "MEG2": ordinary_score}
        if scale:
            assert names == ["MEG1"], "Valid unscaled channels must not be retried"
        values = [0.97 if scale else scores[name] for name in names]
        return AperiodicFit(
            aperiodic_params=pd.DataFrame(
                {"ch_name": names, "Offset": [2 if scale else 1] * len(names)}
            ),
            gof=pd.DataFrame({"ch_name": names, "R2": values}),
            model=pd.DataFrame(
                {"ch_name": names, "aperiodic_model": [1.0] * len(names)}
            ),
        )

    monkeypatch.setattr(AperiodicEpochsSpectrum, "fit_aperiodic_model", fit)
    result = fe._fit_aperiodic_with_retry(source, fit_func="fixed", fit_bounds=(4, 39))
    assert result.gof.R2.tolist() == [0.99, 0.97, ordinary_score]
    assert result.aperiodic_params.Offset.tolist() == [1, 2, 1]


def test_optimizer_exception_retries_only_affected_channel(monkeypatch, caplog):
    source = spectrum([np.full(75, 0.5)] * 2)

    def fit(self, *, scale, **kwargs):
        name = self.ch_names[0]
        if name == "MEG1" and not scale:
            raise RuntimeError("optimizer failed")
        return AperiodicFit(
            aperiodic_params=pd.DataFrame(
                {"ch_name": [name], "Offset": [2 if scale else 1]}
            ),
            gof=pd.DataFrame({"ch_name": [name], "R2": [0.95]}),
            model=pd.DataFrame({"ch_name": [name], "aperiodic_model": [1.0]}),
        )

    monkeypatch.setattr(AperiodicEpochsSpectrum, "fit_aperiodic_model", fit)
    with caplog.at_level(logging.INFO, logger=fe.__name__):
        result = fe._fit_aperiodic_with_retry(
            source, fit_func="fixed", fit_bounds=(4, 39)
        )
    assert result.aperiodic_params.Offset.tolist() == [1, 2]
    assert "optimizer failed" in caplog.text


def test_scaled_optimizer_failure_is_not_hidden(monkeypatch):
    source = spectrum([np.full(75, 0.5)])

    def fail(self, **kwargs):
        raise RuntimeError("no stable fit")

    monkeypatch.setattr(AperiodicEpochsSpectrum, "fit_aperiodic_model", fail)
    with pytest.raises(RuntimeError, match="no stable fit"):
        fe._fit_aperiodic_with_retry(source, fit_func="fixed", fit_bounds=(4, 39))


def test_scaled_fixed_offset_retains_original_log_power_units():
    freqs = np.arange(3, 40.5, 0.5)
    result = fe._fit_aperiodic_with_retry(
        spectrum([1e-26 / freqs**2]), fit_func="fixed", fit_bounds=(4, 39)
    )
    assert result.aperiodic_params.Offset.item() == pytest.approx(-26, abs=0.01)
    assert result.aperiodic_params.Exponent.item() == pytest.approx(2, abs=0.01)


@pytest.mark.parametrize("final_score", [-0.2, np.nan, np.inf, 0.5])
def test_failed_retry_still_obeys_feature_quality_gate(monkeypatch, final_score):
    from pyrasa.irasa_mne.mne_objs import IrasaEpoched, PeriodicEpochsSpectrum
    from meganorm.utils.IO import Config

    source = spectrum([np.full(75, 0.5)])
    periodic = PeriodicEpochsSpectrum(
        np.zeros((1, 1, 75)),
        source.info,
        freqs=source.freqs,
        events=source.events,
        event_id=source.event_id,
    )

    def fit(self, *, scale, **kwargs):
        return AperiodicFit(
            aperiodic_params=pd.DataFrame(
                {"ch_name": ["MEG0"], "Offset": [1.0], "Exponent": [2.0]}
            ),
            gof=pd.DataFrame(
                {"ch_name": ["MEG0"], "R2": [final_score if scale else -800.0]}
            ),
            model=pd.DataFrame({"ch_name": ["MEG0"], "aperiodic_model": [1.0]}),
        )

    monkeypatch.setattr(AperiodicEpochsSpectrum, "fit_aperiodic_model", fit)
    categories = {key: key == "Offset" for key in Config().feature_categories}
    features, _ = fe.feature_extract(
        subject_id="sub-001",
        spectral_models=IrasaEpoched(aperiodic=source, periodic=periodic),
        psds=np.full((1, 75), 0.5),
        feature_categories=categories,
        freqs=source.freqs,
        freq_bands={},
        channel_names=source.ch_names,
        individualized_band_ranges={},
        device="MEGIN",
        which_layout="all",
        which_sensor={"meg": True},
        aperiodic_mode="fixed",
        min_r_squared=0.9,
        power_band_ratios_list=[],
        freq_range_low=3,
        freq_range_high=40,
    )
    assert features.shape == (1, 0)
