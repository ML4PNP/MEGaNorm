from types import SimpleNamespace

import mne
import numpy as np
import pytest

from meganorm.src.psdParameterize import (
    _irasa_welch_kwargs,
    computePsd,
    computePsdIrasa,
    fooof,
    irasa_epochs,
    parameterize_psds,
)


def make_epochs(
    *, sfreq=100.0, duration=4.0, n_epochs=3, n_channels=2, epoch_amplitudes=None
):
    times = np.arange(int(sfreq * duration)) / sfreq
    data = np.empty((n_epochs, n_channels, len(times)))
    if epoch_amplitudes is None:
        epoch_amplitudes = np.ones(n_epochs)
    for epoch in range(n_epochs):
        for channel in range(n_channels):
            data[epoch, channel] = (
                epoch_amplitudes[epoch] * (channel + 1) * np.sin(2 * np.pi * 10 * times)
            )
    info = mne.create_info(
        [f"EEG{channel:03d}" for channel in range(n_channels)], sfreq, "eeg"
    )
    return mne.EpochsArray(data, info, verbose=False)


@pytest.mark.unit
def test_irasa_welch_kwargs_uses_two_second_window_and_half_overlap():
    result = _irasa_welch_kwargs(100.0, 1000, (1.05, 2.0, 0.05))

    assert result == {"nperseg": 200, "noverlap": 100, "nfft": 512}


@pytest.mark.unit
def test_irasa_welch_kwargs_shortens_window_for_short_epochs(caplog):
    result = _irasa_welch_kwargs(100.0, 300, (1.05, 2.0, 0.05))

    assert result == {"nperseg": 150, "noverlap": 75, "nfft": 512}
    assert "Epochs too short" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("method", ["periodogram", "invalid"])
def test_parameterize_psds_rejects_unknown_psd_method(method):
    with pytest.raises(ValueError, match="psd_method"):
        parameterize_psds(None, "fooof", psd_method=method)


@pytest.mark.unit
def test_parameterize_psds_rejects_unknown_aperiodic_mode():
    with pytest.raises(ValueError, match="aperiodic_mode"):
        parameterize_psds(None, "fooof", aperiodic_mode="none")


@pytest.mark.unit
def test_parameterize_psds_rejects_unknown_parametrization_method():
    with pytest.raises(ValueError, match="parametrization_method"):
        parameterize_psds(None, "unknown")


@pytest.mark.unit
def test_irasa_epochs_rejects_non_epochs_input():
    with pytest.raises(TypeError, match="mne.BaseEpochs"):
        irasa_epochs(np.zeros((2, 3, 100)))


@pytest.mark.unit
def test_irasa_epochs_rejects_epochs_with_bad_channels():
    epochs = make_epochs()
    epochs.info["bads"] = [epochs.ch_names[0]]

    with pytest.raises(ValueError, match="bad channels"):
        irasa_epochs(epochs)


@pytest.mark.integration
def test_compute_psd_recovers_known_ten_hz_signal():
    epochs = make_epochs()

    psds, freqs = computePsd(
        epochs,
        freq_range_low=3,
        freq_range_high=20,
        sampling_rate=100,
        psd_n_overlap=1,
        psd_n_fft=2,
        n_per_seg=2,
    )

    assert psds.shape == (2, len(freqs))
    assert freqs[np.argmax(psds[0])] == pytest.approx(10.0, abs=0.1)
    assert freqs[0] >= 3 and freqs[-1] <= 20


@pytest.mark.integration
def test_compute_psd_multitaper_recovers_known_ten_hz_signal():
    psds, freqs = computePsd(
        make_epochs(),
        freq_range_low=3,
        freq_range_high=20,
        sampling_rate=100,
        psd_method="multitaper",
    )

    assert psds.shape == (2, len(freqs))
    assert freqs[np.argmax(psds[0])] == pytest.approx(10.0, abs=0.5)


@pytest.mark.integration
def test_compute_psd_irasa_recovers_peak_and_masks_frequency_range():
    epochs = make_epochs()

    psds, freqs = computePsdIrasa(epochs, freq_range_low=5, freq_range_high=15)

    assert psds.shape == (2, len(freqs))
    assert freqs[np.argmax(psds[0])] == pytest.approx(10.0, abs=0.1)
    assert freqs[0] >= 5 and freqs[-1] <= 15


@pytest.mark.integration
@pytest.mark.parametrize("calculator", [computePsd, computePsdIrasa])
def test_psd_helpers_average_power_across_nonidentical_epochs(calculator):
    combined = make_epochs(
        n_epochs=2, n_channels=1, epoch_amplitudes=np.array([1.0, 3.0])
    )
    first = make_epochs(n_epochs=1, n_channels=1, epoch_amplitudes=np.array([1.0]))
    second = make_epochs(n_epochs=1, n_channels=1, epoch_amplitudes=np.array([3.0]))
    kwargs = {"freq_range_low": 5, "freq_range_high": 15}
    if calculator is computePsd:
        kwargs.update(sampling_rate=100, psd_n_overlap=1, psd_n_fft=2, n_per_seg=2)

    combined_psd, combined_freqs = calculator(combined, **kwargs)
    first_psd, first_freqs = calculator(first, **kwargs)
    second_psd, second_freqs = calculator(second, **kwargs)

    np.testing.assert_array_equal(combined_freqs, first_freqs)
    np.testing.assert_array_equal(combined_freqs, second_freqs)
    np.testing.assert_allclose(combined_psd, (first_psd + second_psd) / 2)


@pytest.mark.integration
def test_fooof_fits_each_channel_and_recovers_alpha_peak():
    freqs = np.arange(2.0, 25.5, 0.5)
    background = 1.0 / freqs
    alpha = 0.8 * np.exp(-0.5 * ((freqs - 10.0) / 0.8) ** 2)
    psds = np.vstack([background + alpha, 2 * (background + alpha)])

    models, returned_psds, returned_freqs = fooof(
        psds, freqs, freq_range_low=2, freq_range_high=25
    )

    assert len(models) == 2
    np.testing.assert_array_equal(returned_psds, psds)
    np.testing.assert_array_equal(returned_freqs, freqs)
    for channel in range(2):
        alpha_peaks = models.get_fooof(channel).get_params("peak_params")
        assert np.any(np.abs(alpha_peaks[:, 0] - 10.0) <= 0.5)


@pytest.mark.integration
@pytest.mark.slow
def test_compute_psd_irasa_matches_real_irasa_frequency_grid_and_decomposition():
    epochs = make_epochs(n_epochs=1, n_channels=1)
    hset = (1.05, 1.5, 0.1)

    comparison_psd, comparison_freqs = computePsdIrasa(
        epochs, hset_info=hset, freq_range_low=5, freq_range_high=15
    )
    raw, irasa_freqs, result = irasa_epochs(epochs, band=(5, 15), hset_info=hset)

    np.testing.assert_array_equal(comparison_freqs, irasa_freqs)
    # Small edge/padding differences are expected between SciPy's direct Welch
    # call and PYRASA's internal calculation; agreement is assessed relative
    # to the maximum spectral power rather than near-zero bins.
    np.testing.assert_allclose(comparison_psd, raw, rtol=0.01, atol=0.01 * np.max(raw))
    reconstructed = result.aperiodic.get_data().squeeze(
        axis=0
    ) + result.periodic.get_data().squeeze(axis=0)
    np.testing.assert_allclose(raw, reconstructed, rtol=1e-10, atol=1e-12)


@pytest.mark.unit
def test_irasa_epochs_averages_epochs_and_preserves_channel_metadata(monkeypatch):
    epochs = make_epochs(n_epochs=2)
    call = {"epoch": 0}

    def fake_irasa(epoch, **kwargs):
        call["epoch"] += 1
        value = float(call["epoch"])
        shape = (epoch.shape[0], 4)
        return SimpleNamespace(
            raw_spectrum=np.full(shape, 10 * value),
            aperiodic=np.full(shape, 3 * value),
            periodic=np.full(shape, 7 * value),
            freqs=np.array([1.0, 2.0, 3.0, 4.0]),
        )

    monkeypatch.setattr("meganorm.src.psdParameterize.irasa", fake_irasa)

    raw, freqs, result = irasa_epochs(epochs, band=(1.0, 4.0))

    np.testing.assert_allclose(raw, 15.0)
    np.testing.assert_allclose(result.aperiodic.get_data(), 4.5)
    np.testing.assert_allclose(result.periodic.get_data(), 10.5)
    assert result.periodic.ch_names == epochs.ch_names
    np.testing.assert_array_equal(freqs, [1.0, 2.0, 3.0, 4.0])


@pytest.mark.integration
def test_parameterize_psds_fooof_path_returns_consistent_shapes():
    epochs = make_epochs()

    models, psds, freqs = parameterize_psds(
        epochs,
        "fooof",
        freq_range_low=3,
        freq_range_high=20,
        sampling_rate=100,
        psd_n_overlap=1,
        psd_n_fft=2,
        n_per_seg=2,
        aperiodic_mode="fixed",
    )

    assert len(models) == len(epochs.ch_names)
    assert psds.shape == (len(epochs.ch_names), len(freqs))


@pytest.mark.unit
def test_parameterize_psds_fooof_forwards_nondefault_configuration(monkeypatch):
    expected_psds = np.ones((2, 3))
    expected_freqs = np.array([4.0, 5.0, 6.0])
    expected_models = object()

    def fake_compute_psd(**kwargs):
        assert kwargs == {
            "segments": "epochs",
            "freq_range_low": 4,
            "freq_range_high": 31,
            "sampling_rate": 200,
            "psd_method": "welch",
            "psd_n_overlap": 2,
            "psd_n_fft": 4,
            "n_per_seg": 3,
        }
        return expected_psds, expected_freqs

    def fake_fooof(**kwargs):
        assert kwargs == {
            "psds": expected_psds,
            "freqs": expected_freqs,
            "freq_range_low": 4,
            "freq_range_high": 31,
            "min_peak_height": 0.2,
            "peak_threshold": 3.0,
            "peak_width_limits": (2.0, 8.0),
            "aperiodic_mode": "fixed",
        }
        return expected_models, expected_psds, expected_freqs

    monkeypatch.setattr("meganorm.src.psdParameterize.computePsd", fake_compute_psd)
    monkeypatch.setattr("meganorm.src.psdParameterize.fooof", fake_fooof)

    result = parameterize_psds(
        "epochs",
        "fooof",
        freq_range_low=4,
        freq_range_high=31,
        min_peak_height=0.2,
        peak_threshold=3.0,
        sampling_rate=200,
        psd_method="welch",
        psd_n_overlap=2,
        psd_n_fft=4,
        n_per_seg=3,
        peak_width_limits=(2.0, 8.0),
        aperiodic_mode="fixed",
    )

    assert result == (expected_models, expected_psds, expected_freqs)


@pytest.mark.unit
def test_parameterize_psds_irasa_path_preserves_return_order(monkeypatch):
    expected_models = SimpleNamespace(
        periodic=SimpleNamespace(get_data=lambda: np.ones((1, 2, 3)))
    )
    expected_psds = np.ones((2, 3))
    expected_freqs = np.array([1.0, 2.0, 3.0])

    def fake_irasa_epochs(segments, band, hset_info):
        assert segments is None
        assert band == (6, 22)
        assert hset_info == (1.1, 1.8, 0.1)
        return expected_psds, expected_freqs, expected_models

    monkeypatch.setattr("meganorm.src.psdParameterize.irasa_epochs", fake_irasa_epochs)

    models, psds, freqs = parameterize_psds(
        None,
        "irasa",
        freq_range_low=6,
        freq_range_high=22,
        irasa_hset=(1.1, 1.8, 0.1),
    )

    assert models is expected_models
    assert psds is expected_psds
    assert freqs is expected_freqs


@pytest.mark.unit
def test_parameterize_psds_rejects_irasa_raw_frequency_mismatch(monkeypatch):
    models = SimpleNamespace(
        periodic=SimpleNamespace(get_data=lambda: np.ones((1, 2, 3)))
    )
    monkeypatch.setattr(
        "meganorm.src.psdParameterize.irasa_epochs",
        lambda *args, **kwargs: (np.ones((2, 4)), np.arange(3.0), models),
    )

    with pytest.raises(ValueError, match="raw spectrum"):
        parameterize_psds(None, "irasa")


@pytest.mark.unit
def test_parameterize_psds_rejects_irasa_periodic_frequency_mismatch(monkeypatch):
    models = SimpleNamespace(
        periodic=SimpleNamespace(get_data=lambda: np.ones((1, 2, 4)))
    )
    monkeypatch.setattr(
        "meganorm.src.psdParameterize.irasa_epochs",
        lambda *args, **kwargs: (np.ones((2, 3)), np.arange(3.0), models),
    )

    with pytest.raises(ValueError, match="periodic"):
        parameterize_psds(None, "irasa")
