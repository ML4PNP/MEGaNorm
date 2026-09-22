import numpy as np
import pytest


@pytest.fixture
def freqs():
    """Integer-spaced frequencies make hand-derived integrals transparent."""
    return np.arange(0.0, 21.0)


@pytest.fixture
def simple_psd(freqs):
    """A linear PSD for which trapezoidal band powers are exact."""
    return freqs.copy()


@pytest.fixture
def zero_psd(freqs):
    return np.zeros_like(freqs)


@pytest.fixture
def synthetic_alpha_psd(freqs):
    """Positive background with a clear dominant bin at 10 Hz."""
    psd = np.ones_like(freqs)
    psd[freqs == 8] = 3.0
    psd[freqs == 10] = 8.0
    return psd


@pytest.fixture
def channel_names():
    return ["MEG001", "MEG002", "MEG003", "MEG004"]


@pytest.fixture
def freq_bands():
    return {
        "Broadband": (1.0, 20.0),
        "Theta": (4.0, 7.0),
        "Alpha": (8.0, 12.0),
        "Beta": (13.0, 20.0),
    }


@pytest.fixture
def individualized_band_ranges():
    return {"Alpha": (-2.0, 2.0)}
