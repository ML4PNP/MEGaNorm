"""Synthetic EEG for the ERP tutorial and tests; no file downloads."""

import mne
import numpy as np


def make_demo_recording(seed=42):
    """Return deterministic synthetic EEG Raw and absolute-sample events.

    Two sensors (Cz, Pz), 250 Hz, first_samp=500 and 40 alternating trials
    (codes 1=standard, 2=target). Negative peaks occur at 100 ms; positive
    peaks at 300 ms have Pz amplitudes 3/6 microvolts by condition. A small
    seeded noise term is added. Used by tests and the source-tree tutorial.
    """
    rng = np.random.default_rng(seed)
    sfreq = 250.0
    first_samp = 500
    onsets = 1.0 + np.arange(40) * 1.4
    n_times = int((onsets[-1] + 1.5) * sfreq)
    data = 2e-6 + rng.normal(0, 0.15e-6, (2, n_times))
    relative_times = np.arange(-50, 201) / sfreq
    negative = np.exp(-0.5 * ((relative_times - 0.1) / 0.015) ** 2)
    positive = np.exp(-0.5 * ((relative_times - 0.3) / 0.04) ** 2)
    events = []
    for trial, onset in enumerate(onsets):
        sample = int(round(onset * sfreq))
        event_code = 1 if trial % 2 == 0 else 2
        amplitude = 3e-6 if event_code == 1 else 6e-6
        waveform = np.vstack(
            [
                -4e-6 * negative + 0.5 * amplitude * positive,
                -2e-6 * negative + amplitude * positive,
            ]
        )
        data[:, sample - 50 : sample + 201] += waveform
        events.append([sample + first_samp, 0, event_code])
    info = mne.create_info(["Cz", "Pz"], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, first_samp=first_samp, verbose=False)
    raw.set_montage("standard_1020")
    return raw, np.asarray(events, dtype=int)
