import mne
import numpy as np


def _ast_get_rs_events(raw, start_channel='STI001', end_channel='STI002'):
    """Returns paired (start_sample, end_sample) for each resting block."""
    starts = mne.find_events(raw, stim_channel=start_channel, shortest_event=1)[:, 0]
    ends = mne.find_events(raw, stim_channel=end_channel, shortest_event=1)[:, 0]

    assert len(starts) == len(ends), f"{len(starts)} starts vs {len(ends)} ends"

    return list(zip(starts, ends))


def _ast_get_rs_block(raw, block_index, start_channel='STI001', end_channel='STI002'):
    """block_index: 0 = first block, 1 = second block."""
    blocks = _ast_get_rs_events(raw, start_channel, end_channel)
    start_samp, end_samp = blocks[block_index]

    tmin = (start_samp - raw.first_samp) / raw.info['sfreq']
    tmax = (end_samp - raw.first_samp) / raw.info['sfreq']

    return raw.copy().crop(tmin=tmin, tmax=tmax)