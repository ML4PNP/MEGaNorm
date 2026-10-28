"""Numerical contracts for fitting AutoReject on existing epochs."""

import mne
import numpy as np
import pandas as pd
import pytest
from autoreject import AutoReject
from meganorm.src import preprocess as prep

pytestmark = pytest.mark.unit


def sample_epochs():
    rng = np.random.default_rng(71)
    values = rng.normal(0, 1e-6, (12, 4, 81))
    values[2, 0, 35:40] += 30e-6
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], 100, "eeg")
    info.set_montage("standard_1020")
    events = np.array([[100 + i * 200, 0, 1 + i % 2] for i in range(12)])
    epochs = mne.EpochsArray(
        values,
        info,
        events=events,
        event_id={"standard": 1, "target": 2},
        tmin=-0.2,
        baseline=(-0.2, 0),
        metadata=pd.DataFrame({"trial": range(12)}),
        verbose=False,
    )
    epochs.drop([1], reason="PREVIOUS", verbose=False)
    return epochs


@pytest.mark.parametrize("load_data", [False, True])
def test_shared_autoreject_matches_direct_fit_and_transform(load_data):
    epochs = sample_epochs()
    before = epochs.copy()
    parameters = dict(
        n_interpolate=[0, 1],
        consensus=[0.5, 1.0],
        cv=2,
        thresh_method="random_search",
        random_state=9,
        picks=np.arange(4),
        n_jobs=1,
        verbose=False,
    )
    expected = AutoReject(**parameters)
    if load_data:
        epochs.load_data()
    expected.fit(epochs)
    actual = prep._fit_autoreject_epochs(
        epochs, estimator=AutoReject, load_data=load_data, **parameters
    )
    assert actual.threshes_ == expected.threshes_
    assert actual.consensus_ == expected.consensus_
    assert actual.n_interpolate_ == expected.n_interpolate_
    np.testing.assert_array_equal(actual.loss_["eeg"], expected.loss_["eeg"])
    cleaned, log = actual.transform(epochs, return_log=True)
    reference, reference_log = expected.transform(epochs, return_log=True)
    np.testing.assert_array_equal(log.bad_epochs, reference_log.bad_epochs)
    np.testing.assert_array_equal(log.labels, reference_log.labels)
    np.testing.assert_array_equal(cleaned.get_data(), reference.get_data())
    np.testing.assert_array_equal(cleaned.events, reference.events)
    np.testing.assert_array_equal(cleaned.selection, reference.selection)
    assert cleaned.drop_log == reference.drop_log
    assert cleaned.baseline == reference.baseline
    pd.testing.assert_frame_equal(cleaned.metadata, reference.metadata)
    np.testing.assert_array_equal(epochs.get_data(), before.get_data())
    np.testing.assert_array_equal(epochs.selection, before.selection)
    assert epochs.drop_log == before.drop_log
    assert epochs.baseline == before.baseline
    pd.testing.assert_frame_equal(epochs.metadata, before.metadata)
