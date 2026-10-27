"""Numerical and selection contracts for shared ICA operations."""

import mne
import numpy as np
import pytest

from meganorm.src import preprocess as prep

pytestmark = pytest.mark.unit


def mixed_raw():
    rng = np.random.default_rng(12)
    t = np.arange(1500) / 100
    sources = np.array(
        [np.sin(2 * np.pi * 3 * t), rng.laplace(size=len(t)), np.cos(2 * np.pi * 7 * t)]
    )
    data = np.array([[1, 0.2, 0.4], [0.3, 1, 0.1], [0.2, 0.4, 1]]) @ sources * 1e-6
    raw = mne.io.RawArray(
        data,
        mne.create_info(["Fz", "Cz", "Pz"], 100, "eeg"),
        first_samp=51,
        verbose=False,
    )
    raw.set_annotations(mne.Annotations([2], [0.4], ["BAD_test"]))
    raw.filter(1, 30, verbose=False)
    return raw


@pytest.mark.parametrize(
    "method,fit_params",
    [("fastica", None), ("infomax", None), ("infomax", {"extended": True})],
)
@pytest.mark.parametrize("reject_annotations", [True, False])
def test_shared_fit_matches_direct_mne(method, fit_params, reject_annotations):
    raw = mixed_raw()
    before = raw.get_data().copy()
    reference = mne.preprocessing.ICA(
        n_components=3,
        method=method,
        max_iter=40,
        random_state=9,
        fit_params=fit_params,
        verbose=False,
    )
    reference.fit(
        raw, picks=["eeg"], reject_by_annotation=reject_annotations, verbose=False
    )
    result = prep._fit_ica(
        raw,
        n_components=3,
        method=method,
        max_iter=40,
        random_state=9,
        fit_params=fit_params,
        picks=["eeg"],
        reject_by_annotation=reject_annotations,
    )
    np.testing.assert_array_equal(result.unmixing_matrix_, reference.unmixing_matrix_)
    np.testing.assert_array_equal(
        result.get_sources(raw).get_data(), reference.get_sources(raw).get_data()
    )
    reference.exclude = result.exclude = [0]
    np.testing.assert_array_equal(
        result.apply(raw.copy(), verbose=False).get_data(),
        reference.apply(raw.copy(), verbose=False).get_data(),
    )
    np.testing.assert_array_equal(raw.get_data(), before)
    assert result.n_samples_ == reference.n_samples_
    assert raw.first_samp == 51


@pytest.mark.parametrize("inclusive, expected", [(False, [1]), (True, [0, 1])])
def test_iclabel_selection_threshold_and_class_policy(inclusive, expected):
    labels = {
        "labels": ["eye blink", "eye blink", "brain", "heart beat", "eye blink"],
        "y_pred_proba": np.array([0.8, 0.81, 0.99, 0.99, 0.79]),
    }
    assert (
        prep._select_iclabel_components(labels, {"eye blink"}, 0.8, inclusive=inclusive)
        == expected
    )
    assert labels["labels"][0] == "eye blink"


def test_iclabel_selection_multiple_artifact_classes():
    names = [
        "eye blink",
        "heart beat",
        "muscle artifact",
        "channel noise",
        "line noise",
        "brain",
        "other",
    ]
    labels = {"labels": names, "y_pred_proba": np.ones(7)}
    assert prep._select_iclabel_components(
        labels, set(names[:5]), 0.8, inclusive=True
    ) == list(range(5))


def test_iclabel_selection_empty_results():
    assert (
        prep._select_iclabel_components(
            {"labels": [], "y_pred_proba": []}, {"heart beat"}, 0.8, inclusive=False
        )
        == []
    )
