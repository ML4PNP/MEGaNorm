"""Compatibility imports for bootstrap utilities now in ``meganorm.utils``."""

from meganorm.utils.erp_bootstrap import (
    ERPBootstrapSummary,
    _validate_bootstrap_settings,
)
from meganorm.utils.erp_bootstrap import bootstrap_erp as _bootstrap_erp

__all__ = ["ERPBootstrapSummary", "bootstrap_erp"]


def bootstrap_erp(
    epochs,
    subject_id,
    components,
    *,
    n_repeats,
    random_state,
    min_trials=1,
    evokeds=None,
    features=None,
):
    """Bootstrap retained trials through the standalone compatibility API.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Event-locked epochs with condition codes in ``event_id``.
    subject_id : str
        Exact participant identifier used for the feature row.
    components : mapping of str to ERPComponent
        Named measurements in output order.
    n_repeats : int
        Number of bootstrap draws, at least two.
    random_state : int
        Nonnegative seed combined with participant and condition identifiers.
    min_trials : int, default=1
        Measurement threshold; uncertainty additionally requires two trials.
    evokeds : mapping, optional
        Original condition averages; computed from epochs when omitted.
    features : pandas.DataFrame, optional
        Authoritative features; must equal extraction from the original averages.

    Returns
    -------
    summary : ERPBootstrapSummary
        Original IDPs, bootstrap quantiles, valid counts, waveform summaries and
        reproducibility metadata. Ranges are descriptive, not calibrated intervals.

    Raises
    ------
    ValueError
        If repetitions, seed or minimum trials are invalid, or original features differ.
    """
    from .erp import compute_evokeds, extract_erp_idps

    _validate_bootstrap_settings(n_repeats, random_state, min_trials)
    if evokeds is None:
        evokeds = compute_evokeds(epochs)
    return _bootstrap_erp(
        epochs,
        subject_id,
        components,
        n_repeats=n_repeats,
        random_state=random_state,
        min_trials=min_trials,
        evokeds=evokeds,
        features=features,
        extract_idps=extract_erp_idps,
    )
