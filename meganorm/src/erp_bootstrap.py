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
    """Preserve the original standalone bootstrap call signature."""
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
