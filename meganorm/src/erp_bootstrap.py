"""Optional within-condition trial bootstrap of already cleaned ERP epochs.

Original averages and measurements remain authoritative. Only summary statistics
are returned; temporary waveform draws are removed, including on failure.
"""

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from meganorm.utils.erp_logging import event as progress_event


@dataclass
class ERPBootstrapSummary:
    """IDP and waveform summaries, with reproducibility metadata."""

    idps: pd.DataFrame
    waveforms: dict
    provenance: dict


def _matching_times(times, reference):
    """Allow FIF float32 sampling-rate roundoff, but not a different grid."""
    times, reference = np.asarray(times), np.asarray(reference)
    if times.shape != reference.shape or not times.size:
        return False
    if not np.isfinite(times).all() or not np.isfinite(reference).all():
        return False
    # FIF stores sfreq in float32. Its relative rounding error propagates to
    # every sample's time, including a nonzero epoch start.
    tolerance = max(1e-8, np.finfo(np.float32).eps * np.max(np.abs(reference)))
    return np.allclose(times, reference, atol=tolerance, rtol=0)


def _condition_rng(seed, subject, condition):
    payload = json.dumps(
        [seed, subject, condition], ensure_ascii=False, separators=(",", ":")
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _waveform_summary(draws):
    """Reduce exact quantiles in chunks without loading all draws into RAM."""
    flat = draws.reshape(len(draws), -1)
    quantiles = np.empty((3, flat.shape[1]))
    counts = np.empty(flat.shape[1], dtype=np.int64)
    width = max(1, (32 * 1024**2) // (len(draws) * 8 * 4))
    for start in range(0, flat.shape[1], width):
        stop = min(start + width, flat.shape[1])
        values = np.array(flat[:, start:stop])
        valid = np.isfinite(values)
        n_valid = valid.sum(axis=0)
        values[~valid] = np.nan
        values[:, n_valid == 0] = 0
        part = np.nanquantile(values, [0.05, 0.5, 0.95], axis=0, method="linear")
        part[:, n_valid == 0] = np.nan
        quantiles[:, start:stop] = part
        counts[start:stop] = n_valid
    return {
        **{
            key: value.reshape(draws.shape[1:])
            for key, value in zip(("p05", "median", "p95"), quantiles)
        },
        "n_valid": counts.reshape(draws.shape[1:]),
    }


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
    """Resample retained trials once per condition and repetition.

    Seeds are stable across participant/condition traversal order. All sensors
    share a trial draw. Peak summaries exclude invalid measurements and expose
    their valid count. Waveform quantiles are pointwise, in native SI units.
    Conditions below max(2, min_trials) have no bootstrap uncertainty estimate.
    """
    from .erp import compute_evokeds, extract_erp_idps

    for name, value, minimum in (
        ("n_repeats", n_repeats, 2),
        ("random_state", random_state, 0),
        ("min_trials", min_trials, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    evokeds = compute_evokeds(epochs) if evokeds is None else evokeds
    original, quality = extract_erp_idps(subject_id, evokeds, components, min_trials)
    if features is not None:
        if not features.equals(original):
            raise ValueError("Bootstrap originals do not match ERP features.")
        original = features
    values = {key: [] for key in original.columns}
    boundaries = dict.fromkeys(original.columns, 0)
    waveforms, trial_counts = {}, {}
    for condition, evoked in evokeds.items():
        code = epochs.event_id.get(condition)
        indices = (
            np.flatnonzero(epochs.events[:, 2] == code)
            if code is not None
            else np.array([], dtype=int)
        )
        n_trials = len(indices)
        trial_counts[condition] = n_trials
        waveforms[condition] = None
        if evoked is None or n_trials < max(2, min_trials):
            continue
        # average() can omit auxiliary/bad channels; align by name, not position.
        picks = [epochs.ch_names.index(ch) for ch in evoked.ch_names]
        data = epochs.get_data(copy=False)[indices][:, picks]
        selected = {
            name: replace(spec, conditions=(condition,))
            for name, spec in components.items()
            if condition in spec.conditions
        }
        rng = _condition_rng(random_state, subject_id, condition)
        with tempfile.TemporaryDirectory(prefix="meganorm-bootstrap-") as directory:
            draws = np.memmap(
                Path(directory) / "waveforms.dat",
                mode="w+",
                dtype="float64",
                shape=(n_repeats, *evoked.data.shape),
            )
            try:
                for repetition in range(n_repeats):
                    sample = rng.integers(n_trials, size=n_trials)
                    # Use the same ordinary trial mean as the original ERP.
                    # Unsampled nonfinite trials must not contaminate this draw.
                    draws[repetition] = data[sample].mean(axis=0)
                    if selected:
                        ev = evoked.copy()
                        ev.data[:] = draws[repetition]
                        features_draw, qc_draw = extract_erp_idps(
                            subject_id, {condition: ev}, selected, min_trials
                        )
                        for row in qc_draw.itertuples(index=False):
                            value = features_draw.loc[subject_id, row.idp]
                            if row.status == "ok" and np.isfinite(value):
                                values[row.idp].append(value)
                                boundaries[row.idp] += int(row.boundary_peak)
                    if (repetition + 1) % max(
                        1, n_repeats // 10
                    ) == 0 or repetition + 1 == n_repeats:
                        progress_event(
                            "bootstrap_progress",
                            condition=condition,
                            completed=repetition + 1,
                            n_repeats=n_repeats,
                        )
                waveforms[condition] = dict(
                    ch_names=tuple(evoked.ch_names),
                    times=evoked.times.copy(),
                    n_trials=n_trials,
                    **_waveform_summary(draws),
                )
            finally:
                draws._mmap.close()
    rows = []
    for row in quality.itertuples(index=False):
        valid = values[row.idp]
        n_trials = trial_counts.get(row.condition, 0)
        if n_trials == 0:
            status = "missing_condition"
        elif n_trials < max(2, min_trials):
            status = "insufficient_trials"
        elif len(valid) == n_repeats:
            status = "ok"
        elif valid:
            status = "partial"
        else:
            status = row.status if row.status != "ok" else "no_valid_repetitions"
        p05, median, p95 = (
            np.quantile(valid, [0.05, 0.5, 0.95], method="linear")
            if valid
            else (np.nan, np.nan, np.nan)
        )
        rows.append(
            dict(
                subject=subject_id,
                condition=row.condition,
                idp=row.idp,
                original=original.loc[subject_id, row.idp],
                median=median,
                p05=p05,
                p95=p95,
                n_valid=len(valid),
                n_repeats=n_repeats,
                n_trials=n_trials,
                status=status,
                n_boundary_peaks=boundaries[row.idp],
            )
        )
    return ERPBootstrapSummary(
        pd.DataFrame(rows),
        waveforms,
        dict(
            schema_version=1,
            subject=subject_id,
            n_repeats=n_repeats,
            random_state=random_state,
            min_trials=min_trials,
            quantiles=[5, 50, 95],
            quantile_method="linear",
            method="trial bootstrap within condition",
            seed_scheme="sha256-json-seed-subject-condition-v1",
            interpretation="Descriptive central 90% bootstrap range; waveform ranges are pointwise, not simultaneous confidence bands.",
        ),
    )
