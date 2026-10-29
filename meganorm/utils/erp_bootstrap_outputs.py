"""Validated, pickle-free storage of ERP bootstrap summaries (never replicates)."""

import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from meganorm.src.erp_bootstrap import _matching_times

from .erp import write_erp_json, write_erp_table

FILES = ("bootstrap.json", "bootstrap_summary.csv", "bootstrap_waveforms.npz")
COLUMNS = (
    "subject",
    "condition",
    "idp",
    "original",
    "median",
    "p05",
    "p95",
    "n_valid",
    "n_repeats",
    "n_trials",
    "status",
    "n_boundary_peaks",
)


def _fail(message):
    raise ValueError("Invalid ERP bootstrap " + message)


def _integer(value, minimum=0):
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value >= minimum
    )


def _quantiles(values, counts):
    low, median, high = (np.asarray(values[key]) for key in ("p05", "median", "p95"))
    if any(
        a.dtype.kind not in "fiu" or a.shape != counts.shape
        for a in (low, median, high)
    ):
        _fail("quantile shapes or types")
    valid = counts > 0
    if any(
        not np.isfinite(a[valid]).all() or not np.isnan(a[~valid]).all()
        for a in (low, median, high)
    ):
        _fail("quantile finite-value counts")
    if np.any(low[valid] > median[valid]) or np.any(median[valid] > high[valid]):
        _fail("quantile order")


def validate_bootstrap(summary, result):
    """Validate against authoritative features, QC and original evoked axes."""
    subject = result.features.index[0]
    p = summary.provenance
    if (
        not isinstance(p, dict)
        or type(p.get("schema_version")) is not int
        or p["schema_version"] != 1
    ):
        _fail("schema version")
    if p.get("subject") != subject:
        _fail("subject identity")
    if (
        not _integer(p.get("random_state"))
        or not _integer(p.get("n_repeats"), 2)
        or not _integer(p.get("min_trials"), 1)
    ):
        _fail("seed, repetitions or minimum trials")
    if p.get("quantiles") != [5, 50, 95] or p.get("quantile_method") != "linear":
        _fail("quantile provenance")
    if any(
        not isinstance(p.get(k), str) or not p[k] for k in ("method", "seed_scheme")
    ):
        _fail("method provenance")
    frame = summary.idps
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != list(COLUMNS):
        _fail("summary columns")
    if frame["idp"].duplicated().any() or set(frame["idp"]) != set(
        result.features.columns
    ):
        _fail("feature identity")
    if not frame["subject"].eq(subject).all():
        _fail("subject identity")
    qc = result.qc.set_index("idp")
    for row in frame.to_dict("records"):
        if row["condition"] != qc.loc[row["idp"], "condition"]:
            _fail("condition identity")
        original = result.features.loc[subject, row["idp"]]
        if not (
            row["original"] == original
            or (pd.isna(row["original"]) and pd.isna(original))
        ):
            _fail("original feature value")
        if any(
            not _integer(row[k])
            for k in ("n_valid", "n_repeats", "n_trials", "n_boundary_peaks")
        ):
            _fail("summary counts")
        if (
            row["n_repeats"] != p["n_repeats"]
            or row["n_valid"] > p["n_repeats"]
            or row["n_boundary_peaks"] > row["n_valid"]
        ):
            _fail("summary counts")
        if row["n_trials"] != qc.loc[row["idp"], "n_trials"]:
            _fail("trial counts")
        if row["n_trials"] < max(2, p["min_trials"]) and row["n_valid"]:
            _fail("insufficient trial counts")
        if not isinstance(row["status"], str) or not row["status"]:
            _fail("summary status")
    _quantiles(frame, frame["n_valid"].to_numpy())
    if not isinstance(summary.waveforms, dict) or set(summary.waveforms) != set(
        result.evokeds
    ):
        _fail("waveform conditions")
    for condition, wave in summary.waveforms.items():
        evoked = result.evokeds[condition]
        trials = 0 if evoked is None else int(evoked.nave)
        if trials < max(2, p["min_trials"]):
            if wave is not None:
                _fail("unavailable waveform")
            continue
        if not isinstance(wave, dict) or set(wave) != {
            "ch_names",
            "times",
            "median",
            "p05",
            "p95",
            "n_valid",
            "n_trials",
        }:
            _fail("waveform schema")
        if not _integer(wave["n_trials"]) or wave["n_trials"] != trials:
            _fail("waveform trial counts")
        if tuple(wave["ch_names"]) != tuple(evoked.ch_names):
            _fail("waveform channel alignment")
        times = np.asarray(wave["times"])
        if not _matching_times(times, evoked.times):
            _fail("waveform time alignment")
        counts = np.asarray(wave["n_valid"])
        if (
            counts.shape != evoked.data.shape
            or counts.dtype.kind not in "iu"
            or np.any(counts < 0)
            or np.any(counts > p["n_repeats"])
        ):
            _fail("waveform counts or shape")
        _quantiles(wave, counts)


def save_bootstrap(result, directory):
    """Write numeric NPZ, summary CSV and JSON metadata; propagate failures."""
    summary = getattr(result, "bootstrap", None)
    if summary is None:
        return None
    validate_bootstrap(summary, result)
    directory = Path(directory)
    arrays, conditions = {}, {}
    for index, (condition, wave) in enumerate(summary.waveforms.items()):
        if wave is None:
            conditions[condition] = None
            continue
        prefix = f"condition_{index}"
        conditions[condition] = {
            "prefix": prefix,
            "ch_names": list(wave["ch_names"]),
            "n_trials": int(wave["n_trials"]),
        }
        for key in ("times", "median", "p05", "p95", "n_valid"):
            arrays[f"{prefix}_{key}"] = np.asarray(wave[key])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, directory / FILES[2])
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    write_erp_table(directory / FILES[1], summary.idps, index=False)
    write_erp_json(
        directory / FILES[0],
        {"provenance": summary.provenance, "conditions": conditions},
    )
    return FILES[0]


def load_bootstrap(result, directory, filename):
    """Only load the descriptor's explicit pointer; ignore unrelated stale files."""
    from meganorm.src.erp_bootstrap import ERPBootstrapSummary

    if filename != FILES[0]:
        _fail("filepath")
    directory = Path(directory)
    with (directory / filename).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"provenance", "conditions"}
        or not isinstance(metadata["conditions"], dict)
    ):
        _fail("metadata schema")
    frame = pd.read_csv(
        directory / FILES[1],
        dtype={k: str for k in ("subject", "condition", "idp", "status")},
        keep_default_na=False,
        na_values={key: [""] for key in ("original", "median", "p05", "p95")},
        float_precision="round_trip",
    )
    waveforms, expected = {}, set()
    with np.load(directory / FILES[2], allow_pickle=False) as archive:
        for condition, spec in metadata["conditions"].items():
            if spec is None:
                waveforms[condition] = None
                continue
            if (
                not isinstance(spec, dict)
                or set(spec) != {"prefix", "ch_names", "n_trials"}
                or not isinstance(spec["prefix"], str)
                or not isinstance(spec["ch_names"], list)
            ):
                _fail("condition mapping")
            wave = {"ch_names": tuple(spec["ch_names"]), "n_trials": spec["n_trials"]}
            for key in ("times", "median", "p05", "p95", "n_valid"):
                name = f"{spec['prefix']}_{key}"
                if name in expected or name not in archive.files:
                    _fail("waveform array mapping")
                expected.add(name)
                wave[key] = archive[name]
            waveforms[condition] = wave
        if set(archive.files) != expected:
            _fail("unexpected waveform arrays")
    summary = ERPBootstrapSummary(frame, waveforms, metadata["provenance"])
    validate_bootstrap(summary, result)
    return summary
