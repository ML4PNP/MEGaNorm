"""Persist and reload ERP averages with their authoritative saved IDPs and QC."""

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import tempfile
import warnings

import mne
import numpy as np
import pandas as pd

from .erp import (
    validate_erp_subject_id,
    write_erp_json,
    write_erp_table,
    _matching_times,
)
from .erp_logging import stage, skipped

_LOG = logging.getLogger(__name__)


def _identity(result):
    if len(result.features) != 1:
        raise ValueError("ERP results require exactly one subject row.")
    subject = validate_erp_subject_id(result.features.index[0])
    if "subject" not in result.qc or not result.qc["subject"].eq(subject).all():
        raise ValueError("ERP QC subject identity does not match features.")
    if not result.features.columns.is_unique:
        raise ValueError("ERP feature columns must be unique.")
    return subject


def save_erp_artifacts(
    result, directory, *, save_evokeds=True, save_report=True, save_figures=False
):
    """Write optional artifacts; report errors remain separate from scientific IDPs.

    FIF and descriptor failures propagate. Report failures are logged and stored
    in report_outputs.json; interrupts always propagate. Returns that status map.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    subject = _identity(result)
    bootstrap_file = save_bootstrap(result, directory)
    outputs = {"evokeds": {"status": "disabled"}, "report": {"status": "disabled"}}
    if save_evokeds:
        with stage("save_evokeds") as detail:
            evokeds, conditions = [], {}
            for condition, evoked in result.evokeds.items():
                if not isinstance(condition, str) or not condition:
                    raise ValueError("ERP condition names must be nonempty strings.")
                conditions[condition] = None if evoked is None else len(evokeds)
                if evoked is not None:
                    copied = evoked.copy()
                    copied.comment = condition
                    evokeds.append(copied)
            filename = "evoked-ave.fif" if evokeds else None
            if evokeds:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=directory, prefix=".erp-", suffix="-ave.fif", delete=False
                    ) as handle:
                        temporary = Path(handle.name)
                    mne.write_evokeds(temporary, evokeds, overwrite=True, verbose=False)
                    os.replace(temporary, directory / filename)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            descriptor = {
                "schema_version": 1,
                "subject": subject,
                "components": {
                    name: asdict(spec) for name, spec in result.components.items()
                },
                "conditions": conditions,
                "feature_names": list(result.features.columns),
                "evokeds_file": filename,
                "feature_index_name": result.features.index.name,
                "qc_dtypes": {
                    column: str(dtype) for column, dtype in result.qc.dtypes.items()
                },
            }
            if bootstrap_file is not None:
                descriptor["bootstrap_file"] = bootstrap_file
            write_erp_json(directory / "erp_result.json", descriptor)
            outputs["evokeds"] = {
                "status": "saved" if evokeds else "empty",
                "file": filename,
            }
            detail.update(outputs["evokeds"])
    else:
        skipped("save_evokeds", reason="save_evokeds is False")
    if save_report or save_figures:
        try:
            from meganorm.plots.erp_report import write_erp_report

            with stage("report", save_figures=save_figures, file="report.html"):
                write_erp_report(
                    result, directory / "report.html", save_figures=save_figures
                )
            outputs["report"] = {"status": "saved", "file": "report.html"}
        except Exception as error:
            outputs["report"] = {"status": "failed", "error": str(error)}
            _LOG.warning("ERP report failed for %s: %s", subject, error)
            warnings.warn(
                f"ERP report failed for {subject}: {error}", UserWarning, stacklevel=2
            )
    else:
        skipped("report", reason="save_report and save_figures are False")
    with stage("save_artifact_status", file="report_outputs.json"):
        write_erp_json(directory / "report_outputs.json", outputs)
    return outputs


def save_erp_result(result, directory, *, report=True, figures=False, overwrite=False):
    """Save reusable averages and unchanged IDPs; never save individual epochs.

    FIF stores floating-point waveforms with MNE's standard single precision.
    CSV feature values remain authoritative for all subsequent annotations.
    """
    directory = Path(directory)
    _identity(result)
    if directory.exists() and not overwrite:
        raise FileExistsError(f"ERP output directory already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in (
            "evoked-ave.fif",
            "erp_result.json",
            "report.html",
            "report_outputs.json",
            "bootstrap.json",
            "bootstrap_summary.csv",
            "bootstrap_waveforms.npz",
        ):
            (directory / name).unlink(missing_ok=True)
        for pattern in ("erp-page-*.png", "erp-page-*.svg"):
            for path in (directory / "plots").glob(pattern):
                path.unlink()
    write_erp_table(directory / "features.csv", result.features)
    write_erp_table(directory / "qc.csv", result.qc, index=False)
    write_erp_json(directory / "preprocessing.json", result.preprocessing_qc)
    save_erp_artifacts(result, directory, save_report=report, save_figures=figures)
    return directory


def load_erp_result(directory):
    """Load saved IDPs, QC and averages without filtering or re-extraction."""
    from meganorm.src.erp import ERPComponent, ERPResult, extract_erp_idps

    directory = Path(directory)
    with (directory / "erp_result.json").open(encoding="utf-8") as handle:
        descriptor = json.load(handle)
    if not isinstance(descriptor, dict) or descriptor.get("schema_version") != 1:
        raise ValueError("Unsupported ERP result schema version.")
    subject = validate_erp_subject_id(descriptor.get("subject"))
    names = descriptor.get("feature_names")
    if (
        not isinstance(names, list)
        or not all(isinstance(n, str) for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Invalid ERP feature columns in descriptor.")
    header = pd.read_csv(directory / "features.csv", nrows=0).columns[0]
    features = pd.read_csv(
        directory / "features.csv",
        dtype={header: str},
        float_precision="round_trip",
        keep_default_na=False,
    )
    features = features.set_index(header)
    features = features.replace("", float("nan"))
    # Preserve the exact participant identifier, including leading zeros.
    if list(features.index) != [subject]:
        raise ValueError(
            "Saved ERP feature subject identity does not match descriptor."
        )
    if list(features.columns) != names:
        raise ValueError("Saved ERP feature columns do not match descriptor.")
    features.index.name = descriptor.get("feature_index_name")
    features = features.astype(float)
    dtypes = descriptor.get("qc_dtypes", {})
    allowed_dtypes = {"object", "int64", "float64", "bool", "string"}
    if not isinstance(dtypes, dict) or any(
        value not in allowed_dtypes for value in dtypes.values()
    ):
        raise ValueError("Unsupported saved QC column dtype.")
    qc = pd.read_csv(
        directory / "qc.csv",
        dtype=dtypes or {"subject": str},
        keep_default_na=False,
        float_precision="round_trip",
    )
    if "subject" not in qc or not qc["subject"].eq(subject).all():
        raise ValueError("Saved ERP QC subject identity does not match descriptor.")
    if "idp" not in qc or set(qc["idp"]) != set(names) or qc["idp"].duplicated().any():
        raise ValueError("Saved ERP QC feature columns do not match descriptor.")
    component_data = descriptor.get("components")
    if not isinstance(component_data, dict):
        raise ValueError("Invalid saved ERP components.")
    components = {}
    for name, settings in component_data.items():
        if not isinstance(settings, dict):
            raise ValueError("Invalid saved ERP component settings.")
        settings = dict(settings)
        for key in ("conditions", "channels", "window", "metrics"):
            if key in settings:
                settings[key] = tuple(settings[key])
        components[name] = ERPComponent(**settings)
    if (
        components
        and list(extract_erp_idps(subject, {}, components)[0].columns) != names
    ):
        raise ValueError("Saved ERP components do not match feature columns.")
    conditions = descriptor.get("conditions")
    filename = descriptor.get("evokeds_file")
    if not isinstance(conditions, dict) or any(
        not isinstance(k, str) or not k for k in conditions
    ):
        raise ValueError("Invalid ERP condition mapping.")
    indices = [i for i in conditions.values() if i is not None]
    if any(type(i) is not int or i < 0 for i in indices) or sorted(indices) != list(
        range(len(indices))
    ):
        raise ValueError("Invalid ERP condition indices.")
    if filename != ("evoked-ave.fif" if indices else None):
        raise ValueError("Invalid ERP evokeds filepath.")
    evoked_list = (
        mne.read_evokeds(
            directory / filename,
            condition=None,
            baseline=None,
            proj=False,
            verbose=False,
        )
        if filename
        else []
    )
    if len(evoked_list) != len(indices):
        raise ValueError("Saved ERP evoked count does not match condition indices.")
    evokeds = {
        condition: None if index is None else evoked_list[index]
        for condition, index in conditions.items()
    }
    if any(
        ev is not None and ev.comment != condition for condition, ev in evokeds.items()
    ):
        raise ValueError("Saved ERP condition identity does not match descriptor.")
    with (directory / "preprocessing.json").open(encoding="utf-8") as handle:
        preprocessing = json.load(handle)
    result = ERPResult(
        features, qc, None, evokeds, preprocessing, components=components
    )
    if descriptor.get("bootstrap_file") is not None:
        result.bootstrap = load_bootstrap(
            result, directory, descriptor["bootstrap_file"]
        )
    return result


BOOTSTRAP_FILES = ("bootstrap.json", "bootstrap_summary.csv", "bootstrap_waveforms.npz")
BOOTSTRAP_COLUMNS = (
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


def _bootstrap_error(message):
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
        _bootstrap_error("quantile shapes or types")
    valid = counts > 0
    if any(
        not np.isfinite(a[valid]).all() or not np.isnan(a[~valid]).all()
        for a in (low, median, high)
    ):
        _bootstrap_error("quantile finite-value counts")
    if np.any(low[valid] > median[valid]) or np.any(median[valid] > high[valid]):
        _bootstrap_error("quantile order")


def validate_bootstrap(summary, result):
    """Validate against authoritative features, QC and original evoked axes."""
    subject = result.features.index[0]
    p = summary.provenance
    if (
        not isinstance(p, dict)
        or type(p.get("schema_version")) is not int
        or p["schema_version"] != 1
    ):
        _bootstrap_error("schema version")
    if p.get("subject") != subject:
        _bootstrap_error("subject identity")
    if (
        not _integer(p.get("random_state"))
        or not _integer(p.get("n_repeats"), 2)
        or not _integer(p.get("min_trials"), 1)
    ):
        _bootstrap_error("seed, repetitions or minimum trials")
    if p.get("quantiles") != [5, 50, 95] or p.get("quantile_method") != "linear":
        _bootstrap_error("quantile provenance")
    if any(
        not isinstance(p.get(k), str) or not p[k] for k in ("method", "seed_scheme")
    ):
        _bootstrap_error("method provenance")
    frame = summary.idps
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != list(
        BOOTSTRAP_COLUMNS
    ):
        _bootstrap_error("summary columns")
    if frame["idp"].duplicated().any() or set(frame["idp"]) != set(
        result.features.columns
    ):
        _bootstrap_error("feature identity")
    if not frame["subject"].eq(subject).all():
        _bootstrap_error("subject identity")
    qc = result.qc.set_index("idp")
    for row in frame.to_dict("records"):
        if row["condition"] != qc.loc[row["idp"], "condition"]:
            _bootstrap_error("condition identity")
        original = result.features.loc[subject, row["idp"]]
        if not (
            row["original"] == original
            or (pd.isna(row["original"]) and pd.isna(original))
        ):
            _bootstrap_error("original feature value")
        if any(
            not _integer(row[k])
            for k in ("n_valid", "n_repeats", "n_trials", "n_boundary_peaks")
        ):
            _bootstrap_error("summary counts")
        if (
            row["n_repeats"] != p["n_repeats"]
            or row["n_valid"] > p["n_repeats"]
            or row["n_boundary_peaks"] > row["n_valid"]
        ):
            _bootstrap_error("summary counts")
        if row["n_trials"] != qc.loc[row["idp"], "n_trials"]:
            _bootstrap_error("trial counts")
        if row["n_trials"] < max(2, p["min_trials"]) and row["n_valid"]:
            _bootstrap_error("insufficient trial counts")
        if not isinstance(row["status"], str) or not row["status"]:
            _bootstrap_error("summary status")
    _quantiles(frame, frame["n_valid"].to_numpy())
    if not isinstance(summary.waveforms, dict) or set(summary.waveforms) != set(
        result.evokeds
    ):
        _bootstrap_error("waveform conditions")
    for condition, wave in summary.waveforms.items():
        evoked = result.evokeds[condition]
        trials = 0 if evoked is None else int(evoked.nave)
        if trials < max(2, p["min_trials"]):
            if wave is not None:
                _bootstrap_error("unavailable waveform")
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
            _bootstrap_error("waveform schema")
        if not _integer(wave["n_trials"]) or wave["n_trials"] != trials:
            _bootstrap_error("waveform trial counts")
        if tuple(wave["ch_names"]) != tuple(evoked.ch_names):
            _bootstrap_error("waveform channel alignment")
        times = np.asarray(wave["times"])
        if not _matching_times(times, evoked.times):
            _bootstrap_error("waveform time alignment")
        counts = np.asarray(wave["n_valid"])
        if (
            counts.shape != evoked.data.shape
            or counts.dtype.kind not in "iu"
            or np.any(counts < 0)
            or np.any(counts > p["n_repeats"])
        ):
            _bootstrap_error("waveform counts or shape")
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
        os.replace(temporary, directory / BOOTSTRAP_FILES[2])
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    write_erp_table(directory / BOOTSTRAP_FILES[1], summary.idps, index=False)
    write_erp_json(
        directory / BOOTSTRAP_FILES[0],
        {"provenance": summary.provenance, "conditions": conditions},
    )
    return BOOTSTRAP_FILES[0]


def load_bootstrap(result, directory, filename):
    """Only load the descriptor's explicit pointer; ignore unrelated stale files."""
    from .erp_bootstrap import ERPBootstrapSummary

    if filename != BOOTSTRAP_FILES[0]:
        _bootstrap_error("filepath")
    directory = Path(directory)
    with (directory / filename).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"provenance", "conditions"}
        or not isinstance(metadata["conditions"], dict)
    ):
        _bootstrap_error("metadata schema")
    frame = pd.read_csv(
        directory / BOOTSTRAP_FILES[1],
        dtype={k: str for k in ("subject", "condition", "idp", "status")},
        keep_default_na=False,
        na_values={key: [""] for key in ("original", "median", "p05", "p95")},
        float_precision="round_trip",
    )
    waveforms, expected = {}, set()
    with np.load(directory / BOOTSTRAP_FILES[2], allow_pickle=False) as archive:
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
                _bootstrap_error("condition mapping")
            wave = {"ch_names": tuple(spec["ch_names"]), "n_trials": spec["n_trials"]}
            for key in ("times", "median", "p05", "p95", "n_valid"):
                name = f"{spec['prefix']}_{key}"
                if name in expected or name not in archive.files:
                    _bootstrap_error("waveform array mapping")
                expected.add(name)
                wave[key] = archive[name]
            waveforms[condition] = wave
        if set(archive.files) != expected:
            _bootstrap_error("unexpected waveform arrays")
    summary = ERPBootstrapSummary(frame, waveforms, metadata["provenance"])
    validate_bootstrap(summary, result)
    return summary
