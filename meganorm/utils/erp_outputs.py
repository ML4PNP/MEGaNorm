"""Persist and reload ERP averages with their authoritative saved IDPs and QC."""

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import tempfile
import warnings

import mne
import pandas as pd

from .erp import validate_erp_subject_id, write_erp_json, write_erp_table

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
    outputs = {"evokeds": {"status": "disabled"}, "report": {"status": "disabled"}}
    if save_evokeds:
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
        write_erp_json(directory / "erp_result.json", descriptor)
        outputs["evokeds"] = {
            "status": "saved" if evokeds else "empty",
            "file": filename,
        }
    if save_report or save_figures:
        try:
            from meganorm.plots.erp_report import write_erp_report

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
    return ERPResult(features, qc, None, evokeds, preprocessing, components=components)
