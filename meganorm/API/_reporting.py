"""Report construction shared by cohort workflows; publication stays local."""

import json
from pathlib import Path

import pandas as pd


def serialize_manifest(manifest, *, structured=False):
    """Return CSV-ready values without changing the discovered manifest.

    REST retains its tuple-of-strings convention. ERP serializes nested
    containers as JSON, preserving scalar types inside those containers.
    """
    serial = manifest.copy()

    def encode(value):
        if structured and isinstance(value, (tuple, dict, list)):
            return json.dumps(value, default=str)
        if not structured and isinstance(value, tuple):
            return json.dumps([str(p) for p in value])
        return str(value) if isinstance(value, Path) else value

    for column in serial:
        serial[column] = serial[column].map(encode)
    return serial


def processing_row(info, *, feature_path, log_path, qc_path=None):
    """Start an attempted participant as interrupted until its outcome is known."""
    row = dict(
        participant_id=info["participant_id"],
        dataset=info["dataset"],
        status="interrupted",
        error_type=None,
        error_message=None,
        feature_path=str(feature_path),
    )
    if qc_path is not None:
        row["qc_path"] = str(qc_path)
    row["log_path"] = str(log_path)
    return row


def build_run_report(
    rows,
    *,
    columns,
    counts,
    discovered,
    demo_names,
    succeeded,
    analysis_type="rest",
    collected=0,
):
    """Build ordered processing tables and workflow-specific summary fields.

    The caller supplies its success count: REST counts accepted feature files,
    while ERP distinguishes complete, partial and collected participants.
    No files are written here; direct and atomic publication remain distinct.
    """
    table = pd.DataFrame(rows, columns=columns).set_index("participant_id")
    summary = dict(counts)
    is_erp = analysis_type == "erp"
    if is_erp:
        summary["analysis_type"] = "erp"
    summary.update(discovered=discovered, attempted=len(rows), succeeded=succeeded)
    if is_erp:
        summary["partial"] = sum(r["status"] == "partial" for r in rows)
    summary.update(
        failed=sum(r["status"] == "failed" for r in rows),
        interrupted=sum(r["status"] == "interrupted" for r in rows),
    )
    if is_erp:
        summary["collected"] = collected
    matched_statuses = {"success", "partial"} if is_erp else {"success"}
    summary["matched_demographics"] = sum(
        r["status"] in matched_statuses and r["dataset"] in demo_names for r in rows
    )
    return table, summary
