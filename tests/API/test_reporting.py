"""Exact report contracts shared by REST and ERP orchestration."""

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("structured", [False, True])
def test_manifest_serialization_preserves_workflow_policy_and_input(structured):
    from meganorm.API._reporting import serialize_manifest

    nested = {"name": "é", "path": Path("/recording")}
    frame = pd.DataFrame(
        {
            "participant_id": ["001"],
            "recording_paths": [(Path("/data a"), 2, None)],
            "selected_recording": [Path("/data a")],
            "spec": [nested],
            "items": [[1, "é"]],
            "missing": [None],
        }
    )
    original = frame.copy(deep=True)
    result = serialize_manifest(frame, structured=structured)
    assert result.loc[0, "participant_id"] == "001"
    assert result.loc[0, "selected_recording"] == "/data a"
    if structured:
        assert result.loc[0, "recording_paths"] == '["/data a", 2, null]'
        assert result.loc[0, "spec"] == json.dumps(nested, default=str)
        assert result.loc[0, "items"] == json.dumps([1, "é"])
    else:
        assert result.loc[0, "recording_paths"] == '["/data a", "2", "None"]'
        assert result.loc[0, "spec"] is nested
        assert result.loc[0, "items"] == [1, "é"]
    assert result.loc[0, "missing"] is None
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("qc", [None, Path("/out/qc.csv")])
def test_processing_row_preserves_paths_order_and_initial_state(qc):
    from meganorm.API._reporting import processing_row

    info = {"participant_id": "001", "dataset": "cohort"}
    row = processing_row(
        info,
        feature_path=Path("/out/features.csv"),
        log_path=Path("/out/log"),
        qc_path=qc,
    )
    assert list(row) == [
        "participant_id",
        "dataset",
        "status",
        "error_type",
        "error_message",
        "feature_path",
        *(["qc_path"] if qc is not None else []),
        "log_path",
    ]
    assert row["participant_id"] == "001" and row["dataset"] == "cohort"
    assert row["status"] == "interrupted"
    assert row["error_type"] is row["error_message"] is None
    assert row["feature_path"] == "/out/features.csv" and row["log_path"] == "/out/log"
    if qc is not None:
        assert row["qc_path"] == "/out/qc.csv"


@pytest.mark.parametrize("analysis", ["rest", "erp"])
@pytest.mark.parametrize("empty", [False, True])
def test_report_counts_columns_and_json_order(analysis, empty):
    from meganorm.API._reporting import build_run_report, processing_row

    rows = []
    for i, status in enumerate(
        [] if empty else ["success", "partial", "failed", "interrupted"]
    ):
        row = processing_row(
            {"participant_id": f"{i:03}", "dataset": "a"},
            feature_path="/f",
            log_path="/l",
            qc_path="/q" if analysis == "erp" else None,
        )
        row["status"] = status
        rows.append(row)
    columns = [
        "participant_id",
        "dataset",
        "status",
        "error_type",
        "error_message",
        "feature_path",
        *(["qc_path"] if analysis == "erp" else []),
        "log_path",
    ]
    counts = {"matched_demographics": 99, "unused_demographics": {"a": 2}}
    table, summary = build_run_report(
        rows,
        columns=columns,
        counts=counts,
        discovered=5,
        demo_names={"a"},
        analysis_type=analysis,
        succeeded=0 if empty else 1,
        collected=0 if empty else 2,
    )
    expected = dict(counts)
    if analysis == "erp":
        expected["analysis_type"] = "erp"
    expected.update(discovered=5, attempted=len(rows), succeeded=0 if empty else 1)
    if analysis == "erp":
        expected["partial"] = 0 if empty else 1
    expected.update(failed=0 if empty else 1, interrupted=0 if empty else 1)
    if analysis == "erp":
        expected["collected"] = 0 if empty else 2
    expected["matched_demographics"] = 0 if empty else (2 if analysis == "erp" else 1)
    assert json.dumps(summary, indent=2) == json.dumps(expected, indent=2)
    assert table.index.name == "participant_id"
    assert list(table.columns) == columns[1:]
    assert table.index.tolist() == [r["participant_id"] for r in rows]
    assert table.to_csv().splitlines()[0] == ",".join(columns)
    assert counts["matched_demographics"] == 99


def test_rest_summary_uses_collection_count_and_requires_demographics_match():
    from meganorm.API._reporting import build_run_report

    rows = [{"participant_id": "001", "dataset": "no_demo", "status": "success"}]
    _, summary = build_run_report(
        rows,
        columns=["participant_id", "dataset", "status"],
        counts={},
        discovered=1,
        demo_names={"other"},
        succeeded=0,
    )
    assert summary["succeeded"] == 0
    assert summary["matched_demographics"] == 0
    assert (
        "partial" not in summary
        and "collected" not in summary
        and "analysis_type" not in summary
    )
