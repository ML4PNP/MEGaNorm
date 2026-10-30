"""Self-contained participant and cohort ERP reports based on saved IDPs and QC."""

from html import escape
import json
from urllib.parse import quote
import os
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import mne
import pandas as pd

from .plot_erp import _measurement_rows, _result_channels, plot_erp_result


def write_erp_report(result, path, *, save_figures=False):
    """Atomically write an MNE HTML report; never open a browser or recompute IDPs.

    Waveform figures contain at most three channels each. Optional PNG and SVG
    copies are saved in a sibling ``plots`` directory. All internally allocated
    figures are closed even when export fails. An existing report is replaced
    only after the complete new HTML has been written successfully.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subject = str(result.features.index[0]) if len(result.features) else "unknown"
    report = mne.Report(title=f"ERP report — {escape(subject)}", verbose=False)
    rows = pd.DataFrame(_measurement_rows(result))
    measurements = rows.to_html(
        index=False,
        escape=True,
        na_rep="unavailable",
        float_format=lambda x: f"{x:.6g}",
    )
    uncertainty_note = (
        "<p>Bootstrap median and p05–p95 summarize the descriptive central 90% "
        "range of estimates across bootstrap repetitions. Waveform shading is "
        "pointwise. These are not "
        "calibrated confidence intervals or simultaneous bands. Valid repetition "
        "counts and status identify missing or partial uncertainty; original "
        "measurements and curves remain authoritative.</p>"
        if getattr(result, "bootstrap", None) is not None
        else ""
    )
    report.add_html(
        "<p>Participant: " + escape(subject) + "</p>"
        "<p>Saved measurements; no features are recomputed for this report. "
        "Peak markers require valid saved amplitude and latency. "
        "Dashed lines show saved window means.</p>" + uncertainty_note + measurements,
        title="Measurements and trial counts",
    )
    report.add_html(
        result.qc.to_html(index=False, escape=True, na_rep="unavailable"),
        title="Quality control",
    )
    channels = _result_channels(result)
    for start in range(0, len(channels), 3):
        fig = None
        try:
            fig, _ = plot_erp_result(result, channels=channels[start : start + 3])
            page = start // 3 + 1
            report.add_figure(
                fig, title=f"ERP waveforms — page {page}", image_format="png"
            )
            if save_figures:
                directory = path.parent / "plots"
                directory.mkdir(exist_ok=True)
                for extension in ("png", "svg"):
                    fig.savefig(
                        directory / f"erp-page-{page:03d}.{extension}",
                        dpi=160,
                        bbox_inches="tight",
                    )
        finally:
            if fig is not None:
                plt.close(fig)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix="." + path.stem + ".", suffix=".html", delete=False
        ) as handle:
            temporary = Path(handle.name)
        report.save(temporary, overwrite=True, open_browser=False, verbose=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def write_erp_index(root, processing):
    """Write an escaped index without changing processing outcomes or tables."""
    root = Path(root)
    rows = []
    for participant, row in processing.iterrows():
        participant = str(participant)
        directory = root / "participants" / participant
        # Do not follow externally supplied paths or symlinks outside this run.
        safe = directory.resolve().is_relative_to((root / "participants").resolve())
        report = directory / "report.html"
        link = "Unavailable"
        artifact = ""
        if safe:
            if report.is_file() and report.resolve().is_relative_to(root.resolve()):
                href = "participants/" + quote(participant, safe="") + "/report.html"
                link = f'<a href="{escape(href, quote=True)}">Open report</a>'
            status_path = directory / "report_outputs.json"
            if status_path.is_file() and status_path.resolve().is_relative_to(
                root.resolve()
            ):
                try:
                    outputs = json.loads(status_path.read_text(encoding="utf-8"))
                    artifact = json.dumps(outputs, ensure_ascii=False)
                except (OSError, ValueError) as error:
                    artifact = f"Artifact status unavailable: {error}"
        cells = [
            participant,
            row.get("dataset", ""),
            row.get("status", ""),
            row.get("error_message", ""),
        ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{escape(str(value))}</td>" for value in cells)
            + f"<td>{link}</td><td>{escape(artifact)}</td></tr>"
        )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ERP cohort report</title><style>
body{font-family:system-ui,sans-serif;margin:2rem;color:#192d3a;background:#fafcfd}
table{border-collapse:collapse;width:100%;background:white}
th,td{padding:.65rem;border:1px solid #ccd6dd;text-align:left;vertical-align:top}
td{overflow-wrap:anywhere}th{background:#e9f1f5}a{color:#075e94}
</style></head><body><h1>ERP cohort report</h1>
<p>Participant processing outcomes and available reports. Artifact status is
separate from feature extraction status.</p>
<table><thead><tr><th>Participant</th><th>Dataset</th><th>Processing status</th>
<th>Processing error</th><th>Report</th><th>Artifact status</th></tr></thead><tbody>
""" + "\n".join(rows) + "</tbody></table></body></html>\n"
    path = root / "index.html"
    temporary = path.with_suffix(".html.tmp")
    try:
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
