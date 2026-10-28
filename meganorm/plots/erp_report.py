"""Self-contained participant ERP reports based on saved IDPs and QC."""

from html import escape
import json
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
    report.add_html(
        "<p>Participant: " + escape(subject) + "</p>"
        "<p>Saved measurements; no features are recomputed for this report. "
        "Peak markers require valid saved amplitude and latency. "
        "Dashed lines show saved window means.</p>" + measurements,
        title="Measurements and trial counts",
    )
    report.add_html(
        result.qc.to_html(index=False, escape=True, na_rep="unavailable"),
        title="Quality control",
    )
    details = json.dumps(
        result.preprocessing_qc, indent=2, default=str, ensure_ascii=False
    )
    report.add_html(
        "<pre>" + escape(details) + "</pre>", title="Processing and montage"
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
