"""Self-contained cohort index for optional ERP participant reports."""

from html import escape
import json
from pathlib import Path
from urllib.parse import quote


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
