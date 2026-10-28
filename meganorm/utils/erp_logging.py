"""Observational, context-local progress for ERP participant processing.

No handlers or warning filters are installed globally. An inactive context is
silent, so standalone numerical helpers keep their existing behavior. Each
stage duration includes nested work; nested durations must not be summed.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
import traceback
from uuid import uuid4

_LOG = logging.getLogger(__name__)
_CURRENT = ContextVar("meganorm_erp_progress", default=None)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class ERPProgress:
    """Append flushed text/JSONL events for a single ERP worker invocation.

    A sink failure disables only that sink and emits one logging warning.
    Progress output must never replace a scientific exception. The caller owns
    directory creation and the completion marker for scientific outputs.
    """

    def __init__(self, directory, subject, **details):
        self.directory = Path(directory)
        self.subject = subject
        self.details = details
        self.run_id = uuid4().hex
        self._disabled = set()
        self._sequence = 0
        self._span = 0
        self._stack = [0]

    def _append(self, filename, text):
        if filename in self._disabled:
            return
        try:
            with (self.directory / filename).open("a", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
        except Exception as error:
            self._disabled.add(filename)
            _LOG.warning("ERP progress sink %s disabled: %s", filename, error)

    def emit(
        self,
        stage,
        status,
        *,
        details=None,
        duration=None,
        span_id=None,
        parent_span_id=None,
    ):
        # Serialize once before writing either sink so both describe one event.
        try:
            self._sequence += 1
            entry = dict(
                schema_version=1,
                run_id=self.run_id,
                subject=self.subject,
                sequence=self._sequence,
                timestamp=datetime.now(timezone.utc).isoformat(),
                stage=stage,
                status=status,
                span_id=span_id,
                parent_span_id=parent_span_id,
                details=details or {},
            )
            if duration is not None:
                entry["duration_seconds"] = duration
            encoded = json.dumps(
                entry, ensure_ascii=True, allow_nan=False, default=_json_default
            )
            detail = json.dumps(
                entry["details"],
                ensure_ascii=True,
                allow_nan=False,
                default=_json_default,
            )
            elapsed = "" if duration is None else f" ({duration:.3f} s)"
            line = (
                f"{entry['timestamp']} [{status.upper()}] {stage}{elapsed} {detail}\n"
            )
            self._append("processing.log", line)
            self._append("processing_events.jsonl", encoded + "\n")
        except Exception as error:
            if "serialization" not in self._disabled:
                self._disabled.add("serialization")
                _LOG.warning("Could not serialize ERP progress event: %s", error)

    def __enter__(self):
        self._start = time.perf_counter()
        self._token = _CURRENT.set(self)
        self._append(
            "processing.log",
            f"Participant: {self.subject}\nRecording: {self.details.get('recording', 'unspecified')}\n",
        )
        self.emit("run", "started", details=self.details, span_id=0)
        return self

    def __exit__(self, error_type, error, tb):
        try:
            details = dict(self.details)
            status = "completed"
            if error_type is not None:
                status = "failed" if isinstance(error, Exception) else "interrupted"
                details.update(
                    error_type=error_type.__name__,
                    error=str(error),
                    traceback="".join(
                        traceback.format_exception(error_type, error, tb)
                    ),
                )
            if error_type is not None:
                self._append("processing.log", details["traceback"])
            elif "processing_outcome" in details:
                self._append(
                    "processing.log", f"Outcome: {details['processing_outcome']}\n"
                )
            self.emit(
                "run",
                status,
                details=details,
                duration=time.perf_counter() - self._start,
                span_id=0,
            )
        finally:
            _CURRENT.reset(self._token)
        return False


@contextmanager
def stage(name, **details):
    """Emit a live start and a terminal event; yield mutable completion details."""
    progress = _CURRENT.get()
    if progress is None:
        yield details
        return
    progress._span += 1
    span_id, parent = progress._span, progress._stack[-1]
    started = time.perf_counter()
    progress.emit(
        name, "started", details=details, span_id=span_id, parent_span_id=parent
    )
    progress._stack.append(span_id)
    try:
        yield details
    except BaseException as error:
        progress.emit(
            name,
            "failed" if isinstance(error, Exception) else "interrupted",
            details={
                **details,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            duration=time.perf_counter() - started,
            span_id=span_id,
            parent_span_id=parent,
        )
        raise
    else:
        progress.emit(
            name,
            "completed",
            details=details,
            duration=time.perf_counter() - started,
            span_id=span_id,
            parent_span_id=parent,
        )
    finally:
        progress._stack.pop()


def event(name, **details):
    """Record a concise summary or warning in the active ERP context."""
    progress = _CURRENT.get()
    if progress is not None:
        progress.emit(name, "info", details=details, parent_span_id=progress._stack[-1])


def skipped(name, *, reason):
    """Distinguish a configured omission from an unobserved operation."""
    progress = _CURRENT.get()
    if progress is not None:
        progress.emit(
            name,
            "skipped",
            details={"reason": reason},
            parent_span_id=progress._stack[-1],
        )


def runtime_versions():
    """Record installed versions without importing optional scientific backends."""
    from importlib.metadata import PackageNotFoundError, version
    import platform

    versions = {"python": platform.python_version()}
    for package in ("meganorm", "mne", "numpy", "scipy", "autoreject", "mne-icalabel"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed as a distribution"
    return versions
