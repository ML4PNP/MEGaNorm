"""ERP settings, event resolution and synthetic EEG tutorial helpers."""

import mne
import numpy as np
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


def make_demo_recording(seed=42):
    """Return deterministic synthetic EEG Raw and absolute-sample events.

    Two sensors (Cz, Pz), 250 Hz, first_samp=500 and 40 alternating trials
    (codes 1=standard, 2=target). Negative peaks occur at 100 ms; positive
    peaks at 300 ms have Pz amplitudes 3/6 microvolts by condition. A small
    seeded noise term is added. Used by tests and the source-tree tutorial.
    """
    rng = np.random.default_rng(seed)
    sfreq = 250.0
    first_samp = 500
    onsets = 1.0 + np.arange(40) * 1.4
    n_times = int((onsets[-1] + 1.5) * sfreq)
    data = 2e-6 + rng.normal(0, 0.15e-6, (2, n_times))
    relative_times = np.arange(-50, 201) / sfreq
    negative = np.exp(-0.5 * ((relative_times - 0.1) / 0.015) ** 2)
    positive = np.exp(-0.5 * ((relative_times - 0.3) / 0.04) ** 2)
    events = []
    for trial, onset in enumerate(onsets):
        sample = int(round(onset * sfreq))
        event_code = 1 if trial % 2 == 0 else 2
        amplitude = 3e-6 if event_code == 1 else 6e-6
        waveform = np.vstack(
            [
                -4e-6 * negative + 0.5 * amplitude * positive,
                -2e-6 * negative + amplitude * positive,
            ]
        )
        data[:, sample - 50 : sample + 201] += waveform
        events.append([sample + first_samp, 0, event_code])
    info = mne.create_info(["Cz", "Pz"], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, first_samp=first_samp, verbose=False)
    raw.set_montage("standard_1020")
    return raw, np.asarray(events, dtype=int)


def _name(value):
    if not value or value != value.strip() or "__" in value:
        raise ValueError(
            "Names must be nonempty, trimmed and contain no double underscore."
        )
    return value


def _names(values):
    if not values or len(values) != len(set(values)):
        raise ValueError("Names must be nonempty and unique.")
    for value in values:
        _name(value)


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ERPMontageSettings(_Settings):
    """Dataset-level position fallback; inputs are never rewritten."""

    template: str | None = None
    exclude_channels: tuple[str, ...] = ()
    supplied_kind: Literal["unknown", "measured", "template"] = "unknown"

    @model_validator(mode="after")
    def validate_montage(self):
        if self.template is not None:
            available = mne.channels.get_builtin_montages()
            if self.template not in {
                *available,
                "auto",
                "standard_1005",
                "colin27_1005",
            }:
                raise ValueError(f"Unknown EEG template: {self.template!r}.")
        if self.exclude_channels:
            _names(self.exclude_channels)
        return self


class ERPComponentSettings(_Settings):
    """One EEG component and its ordered condition, channel and metric selections."""

    conditions: tuple[str, ...]
    channels: tuple[str, ...]
    window: tuple[float, float]
    polarity: Literal["pos", "neg", "abs"]
    metrics: tuple[Literal["peak_amplitude", "peak_latency", "mean_amplitude"], ...] = (
        "peak_amplitude",
        "peak_latency",
    )
    sensor_type: Literal["eeg"] = "eeg"

    @model_validator(mode="after")
    def validate_component(self):
        _names(self.conditions)
        _names(self.channels)
        _names(self.metrics)
        if self.window[0] >= self.window[1]:
            raise ValueError("Component window must be increasing.")
        return self


def resolve_ica_settings(config):
    """Return the validated canonical ICA mode and scientific QC thresholds."""
    mode = config.ica_mode
    if not config.apply_ica:
        if mode not in ("auto", "none"):
            raise ValueError("apply_ica=False conflicts with the explicit ica_mode.")
        mode = "none"
    return mode, {
        "correlation": config.auto_ica_corr_thr,
        "label_probability": config.ica_label_threshold,
    }


def resolve_ica_fit_settings(config, sfreq=None):
    """Validate the ICA fit copy independently of analysis resampling."""
    high = config.ica_h_freq
    if resolve_ica_settings(config)[0] != "none":
        if high is not None and config.ica_l_freq >= high:
            raise ValueError("ica_l_freq must be below the ICA fit h_freq.")
        for rate in (sfreq, config.ica_resample_sfreq):
            if rate is not None and any(
                cutoff is not None and cutoff >= rate / 2
                for cutoff in (config.ica_l_freq, high)
            ):
                raise ValueError("ICA fit cutoffs must be below the Nyquist frequency.")
    return high


class ERPEventSpec(_Settings):
    """Dataset-specific mapping from logical conditions to acquisition events.

    Annotation maps run from logical condition to exact annotation description.
    Stimulus maps run from logical condition to integer trigger code.
    """

    mode: Literal["annotations", "stim"]
    annotation_map: dict[str, str] | None = None
    event_id: dict[str, StrictInt] | None = None
    stim_channel: str | None = None
    consecutive: bool | Literal["increasing"] = "increasing"
    shortest_event: StrictInt = Field(default=2, ge=1)
    min_duration: float = Field(default=0.0, ge=0)
    initial_event: bool = False

    @model_validator(mode="after")
    def validate_mapping(self):
        if self.mode == "annotations":
            if (
                not self.annotation_map
                or self.event_id is not None
                or self.stim_channel is not None
            ):
                raise ValueError("Annotation mode requires only annotation_map.")
            if (
                self.consecutive,
                self.shortest_event,
                self.min_duration,
                self.initial_event,
            ) != ("increasing", 2, 0.0, False):
                raise ValueError(
                    "Stimulus extraction options are invalid for annotations."
                )
            values = list(self.annotation_map.values())
            if any(not value for value in values):
                raise ValueError("Annotation descriptions must be nonempty.")
            mapping = self.annotation_map
        else:
            if (
                not self.event_id
                or self.annotation_map is not None
                or not self.stim_channel
            ):
                raise ValueError("Stim mode requires event_id and stim_channel only.")
            _name(self.stim_channel)
            mapping = self.event_id
            values = list(mapping.values())
            if any(value < 0 for value in values):
                raise ValueError("Trigger codes must be nonnegative.")
        _names(tuple(mapping))
        if len(values) != len(set(values)):
            raise ValueError("Each logical condition must have a unique event mapping.")
        return self

    @property
    def conditions(self):
        return tuple(
            self.annotation_map if self.mode == "annotations" else self.event_id
        )


class _ERPTaskSettings(_Settings):
    """Private ERP epoch and measurement schema; processing belongs to Config."""

    tmin: float = -0.2
    tmax: float = 0.8
    baseline: tuple[float | None, float | None] | None = (None, 0.0)
    save_evokeds: bool = True
    save_report: bool = True
    save_figures: bool = False
    min_trials: StrictInt = Field(default=10, ge=1)
    bootstrap: bool = False
    bootstrap_n_repeats: StrictInt = Field(default=100, ge=2)
    components: dict[str, ERPComponentSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_task(self):
        if self.tmin >= self.tmax:
            raise ValueError("tmin must be smaller than tmax.")
        if self.baseline is not None:
            lo, hi = self.baseline
            lo = self.tmin if lo is None else lo
            hi = self.tmax if hi is None else hi
            if not self.tmin <= lo <= hi <= self.tmax:
                raise ValueError("Baseline must lie within the epoch.")
        if self.components:
            _names(tuple(self.components))
        for component in self.components.values():
            if not self.tmin <= component.window[0] < component.window[1] <= self.tmax:
                raise ValueError("Component windows must lie within the epoch.")
        return self

    def validate_events(self, spec: ERPEventSpec):
        missing = {
            condition
            for component in self.components.values()
            for condition in component.conditions
        } - set(spec.conditions)
        if missing:
            raise ValueError(
                f"ERP conditions missing from dataset events: {sorted(missing)}"
            )


def require_rest_runner(config):
    """Prevent ERP settings entering a runner before ERP dispatch is implemented."""
    if config.analysis_type == "erp":
        raise NotImplementedError(
            "This runner does not support ERP execution. Use the local Pipeline API or the single-participant CLI with --erp_events."
        )
    if config.analysis_type != "rest":
        raise ValueError(f"Unknown analysis_type: {config.analysis_type!r}")


def resolve_erp_events(raw, spec):
    """Resolve requested conditions without changing the recording.

    Returns an integer MNE events array at absolute original samples and the
    complete logical condition-to-code mapping, including absent conditions.
    Annotation descriptions match exactly. Stimulus mode uses MNE onset
    detection; code zero is its idle state, so zero-coded experimental events
    must be supplied explicitly to run_erp_pipeline instead.
    """
    spec = ERPEventSpec.model_validate(spec)
    if spec.mode == "annotations":
        event_id = {name: i + 1 for i, name in enumerate(spec.annotation_map)}
        descriptions = {
            description: event_id[name]
            for name, description in spec.annotation_map.items()
        }
        if not any(
            description in descriptions for description in raw.annotations.description
        ):
            raise ValueError("No requested condition events were found.")
        events, _ = mne.events_from_annotations(
            raw, event_id=descriptions, regexp=None, use_rounding=True, verbose=False
        )
    else:
        event_id = dict(spec.event_id)
        if 0 in event_id.values():
            raise ValueError(
                "Stimulus onset detection treats code zero as idle; supply explicit events and event_id for zero-coded conditions."
            )
        if spec.stim_channel not in raw.ch_names:
            raise ValueError(f"Stimulus channel {spec.stim_channel!r} is missing.")
        events = mne.find_events(
            raw,
            stim_channel=spec.stim_channel,
            output="onset",
            consecutive=spec.consecutive,
            shortest_event=spec.shortest_event,
            min_duration=spec.min_duration,
            initial_event=spec.initial_event,
            verbose=False,
        )
    events = events[np.isin(events[:, 2], list(event_id.values()))]
    if not len(events):
        raise ValueError("No requested condition events were found.")
    if np.any(np.diff(events[:, 0]) <= 0):
        raise ValueError("Event samples must be strictly increasing and unique.")
    if events[0, 0] < raw.first_samp or events[-1, 0] > raw.last_samp:
        raise ValueError("Event samples must be within the Raw absolute sample range.")
    return events.copy(), event_id


def validate_erp_subject_id(subject_id):
    """Require a literal participant identifier safe for a single directory."""
    if (
        not isinstance(subject_id, str)
        or not subject_id.strip()
        or subject_id != subject_id.strip()
        or subject_id in {".", ".."}
        or any(c in subject_id for c in "/\\\x00")
    ):
        raise ValueError("Participant ID must be a nonempty literal directory name.")
    return subject_id


def erp_metadata_paths(recording_path):
    """Find unambiguous local sidecars for the ERP workflow."""
    from .eeg import eeg_metadata_paths

    return eeg_metadata_paths(recording_path)


def write_erp_json(path, value):
    """Publish strict JSON atomically, including NumPy QC values."""
    import json

    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__} to ERP JSON.")

    _write_erp_atomic(
        path,
        lambda handle: json.dump(
            value, handle, indent=2, allow_nan=False, default=encode
        ),
    )


def write_erp_table(path, table, *, index=True):
    """Publish a CSV atomically."""
    _write_erp_atomic(path, lambda handle: table.to_csv(handle, index=index))


def _write_erp_atomic(path, writer):
    import os
    import tempfile
    from pathlib import Path

    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix="." + path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer(handle)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_erp_montage(
    raw, metadata, settings=None, *, required_channels=(), notify=None
):
    """Validate ERP dataset policy and apply shared EEG position handling."""
    from .eeg import prepare_eeg_montage

    settings = ERPMontageSettings.model_validate(settings or {})
    return prepare_eeg_montage(
        raw,
        metadata,
        **settings.model_dump(),
        required_channels=required_channels,
        notify=notify,
    )


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
