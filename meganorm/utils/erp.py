"""ERP settings, event resolution and synthetic EEG tutorial helpers."""

import mne
import numpy as np
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


def make_demo_recording(seed=42):
    """Return deterministic synthetic EEG Raw and absolute-sample events.

    Parameters
    ----------
    seed : int, default=42
        Seed passed to NumPy default_rng for additive noise.

    Returns
    -------
    raw : mne.io.RawArray
        Two-channel EEG recording with a standard montage.
    events : ndarray of int, shape (40, 3)
        Absolute-sample events with alternating codes 1 and 2.

    Notes
    -----
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
    """Validate and return a trimmed name without reserved separators."""
    if not value or value != value.strip() or "__" in value:
        raise ValueError(
            "Names must be nonempty, trimmed and contain no double underscore."
        )
    return value


def _names(values):
    """Validate a nonempty sequence of unique names."""
    if not values or len(values) != len(set(values)):
        raise ValueError("Names must be nonempty and unique.")
    for value in values:
        _name(value)


class _Settings(BaseModel):
    """Forbid unknown configuration fields and nonfinite numeric values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ERPMontageSettings(_Settings):
    """Dataset-level position fallback; inputs are never rewritten.

    Parameters
    ----------
    template : str or None, optional
        Built-in or supported template fallback; None disables the fallback.
    exclude_channels : tuple of str, optional
        EEG channels marked bad and excluded from position requirements.
    supplied_kind : {"unknown", "measured", "template"}, default="unknown"
        Provenance label for supplied positions.

    Raises
    ------
    ValueError
        If the template is unknown or excluded names are invalid.
    """

    template: str | None = None
    exclude_channels: tuple[str, ...] = ()
    supplied_kind: Literal["unknown", "measured", "template"] = "unknown"

    @model_validator(mode="after")
    def validate_montage(self):
        """Validate the configured montage policy.

        Returns
        -------
        self : ERPMontageSettings
            The validated settings instance.

        Raises
        ------
        ValueError
            If the configured names or settings violate the schema.
        """
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
    """One EEG component and its ordered condition, channel and metric selections.

    Parameters
    ----------
    conditions : tuple of str
        Unique logical condition names.
    channels : tuple of str
        Unique EEG sensors, measured individually in this order.
    window : tuple of float
        Finite increasing inclusive measurement interval in seconds.
    polarity : {"pos", "neg", "abs"}
        Positive, negative or absolute peak selection; amplitudes remain signed.
    metrics : tuple of str, optional
        Peak amplitude, peak latency and/or mean amplitude; defaults to both peaks.
    sensor_type : {"eeg"}, default="eeg"
        Sensor type supported by the raw ERP workflow.

    Raises
    ------
    ValueError
        If selections are empty, duplicated or invalid, or the window is not increasing.
    """

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
        """Validate the configured component selections.

        Returns
        -------
        self : ERPComponentSettings
            The validated settings instance.

        Raises
        ------
        ValueError
            If the configured names or settings violate the schema.
        """
        _names(self.conditions)
        _names(self.channels)
        _names(self.metrics)
        if self.window[0] >= self.window[1]:
            raise ValueError("Component window must be increasing.")
        return self


def resolve_ica_settings(config):
    """Return the validated canonical ICA mode and scientific QC thresholds.

    Parameters
    ----------
    config : Config
        Validated configuration with ICA settings.

    Returns
    -------
    mode : str
        Requested strategy, or "none" when ICA is disabled.
    thresholds : dict
        Correlation and label-probability thresholds.

    Raises
    ------
    ValueError
        If disabled ICA conflicts with an explicit strategy.
    """
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
    """Validate the ICA fit copy independently of analysis resampling.

    Parameters
    ----------
    config : Config
        Validated configuration with ICA filter and resampling settings.
    sfreq : float, optional
        Original recording sampling rate in Hz.

    Returns
    -------
    high : float or None
        Configured ICA fit low-pass cutoff in Hz.

    Raises
    ------
    ValueError
        If enabled ICA has unordered cutoffs or a cutoff reaches Nyquist.
    """
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

    Parameters
    ----------
    mode : {"annotations", "stim"}
        Source of recording events.
    annotation_map : dict of str to str or None, optional
        Logical conditions to unique exact descriptions; required for annotations.
    event_id : dict of str to int or None, optional
        Logical conditions to unique nonnegative trigger codes; required for stim.
    stim_channel : str or None, optional
        Trigger channel required for stim mode.
    consecutive : bool or {"increasing"}, default="increasing"
        MNE stimulus transition policy.
    shortest_event : int, default=2
        Minimum event length in samples for MNE stimulus detection.
    min_duration : float, default=0.0
        Minimum stimulus event duration in seconds.
    initial_event : bool, default=False
        Detect an event at the first sample when the stimulus channel is nonzero.

    Raises
    ------
    ValueError
        If mappings are missing, conflicting or duplicated, or options are invalid.

    Notes
    -----
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
        """Validate the configured event mapping.

        Returns
        -------
        self : ERPEventSpec
            The validated settings instance.

        Raises
        ------
        ValueError
            If the configured names or settings violate the schema.
        """
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
        """Return the logical condition names.

        Returns
        -------
        conditions : tuple of str
            Logical condition names in mapping insertion order.
        """
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
        """Validate epoch, baseline and component-window consistency."""
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
        """Check that every component condition has an event mapping.

        Parameters
        ----------
        spec : ERPEventSpec
            Dataset-level annotation or stimulus event specification.

        Raises
        ------
        ValueError
            If any component condition is absent from the specification.
        """
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
    """Reject non-rest configurations in rest-only runners.

    Parameters
    ----------
    config : Config
        Configuration whose analysis_type is checked.

    Returns
    -------
    None
        This function is called for its side effects.

    Raises
    ------
    NotImplementedError
        If ERP execution is requested for a rest-only runner.
    ValueError
        If analysis_type is neither rest nor erp.
    """
    if config.analysis_type == "erp":
        raise NotImplementedError(
            "This runner does not support ERP execution. Use the local Pipeline API or the single-participant CLI with --erp_events."
        )
    if config.analysis_type != "rest":
        raise ValueError(f"Unknown analysis_type: {config.analysis_type!r}")


def resolve_erp_events(raw, spec):
    """Resolve requested conditions without changing the recording.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording with EEG channel types and required sensor locations already set.
    spec : ERPEventSpec or mapping
        Validated annotation or stimulus event mapping.

    Returns
    -------
    events : ndarray of int, shape (n_events, 3)
        Selected events at original absolute sample indices.
    event_id : dict of str to int
        Complete logical condition mapping, including conditions without events.

    Raises
    ------
    ValueError
        If no requested events exist, samples are invalid, a stimulus channel
        is missing, or stimulus onset detection is asked to detect code zero.

    Notes
    -----
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
    """Require a literal participant identifier safe for a single directory.

    Parameters
    ----------
    subject_id : str
        Exact participant identifier used for the feature row.

    Returns
    -------
    subject_id : str
        The unchanged identifier.

    Raises
    ------
    ValueError
        If the identifier is empty, untrimmed or unsafe as a directory name.
    """
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
    """Find unambiguous local sidecars for the ERP workflow.

    Parameters
    ----------
    recording_path : path-like
        Recording whose local sidecars should be located.

    Returns
    -------
    metadata : dict
        Paths under channels, montage, electrodes and coordsystem, with None
        for absent sidecars; bids_error describes ambiguous or incomplete BIDS
        coordinate metadata when present.

    Raises
    ------
    ValueError
        If local channel tables or montage CSV files are ambiguous.
    """
    from .eeg import eeg_metadata_paths

    return eeg_metadata_paths(recording_path)


def write_erp_json(path, value):
    """Publish strict JSON atomically, including NumPy QC values.

    Parameters
    ----------
    path : path-like
        Destination in an existing parent directory.
    value : object
        JSON-compatible data; NumPy scalars and arrays are supported.

    Returns
    -------
    None
        This function is called for its side effects.

    Raises
    ------
    TypeError
        If a value is not serializable.
    ValueError
        If a value contains NaN or infinity.
    """
    import json

    def encode(value):
        """Convert NumPy values to JSON-compatible Python values."""
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
    """Publish a CSV atomically.

    Parameters
    ----------
    path : path-like
        CSV destination in an existing parent directory.
    table : pandas.DataFrame
        Table to serialize.
    index : bool, default=True
        Include row labels in the CSV.

    Returns
    -------
    None
        This function is called for its side effects.
    """
    _write_erp_atomic(path, lambda handle: table.to_csv(handle, index=index))


def _write_erp_atomic(path, writer):
    """Write text to a temporary sibling and atomically replace the target."""
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
    """Validate ERP dataset policy and apply shared EEG positions in place.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording with EEG channel types and required sensor locations already set.
    metadata : dict
        Channel and montage sidecar paths.
    settings : ERPMontageSettings or mapping, optional
        Position fallback policy; None uses defaults.
    required_channels : sequence of str, optional
        Sensors whose position availability is reported.
    notify : callable, optional
        Callback accepting a position-handling notice string.

    Returns
    -------
    qc : dict
        Position-source and availability information from shared EEG preparation.

    Raises
    ------
    ValueError
        If montage settings or supplied metadata are invalid.
    """
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
