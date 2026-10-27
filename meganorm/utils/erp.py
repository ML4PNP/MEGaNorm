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
    """Resolve ERP-only legacy switches and thresholds without mutating settings."""
    import math

    mode = config.ica_mode
    if mode not in ("auto", "physiological", "iclabel", "none"):
        raise ValueError("ica_mode must be auto, physiological, iclabel or none.")
    if not config.apply_ica:
        if mode not in ("auto", "none"):
            raise ValueError("apply_ica=False conflicts with the explicit ica_mode.")
        mode = "none"
    for value in (
        config.ica_threshold,
        config.ica_corr_threshold,
        config.ica_label_threshold,
    ):
        if value is not None and (not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError("ICA thresholds must be finite values in (0, 1].")
    n = config.ica_n_components
    if n is not None and not (
        type(n) is int and n >= 2 or type(n) is float and 0 < n < 1
    ):
        raise ValueError(
            "ICA components must be an integer >=2 or a fraction in (0, 1)."
        )
    return mode, {
        "correlation": (
            config.ica_threshold
            if config.ica_corr_threshold is None
            else config.ica_corr_threshold
        ),
        "label_probability": (
            config.ica_threshold
            if config.ica_label_threshold is None
            else config.ica_label_threshold
        ),
    }


def resolve_ica_fit_settings(config, sfreq=None):
    """Validate the ERP-only fit copy, independently of analysis resampling.

    A missing ICA low-pass inherits the analysis low-pass for compatibility.
    A missing fit sample rate leaves the copy at the recording's sample rate.
    """
    import math

    high = config.h_freq if config.ica_h_freq is None else config.ica_h_freq
    for value in (config.ica_h_freq, config.ica_resample_sfreq):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("ICA fit frequencies must be positive and finite.")
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


class ERPPreprocessing(_Settings):
    """Settings matching the standalone ERPConfig, with early validation."""

    tmin: float = -0.2
    tmax: float = 0.8
    baseline: tuple[float | None, float | None] | None = (None, 0.0)
    l_freq: float | None = Field(default=0.1, gt=0)
    h_freq: float | None = Field(default=35.0, gt=0)
    ica_l_freq: float = Field(default=1.0, gt=0)
    ica_h_freq: float | None = Field(default=None, gt=0)
    ica_resample_sfreq: float | None = Field(default=None, gt=0)
    reference: str | tuple[str, ...] | None = "average"
    resample_sfreq: float | None = Field(default=None, gt=0)
    apply_ica: bool = True
    ica_n_components: StrictInt | float | None = None
    ica_threshold: float = Field(default=0.8, gt=0, le=1)
    random_state: StrictInt = Field(default=42, ge=0)
    min_trials: StrictInt = Field(default=10, ge=1)
    reject: dict[Literal["eeg"], float] | None = Field(
        default_factory=lambda: {"eeg": 150e-6}
    )
    flat: dict[Literal["eeg"], float] | None = None

    ica_mode: Literal["auto", "physiological", "iclabel", "none"] = "auto"
    ica_corr_threshold: float | None = Field(default=None, gt=0, le=1)
    ica_label_threshold: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def validate_preprocessing(self):
        mode, _ = resolve_ica_settings(self)
        if self.tmin >= self.tmax:
            raise ValueError("tmin must be smaller than tmax.")
        if self.baseline is not None:
            lo, hi = self.baseline
            lo = self.tmin if lo is None else lo
            hi = self.tmax if hi is None else hi
            if not self.tmin <= lo <= hi <= self.tmax:
                raise ValueError("Baseline must lie within the epoch.")
        if (
            self.l_freq is not None
            and self.h_freq is not None
            and self.l_freq >= self.h_freq
        ):
            raise ValueError("l_freq must be below h_freq.")
        resolve_ica_fit_settings(self)
        n = self.ica_n_components
        if n is not None and not (
            type(n) is int and n > 0 or type(n) is float and 0 < n < 1
        ):
            raise ValueError(
                "ICA components must be a positive integer or variance fraction in (0, 1)."
            )
        if isinstance(self.reference, str):
            _name(self.reference)
        elif self.reference is not None:
            _names(self.reference)
        for thresholds in (self.reject, self.flat):
            if thresholds is not None and any(
                value <= 0 for value in thresholds.values()
            ):
                raise ValueError("Rejection thresholds must be positive.")
        if self.reject and self.flat and self.flat["eeg"] >= self.reject["eeg"]:
            raise ValueError("Flat threshold must be below rejection threshold.")
        if self.resample_sfreq is not None:
            self.validate_sampling_rate(self.resample_sfreq, include_ica=False)
        return self

    def validate_sampling_rate(self, sfreq, *, include_ica=True):
        """Validate filter limits against a recording or requested sample rate."""
        import math

        if not math.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("Sampling rate must be finite and positive.")
        cutoffs = [self.l_freq, self.h_freq]
        if include_ica:
            resolve_ica_fit_settings(self, sfreq)
        if any(value is not None and value >= sfreq / 2 for value in cutoffs):
            raise ValueError("Filter cutoffs must be below the Nyquist frequency.")


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


class ERPAutoRejectSettings(_Settings):
    """Participant-level cross-validation choices for EEG epoch repair."""

    cv: StrictInt = Field(default=5, ge=2)
    n_interpolate: tuple[StrictInt, ...] | None = None
    consensus: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)

    @model_validator(mode="after")
    def validate_candidates(self):
        if self.n_interpolate is not None and (
            not self.n_interpolate
            or min(self.n_interpolate) < 0
            or len(set(self.n_interpolate)) != len(self.n_interpolate)
        ):
            raise ValueError("n_interpolate must contain unique nonnegative integers.")
        if (
            not self.consensus
            or len(set(self.consensus)) != len(self.consensus)
            or any(not 0 < value <= 1 for value in self.consensus)
        ):
            raise ValueError(
                "consensus must contain unique finite fractions in (0, 1]."
            )
        return self


def resolve_rejection_settings(mode, options):
    """Validate the shared standalone/cohort epoch-rejection contract."""
    if mode not in ("threshold", "autoreject"):
        raise ValueError("rejection_mode must be threshold or autoreject.")
    if mode == "threshold":
        if options is not None:
            raise ValueError("autoreject settings require rejection_mode='autoreject'.")
        return None
    return ERPAutoRejectSettings.model_validate({} if options is None else options)


class ERPSettings(_Settings):
    """ERP analysis definition independent of dataset trigger encodings."""

    preprocessing: ERPPreprocessing = Field(default_factory=ERPPreprocessing)
    components: dict[str, ERPComponentSettings]
    rejection_mode: Literal["threshold", "autoreject"] = "threshold"
    autoreject: ERPAutoRejectSettings | None = None

    @model_validator(mode="after")
    def validate_components(self):
        resolve_rejection_settings(self.rejection_mode, self.autoreject)
        _names(tuple(self.components))
        for component in self.components.values():
            if (
                not self.preprocessing.tmin
                <= component.window[0]
                < component.window[1]
                <= self.preprocessing.tmax
            ):
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

    def to_runtime(self):
        """Convert validated definitions for the existing standalone ERP API."""
        from meganorm.src.erp import ERPConfig, ERPComponent

        return ERPConfig(
            **self.preprocessing.model_dump(),
            rejection_mode=self.rejection_mode,
            autoreject=(
                None if self.autoreject is None else self.autoreject.model_dump()
            ),
        ), {
            name: ERPComponent(**component.model_dump())
            for name, component in self.components.items()
        }


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
    """Find unambiguous local sidecars using the existing EEG conventions."""
    from pathlib import Path

    path = Path(recording_path)
    stem = path.stem
    for suffix in ("_raw", "_eeg"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    exact = path.with_name(stem + "_channels.tsv")
    channels = (
        [exact]
        if exact.is_file()
        else sorted(path.parent.glob(f'*_{stem.split("_")[-1]}_channels.tsv'))
    )
    exact_montage = path.with_name(stem + "_montage.csv")
    montages = (
        [exact_montage]
        if exact_montage.is_file()
        else sorted(path.parent.glob("*_montage.csv"))
    )
    # Match adjacent BIDS entities; do not choose an arbitrary coordinate space.
    entities = dict(token.split("-", 1) for token in stem.split("_") if "-" in token)
    electrodes = []
    bids_error = None
    for candidate in path.parent.glob("*_electrodes.tsv"):
        tags = dict(
            token.split("-", 1) for token in candidate.stem.split("_") if "-" in token
        )
        if all(
            key == "space" or entities.get(key) == value for key, value in tags.items()
        ):
            electrodes.append((len(tags), candidate))
    if electrodes:
        specificity = max(n for n, _ in electrodes)
        electrodes = [p for n, p in electrodes if n == specificity]
        if len(electrodes) != 1:
            bids_error = f"Ambiguous BIDS electrode sidecars for {path}."
    electrode = electrodes[0] if len(electrodes) == 1 else None
    coordsystem = (
        electrode.with_name(
            electrode.name.replace("_electrodes.tsv", "_coordsystem.json")
        )
        if electrode
        else None
    )
    if coordsystem is not None and not coordsystem.is_file():
        bids_error = f"Missing coordinate-system JSON for {electrode}."
    if len(channels) > 1 or len(montages) > 1:
        raise ValueError(f"Ambiguous EEG metadata sidecars for {path}.")
    return {
        "channels": channels[0] if channels else None,
        "montage": montages[0] if montages else None,
        "electrodes": electrode,
        "coordsystem": coordsystem,
        "bids_error": bids_error,
    }


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
    """Apply supplied or explicitly selected template positions, in memory.

    BIDS support covers adjacent electrode/coordinate-system pairs in CapTrak
    head coordinates. Other frames require an explicit external transformation.
    """
    import json
    import warnings
    import pandas as pd

    settings = ERPMontageSettings.model_validate(settings or {})
    notify = notify or (
        lambda message: warnings.warn(message, UserWarning, stacklevel=2)
    )
    eeg = [
        raw.ch_names[p]
        for p in mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    ]
    excluded = [name for name in settings.exclude_channels if name in eeg]
    if set(excluded) & set(required_channels):
        raise ValueError(
            "Montage exclusions include requested IDP channels: "
            f"{sorted(set(excluded) & set(required_channels))}."
        )
    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + excluded))
    usable = [name for name in eeg if name not in raw.info["bads"]]
    qc = dict(
        source="none",
        kind="unknown",
        template=None,
        aliases={},
        excluded_channels=excluded,
        matched_channels=[],
        coordinate_frame=None,
        units=None,
    )

    def valid_names():
        return [
            name
            for name in usable
            if np.isfinite(raw.info["chs"][raw.ch_names.index(name)]["loc"][:3]).all()
            and np.linalg.norm(raw.info["chs"][raw.ch_names.index(name)]["loc"][:3]) > 0
        ]

    embedded = valid_names()
    # Defer unused BIDS metadata errors until embedded positions are ruled out.
    if not embedded and metadata.get("bids_error"):
        raise ValueError(metadata["bids_error"])
    if embedded:
        if len(embedded) != len(usable):
            raise ValueError(
                "Embedded EEG positions are incomplete; supply a complete montage rather than mixing measured and template coordinates."
            )
        from mne._fiff.constants import FIFF

        if any(
            raw.info["chs"][raw.ch_names.index(name)]["coord_frame"]
            != FIFF.FIFFV_COORD_HEAD
            for name in embedded
        ):
            raise ValueError(
                "Embedded EEG positions must be in head coordinates; transform them explicitly before loading."
            )
        qc.update(
            source="embedded", kind=settings.supplied_kind, matched_channels=embedded
        )
    elif metadata.get("electrodes") is not None:
        spec = json.loads(metadata["coordsystem"].read_text(encoding="utf-8-sig"))
        if spec.get("EEGCoordinateSystem") != "CapTrak":
            raise ValueError(
                "BIDS EEG positions must use CapTrak head coordinates; transform other coordinate systems explicitly before loading."
            )
        scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}.get(spec.get("EEGCoordinateUnits"))
        if scale is None:
            raise ValueError(
                "Unsupported BIDS EEG coordinate units; expected m, cm or mm."
            )
        table = pd.read_csv(metadata["electrodes"], sep="\t", dtype={"name": str})
        if (
            not {"name", "x", "y", "z"}.issubset(table)
            or table["name"].isna().any()
            or table["name"].duplicated().any()
        ):
            raise ValueError("BIDS electrodes require unique name and x/y/z columns.")
        coordinates = table[["x", "y", "z"]].to_numpy(dtype=float) * scale
        raw.set_montage(
            mne.channels.make_dig_montage(
                ch_pos=dict(zip(table["name"], coordinates)), coord_frame="head"
            ),
            on_missing="ignore",
        )
        qc.update(
            source="bids",
            kind=settings.supplied_kind,
            input_coordinate_system=spec["EEGCoordinateSystem"],
            input_units=spec["EEGCoordinateUnits"],
        )
    elif metadata.get("montage") is not None:
        table = pd.read_csv(metadata["montage"], dtype={"Channel": str})
        if (
            not {"Channel", "X", "Y", "Z"}.issubset(table)
            or table.Channel.isna().any()
            or table.Channel.duplicated().any()
        ):
            raise ValueError("Montage CSV requires unique Channel and X/Y/Z columns.")
        coordinates = table[["X", "Y", "Z"]].to_numpy(dtype=float)
        if not np.isfinite(coordinates).all():
            raise ValueError(
                "Montage coordinates must be finite head coordinates in metres."
            )
        raw.set_montage(
            mne.channels.make_dig_montage(
                ch_pos=dict(zip(table.Channel, coordinates)), coord_frame="head"
            ),
            on_missing="ignore",
        )
        qc.update(source="sidecar", kind=settings.supplied_kind)
    elif settings.template is not None and usable:
        template = settings.template
        available = mne.channels.get_builtin_montages()
        if template in {"auto", "standard_1005", "colin27_1005"}:
            template = (
                "colin27_1005" if "colin27_1005" in available else "standard_1005"
            )
        montage = mne.channels.make_standard_montage(template)
        montage.apply_trans(mne.channels.compute_native_head_t(montage))
        positions = montage.get_positions()["ch_pos"]
        aliases = {"TP9 LEFT EAR": "TP9", "TP10 RIGHT EAR": "TP10"}
        mapped = {name: aliases.get(name, name) for name in usable}
        missing = [name for name, target in mapped.items() if target not in positions]
        if missing:
            raise ValueError(
                f"Template {template!r} has no positions for {missing}; supply positions or explicitly list non-IDP channels in exclude_channels."
            )
        if len(set(mapped.values())) != len(mapped):
            raise ValueError(
                "Multiple EEG channels map to the same template electrode."
            )
        raw.set_montage(
            mne.channels.make_dig_montage(
                ch_pos={name: positions[target] for name, target in mapped.items()},
                coord_frame="head",
            ),
            on_missing="ignore",
        )
        qc.update(
            source="template",
            kind="template",
            template=template,
            aliases={name: target for name, target in mapped.items() if name != target},
        )
    if qc["source"] != "none":
        qc["matched_channels"] = valid_names()
        missing = sorted(set(usable) - set(qc["matched_channels"]))
        if missing:
            raise ValueError(
                f"Supplied EEG positions are incomplete or invalid for usable channels: {missing}."
            )
        qc.update(coordinate_frame="head", units="m")
    if qc["kind"] == "template" or excluded:
        notify(
            f"EEG montage source={qc['source']}, kind={qc['kind']}, template={qc['template']}; "
            f"{len(qc['matched_channels'])} usable EEG positions; excluded channels={excluded}."
        )
    qc["bad_channels"] = list(raw.info["bads"])
    return qc
