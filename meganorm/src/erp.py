"""Automatic EEG ERP preparation and configurable sensor-space IDPs.

Time settings are in seconds. EEG amplitudes are exported in microvolts, magnetometer amplitudes
in tesla, gradiometer amplitudes in tesla/metre, and latencies in milliseconds.
"""

from dataclasses import dataclass, field
from typing import Mapping

import mne
import numpy as np
import pandas as pd
from meganorm.utils.erp import ERPEventSpec, resolve_erp_events


def _name(value):
    if not isinstance(value, str) or not value or "__" in value:
        raise ValueError("Names must be nonempty strings without '__'.")


def _window(value):
    if len(value) != 2 or not np.all(np.isfinite(value)) or value[0] >= value[1]:
        raise ValueError("Windows must contain two finite, increasing times.")


@dataclass(frozen=True)
class ERPComponent:
    """A named component's conditions, sensors and measurement definition.

    Parameters
    ----------
    conditions : tuple of str
        Exact condition keys in the evoked dictionary.
    channels : tuple of str
        Individual sensors to measure, in output order. No spatial averaging.
    window : tuple of float
        Inclusive measurement window in seconds relative to the event.
    polarity : {'pos', 'neg', 'abs'}
        Select the largest positive, most negative, or largest absolute sample.
        Amplitude remains signed. Ties select the earliest sample.
    metrics : tuple of str
        Any combination of peak_amplitude, peak_latency and mean_amplitude.
    sensor_type : {'eeg', 'mag', 'grad'}
        Expected type of every requested sensor; determines amplitude units.
    """

    conditions: tuple[str, ...]
    channels: tuple[str, ...]
    window: tuple[float, float]
    polarity: str
    metrics: tuple[str, ...] = ("peak_amplitude", "peak_latency")
    sensor_type: str = "eeg"

    def __post_init__(self):
        for values in (self.conditions, self.channels, self.metrics):
            if isinstance(values, str) or not values or len(set(values)) != len(values):
                raise ValueError(
                    "Conditions, channels and metrics must be nonempty unique sequences."
                )
            for value in values:
                _name(value)
        _window(self.window)
        if self.polarity not in ("pos", "neg", "abs"):
            raise ValueError("polarity must be pos, neg or abs.")
        if self.sensor_type not in ("eeg", "mag", "grad"):
            raise ValueError("sensor_type must be eeg, mag or grad.")
        if set(self.metrics) - {"peak_amplitude", "peak_latency", "mean_amplitude"}:
            raise ValueError("Unknown ERP metric.")


@dataclass(frozen=True)
class ERPConfig:
    """Settings for the automatic EEG raw-data pipeline.

    Filters are configurable protocol choices, not universal ERP defaults.
    ICA uses infomax on a separately filtered copy with the same reference and
    channels. Without EOG/ECG channels, automatic ICLabel classification is used.
    Rejection uses MNE peak-to-peak thresholds in SI units plus BAD annotations.
    Set apply_ica=False only for already-clean data or controlled tests.
    """

    tmin: float = -0.2
    tmax: float = 0.8
    baseline: tuple[float | None, float | None] | None = (None, 0.0)
    l_freq: float | None = 0.1
    h_freq: float | None = 35.0
    ica_l_freq: float = 1.0
    reference: str | tuple[str, ...] | None = "average"
    resample_sfreq: float | None = None
    apply_ica: bool = True
    ica_n_components: int | float | None = None
    ica_threshold: float = 0.8
    random_state: int = 42
    min_trials: int = 10
    reject: dict[str, float] | None = field(default_factory=lambda: {"eeg": 150e-6})
    flat: dict[str, float] | None = None

    def __post_init__(self):
        _window((self.tmin, self.tmax))
        if self.baseline is not None:
            if len(self.baseline) != 2:
                raise ValueError("baseline must contain two times.")
            lo = self.tmin if self.baseline[0] is None else self.baseline[0]
            hi = self.tmax if self.baseline[1] is None else self.baseline[1]
            if (
                not np.all(np.isfinite([lo, hi]))
                or not self.tmin <= lo <= hi <= self.tmax
            ):
                raise ValueError("baseline must lie inside the epoch.")
        for value in (self.l_freq, self.h_freq, self.resample_sfreq, self.ica_l_freq):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(
                    "Filter and sampling frequencies must be positive and finite."
                )
        if self.h_freq is not None:
            if self.l_freq is not None and self.l_freq >= self.h_freq:
                raise ValueError("l_freq must be below h_freq.")
            if self.apply_ica and self.ica_l_freq >= self.h_freq:
                raise ValueError("ica_l_freq must be below h_freq.")
        if not isinstance(self.min_trials, int) or self.min_trials < 1:
            raise ValueError("min_trials must be a positive integer.")
        if not np.isfinite(self.ica_threshold) or not 0 < self.ica_threshold <= 1:
            raise ValueError("ica_threshold must be in (0, 1].")
        for thresholds in (self.reject, self.flat):
            if thresholds is not None and any(
                key != "eeg" or not np.isfinite(value) or value <= 0
                for key, value in thresholds.items()
            ):
                raise ValueError(
                    "EEG rejection thresholds must be positive, finite EEG values."
                )


def _validate_events(raw, events, event_id):
    if not event_id:
        raise ValueError("Provide an explicit condition-to-integer event_id mapping.")
    for name, code in event_id.items():
        _name(name)
        if (
            not isinstance(code, (int, np.integer))
            or isinstance(code, bool)
            or code < 0
        ):
            raise ValueError("Event IDs must be nonnegative integers.")
    if len(set(event_id.values())) != len(event_id):
        raise ValueError("Condition event IDs must be unique.")
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 3 or not len(events):
        raise ValueError("events must be a nonempty (n_events, 3) array.")
    if not np.issubdtype(events.dtype, np.integer):
        raise ValueError("Event samples and codes must be integers.")
    if np.any(np.diff(events[:, 0]) <= 0):
        raise ValueError("Event samples must be strictly increasing and unique.")
    if events[0, 0] < raw.first_samp or events[-1, 0] > raw.last_samp:
        raise ValueError("Event samples must be within the Raw absolute sample range.")
    return events.copy()


def preprocess_erp(raw, config=None):
    """Return an automatically cleaned EEG copy and a preprocessing QC dict.

    Reuses MEGaNorm's physiological correlation helper. Existing bad-channel
    markings, annotations and auxiliary channels are retained. Channel types
    and montage should already be set by the loader or prepare_eeg_data().
    Stimulus events must be resolved before this step by the caller; resampling
    is performed jointly with events by run_erp_pipeline().
    """
    from .preprocess import find_ica_component

    config = config or ERPConfig()
    data = raw.copy().load_data()
    eeg = mne.pick_types(data.info, meg=False, eeg=True, exclude="bads")
    if not len(eeg):
        raise ValueError(
            "No usable EEG channels; raw preprocessing currently supports EEG."
        )
    if not np.isfinite(data.get_data(picks=eeg)).all():
        raise ValueError(
            "Nonfinite EEG samples: repair or annotate/handle these before ERP processing."
        )
    if config.reference is not None:
        reference = (
            list(config.reference)
            if isinstance(config.reference, tuple)
            else config.reference
        )
        data.set_eeg_reference(reference, projection=False, verbose=False)
    fit_data = data.copy() if config.apply_ica else None
    if config.l_freq is not None or config.h_freq is not None:
        data.filter(config.l_freq, config.h_freq, picks=eeg, verbose=False)
    qc = {"ica_applied": False, "ica_excluded": [], "ica_method": None}
    if config.apply_ica:
        fit_data.filter(config.ica_l_freq, config.h_freq, picks=eeg, verbose=False)
        ica = mne.preprocessing.ICA(
            n_components=config.ica_n_components,
            method="infomax",
            fit_params={"extended": True},
            random_state=config.random_state,
            max_iter=800,
            verbose=False,
        )
        ica.fit(fit_data, picks=eeg, reject_by_annotation=True, verbose=False)
        auxiliary = mne.pick_types(
            fit_data.info, meg=False, eog=True, ecg=True, exclude="bads"
        )
        excluded = set()
        if len(auxiliary):
            for signal in fit_data.get_data(picks=auxiliary):
                indices, _ = find_ica_component(
                    ica, fit_data, signal, config.ica_threshold
                )
                excluded.update(indices)
            qc["ica_method"] = "physiological_correlation"
        else:
            from mne_icalabel import label_components

            if fit_data.get_montage() is None:
                raise ValueError(
                    "Automatic ICLabel needs EEG sensor locations when auxiliary channels are absent."
                )
            labels = label_components(fit_data, ica, method="iclabel")
            excluded.update(
                i
                for i, (label, probability) in enumerate(
                    zip(labels["labels"], labels["y_pred_proba"])
                )
                if label
                in {
                    "eye blink",
                    "heart beat",
                    "muscle artifact",
                    "channel noise",
                    "line noise",
                }
                and probability >= config.ica_threshold
            )
            qc["ica_method"] = "iclabel"
        ica.exclude = sorted(excluded)
        ica.apply(data, verbose=False)
        qc.update(ica_applied=True, ica_excluded=ica.exclude.copy())
    return data, qc


def epoch_erp(raw, events, event_id, config=None):
    """Build EEG event-locked epochs with automatic threshold/annotation rejection.

    Events use absolute samples at raw.info['sfreq'], including raw.first_samp.
    Missing conditions are allowed and become unavailable IDPs downstream.
    """
    config = config or ERPConfig()
    events = _validate_events(raw, events, event_id)
    if not np.isin(events[:, 2], list(event_id.values())).any():
        raise ValueError("No requested condition events were found.")
    return mne.Epochs(
        raw,
        events,
        event_id=dict(event_id),
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=config.baseline,
        picks="eeg",
        reject=config.reject,
        flat=config.flat,
        reject_by_annotation=True,
        preload=True,
        on_missing="ignore",
        event_repeated="error",
        verbose=False,
    )


def compute_evokeds(epochs):
    """Average each exact event code, retaining absent conditions as None."""
    return {
        condition: (
            epochs[epochs.events[:, 2] == code].average()
            if np.any(epochs.events[:, 2] == code)
            else None
        )
        for condition, code in epochs.event_id.items()
    }


def extract_erp_idps(subject_id, evokeds, components, min_trials=1):
    """Return (wide subject DataFrame, per-IDP QC DataFrame).

    components maps arbitrary names to ERPComponent objects. evokeds maps exact
    condition names to MNE Evoked objects (or None). Missing conditions, bad or
    absent sensors, nonfinite waveforms and too few trials yield NaN with a QC
    reason. Invalid windows or sensor-type mismatches raise ValueError. Window
    means do not depend on peak polarity. No eligible signed peak yields NaN
    for both peak metrics. Boundary peaks retain their values and have
    boundary_peak=True in peak-metric QC rows (False for means or unavailable
    peaks). Input objects are never modified.
    """
    if not isinstance(subject_id, str) or not subject_id:
        raise ValueError("subject_id must be a nonempty string.")
    if not isinstance(min_trials, int) or min_trials < 1:
        raise ValueError("min_trials must be a positive integer.")
    if not components:
        raise ValueError("At least one component is required.")
    values, quality = {}, []
    units = {"eeg": (1e6, "uV"), "mag": (1.0, "T"), "grad": (1.0, "T_per_m")}
    for name, component in components.items():
        _name(name)
        if not isinstance(component, ERPComponent):
            raise TypeError("Each component must be an ERPComponent.")
        scale, amplitude_unit = units[component.sensor_type]
        for condition in component.conditions:
            evoked = evokeds.get(condition)
            nave = 0 if evoked is None else evoked.nave
            for channel in component.channels:
                reason = "ok"
                measures = {}
                boundary_peak = False
                if evoked is None:
                    reason = "missing_condition"
                elif nave < min_trials:
                    reason = "insufficient_trials"
                elif channel not in evoked.ch_names:
                    reason = "missing_channel"
                elif channel in evoked.info["bads"]:
                    reason = "bad_channel"
                else:
                    index = evoked.ch_names.index(channel)
                    if evoked.get_channel_types()[index] != component.sensor_type:
                        raise ValueError(f"Sensor type mismatch for {channel}.")
                    lo, hi = component.window
                    eps = 1e-12
                    if lo < evoked.times[0] - eps or hi > evoked.times[-1] + eps:
                        raise ValueError(f"Window for {name} lies outside evoked data.")
                    mask = (evoked.times >= lo - eps) & (evoked.times <= hi + eps)
                    if not mask.any():
                        raise ValueError(f"Window for {name} contains no samples.")
                    signal = evoked.data[index, mask]
                    times = evoked.times[mask]
                    if not np.isfinite(signal).all():
                        reason = "nonfinite_data"
                    else:
                        measures["mean_amplitude"] = signal.mean() * scale
                        peak = (
                            int(np.argmax(np.abs(signal)))
                            if component.polarity == "abs"
                            else (
                                int(np.argmax(signal))
                                if component.polarity == "pos"
                                else int(np.argmin(signal))
                            )
                        )
                        valid = (
                            signal[peak] > 0
                            if component.polarity == "pos"
                            else (
                                signal[peak] < 0
                                if component.polarity == "neg"
                                else signal[peak] != 0
                            )
                        )
                        if valid:
                            boundary_peak = peak in (0, len(signal) - 1)
                            measures.update(
                                peak_amplitude=signal[peak] * scale,
                                peak_latency=times[peak] * 1e3,
                            )
                for metric in component.metrics:
                    unit = "ms" if metric == "peak_latency" else amplitude_unit
                    column = f"ERP__{condition}__{name}__{metric}_{unit}__{channel}"
                    values[column] = measures.get(metric, np.nan)
                    status = (
                        "no_peak_with_requested_polarity"
                        if reason == "ok" and metric not in measures
                        else reason
                    )
                    quality.append(
                        {
                            "subject": subject_id,
                            "condition": condition,
                            "idp": column,
                            "status": status,
                            "n_trials": nave,
                            "boundary_peak": (
                                boundary_peak if metric.startswith("peak_") else False
                            ),
                        }
                    )
    return pd.DataFrame([values], index=[subject_id], dtype=float), pd.DataFrame(
        quality
    )


@dataclass
class ERPResult:
    """IDPs and separate QC, with epochs/evokeds available for optional saving."""

    features: pd.DataFrame
    qc: pd.DataFrame
    epochs: mne.BaseEpochs
    evokeds: Mapping
    preprocessing_qc: dict


def extract_recording(
    recording_path,
    subject_id,
    components,
    *,
    annotation_map=None,
    event_spec=None,
    events=None,
    event_id=None,
    config=None,
):
    """Load an EEG recording and return ERP IDPs and QC without saving files.

    Parameters
    ----------
    recording_path : str or path-like
        A recording supported by MNE's generic read_raw reader. Channel types
        and sensor locations must be present in the file when needed. For
        custom metadata preparation, load the Raw object separately, use
        existing preparation helpers, and call run_erp_pipeline instead.
    subject_id : str
        Subject identifier for the single-row feature table.
    components : mapping of str to ERPComponent
        Named component specifications.
    annotation_map : mapping, optional
        Condition names to exact annotation descriptions. Supply this OR
        events and event_id, as for run_erp_pipeline.
    events : ndarray, optional
        Integer MNE events at the original recording sample rate.
    event_id : mapping, optional
        Condition names to integer event codes.
    event_spec : ERPEventSpec or mapping, optional
        Dataset annotation or stimulus specification. Mutually exclusive with
        annotation_map and explicit events/event_id.
    config : ERPConfig, optional
        ERP preprocessing and epoch settings. Defaults to ERPConfig().

    Returns
    -------
    ERPResult
        Wide IDPs, separate QC, epochs and condition averages.
    """
    recording = mne.io.read_raw(str(recording_path), preload=True)
    return run_erp_pipeline(
        recording,
        subject_id,
        components,
        annotation_map=annotation_map,
        event_spec=event_spec,
        events=events,
        event_id=event_id,
        config=config,
    )


def run_erp_pipeline(
    raw,
    subject_id,
    components,
    *,
    events=None,
    event_id=None,
    annotation_map=None,
    event_spec=None,
    config=None,
):
    """Run automatic EEG preparation, epoching and multi-component extraction.

    Supply event_spec, events plus condition-to-integer event_id, or annotation_map
    mapping condition names to exact annotation descriptions. Events are read
    before cleaning; optional resampling updates Raw and events together.
    Save result.features using to_csv() to retain MEGaNorm's subject-row format.
    """
    config = config or ERPConfig()
    if event_spec is not None:
        if annotation_map is not None or events is not None or event_id is not None:
            raise ValueError(
                "Supply one event source: event_spec OR annotation_map OR events and event_id."
            )
        events, event_id = resolve_erp_events(raw, event_spec)
    elif annotation_map is not None:
        if events is not None or event_id is not None:
            raise ValueError("Use annotation_map OR events and event_id.")
        events, event_id = resolve_erp_events(
            raw, ERPEventSpec(mode="annotations", annotation_map=annotation_map)
        )
    if events is None or event_id is None:
        raise ValueError("Provide events and event_id, or annotation_map.")
    events = _validate_events(raw, events, event_id)
    events = events[np.isin(events[:, 2], list(event_id.values()))]
    if not len(events):
        raise ValueError("No requested condition events were found.")
    if config.resample_sfreq is not None:
        ratio = config.resample_sfreq / raw.info["sfreq"]
        projected = np.round(events[:, 0] * ratio).astype(int)
        if np.any(np.diff(projected) <= 0):
            raise ValueError(
                "Requested events collide after resampling; use a higher sampling rate."
            )
    if not components:
        raise ValueError("At least one component is required.")
    for name, component in components.items():
        _name(name)
        if not isinstance(component, ERPComponent):
            raise TypeError("Each component must be an ERPComponent.")
        if component.window[0] < config.tmin or component.window[1] > config.tmax:
            raise ValueError(f"Window for {name} lies outside the epoch.")
        if set(component.conditions) - set(event_id):
            raise ValueError(
                "Component conditions must be declared in the event mapping."
            )
        if component.sensor_type != "eeg":
            raise ValueError(
                "Raw pipeline supports EEG; use extract_erp_idps for prepared MEG evokeds."
            )
    cleaned, preprocessing_qc = preprocess_erp(raw, config)
    if config.resample_sfreq is not None:
        cleaned, events = cleaned.resample(
            config.resample_sfreq, events=events, verbose=False
        )
    events = _validate_events(cleaned, events, event_id)
    epochs = epoch_erp(cleaned, events, event_id, config)
    evokeds = compute_evokeds(epochs)
    features, qc = extract_erp_idps(subject_id, evokeds, components, config.min_trials)
    counts = {
        condition: int(np.sum(events[:, 2] == code))
        for condition, code in event_id.items()
    }
    qc["n_events"] = qc["condition"].map(counts)
    # Mutually exclusive categories reflect MNE's recorded drop reason, not
    # hypothetical additional reasons for an epoch already dropped earlier.
    losses = {
        name: dict(
            n_edge_dropped=0,
            n_annotation_rejected=0,
            n_amplitude_rejected=0,
            n_other_dropped=0,
        )
        for name in event_id
    }
    code_names = {code: name for name, code in event_id.items()}
    # MNE stores annotation descriptions and rejected channel names as plain
    # strings. Do not guess the cause when those labels are indistinguishable.
    ambiguous_labels = set(epochs.ch_names) & (
        set(cleaned.annotations.description) | {"NO_DATA", "TOO_SHORT"}
    )
    for event, reasons in zip(events, epochs.drop_log):
        if not reasons:
            continue
        if any(reason in ambiguous_labels for reason in reasons):
            category = "n_other_dropped"
        elif any(reason in {"NO_DATA", "TOO_SHORT"} for reason in reasons):
            category = "n_edge_dropped"
        elif any(
            reason.lower().startswith("bad") and reason not in epochs.ch_names
            for reason in reasons
        ):
            category = "n_annotation_rejected"
        elif any(reason in epochs.ch_names for reason in reasons):
            category = "n_amplitude_rejected"
        else:
            category = "n_other_dropped"
        losses[code_names[event[2]]][category] += 1
    for category in next(iter(losses.values())):
        qc[category] = qc["condition"].map(
            {name: value[category] for name, value in losses.items()}
        )
    qc["n_rejected"] = qc["n_events"] - qc["n_trials"]
    qc.loc[(qc.status == "missing_condition") & (qc.n_events > 0), "status"] = (
        "no_retained_trials"
    )
    preprocessing_qc["drop_log"] = epochs.drop_log
    return ERPResult(features, qc, epochs, evokeds, preprocessing_qc)


def process_erp_recording(
    recording_path, subject_id, *, settings, event_spec, output_root, line_freq=None
):
    """Process and publish one EEG participant in a fresh run directory.

    Sidecar channel types/status are explicit overrides; an embedded montage
    is preserved. Montage CSV coordinates follow the existing head/metre
    convention (Channel, X, Y, Z). No montage is inferred from channel names.
    Features and QC are published before complete.json. A failed or interrupted
    write never has a completion record. Existing participant outputs are never
    overwritten. No-usable-IDP results are saved with status='failed' for audit.
    """
    from pathlib import Path
    import traceback
    from meganorm.utils.erp import (
        ERPSettings,
        erp_metadata_paths,
        validate_erp_subject_id,
        write_erp_json,
        write_erp_table,
    )

    validate_erp_subject_id(subject_id)
    settings = ERPSettings.model_validate(
        settings.model_dump() if isinstance(settings, ERPSettings) else settings
    )
    event_spec = ERPEventSpec.model_validate(
        event_spec.model_dump() if isinstance(event_spec, ERPEventSpec) else event_spec
    )
    settings.validate_events(event_spec)
    path = Path(recording_path).resolve()
    metadata = erp_metadata_paths(path)
    directory = Path(output_root) / "participants" / subject_id
    directory.mkdir(parents=True, exist_ok=False)
    log = directory / "processing.log"
    log.write_text(f"Participant: {subject_id}\nRecording: {path}\n", encoding="utf-8")
    try:
        raw = mne.io.read_raw(str(path), preload=True, verbose=False)
        with raw:
            if metadata["channels"] is not None:
                channels = pd.read_csv(
                    metadata["channels"], sep="\t", dtype={"name": str}
                )
                if (
                    not {"name", "type"}.issubset(channels)
                    or channels.name.isna().any()
                    or channels.name.duplicated().any()
                ):
                    raise ValueError(
                        "Channel sidecar requires unique name and type columns."
                    )
                mapping = channels.set_index("name")["type"].str.lower().to_dict()
                if set(mapping) - set(raw.ch_names):
                    raise ValueError(
                        "Channel sidecar names do not match recording channels."
                    )
                raw.set_channel_types(mapping)
                if "status" in channels:
                    bad = channels.loc[
                        channels.status.str.lower() == "bad", "name"
                    ].tolist()
                    raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad))
            if raw.get_montage() is None and metadata["montage"] is not None:
                montage = pd.read_csv(metadata["montage"], dtype={"Channel": str})
                if (
                    not {"Channel", "X", "Y", "Z"}.issubset(montage)
                    or montage.Channel.isna().any()
                    or montage.Channel.duplicated().any()
                ):
                    raise ValueError(
                        "Montage CSV requires unique Channel and X/Y/Z columns."
                    )
                coordinates = montage[["X", "Y", "Z"]].to_numpy(dtype=float)
                if not np.isfinite(coordinates).all():
                    raise ValueError(
                        "Montage coordinates must be finite head coordinates in metres."
                    )
                raw.set_montage(
                    mne.channels.make_dig_montage(
                        ch_pos=dict(zip(montage.Channel, coordinates)),
                        coord_frame="head",
                    )
                )
            if raw.info.get("line_freq") is None and line_freq is not None:
                raw.info["line_freq"] = float(line_freq)
            settings.preprocessing.validate_sampling_rate(raw.info["sfreq"])
            config, components = settings.to_runtime()
            result = run_erp_pipeline(
                raw, subject_id, components, event_spec=event_spec, config=config
            )
        finite = np.isfinite(result.features.to_numpy())
        status = "success" if finite.all() else "partial" if finite.any() else "failed"
        result.features.index.name = "participant_id"
        write_erp_table(directory / "features.csv", result.features)
        write_erp_table(directory / "qc.csv", result.qc, index=False)
        write_erp_json(directory / "preprocessing.json", result.preprocessing_qc)
        write_erp_json(
            directory / "provenance.json",
            dict(
                recording=str(path),
                settings=settings.model_dump(mode="json"),
                event_spec=event_spec.model_dump(mode="json"),
                metadata={k: str(v) if v else None for k, v in metadata.items()},
                line_freq=line_freq,
            ),
        )
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"Outcome: {status}\n")
        write_erp_json(
            directory / "complete.json",
            dict(
                subject=subject_id,
                status=status,
                feature_names=list(result.features.columns),
            ),
        )
        return result
    except BaseException:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        raise
