"""Automatic EEG ERP preparation and configurable sensor-space IDPs.

Time settings are in seconds. EEG amplitudes are exported in microvolts, magnetometer amplitudes
in tesla, gradiometer amplitudes in tesla/metre, and latencies in milliseconds.
"""

from dataclasses import dataclass, field
from meganorm.utils.erp_bootstrap import ERPBootstrapSummary
from typing import Mapping
from meganorm.utils.erp_logging import (
    ERPProgress,
    stage,
    event as progress_event,
    skipped,
    runtime_versions,
)

import mne
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from meganorm.utils.erp import (
    ERPEventSpec,
    resolve_erp_events,
    resolve_ica_settings,
    resolve_ica_fit_settings,
)


def _name(value):
    """Validate a nonempty feature name without reserved separators."""
    if not isinstance(value, str) or not value or "__" in value:
        raise ValueError("Names must be nonempty strings without '__'.")


def _window(value):
    """Validate two finite increasing window bounds."""
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
        Select the largest positive, most negative, or largest absolute local peak.
        Amplitude remains signed. Peaks require support on both sides inside
        the window. Flat tops use their midpoint; ties select the earlier peak.
    metrics : tuple of str
        Any combination of peak_amplitude, peak_latency and mean_amplitude.
    sensor_type : {'eeg', 'mag', 'grad'}
        Expected type of every requested sensor; determines amplitude units.

    Raises
    ------
    ValueError
        If names or selections are invalid, the window is not increasing, or
        polarity, metrics or sensor type are unsupported.
    """

    conditions: tuple[str, ...]
    channels: tuple[str, ...]
    window: tuple[float, float]
    polarity: str
    metrics: tuple[str, ...] = ("peak_amplitude", "peak_latency")
    sensor_type: str = "eeg"

    def __post_init__(self):
        """Validate component names, window, polarity, metrics and sensor type."""
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


def _validate_erp_config(config):
    """Use the same canonical validation for standalone, CLI and cohort calls."""
    from meganorm.utils.IO import Config

    if config is None:
        config = Config(analysis_type="erp", which_sensor="eeg", erp={})
    else:
        config = Config.model_validate(
            config.model_dump() if isinstance(config, Config) else config
        )
    if config.analysis_type != "erp":
        raise ValueError("ERP processing requires analysis_type='erp'.")
    return config


def _erp_components(config):
    """Materialize measurement objects from canonical task definitions."""
    return {
        name: ERPComponent(**component.model_dump())
        for name, component in config.erp.components.items()
    }


def _validate_events(raw, events, event_id):
    """Validate absolute event samples and codes, then return an array copy."""
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

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording with EEG channel types and required sensor locations already set.
    config : Config or mapping, optional
        Canonical ERP configuration. None constructs the default ERP configuration.

    Returns
    -------
    cleaned : mne.io.BaseRaw
        Loaded, rereferenced, filtered and optionally ICA-cleaned copy.
    qc : dict
        ICA strategy, fitted rank, component decisions and existing bad channels.

    Raises
    ------
    ValueError
        If configuration, EEG samples, auxiliary signals, montage or ICA rank are invalid.

    Notes
    -----
    Reuses MEGaNorm's physiological correlation helper. Existing bad-channel
    markings, annotations and auxiliary channels are retained. Channel types
    and montage should already be set by the loader or prepare_eeg_data().
    Stimulus events must be resolved before this step by the caller; resampling
    is performed jointly with events by run_erp_pipeline().
    """
    from .preprocess import find_ica_component, _fit_ica, _select_iclabel_components

    config = _validate_erp_config(config)
    mode, thresholds = resolve_ica_settings(config)
    fit_h_freq = resolve_ica_fit_settings(config, raw.info["sfreq"])
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
    if config.rereference_method is not None:
        reference = (
            list(config.rereference_method)
            if isinstance(config.rereference_method, tuple)
            else config.rereference_method
        )
        with stage("reference", reference=reference):
            data.set_eeg_reference(reference, projection=False, verbose=False)
    else:
        skipped("reference", reason="rereference_method is None")
    fit_data = data.copy() if mode != "none" else None
    if config.digital_filter and (
        config.cutoffFreqLow is not None or config.cutoffFreqHigh is not None
    ):
        with stage(
            "analysis_filter",
            l_freq=config.cutoffFreqLow,
            h_freq=config.cutoffFreqHigh,
            n_eeg_channels=len(eeg),
        ):
            data.filter(
                config.cutoffFreqLow, config.cutoffFreqHigh, picks=eeg, verbose=False
            )
    else:
        skipped(
            "analysis_filter", reason="digital_filter disabled or both cutoffs unset"
        )
    qc = {
        "ica_applied": False,
        "ica_excluded": [],
        "ica_method": None,
        "ica_mode_requested": config.ica_mode,
        "ica_mode_effective": mode,
        "ica_selection_reason": "disabled by configuration",
        "ica_thresholds": thresholds,
        "ica_auxiliary": {},
        "ica_component_scores": [],
        "ica_warnings": [],
        "ica_rank": None,
        "ica_rank_estimated": None,
        "ica_fit_sfreq": None,
        "ica_n_components": None,
        "ica_reference": config.rereference_method,
        "ica_fit_filter": None,
        "bad_channel_detection": "not_performed",
        "bad_channels": list(data.info["bads"]),
    }
    if mode != "none":
        with stage("ica_strategy", requested_mode=config.ica_mode) as detail:
            annotation_selection = (
                "omit" if config.ica_if_reject_by_annotation else None
            )
            # Auxiliary quality and correlations use the same annotation-retained
            # samples as ICA fitting. Do not filter nonfinite auxiliary channels.
            auxiliary = mne.pick_types(
                fit_data.info, meg=False, eog=True, ecg=True, exclude=[]
            )
            usable = []
            for pick in auxiliary:
                name = fit_data.ch_names[pick]
                signal = fit_data.get_data(
                    picks=[pick], reject_by_annotation=annotation_selection
                )[0]
                if name in fit_data.info["bads"]:
                    status = "marked_bad"
                elif len(signal) < 2:
                    status = "insufficient_samples"
                elif not np.isfinite(signal).all():
                    status = "nonfinite"
                elif np.ptp(signal) == 0:
                    status = "flat"
                else:
                    status = "usable"
                    usable.append(pick)
                qc["ica_auxiliary"][name] = status
            if mode == "auto":
                mode = "physiological" if usable else "iclabel"
                reason = (
                    "usable EOG/ECG channels"
                    if usable
                    else "no usable EOG/ECG channels"
                )
            else:
                reason = "explicit strategy"
            qc.update(ica_mode_effective=mode, ica_selection_reason=reason)
            if mode == "physiological" and not usable:
                raise ValueError(
                    "Physiological ICA requires usable EOG/ECG channels; "
                    f"quality: {qc['ica_auxiliary']}."
                )
            if mode == "iclabel":
                positions = np.array([fit_data.info["chs"][p]["loc"][:3] for p in eeg])
                if (
                    fit_data.get_montage() is None
                    or not np.isfinite(positions).all()
                    or np.any(np.linalg.norm(positions, axis=1) == 0)
                ):
                    raise ValueError(
                        "ICLabel requires finite EEG sensor locations for every "
                        f"usable EEG channel; auxiliary quality: {qc['ica_auxiliary']}."
                    )
                if config.rereference_method != "average" or config.ica_l_freq < 1:
                    raise ValueError(
                        "ICLabel requires reference='average' and ica_l_freq >= 1 Hz."
                    )
                if config.ica_method != "infomax":
                    qc["ica_warnings"].append(
                        "ICLabel was trained using extended Infomax ICA; the configured "
                        "ICA method differs and classification performance may be affected."
                    )
                if config.ica_l_freq != 1 or fit_h_freq != 100:
                    qc["ica_warnings"].append(
                        "ICLabel was trained on 1–100 Hz data; the configured ICA fit "
                        "filter differs and classification performance may be affected."
                    )
            detail.update(
                effective_mode=mode, reason=reason, auxiliary=qc["ica_auxiliary"]
            )
        with stage(
            "ica_fit_preparation",
            l_freq=config.ica_l_freq,
            h_freq=fit_h_freq,
            resample_sfreq=config.ica_resample_sfreq,
        ) as detail:
            fit_data.filter(config.ica_l_freq, fit_h_freq, picks=eeg, verbose=False)
            if config.ica_resample_sfreq is not None:
                fit_data.resample(config.ica_resample_sfreq, verbose=False)
                if mode == "physiological":
                    for pick in usable:
                        signal = fit_data.get_data(
                            picks=[pick], reject_by_annotation=annotation_selection
                        )[0]
                        if not np.isfinite(signal).all() or np.ptp(signal) == 0:
                            raise ValueError(
                                f"Auxiliary {fit_data.ch_names[pick]!r} became nonfinite "
                                "or flat after ICA fit resampling; repair the auxiliary "
                                "signal or leave ica_resample_sfreq=None."
                            )
            qc["ica_fit_filter"] = {"l_freq": config.ica_l_freq, "h_freq": fit_h_freq}
            qc["ica_fit_sfreq"] = float(fit_data.info["sfreq"])
            # Loaded samples already include active projectors. Inactive SSP must
            # not reduce rank: ICA does not apply those projectors during fitting.
            rank = int(
                mne.compute_rank(fit_data.copy().pick(eeg), proj=False, verbose=False)[
                    "eeg"
                ]
            )
            qc["ica_rank_estimated"] = rank
            # Average referencing imposes an exact dependency on the usable EEG
            # channels, even when numerical rank estimation misses that dependency.
            if config.rereference_method == "average":
                rank = min(rank, len(eeg) - 1)
            qc["ica_rank"] = rank
            n_components = config.ica_n_component
            if rank < 2 or (type(n_components) is int and n_components > rank):
                raise ValueError(
                    f"ICA needs EEG rank >=2 and n_components <= rank; "
                    f"rank={rank}, requested={n_components}."
                )
            detail.update(
                rank=rank,
                rank_estimated=qc["ica_rank_estimated"],
                fit_sfreq=qc["ica_fit_sfreq"],
            )
        with stage(
            "ica_fit",
            method=config.ica_method,
            n_components=rank if n_components is None else n_components,
            random_state=config.random_state,
            max_iter=config.ica_max_iter,
            reject_by_annotation=config.ica_if_reject_by_annotation,
        ) as detail:
            ica = _fit_ica(
                fit_data,
                n_components=rank if n_components is None else n_components,
                method=config.ica_method,
                fit_params=(
                    {"extended": True} if config.ica_method == "infomax" else None
                ),
                random_state=config.random_state,
                max_iter=config.ica_max_iter,
                picks=eeg,
                reject_by_annotation=config.ica_if_reject_by_annotation,
            )
            if ica.n_components_ > rank:
                raise ValueError("Fitted ICA components exceed the effective EEG rank.")
            qc["ica_n_components"] = int(ica.n_components_)
            detail["n_components_fitted"] = qc["ica_n_components"]
        with stage(
            "ica_classification", effective_mode=mode, thresholds=thresholds
        ) as detail:
            excluded = set()
            if mode == "physiological":
                # The shared helper chooses the strongest correlated component per
                # auxiliary channel. Preserve that established rest/ERP behavior.
                retained = fit_data.get_data(reject_by_annotation=annotation_selection)
                correlation_data = mne.io.RawArray(
                    retained, fit_data.info.copy(), verbose=False
                )
                for pick in usable:
                    indices, scores = find_ica_component(
                        ica, correlation_data, retained[pick], thresholds["correlation"]
                    )
                    excluded.update(indices)
                    qc["ica_component_scores"].extend(
                        {
                            "channel": fit_data.ch_names[pick],
                            "component": int(i),
                            "correlation": float(score),
                        }
                        for i, score in zip(indices, scores)
                    )
                qc["ica_method"] = "physiological_correlation"
            else:
                from mne_icalabel import label_components

                labels = label_components(fit_data, ica, method="iclabel")
                for i, (label, probability) in enumerate(
                    zip(labels["labels"], labels["y_pred_proba"])
                ):
                    qc["ica_component_scores"].append(
                        {
                            "component": i,
                            "label": str(label),
                            "probability": float(probability),
                        }
                    )
                excluded.update(
                    _select_iclabel_components(
                        labels,
                        {
                            "eye blink",
                            "heart beat",
                            "muscle artifact",
                            "channel noise",
                            "line noise",
                        },
                        thresholds["label_probability"],
                        inclusive=True,
                    )
                )
                qc["ica_method"] = "iclabel"
            detail.update(
                excluded_components=sorted(excluded),
                component_scores=qc["ica_component_scores"],
                warnings=qc["ica_warnings"],
            )
        with stage("ica_apply", excluded_components=sorted(excluded)):
            ica.exclude = sorted(excluded)
            ica.apply(data, verbose=False)
            qc.update(ica_applied=True, ica_excluded=ica.exclude.copy())
    else:
        skipped("ica", reason="disabled by configuration")
    progress_event("preprocessing_summary", **qc)
    return data, qc


def reject_erp_epochs(epochs, config=None):
    """Clean a copy of existing event-locked epochs and return rejection QC.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Event-locked epochs with condition codes in ``event_id``.
    config : Config or mapping, optional
        Canonical ERP configuration. None constructs the default ERP configuration.

    Returns
    -------
    cleaned : mne.BaseEpochs
        Loaded copy after configured rejection and optional interpolation.
    qc : dict
        Retained counts, rejected event indices and per-trial interpolation details.

    Raises
    ------
    ValueError
        If configuration or the data required for AutoReject are invalid.

    Notes
    -----
    Times, baseline, events, metadata and original selection indices are kept.
    AutoReject replaces fixed amplitude rejection; flat thresholds still apply
    before fitting. Inputs already dropped by a caller cannot be recovered.
    """
    config = _validate_erp_config(config)
    use_autoreject = config.bad_segment_removal_method == "autoreject"
    reject = (
        {"eeg": config.eeg_var_threshold}
        if config.eeg_var_threshold is not None
        else None
    )
    flat = (
        {"eeg": config.eeg_flat_threshold}
        if config.eeg_flat_threshold is not None
        else None
    )
    cleaned = epochs.copy().load_data()
    n_input = len(cleaned)
    report = dict(
        mode={None: "none", "fixed_thr": "threshold", "autoreject": "autoreject"}[
            config.bad_segment_removal_method
        ],
        n_input_epochs=n_input,
        n_retained_epochs=n_input,
        n_prefit_flat_rejected=0,
        autoreject_rejected_event_indices=[],
        n_interpolated_trials=0,
        n_interpolated_channel_epochs=0,
        trials=[],
    )
    if not use_autoreject:
        cleaned.drop_bad(
            reject=reject if config.bad_segment_removal_method == "fixed_thr" else None,
            flat=flat if config.bad_segment_removal_method is not None else None,
            verbose=False,
        )
        report["n_retained_epochs"] = len(cleaned)
        return cleaned, report

    # Preserve annotations/edge exclusions made by Epochs construction. Do not
    # apply the fixed amplitude cutoff before a repair-capable algorithm.
    cleaned.drop_bad(reject=None, flat=flat, verbose=False)
    report["n_prefit_flat_rejected"] = n_input - len(cleaned)
    minimum = max(5, config.autoreject_cv)
    if len(cleaned) < minimum:
        raise ValueError(
            f"AutoReject requires at least {minimum} retained epochs "
            "after annotation/edge/flat rejection; no fallback is applied."
        )
    picks = mne.pick_types(cleaned.info, meg=False, eeg=True, exclude="bads")
    if len(picks) < 4:
        raise ValueError("AutoReject requires at least 4 usable EEG channels.")
    values = cleaned.get_data(picks=picks)
    if not np.isfinite(values).all():
        raise ValueError("AutoReject requires finite EEG epoch samples.")
    positions = np.array([cleaned.info["chs"][p]["loc"][:3] for p in picks])
    if (
        not np.isfinite(positions).all()
        or np.any(np.linalg.norm(positions, axis=1) == 0)
        or len(np.unique(positions, axis=0)) != len(picks)
    ):
        raise ValueError(
            "AutoReject requires distinct finite nonzero EEG sensor locations."
        )
    candidates = (
        tuple(value for value in (1, 2, 4) if value < len(picks))
        if config.autoreject_n_interpolates is None
        else config.autoreject_n_interpolates
    )
    if any(value >= len(picks) for value in candidates):
        raise ValueError(
            "AutoReject n_interpolate must be below the usable EEG channel count."
        )
    if not any(
        fraction * len(picks) > count
        for fraction in config.autoreject_consensus_percs
        for count in candidates
    ):
        raise ValueError(
            "AutoReject consensus and n_interpolate have no valid combination."
        )
    from autoreject import AutoReject
    from .preprocess import _fit_autoreject_epochs

    with stage(
        "autoreject_fit",
        cv=config.autoreject_cv,
        n_epochs=len(cleaned),
        n_channels=len(picks),
        n_interpolate=list(candidates),
        consensus=list(config.autoreject_consensus_percs),
        thresh_method=config.autoreject_thresh_method,
        random_state=config.random_state,
    ):
        model = _fit_autoreject_epochs(
            cleaned,
            estimator=AutoReject,
            n_interpolate=list(candidates),
            consensus=list(config.autoreject_consensus_percs),
            cv=config.autoreject_cv,
            thresh_method=config.autoreject_thresh_method,
            random_state=config.random_state,
            picks=picks,
            n_jobs=1,
            verbose=False,
        )
    if not np.isfinite(np.mean(model.loss_["eeg"], axis=-1)).any():
        raise ValueError(
            "AutoReject could not find a finite cross-validation loss; "
            "inspect the epochs and candidate settings."
        )
    log = model.get_reject_log(cleaned)
    bad = np.asarray(log.bad_epochs, dtype=bool)
    labels = np.asarray(log.labels)
    if bad.shape != (len(cleaned),) or labels.shape != (
        len(cleaned),
        len(cleaned.ch_names),
    ):
        raise ValueError(
            "AutoReject returned a rejection log with an unexpected shape."
        )
    for i, (selection, event) in enumerate(zip(cleaned.selection, cleaned.events)):
        interpolated = (
            [] if bad[i] else [cleaned.ch_names[p] for p in picks if labels[i, p] == 2]
        )
        report["trials"].append(
            dict(
                event_index=int(selection),
                event_sample=int(event[0]),
                event_code=int(event[2]),
                retained=not bool(bad[i]),
                bad_channels=[
                    cleaned.ch_names[p] for p in picks if labels[i, p] in (1, 2)
                ],
                interpolated_channels=interpolated,
            )
        )
    report.update(
        autoreject_rejected_event_indices=cleaned.selection[bad].astype(int).tolist(),
        n_interpolated_trials=sum(
            bool(t["interpolated_channels"]) for t in report["trials"]
        ),
        n_interpolated_channel_epochs=sum(
            len(t["interpolated_channels"]) for t in report["trials"]
        ),
        cv=config.autoreject_cv,
        thresh_method=config.autoreject_thresh_method,
        random_state=config.random_state,
        n_interpolate_candidates=list(candidates),
        consensus_candidates=list(config.autoreject_consensus_percs),
        n_interpolate=int(model.n_interpolate_["eeg"]),
        consensus=float(model.consensus_["eeg"]),
        thresholds={name: float(value) for name, value in model.threshes_.items()},
        fitted_channels=[cleaned.ch_names[p] for p in picks],
    )
    if bad.all():
        # Keep normal all-rejected IDP/QC reporting; there are no retained
        # channels to repair, and no need to interpolate discarded data.
        cleaned.drop(np.arange(len(cleaned)), reason="AUTOREJECT", verbose=False)
    else:
        with stage("autoreject_transform", n_rejected=int(bad.sum())):
            cleaned = model.transform(cleaned, reject_log=log)
    report["n_retained_epochs"] = len(cleaned)
    return cleaned, report


def epoch_erp(raw, events, event_id, config=None, *, return_qc=False):
    """Build and clean EEG event-locked epochs at absolute event samples.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording with EEG channel types and required sensor locations already set.
    events : ndarray of int, shape (n_events, 3)
        MNE events with unique increasing absolute sample indices.
    event_id : mapping of str to int
        Logical condition names and unique nonnegative event codes.
    config : Config or mapping, optional
        Canonical ERP configuration. None constructs the default ERP configuration.
    return_qc : bool, default=False
        Include the epoch-rejection QC dictionary in the return value.

    Returns
    -------
    epochs : mne.Epochs
        Baseline-corrected EEG epochs after annotation, edge and configured rejection.
    qc : dict
        Rejection details; returned only when ``return_qc=True``.

    Raises
    ------
    ValueError
        If events are invalid, no requested events exist, or configuration is invalid.

    Notes
    -----
    Return Epochs by default, or (Epochs, rejection QC) with return_qc=True.
    Missing conditions remain in the event mapping for downstream IDP reporting.
    """
    config = _validate_erp_config(config)
    events = _validate_events(raw, events, event_id)
    if not np.isin(events[:, 2], list(event_id.values())).any():
        raise ValueError("No requested condition events were found.")
    with stage(
        "epoching",
        tmin=config.erp.tmin,
        tmax=config.erp.tmax,
        baseline=config.erp.baseline,
        reject_by_annotation=True,
        n_events=len(events),
    ) as detail:
        epochs = mne.Epochs(
            raw,
            events,
            event_id=dict(event_id),
            tmin=config.erp.tmin,
            tmax=config.erp.tmax,
            baseline=config.erp.baseline,
            picks="eeg",
            reject=None,
            flat=None,
            reject_by_annotation=True,
            preload=True,
            on_missing="ignore",
            event_repeated="error",
            verbose=False,
        )
        detail["n_epochs_after_annotation_and_edge_rejection"] = len(epochs)
    with stage(
        "epoch_cleaning",
        method=config.bad_segment_removal_method,
        amplitude_threshold=(
            config.eeg_var_threshold
            if config.bad_segment_removal_method == "fixed_thr"
            else None
        ),
        flat_threshold=(
            config.eeg_flat_threshold
            if config.bad_segment_removal_method is not None
            else None
        ),
    ) as detail:
        epochs, report = reject_erp_epochs(epochs, config)
        detail.update(
            {
                key: value
                for key, value in report.items()
                if key not in {"trials", "autoreject_rejected_event_indices"}
            }
        )
    return (epochs, report) if return_qc else epochs


def compute_evokeds(epochs):
    """Average each exact event code, retaining absent conditions as None.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Event-locked epochs with condition codes in ``event_id``.

    Returns
    -------
    evokeds : dict of str to mne.Evoked or None
        Exact-code averages in event mapping order; None for absent conditions.
    """
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

    Parameters
    ----------
    subject_id : str
        Exact participant identifier used for the feature row.
    evokeds : mapping of str to mne.Evoked or None
        Prepared condition averages, with native SI amplitudes.
    components : mapping of str to ERPComponent
        Named measurements in output order.
    min_trials : int, default=1
        Minimum ``nave`` required for measurements.

    Returns
    -------
    features : pandas.DataFrame
        One participant row with ordered ``ERP__`` feature columns. EEG amplitudes
        are in microvolts, MEG amplitudes in T or T/m, and latencies in ms.
    qc : pandas.DataFrame
        One row per feature, including status, trial count and boundary-peak flag.

    Raises
    ------
    ValueError
        If identifiers, minimum trials, windows or sensor types are invalid.
    TypeError
        If a component is not an ERPComponent.

    Notes
    -----
    components maps arbitrary names to ERPComponent objects. evokeds maps exact
    condition names to MNE Evoked objects (or None). Missing conditions, bad or
    absent sensors, nonfinite waveforms and too few trials yield NaN with a QC
    reason. Invalid windows or sensor-type mismatches raise ValueError. Window
    means do not depend on peak polarity. No eligible signed peak yields NaN
    for both peak metrics. Peaks require lower polarity-adjusted samples on both
    sides within the inclusive window; endpoints cannot qualify. Flat tops use
    their midpoint sample (rounded down); equal extrema use the earlier peak.
    The strongest eligible local extremum is selected without extra smoothing
    or prominence thresholds. QC records peak_method="local_extremum_v1".
    The legacy boundary_peak column remains False. Input objects are unchanged.
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
                        score = (
                            np.abs(signal)
                            if component.polarity == "abs"
                            else signal if component.polarity == "pos" else -signal
                        )
                        candidates, _ = find_peaks(score)
                        candidates = candidates[score[candidates] > 0]
                        if candidates.size:
                            peak = candidates[np.argmax(score[candidates])]
                            measures.update(
                                peak_amplitude=signal[peak] * scale,
                                peak_latency=times[peak] * 1e3,
                            )
                for metric in component.metrics:
                    unit = "ms" if metric == "peak_latency" else amplitude_unit
                    column = f"ERP__{condition}__{name}__{metric}_{unit}__{channel}"
                    values[column] = measures.get(metric, np.nan)
                    status = (
                        "no_local_peak_with_requested_polarity"
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
                            "peak_method": "local_extremum_v1",
                            "boundary_peak": False,
                        }
                    )
    return pd.DataFrame([values], index=[subject_id], dtype=float), pd.DataFrame(
        quality
    )


@dataclass
class ERPResult:
    """IDPs and separate QC, with epochs/evokeds available for optional saving.

    Parameters
    ----------
    features : pandas.DataFrame
        One participant row of authoritative measurements.
    qc : pandas.DataFrame
        Per-feature statuses and trial counts.
    epochs : mne.BaseEpochs or None
        Retained epochs, or None for a result loaded from disk.
    evokeds : mapping
        Condition names to averages or None for unavailable conditions.
    preprocessing_qc : dict
        Preprocessing decisions and rejection details.
    components : mapping, optional
        Named ERPComponent definitions; defaults to an empty mapping.
    bootstrap : ERPBootstrapSummary or None, optional
        Descriptive uncertainty summaries, if computed.
    """

    features: pd.DataFrame
    qc: pd.DataFrame
    epochs: mne.BaseEpochs
    evokeds: Mapping
    preprocessing_qc: dict
    components: Mapping = field(default_factory=dict)
    bootstrap: ERPBootstrapSummary | None = None

    def plot(self, *, channels=None, show=False):
        """Plot averages with the saved IDP measurements and QC.

        Parameters
        ----------
        channels : sequence of str, optional
            Unique sensor names in panel order; defaults to sensors in the result.
        show : bool, default=False
            Display the figure with pyplot.show().

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure owned by the caller.
        axes : numpy.ndarray
            One-dimensional array of sensor axes.

        Raises
        ------
        ValueError
            If channel names are empty or duplicated, or sensor types are inconsistent.
        """
        from meganorm.plots.plot_erp import plot_erp_result

        return plot_erp_result(self, channels=channels, show=show)

    def save(self, directory, *, report=True, figures=False, overwrite=False):
        """Save reusable averages, measurements and an optional HTML report.

        Parameters
        ----------
        directory : path-like
            Output directory.
        report : bool, default=True
            Generate an HTML report.
        figures : bool, default=False
            Also export PNG/SVG figures; this enables report generation.
        overwrite : bool, default=False
            Permit reuse of an existing directory and remove known stale artifacts.

        Returns
        -------
        directory : pathlib.Path
            Directory containing saved measurements and reusable averages.

        Raises
        ------
        FileExistsError
            If the directory exists and overwrite is False.
        ValueError
            If subject identity or bootstrap summaries are inconsistent.
        """
        from meganorm.utils.erp_outputs import save_erp_result

        return save_erp_result(
            self, directory, report=report, figures=figures, overwrite=overwrite
        )


def _prepare_erp_raw(
    raw,
    metadata,
    components,
    config,
    *,
    montage_settings=None,
    line_freq=None,
    notify=None,
):
    """Apply the same EEG metadata preparation in standalone and cohort paths."""
    from meganorm.utils.eeg import read_channel_table, apply_channel_table
    from meganorm.utils.erp import prepare_erp_montage

    if metadata["channels"] is not None:
        apply_channel_table(
            raw,
            read_channel_table(metadata["channels"]),
            strict=True,
            apply_status=True,
        )
    montage_qc = prepare_erp_montage(
        raw,
        metadata,
        montage_settings,
        required_channels=[
            channel
            for component in components.values()
            for channel in component.channels
        ],
        notify=notify,
    )
    if raw.info.get("line_freq") is None and line_freq is not None:
        raw.info["line_freq"] = float(line_freq)
    config.validate_sampling_rate(raw.info["sfreq"])
    return montage_qc


def extract_recording(
    recording_path,
    subject_id,
    components=None,
    *,
    annotation_map=None,
    event_spec=None,
    events=None,
    event_id=None,
    config=None,
    montage_settings=None,
    line_freq=None,
):
    """Extract one recording using local channel and montage sidecars.

    Parameters
    ----------
    recording_path : path-like
        EEG recording supported by mne.io.read_raw.
    subject_id : str
        Exact participant identifier used for the feature row.
    components : mapping of str to ERPComponent, optional
        Named measurements; defaults to components in config.erp.
    events : ndarray of int, shape (n_events, 3), optional
        Explicit absolute-sample MNE events; requires event_id.
    event_id : mapping of str to int, optional
        Logical conditions and unique nonnegative codes for explicit events.
    annotation_map : mapping of str to str, optional
        Logical conditions mapped to exact annotation descriptions.
    event_spec : ERPEventSpec or mapping, optional
        Annotation or stimulus extraction settings. Supply only one event source.
    config : Config or mapping, optional
        Canonical ERP configuration. None constructs the default ERP configuration.
    montage_settings : ERPMontageSettings or mapping, optional
        Template fallback and channel exclusions for missing positions.
    line_freq : float, optional
        Line frequency in Hz, assigned only when the recording has none.

    Returns
    -------
    result : ERPResult
        Extracted measurements and QC, including montage provenance.

    Raises
    ------
    ValueError
        If the participant identifier, metadata, events or ERP settings are invalid.

    Notes
    -----
    Subject IDs are exact strings. ``montage_settings`` and ``line_freq``
    have the same meaning as in Dataset. No files are written. Sensor locations
    required by ICA must be embedded, supplied by sidecars, or configured via
    montage_settings; locations are never invented for unmatched channels.
    """
    from meganorm.utils.erp import erp_metadata_paths, validate_erp_subject_id

    validate_erp_subject_id(subject_id)
    config = _validate_erp_config(config)
    components = _erp_components(config) if components is None else components
    metadata = erp_metadata_paths(recording_path)
    with mne.io.read_raw(str(recording_path), preload=True) as recording:
        montage_qc = _prepare_erp_raw(
            recording,
            metadata,
            components,
            config,
            montage_settings=montage_settings,
            line_freq=line_freq,
        )
        result = run_erp_pipeline(
            recording,
            subject_id,
            components,
            annotation_map=annotation_map,
            event_spec=event_spec,
            events=events,
            event_id=event_id,
            config=config,
        )
    result.preprocessing_qc["montage"] = montage_qc
    return result


def run_erp_pipeline(
    raw,
    subject_id,
    components=None,
    *,
    events=None,
    event_id=None,
    annotation_map=None,
    event_spec=None,
    config=None,
):
    """Run automatic EEG preparation, epoching and multi-component extraction.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording with EEG channel types and required sensor locations already set.
    subject_id : str
        Exact participant identifier used for the feature row.
    components : mapping of str to ERPComponent, optional
        Named measurements; defaults to components in config.erp.
    events : ndarray of int, shape (n_events, 3), optional
        Explicit absolute-sample MNE events; requires event_id.
    event_id : mapping of str to int, optional
        Logical conditions and unique nonnegative codes for explicit events.
    annotation_map : mapping of str to str, optional
        Logical conditions mapped to exact annotation descriptions.
    event_spec : ERPEventSpec or mapping, optional
        Annotation or stimulus extraction settings. Supply only one event source.
    config : Config or mapping, optional
        Canonical ERP configuration. None constructs the default ERP configuration.

    Returns
    -------
    result : ERPResult
        Features, QC, retained epochs, condition averages and optional bootstrap.

    Raises
    ------
    ValueError
        If event sources conflict, events are invalid or collide after resampling,
        components lie outside the epoch, or preprocessing requirements are unmet.
    TypeError
        If a component is not an ERPComponent.

    Notes
    -----
    Supply event_spec, events plus condition-to-integer event_id, or annotation_map
    mapping condition names to exact annotation descriptions. Events are read
    before cleaning; optional resampling updates Raw and events together.
    Save result.features using to_csv() to retain MEGaNorm's subject-row format.
    """
    config = _validate_erp_config(config)
    if components is None:
        components = _erp_components(config)
    with stage("resolve_events") as detail:
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
        detail["condition_counts"] = {
            name: int(np.sum(events[:, 2] == code)) for name, code in event_id.items()
        }
    if config.resampling_rate is not None:
        ratio = config.resampling_rate / raw.info["sfreq"]
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
        if (
            component.window[0] < config.erp.tmin
            or component.window[1] > config.erp.tmax
        ):
            raise ValueError(f"Window for {name} lies outside the epoch.")
        if set(component.conditions) - set(event_id):
            raise ValueError(
                "Component conditions must be declared in the event mapping."
            )
        if component.sensor_type != "eeg":
            raise ValueError(
                "Raw pipeline supports EEG; use extract_erp_idps for prepared MEG evokeds."
            )
    with stage("preprocessing"):
        cleaned, preprocessing_qc = preprocess_erp(raw, config)
    if config.resampling_rate is not None:
        with stage(
            "resample",
            from_sfreq=cleaned.info["sfreq"],
            to_sfreq=config.resampling_rate,
        ):
            cleaned, events = cleaned.resample(
                config.resampling_rate, events=events, verbose=False
            )
    else:
        skipped("resample", reason="resampling_rate is None")
    events = _validate_events(cleaned, events, event_id)
    epochs, rejection_qc = epoch_erp(cleaned, events, event_id, config, return_qc=True)
    preprocessing_qc["epoch_rejection"] = rejection_qc
    with stage("average_conditions") as detail:
        evokeds = compute_evokeds(epochs)
        detail["n_trials"] = {
            name: 0 if evoked is None else evoked.nave
            for name, evoked in evokeds.items()
        }
    with stage(
        "extract_idps",
        min_trials=config.erp.min_trials,
        components={
            name: dict(
                conditions=spec.conditions,
                channels=spec.channels,
                window_seconds=spec.window,
                metrics=spec.metrics,
                polarity=spec.polarity,
                sensor_type=spec.sensor_type,
            )
            for name, spec in components.items()
        },
    ):
        features, qc = extract_erp_idps(
            subject_id, evokeds, components, config.erp.min_trials
        )
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
            n_autoreject_rejected=0,
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
    ar_dropped = set(rejection_qc["autoreject_rejected_event_indices"])
    for event_index, (event, reasons) in enumerate(zip(events, epochs.drop_log)):
        if not reasons:
            continue
        if event_index in ar_dropped:
            category = "n_autoreject_rejected"
        elif any(reason in ambiguous_labels for reason in reasons):
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
    interpolated = {
        name: sum(
            trial["event_code"] == code and bool(trial["interpolated_channels"])
            for trial in rejection_qc["trials"]
        )
        for name, code in event_id.items()
    }
    qc["n_interpolated_trials"] = qc["condition"].map(interpolated)
    qc["n_rejected"] = qc["n_events"] - qc["n_trials"]
    qc.loc[(qc.status == "missing_condition") & (qc.n_events > 0), "status"] = (
        "no_retained_trials"
    )
    progress_event(
        "erp_summary",
        conditions={
            name: dict(
                n_events=counts[name],
                n_trials=0 if evokeds[name] is None else evokeds[name].nave,
                **losses[name],
                n_interpolated_trials=interpolated[name],
            )
            for name in event_id
        },
        idps=dict(
            total=features.size,
            valid=int(np.isfinite(features.to_numpy()).sum()),
            statuses=qc.status.value_counts().to_dict(),
        ),
    )
    preprocessing_qc["drop_log"] = epochs.drop_log
    bootstrap = None
    if config.erp.bootstrap:
        from meganorm.utils.erp_bootstrap import bootstrap_erp

        with stage(
            "bootstrap",
            n_repeats=config.erp.bootstrap_n_repeats,
            random_state=config.random_state,
        ):
            bootstrap = bootstrap_erp(
                epochs,
                subject_id,
                components,
                n_repeats=config.erp.bootstrap_n_repeats,
                random_state=config.random_state,
                min_trials=config.erp.min_trials,
                evokeds=evokeds,
                features=features,
                extract_idps=extract_erp_idps,
            )
    return ERPResult(
        features, qc, epochs, evokeds, preprocessing_qc, dict(components), bootstrap
    )


def process_erp_recording(
    recording_path,
    subject_id,
    *,
    config,
    event_spec,
    output_root,
    line_freq=None,
    montage_settings=None,
):
    """Process and publish one EEG participant in a fresh run directory.

    Parameters
    ----------
    recording_path : path-like
        EEG recording supported by mne.io.read_raw.
    subject_id : str
        Exact participant identifier used for the feature row.
    config : Config or mapping
        Canonical ERP configuration with component definitions.
    event_spec : ERPEventSpec or mapping
        Dataset annotation or stimulus mapping.
    output_root : path-like
        Run root; creates participants/<subject_id> beneath it.
    montage_settings : ERPMontageSettings or mapping, optional
        Template fallback and channel exclusions for missing positions.
    line_freq : float, optional
        Line frequency in Hz, assigned only when the recording has none.

    Returns
    -------
    result : ERPResult
        Computed measurements and QC; artifacts are written before completion.

    Raises
    ------
    FileExistsError
        If the participant directory already exists.
    ValueError
        If participant identity, event settings or scientific inputs are invalid.

    Notes
    -----
    Sidecar channel types/status are explicit overrides; an embedded montage
    is preserved. Montage CSV coordinates follow the existing head/metre
    convention (Channel, X, Y, Z). Optional template assignment is recorded in QC.
    Features and QC are published before complete.json. A failed or interrupted
    write never has a completion record. Existing participant outputs are never
    overwritten. No-usable-IDP results are saved with status='failed' for audit.
    """
    from pathlib import Path
    from meganorm.utils.erp import (
        prepare_erp_montage,
        erp_metadata_paths,
        validate_erp_subject_id,
        write_erp_json,
        write_erp_table,
    )

    validate_erp_subject_id(subject_id)
    config = _validate_erp_config(config)
    settings = config.erp
    event_spec = ERPEventSpec.model_validate(
        event_spec.model_dump() if isinstance(event_spec, ERPEventSpec) else event_spec
    )
    settings.validate_events(event_spec)
    path = Path(recording_path).resolve()
    metadata = erp_metadata_paths(path)
    directory = Path(output_root) / "participants" / subject_id
    directory.mkdir(parents=True, exist_ok=False)
    with ERPProgress(
        directory,
        subject_id,
        recording=str(path),
        versions=runtime_versions(),
        settings_file="provenance.json",
        analysis_type="erp",
    ) as progress:
        with stage("load_recording", recording=str(path)) as detail:
            raw = mne.io.read_raw(str(path), preload=True, verbose=False)
            from collections import Counter

            detail.update(
                sfreq=raw.info["sfreq"],
                n_samples=raw.n_times,
                duration_seconds=raw.n_times / raw.info["sfreq"],
                n_channels=len(raw.ch_names),
                channel_types=dict(Counter(raw.get_channel_types())),
                bad_channels=list(raw.info["bads"]),
            )
        with raw:

            def notify_montage(message):
                """Record a montage notice and emit a participant-labelled warning."""
                import warnings

                progress_event("montage_notice", message=message)
                warnings.warn(f"{subject_id}: {message}", UserWarning, stacklevel=2)

            components = _erp_components(config)
            with stage(
                "prepare_metadata",
                sidecars={
                    key: str(value) if value else None
                    for key, value in metadata.items()
                },
            ) as detail:
                montage_qc = _prepare_erp_raw(
                    raw,
                    metadata,
                    components,
                    config,
                    montage_settings=montage_settings,
                    line_freq=line_freq,
                    notify=notify_montage,
                )
                write_erp_json(directory / "montage.json", montage_qc)
                detail.update(
                    montage=montage_qc,
                    bad_channels=list(raw.info["bads"]),
                    channel_types=dict(zip(raw.ch_names, raw.get_channel_types())),
                    line_freq=raw.info.get("line_freq"),
                )
            result = run_erp_pipeline(
                raw, subject_id, components, event_spec=event_spec, config=config
            )
        result.preprocessing_qc["montage"] = montage_qc
        finite = np.isfinite(result.features.to_numpy())
        status = "success" if finite.all() else "partial" if finite.any() else "failed"
        result.features.index.name = "participant_id"
        with stage(
            "save_tables",
            files=["features.csv", "qc.csv", "preprocessing.json", "provenance.json"],
        ):
            write_erp_table(directory / "features.csv", result.features)
            write_erp_table(directory / "qc.csv", result.qc, index=False)
            write_erp_json(directory / "preprocessing.json", result.preprocessing_qc)
            write_erp_json(
                directory / "provenance.json",
                dict(
                    peak_method="local_extremum_v1",
                    recording=str(path),
                    montage=montage_qc,
                    settings=config.model_dump(mode="json"),
                    event_spec=event_spec.model_dump(mode="json"),
                    metadata={k: str(v) if v else None for k, v in metadata.items()},
                    line_freq=line_freq,
                ),
            )
        from meganorm.utils.erp_outputs import save_erp_artifacts

        outputs = save_erp_artifacts(
            result,
            directory,
            save_evokeds=settings.save_evokeds,
            save_report=settings.save_report,
            save_figures=settings.save_figures,
        )
        with stage(
            "publish_completion", file="complete.json", processing_outcome=status
        ):
            write_erp_json(
                directory / "complete.json",
                dict(
                    subject=subject_id,
                    status=status,
                    feature_names=list(result.features.columns),
                ),
            )
        progress.details.update(processing_outcome=status, artifacts=outputs)
        return result
