"""Sensor-space evoked waveform inspection, independent of pipeline execution."""

import numpy as np
import matplotlib.pyplot as plt

from meganorm.utils.erp import _matching_times


def plot_erp(evokeds, *, components=None, channels=None, show=False):
    """Plot condition waveforms and configured component windows per sensor.

    Parameters
    ----------
    evokeds : mapping
        Condition names to MNE Evoked objects or None for absent conditions.
    components : mapping, optional
        Names to ERPComponent objects. Shade a component window only for its
        configured sensors and conditions that have a waveform in that panel.
    channels : sequence of str, optional
        Ordered sensors. Defaults to component channels, or all channels in
        the first available Evoked when components are not provided.
    show : bool, default=False
        Call pyplot.show() when True. Never saves files or modifies inputs.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure, available for layout adjustment or savefig().
    axes : ndarray
        One-dimensional array of axes, one per requested sensor. Bad and
        missing sensors are omitted; a panel with no usable data is labelled.

    Raises
    ------
    ValueError
        If no averages exist, channel names are empty or duplicated, or sensor
        types are inconsistent or unsupported.

    Notes
    -----
    EEG is displayed in microvolts, magnetometers in tesla and gradiometers in
    tesla/metre. Times are milliseconds. This shows evoked waveforms and windows,
    not extracted peak markers. Conditions can have different time axes.
    """
    available = {name: evoked for name, evoked in evokeds.items() if evoked is not None}
    if not available:
        raise ValueError("No evoked responses are available to plot.")
    components = components or {}
    if channels is None:
        channels = (
            list(
                dict.fromkeys(
                    ch for spec in components.values() for ch in spec.channels
                )
            )
            if components
            else next(iter(available.values())).ch_names
        )
    if isinstance(channels, str) or not channels or len(set(channels)) != len(channels):
        raise ValueError("channels must be a nonempty sequence of unique names.")
    units = {"eeg": (1e6, "µV"), "mag": (1.0, "T"), "grad": (1.0, "T/m")}
    # Validate before allocating a figure, and never mix units within a panel.
    sensor_types = {}
    for channel in channels:
        types = {
            ev.get_channel_types()[ev.ch_names.index(channel)]
            for ev in available.values()
            if channel in ev.ch_names and channel not in ev.info["bads"]
        }
        if len(types) > 1 or types - units.keys():
            raise ValueError(f"Inconsistent or unsupported sensor type for {channel}.")
        sensor_types[channel] = next(iter(types), None)
    fig, axes = plt.subplots(
        1, len(channels), figsize=(5 * len(channels), 3.6), squeeze=False
    )
    axes = axes.ravel()
    colors = plt.get_cmap("tab10")
    for ax, channel in zip(axes, channels):
        plotted = set()
        for i, (condition, evoked) in enumerate(available.items()):
            if channel not in evoked.ch_names or channel in evoked.info["bads"]:
                continue
            scale, unit = units[sensor_types[channel]]
            index = evoked.ch_names.index(channel)
            ax.plot(
                evoked.times * 1000,
                evoked.data[index] * scale,
                label=condition,
                color=colors(i % 10),
            )
            plotted.add(condition)
        for i, (name, spec) in enumerate(components.items()):
            if channel in spec.channels and plotted.intersection(spec.conditions):
                ax.axvspan(
                    np.asarray(spec.window)[0] * 1000,
                    np.asarray(spec.window)[1] * 1000,
                    color=colors(i % 10),
                    alpha=0.12,
                    label=f"{name} window",
                )
        ax.axvline(0, color="0.5", linewidth=0.7)
        ax.axhline(0, color="0.5", linewidth=0.7)
        ax.set(title=channel, xlabel="Time from event (ms)")
        if plotted:
            ax.set_ylabel(f"Amplitude ({units[sensor_types[channel]][1]})")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No usable data", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


_UNITS = {"eeg": (1e6, "uV"), "mag": (1.0, "T"), "grad": (1.0, "T_per_m")}


def _result_channels(result):
    """Choose ordered component sensors, falling back to evoked channels."""
    channels = list(
        dict.fromkeys(ch for spec in result.components.values() for ch in spec.channels)
    )
    if not channels:
        channels = list(
            dict.fromkeys(
                ch
                for ev in result.evokeds.values()
                if ev is not None
                for ch in ev.ch_names
            )
        )
    return channels or ["No configured channels"]


def _measurement_rows(result):
    """Join configured metrics to persisted values/QC, without extraction."""
    rows = []
    bootstrap = getattr(result, "bootstrap", None)
    for name, spec in result.components.items():
        for condition in spec.conditions:
            for channel in spec.channels:
                for metric in spec.metrics:
                    unit = (
                        "ms"
                        if metric == "peak_latency"
                        else _UNITS[spec.sensor_type][1]
                    )
                    key = f"ERP__{condition}__{name}__{metric}_{unit}__{channel}"
                    matches = (
                        result.qc.loc[result.qc["idp"] == key]
                        if "idp" in result.qc
                        else result.qc.iloc[:0]
                    )
                    quality = matches.iloc[0].to_dict() if len(matches) == 1 else {}
                    value = (
                        result.features.iloc[0].get(key, np.nan)
                        if len(result.features)
                        else np.nan
                    )
                    status = quality.get("status", "missing_qc")
                    if status == "ok" and not np.isfinite(value):
                        status = "missing_value"
                    uncertainty = {}
                    if bootstrap is not None:
                        matches = bootstrap.idps.loc[bootstrap.idps["idp"] == key]
                        summary = matches.iloc[0].to_dict() if len(matches) == 1 else {}
                        uncertainty = {
                            f"bootstrap_{field}": summary.get(field, np.nan)
                            for field in (
                                "median",
                                "p05",
                                "p95",
                                "n_valid",
                                "n_repeats",
                            )
                        }
                        uncertainty["bootstrap_status"] = summary.get(
                            "status", "unavailable"
                        )
                    rows.append(
                        dict(
                            component=name,
                            condition=condition,
                            channel=channel,
                            metric=metric,
                            unit=unit,
                            value=value,
                            status=status,
                            peak_method=quality.get(
                                "peak_method", "legacy/unspecified"
                            ),
                            boundary_peak=bool(quality.get("boundary_peak", False)),
                            n_events=quality.get("n_events", "unknown"),
                            n_trials=quality.get("n_trials", "unknown"),
                            window_ms=f"{spec.window[0] * 1000:g}–{spec.window[1] * 1000:g}",
                            **uncertainty,
                        )
                    )
    return rows


def _plot_bootstrap_band(ax, bootstrap, condition, evoked, channel, scale, color):
    """Add a saved pointwise range, returning any uncertainty warning."""
    if bootstrap is None:
        return None
    wave = bootstrap.waveforms.get(condition)
    unavailable = f"{condition}: bootstrap uncertainty unavailable"
    if wave is None or channel not in wave["ch_names"]:
        return unavailable
    times = np.asarray(wave["times"])
    if not _matching_times(times, evoked.times):
        return unavailable + " (time mismatch)"
    index = list(wave["ch_names"]).index(channel)
    low = np.asarray(wave["p05"])[index]
    high = np.asarray(wave["p95"])[index]
    counts = np.asarray(wave["n_valid"])[index]
    if any(array.shape != times.shape for array in (low, high, counts)):
        return unavailable
    valid = np.isfinite(low) & np.isfinite(high) & (counts >= 2) & (low <= high)
    # An isolated point cannot define an area; do not advertise a visible band.
    if not np.any(valid[:-1] & valid[1:]):
        return unavailable
    ax.fill_between(
        evoked.times * 1000,
        low * scale,
        high * scale,
        where=valid,
        color=color,
        alpha=0.2,
        linewidth=0,
        label=f"{condition} bootstrap pointwise p05–p95",
    )
    repeats = bootstrap.provenance.get("n_repeats", counts.max())
    if not valid.all() or np.any(counts < repeats):
        return f"{condition}: bootstrap uncertainty partial"
    return None


def plot_erp_result(result, *, channels=None, show=False):
    """Plot saved ERP waveforms and IDPs, one vertically stacked sensor per panel.

    Parameters
    ----------
    result : ERPResult
        Saved or freshly computed measurements, QC, averages and component definitions.
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
        If channels are empty or duplicated, or sensor types are inconsistent.

    Notes
    -----
    Peak markers require both saved peak metrics with valid QC. Dashed mean
    lines use saved means. No measurement is recomputed. EEG amplitudes are
    microvolts, magnetic amplitudes T or T/m, and latencies milliseconds.
    Bootstrap shading shows descriptive pointwise central 90% ranges, not
    calibrated confidence intervals or simultaneous bands.
    The caller owns the returned figure. Missing responses yield labelled
    panels; configured metrics and QC are available in the HTML report.
    """
    channels = _result_channels(result) if channels is None else channels
    if (
        isinstance(channels, str)
        or not len(channels)
        or len(set(channels)) != len(channels)
    ):
        raise ValueError("channels must be a nonempty sequence of unique names.")
    rows = _measurement_rows(result)
    conditions = list(dict.fromkeys([*result.evokeds, *(r["condition"] for r in rows)]))
    palette = plt.get_cmap("tab10")
    colors = {condition: palette(i % 10) for i, condition in enumerate(conditions)}
    colors.update(
        {
            c: color
            for c, color in [("standard", "#2976b8"), ("target", "#d4533b")]
            if c in colors
        }
    )
    types = {}
    for channel in channels:
        found = {
            ev.get_channel_types()[ev.ch_names.index(channel)]
            for ev in result.evokeds.values()
            if ev is not None
            and channel in ev.ch_names
            and channel not in ev.info["bads"]
        }
        found.update(
            spec.sensor_type
            for spec in result.components.values()
            if channel in spec.channels
        )
        if len(found) > 1 or found - _UNITS.keys():
            raise ValueError(f"Inconsistent or unsupported sensor type for {channel}.")
        types[channel] = next(iter(found), "eeg")
    fig, grid = plt.subplots(
        len(channels), 1, figsize=(10, 3.8 * len(channels)), squeeze=False
    )
    axes = grid.ravel()
    try:
        for ax, channel in zip(axes, channels):
            usable = set()
            uncertainty_flags = []
            for condition, ev in result.evokeds.items():
                if (
                    ev is None
                    or channel not in ev.ch_names
                    or channel in ev.info["bads"]
                ):
                    continue
                signal = ev.data[ev.ch_names.index(channel)]
                if not np.isfinite(signal).all():
                    continue
                ax.plot(
                    ev.times * 1000,
                    signal * _UNITS[types[channel]][0],
                    color=colors[condition],
                    label=f"{condition} (n={ev.nave})",
                )
                warning = _plot_bootstrap_band(
                    ax,
                    getattr(result, "bootstrap", None),
                    condition,
                    ev,
                    channel,
                    _UNITS[types[channel]][0],
                    colors[condition],
                )
                if warning:
                    uncertainty_flags.append(warning)
                usable.add(condition)
            for i, (name, spec) in enumerate(result.components.items()):
                if channel not in spec.channels:
                    continue
                lo, hi = np.asarray(spec.window) * 1000
                ax.axvspan(
                    lo, hi, alpha=0.09, color=palette(i % 10), label=f"{name} window"
                )
                for condition in spec.conditions:
                    selected = {
                        r["metric"]: r
                        for r in rows
                        if r["channel"] == channel
                        and r["condition"] == condition
                        and r["component"] == name
                    }
                    valid = {
                        metric: row
                        for metric, row in selected.items()
                        if row["status"] == "ok" and np.isfinite(row["value"])
                    }
                    if condition not in usable:
                        continue
                    if {"peak_amplitude", "peak_latency"} <= valid.keys():
                        amp, latency = valid["peak_amplitude"], valid["peak_latency"]
                        boundary = (
                            " [boundary]"
                            if amp["boundary_peak"] or latency["boundary_peak"]
                            else ""
                        )
                        ax.plot(
                            [latency["value"]],
                            [amp["value"]],
                            marker="o",
                            linestyle="none",
                            color=colors[condition],
                            label=f"{condition} {name} peak{boundary}",
                        )
                    if "mean_amplitude" in valid:
                        ax.hlines(
                            valid["mean_amplitude"]["value"],
                            lo,
                            hi,
                            color=colors[condition],
                            linestyles="dashed",
                            label=f"{condition} {name} mean",
                        )
            flags = list(
                dict.fromkeys(
                    f"{r['condition']} / {r['component']}: {r['status']}"
                    for r in rows
                    if r["channel"] == channel and r["status"] != "ok"
                )
            )
            flags.extend(uncertainty_flags)
            if not usable:
                ax.text(
                    0.5,
                    0.5,
                    "No usable evoked data",
                    transform=ax.transAxes,
                    ha="center",
                )
            if flags:
                ax.text(
                    1.01,
                    0.98,
                    "\n".join(flags),
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                )
            ax.axvline(0, color="0.6", linewidth=0.7)
            ax.axhline(0, color="0.6", linewidth=0.7)
            unit = (
                _UNITS[types[channel]][1].replace("uV", "µV").replace("T_per_m", "T/m")
            )
            ax.set(
                title=channel,
                xlabel="Time from event (ms)",
                ylabel=f"Amplitude ({unit})",
            )
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes
    except BaseException:
        plt.close(fig)
        raise
