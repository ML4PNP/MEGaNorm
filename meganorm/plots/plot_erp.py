"""Sensor-space evoked waveform inspection, independent of pipeline execution."""

import numpy as np
import matplotlib.pyplot as plt


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
