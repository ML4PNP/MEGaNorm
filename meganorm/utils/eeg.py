"""Shared EEG metadata operations with explicit workflow policies.

Legacy REST callers retain their discovery, decoding and error handling.
Strict supplied/template resolution is opt-in and never rewrites input files.
"""

import warnings

import mne
import numpy as np
import pandas as pd


def read_channel_table(path, *, legacy=False):
    """Read a channel TSV with an explicit decoding policy.

    Parameters
    ----------
    path : path-like
        Channel table with name and type columns and optional status.
    legacy : bool, default=False
        Retain pandas default decoding and dtype inference when True.

    Returns
    -------
    table : pandas.DataFrame
        Parsed channel metadata; strict mode requests string channel names.

    Notes
    -----
    Strict-workflow reading accepts UTF-8/BOM and warns on a Windows-1252
    fallback. Parsing and decoding errors are never silently discarded.
    """
    if legacy:
        return pd.read_csv(path, sep="\t")
    try:
        return pd.read_csv(path, sep="\t", dtype={"name": str}, encoding="utf-8-sig")
    except UnicodeDecodeError:
        table = pd.read_csv(path, sep="\t", dtype={"name": str}, encoding="cp1252")
        warnings.warn(
            f"Channel sidecar {path} is not UTF-8; read using Windows-1252. "
            "Consider exporting future channel tables as UTF-8.",
            UserWarning,
            stacklevel=2,
        )
        return table


def apply_channel_table(raw, table, *, strict=True, apply_status=True):
    """Apply channel types and optional bad status without changing samples.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording whose channel metadata are updated in place.
    table : pandas.DataFrame
        Channel names and types, with an optional status column.
    strict : bool, default=True
        Require unique nonmissing names and reject names absent from raw.
    apply_status : bool, default=True
        Append channels labelled bad to existing bad-channel markings.

    Returns
    -------
    None
        Metadata are updated in place.

    Raises
    ------
    ValueError
        If strict validation fails or MNE rejects channel names or types.

    Notes
    -----
    Legacy callers disable both flags to retain last-row duplicate handling,
    MNE's name/type errors, and existing bad-channel markings.
    """
    if strict and (
        not {"name", "type"}.issubset(table)
        or table.name.isna().any()
        or table.name.duplicated().any()
    ):
        raise ValueError("Channel sidecar requires unique name and type columns.")

    channel_types = table.set_index("name")["type"].str.lower()
    if strict:
        channel_types = channel_types.replace({"trig": "stim"})
    mapping = channel_types.to_dict()
    
    if strict and set(mapping) - set(raw.ch_names):
        raise ValueError("Channel sidecar names do not match recording channels.")
    raw.set_channel_types(mapping)
    if apply_status and "status" in table:
        bad = table.loc[table.status.str.lower() == "bad", "name"].tolist()
        raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + bad))


def read_montage_positions(path, *, strict=True):
    """Read Channel/X/Y/Z positions in head-frame metres.

    Parameters
    ----------
    path : path-like
        CSV with Channel, X, Y and Z columns.
    strict : bool, default=True
        Require unique nonmissing channel names and finite numeric coordinates.

    Returns
    -------
    positions : dict
        Channel names mapped to three head-coordinate values in metres.

    Raises
    ------
    ValueError
        If strict-mode columns, names or coordinates are invalid.

    Notes
    -----
    Legacy parsing intentionally retains pandas inference and last-row wins.
    Coordinates are neither transformed nor rescaled.
    """
    if not strict:
        table = pd.read_csv(path)
        return {
            row["Channel"]: [row["X"], row["Y"], row["Z"]]
            for _, row in table.iterrows()
        }
    table = pd.read_csv(path, dtype={"Channel": str})
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
    return dict(zip(table.Channel, coordinates))


def set_head_montage(raw, positions, *, allow_missing=False):
    """Assign head-frame metre coordinates with an explicit missing policy.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording to update in place.
    positions : mapping of str to array-like
        Channel names and three-dimensional head coordinates in metres.
    allow_missing : bool, default=False
        Ignore recording channels missing from the montage when True.

    Returns
    -------
    None
        The montage is assigned to raw.

    Raises
    ------
    ValueError
        If MNE rejects the montage or required channel positions are missing.
    """
    montage = mne.channels.make_dig_montage(ch_pos=positions, coord_frame="head")
    raw.set_montage(montage, on_missing="ignore" if allow_missing else "raise")


def eeg_metadata_paths(recording_path):
    """Find unambiguous local sidecars using the existing EEG conventions.

    Parameters
    ----------
    recording_path : path-like
        Recording path used to match adjacent sidecars.

    Returns
    -------
    metadata : dict
        Channel TSV, montage CSV, BIDS electrodes TSV and coordinate-system JSON
        paths under channels, montage, electrodes and coordsystem. Unmatched
        paths are None; bids_error records ambiguous electrodes or missing JSON.

    Raises
    ------
    ValueError
        If channel tables or montage CSV candidates are ambiguous.

    Notes
    -----
    Exact channel/montage filenames take precedence over directory fallbacks.
    BIDS electrodes match recording entities, allowing a space entity, and the
    most specific matching set is selected. BIDS errors are deferred so callers
    can prefer embedded positions without using invalid sidecars.
    """
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


def prepare_eeg_montage(
    raw,
    metadata,
    *,
    template=None,
    exclude_channels=(),
    supplied_kind="unknown",
    required_channels=(),
    notify=None,
):
    """Apply supplied or explicitly selected template positions, in memory.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording whose montage and bad-channel markings are updated in place.
    metadata : dict
        Sidecar paths and deferred BIDS errors from eeg_metadata_paths.
    template : str or None, optional
        Explicit standard montage fallback when supplied positions are absent.
        The auto/standard_1005/colin27_1005 choices prefer colin27_1005 when available.
    exclude_channels : sequence of str, optional
        EEG channels to mark bad and exclude from position requirements.
    supplied_kind : {'unknown', 'measured', 'template'}, default='unknown'
        Provenance label for embedded or sidecar positions.
    required_channels : sequence of str, optional
        Requested measurement sensors; these cannot be explicitly excluded.
    notify : callable, optional
        Callback receiving a notice string for template positions or exclusions.
        Defaults to emitting a UserWarning.

    Returns
    -------
    qc : dict
        Position source, kind, template aliases, matched/excluded channels,
        coordinate frame, units and final bad-channel markings.

    Raises
    ------
    ValueError
        If usable positions are incomplete, unsupported or invalid, template
        mappings collide, or exclusions include required measurement sensors.

    Notes
    -----
    Embedded positions take precedence over BIDS positions, then montage CSV,
    then the explicit template. BIDS support covers adjacent electrode and
    coordinate-system pairs in CapTrak head coordinates, with m/cm/mm units.
    Other frames require an explicit external transformation. No source leaves
    positions unset; sources are never mixed or written back to input files.
    """
    import json
    import warnings
    import pandas as pd

    notify = notify or (
        lambda message: warnings.warn(message, UserWarning, stacklevel=2)
    )
    eeg = [
        raw.ch_names[p]
        for p in mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    ]
    excluded = [name for name in exclude_channels if name in eeg]
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
        """Return usable EEG channels with finite nonzero coordinates."""
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
        qc.update(source="embedded", kind=supplied_kind, matched_channels=embedded)
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
        set_head_montage(raw, dict(zip(table["name"], coordinates)), allow_missing=True)
        qc.update(
            source="bids",
            kind=supplied_kind,
            input_coordinate_system=spec["EEGCoordinateSystem"],
            input_units=spec["EEGCoordinateUnits"],
        )
    elif metadata.get("montage") is not None:
        set_head_montage(
            raw,
            read_montage_positions(metadata["montage"], strict=True),
            allow_missing=True,
        )
        qc.update(source="sidecar", kind=supplied_kind)
    elif template is not None and usable:
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
        set_head_montage(
            raw,
            {name: positions[target] for name, target in mapped.items()},
            allow_missing=True,
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
