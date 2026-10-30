"""Adapt legacy single-participant processing without leaking root logging."""

import logging
from meganorm.src.mainParallel import main

_FLAGS = {
    "device": "device_type",
    "line_freq": "line_freq",
    "empty_room_record": "empty_room_recording_path",
    "mri_surface": "surfaces_dir",
    "event_record": "event_record",
    "event_of_interest": "event_of_interest",
    "pos_path": "pos_file",
    "trans_path": "trans_file",
    "annotation_path": "annotation_path",
    "layout_path": "layout_path",
    "demographic_path": "demographic_path",
}


def process_participant(info, *, participant_id, config_path, temp_dir):
    """Run the legacy engine, retaining all optional scientific inputs."""
    args = [
        str(info["rest_record"]),
        str(temp_dir),
        str(participant_id),
        str(config_path),
    ]
    for key, flag in _FLAGS.items():
        value = info.get(key)
        if value is not None and str(value) != "None":
            args.extend([f"--{flag}", str(value)])
        elif key == "line_freq":
            args.extend(["--line_freq", "None"])
    names = ["", "mne", "numexpr", "dipy"]
    root = logging.getLogger()
    handlers = list(root.handlers)
    levels = {name: logging.getLogger(name).level for name in names}
    try:
        main(args)
    finally:
        for handler in list(root.handlers):
            if handler not in handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = handlers
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


def process_erp_participant(info, *, participant_id, config, output_root):
    """Run the shared ERP worker using the preflight-selected recording.

    Parameters
    ----------
    info : dict
        Preflight metadata containing selected_recording and erp_events, with
        optional erp_montage dictionary and line_freq in Hz.
    participant_id : str
        Exact participant identifier used for outputs and feature rows.
    config : Config or mapping
        Canonical ERP configuration with component definitions.
    output_root : path-like
        Run root beneath which participants/<participant_id> is created.

    Returns
    -------
    result : ERPResult
        Extracted measurements, QC, retained epochs and condition averages.

    Raises
    ------
    FileExistsError
        If the participant output directory already exists.
    ValueError
        If metadata, event settings or scientific inputs fail validation.

    Notes
    -----
    The worker publishes participant artifacts and a completion record. This
    adapter preserves the selected recording and does not rediscover inputs.
    """
    from meganorm.src.erp import process_erp_recording

    return process_erp_recording(
        info["selected_recording"],
        participant_id,
        config=config,
        event_spec=info["erp_events"],
        montage_settings=(
            info.get("erp_montage")
            if isinstance(info.get("erp_montage"), dict)
            else None
        ),
        output_root=output_root,
        line_freq=info.get("line_freq"),
    )
