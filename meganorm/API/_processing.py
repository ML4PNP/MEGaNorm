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
