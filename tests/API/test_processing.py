import logging
from pathlib import Path
import pytest
from meganorm.src.mainParallel import main_argparser


@pytest.mark.unit
def test_processing_passes_all_auxiliary_inputs_to_real_parser(tmp_path, monkeypatch):
    from meganorm.API import _processing

    info = {
        "rest_record": "record.fif*",
        "device": "MEGIN",
        "line_freq": 50,
        "empty_room_record": "empty.fif*",
        "mri_surface": "surfaces",
        "event_record": "events.tsv*",
        "event_of_interest": 16,
        "pos_path": "head.pos*",
        "trans_path": "trans.fif*",
        "annotation_path": "annotations.txt*",
        "layout_path": "layout.json",
        "demographic_path": "participants.tsv",
    }
    parsed = []
    monkeypatch.setattr(
        _processing, "main", lambda args: parsed.append(main_argparser(args))
    )
    _processing.process_participant(
        info,
        participant_id="001",
        config_path=tmp_path / "config.json",
        temp_dir=tmp_path / "temp",
    )
    args = parsed[0]
    assert (
        args.subject == "001" and args.device_type == "MEGIN" and args.line_freq == "50"
    )
    assert (
        args.surfaces_dir == "surfaces"
        and args.empty_room_recording_path == "empty.fif*"
    )
    assert args.event_record == "events.tsv*" and args.event_of_interest == "16"
    assert args.pos_file == "head.pos*" and args.trans_file == "trans.fif*"
    assert (
        args.annotation_path == "annotations.txt*" and args.layout_path == "layout.json"
    )
    assert args.demographic_path == "participants.tsv"


@pytest.mark.unit
@pytest.mark.parametrize(
    "exception", [None, RuntimeError, KeyboardInterrupt, SystemExit]
)
def test_processing_restores_logging_on_every_exit(tmp_path, monkeypatch, exception):
    from meganorm.API import _processing

    root = logging.getLogger()
    previous = list(root.handlers)
    levels = {
        name: logging.getLogger(name).level for name in ["", "mne", "numexpr", "dipy"]
    }
    handler = logging.StreamHandler()

    def fake_main(args):
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        for name in ["mne", "numexpr", "dipy"]:
            logging.getLogger(name).setLevel(logging.WARNING)
        if exception:
            raise exception("stop")

    monkeypatch.setattr(_processing, "main", fake_main)
    info = {"rest_record": "record.fif", "line_freq": None}
    if exception:
        with pytest.raises(exception):
            _processing.process_participant(
                info,
                participant_id="001",
                config_path=tmp_path / "config.json",
                temp_dir=tmp_path / "temp",
            )
    else:
        _processing.process_participant(
            info,
            participant_id="001",
            config_path=tmp_path / "config.json",
            temp_dir=tmp_path / "temp",
        )
    assert root.handlers == previous
    assert {name: logging.getLogger(name).level for name in levels} == levels
    assert handler._closed
