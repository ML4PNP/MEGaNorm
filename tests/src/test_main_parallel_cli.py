import logging
from types import SimpleNamespace

import pytest

from meganorm.src.mainParallel import main_argparser, set_logger

pytestmark = pytest.mark.unit


def test_main_argparser_parses_required_arguments_and_defaults():
    args = main_argparser(
        ["raw/sub-01.fif", "outputs/features", "sub-01", "config.json"]
    )

    assert args.dir == "raw/sub-01.fif"
    assert args.save_dir == "outputs/features"
    assert args.subject == "sub-01"
    assert args.configs == "config.json"
    assert args.line_freq == 60
    assert args.surfaces_dir is None
    assert args.empty_room_recording_path is None
    assert args.event_record is None
    assert args.event_of_interest is None
    assert args.device_type is None
    assert args.pos_file is None
    assert args.trans_file is None
    assert args.annotation_path is None
    assert args.layout_path is None
    assert args.demographic_path is None


def test_main_argparser_preserves_explicit_optional_values():
    args = main_argparser(
        [
            "raw/sub-01.fif",
            "outputs/features",
            "sub-01",
            "config.json",
            "--line_freq",
            "None",
            "--surfaces_dir",
            "subjects",
            "--empty_room_recording_path",
            "empty-room.fif",
            "--event_record",
            "events.tsv",
            "--event_of_interest",
            "16",
            "--device_type",
            "MEGIN",
            "--pos_file",
            "headshape.pos",
            "--trans_file",
            "sub-01-trans.fif",
            "--annotation_path",
            "annotations.csv",
            "--layout_path",
            "layout.json",
            "--demographic_path",
            "participants.tsv",
        ]
    )

    assert args.line_freq == "None"
    assert args.surfaces_dir == "subjects"
    assert args.empty_room_recording_path == "empty-room.fif"
    assert args.event_record == "events.tsv"
    assert args.event_of_interest == "16"
    assert args.device_type == "MEGIN"
    assert args.pos_file == "headshape.pos"
    assert args.trans_file == "sub-01-trans.fif"
    assert args.annotation_path == "annotations.csv"
    assert args.layout_path == "layout.json"
    assert args.demographic_path == "participants.tsv"


def test_main_argparser_requires_all_positional_arguments():
    with pytest.raises(SystemExit) as error:
        main_argparser(["raw/sub-01.fif", "outputs/features", "sub-01"])

    assert error.value.code == 2


def test_set_logger_writes_subject_log_and_silences_requested_package(tmp_path):
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_root_level = root_logger.level
    dependency_logger = logging.getLogger("test_noisy_dependency")
    original_dependency_level = dependency_logger.level

    args = SimpleNamespace(
        save_dir=str(tmp_path / "project" / "features"), subject="sub-01"
    )
    log_path = (
        tmp_path
        / "project"
        / "Saved_outputs"
        / "log_summary"
        / "subject_sub-01_report.log"
    )

    try:
        logger = set_logger(args, ["test_noisy_dependency"])
        logger.info("CLI logging is configured")
        for handler in root_logger.handlers:
            handler.flush()

        assert dependency_logger.level == logging.WARNING
        assert log_path.exists()
        assert (
            "meganorm.src.mainParallel - INFO - "
            "test_set_logger_writes_subject_log_and_silences_requested_package - "
            "CLI logging is configured"
        ) in log_path.read_text(encoding="utf-8")
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            if handler not in original_handlers:
                handler.close()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_root_level)
        dependency_logger.setLevel(original_dependency_level)
