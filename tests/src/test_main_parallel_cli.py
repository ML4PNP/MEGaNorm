import logging
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest

from meganorm.src import mainParallel as cli
from meganorm.src.mainParallel import main_argparser, set_logger
from meganorm.utils.IO import Config

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


@pytest.fixture
def numerical_cli_stubs(monkeypatch):
    seen_annotations = []

    def preprocess(**kwargs):
        raw = kwargs["data"]
        return raw.copy(), raw.ch_names, int(raw.info["sfreq"]), None, None, None

    def segment(**kwargs):
        raw = kwargs["data"]
        seen_annotations.extend(raw.annotations.description)
        return mne.EpochsArray(np.zeros((2, len(raw.ch_names), 100)), raw.info.copy())

    monkeypatch.setattr(cli, "preprocess", preprocess)
    monkeypatch.setattr(cli, "segment_epoch", segment)
    monkeypatch.setattr(
        cli,
        "source_localization",
        lambda **kwargs: (np.zeros((2, 2, 100)), ["region-lh", "region-rh"]),
    )
    monkeypatch.setattr(
        cli, "parameterize_psds", lambda **kwargs: ([], np.zeros((2, 3)), np.arange(3))
    )
    monkeypatch.setattr(
        cli,
        "feature_extract",
        lambda **kwargs: (
            pd.DataFrame({"feature": [0.5]}, index=[kwargs["subject_id"]]),
            None,
        ),
    )
    monkeypatch.setattr(cli, "set_logger", lambda *args: logging.getLogger("cli-test"))
    return seen_annotations


def _write_cli_inputs(tmp_path, config):
    raw = mne.io.RawArray(
        np.zeros((2, 200)), mne.create_info(["MEG0111", "MEG0121"], 100, "mag")
    )
    raw.set_annotations(
        mne.Annotations([0.2, 0.5], [0.1, 0.1], ["UADC001", "eyes-closed"])
    )
    recording = tmp_path / "input-raw.fif"
    raw.save(recording, overwrite=True)
    config_path = tmp_path / "config.json"
    config.save(config_path)
    surfaces = tmp_path / "surfaces"
    (surfaces / "sub-01").mkdir(parents=True)
    return recording, config_path, surfaces


@pytest.mark.parametrize(
    ("save_preprocessed", "save_source_epochs"),
    [(False, False), (True, False), (False, True)],
)
def test_main_creates_requested_output_directories_and_writes_readable_files(
    tmp_path,
    numerical_cli_stubs,
    save_preprocessed,
    save_source_epochs,
):
    config = Config(
        drop_noisy_flat_channel=False,
        bad_segment_removal_method=None,
        apply_source_localization=save_source_epochs,
        save_preprocessed_data=save_preprocessed,
        save_source_localized_epochs=save_source_epochs,
    )
    recording, config_path, surfaces = _write_cli_inputs(tmp_path, config)
    output = tmp_path / "new-output" / "features"

    cli.main(
        [
            str(recording),
            str(output),
            "sub-01",
            str(config_path),
            "--surfaces_dir",
            str(surfaces),
        ]
    )

    assert (
        pd.read_csv(output / "sub-01.csv", index_col=0).loc["sub-01", "feature"] == 0.5
    )
    saved = output.parent / "Saved_outputs"
    if save_preprocessed:
        preprocessed = mne.io.read_raw_fif(
            saved / "Preprocessed_data" / "sub-01_preproc-raw.fif"
        )
        assert preprocessed.ch_names == ["MEG0111", "MEG0121"]
    if save_source_epochs:
        epochs = mne.read_epochs(saved / "Epochs" / "sub-01" / "sub-01-SL-epo.fif")
        assert epochs.ch_names == ["region-lh", "region-rh"]
        assert len(epochs) == 2


def test_main_preserves_acquisition_annotations(tmp_path, numerical_cli_stubs):
    config = Config(drop_noisy_flat_channel=False, bad_segment_removal_method=None)
    recording, config_path, _ = _write_cli_inputs(tmp_path, config)
    output = tmp_path / "features"
    output.mkdir()

    cli.main([str(recording), str(output), "sub-01", str(config_path)])

    assert numerical_cli_stubs == ["UADC001", "eyes-closed"]
