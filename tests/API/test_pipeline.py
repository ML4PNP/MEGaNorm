import json
from pathlib import Path
import pandas as pd
import pytest
from meganorm.API.datasets import Dataset
from meganorm.utils.IO import Config


@pytest.fixture
def cohort(tmp_path):
    root = tmp_path / "data"
    for subject in ["001", "002"]:
        path = root / subject / f"{subject}_rest.fif"
        path.parent.mkdir(parents=True)
        path.touch()
    (root / "participants.tsv").write_text("participant_id\tage\n001\t20\n002\t30\n")
    return Dataset(
        name="cohort",
        root=root,
        task="rest",
        extension=".fif",
        demographics="participants.tsv",
    )


def write_features(info, *, participant_id, config_path, temp_dir):
    pd.DataFrame(
        {
            "OriginalPSD_Canonical_Relative_Power__Alpha__MEG001": [0.2],
            "scanner": ["MEGIN"],
        },
        index=[participant_id],
    ).to_csv(temp_dir / f"{participant_id}.csv")


@pytest.mark.unit
def test_pipeline_collects_features_and_reports_without_changing_config(
    tmp_path, cohort, monkeypatch
):
    from meganorm.API import pipeline

    config = Config(apply_source_localization=False)
    run = pipeline.Pipeline(config=config, output_dir=tmp_path / "out", progress=False)
    config.which_meg_session = 10
    monkeypatch.setattr(pipeline, "process_participant", write_features)
    result = run.run(cohort)
    assert result.data.index.tolist() == ["001", "002"]
    assert result.data.age.tolist() == [20, 30]
    assert result.feature_names == (
        "OriginalPSD_Canonical_Relative_Power__Alpha__MEG001",
    )
    assert result.summary["succeeded"] == 2
    assert result.processing.status.tolist() == ["success", "success"]
    assert all(p.exists() for p in result.paths.values())
    assert Config.load(str(result.paths["config"])).which_meg_session == 0
    with pytest.raises(FileExistsError):
        run.run(cohort)


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_mode",
    ["exception", "after_csv", "wrong_id", "two_rows", "unreadable", "no_features"],
)
def test_pipeline_continue_never_collects_failed_files(
    tmp_path, cohort, monkeypatch, bad_mode
):
    from meganorm.API import pipeline

    def process(info, **kwargs):
        if kwargs["participant_id"] == "001":
            return write_features(info, **kwargs)
        path = kwargs["temp_dir"] / "002.csv"
        if bad_mode == "after_csv":
            write_features(info, **kwargs)
        if bad_mode in ["exception", "after_csv"]:
            raise RuntimeError("failed participant")
        if bad_mode == "wrong_id":
            pd.DataFrame({"f__x": [1]}, index=["wrong"]).to_csv(path)
        if bad_mode == "two_rows":
            pd.DataFrame({"f__x": [1, 2]}, index=["002", "002"]).to_csv(path)
        if bad_mode == "unreadable":
            path.write_text("garbage")
        if bad_mode == "no_features":
            pd.DataFrame({"scanner": ["MEGIN"]}, index=["002"]).to_csv(path)

    monkeypatch.setattr(pipeline, "process_participant", process)
    result = pipeline.Pipeline(
        config=Config(), output_dir=tmp_path / "out", on_error="continue"
    ).run(cohort)
    assert result.data.index.tolist() == ["001"]
    assert result.summary["failed"] == 1
    assert result.processing.loc["002", "status"] == "failed"


@pytest.mark.unit
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_pipeline_stops_with_partial_report_and_preserves_exception(
    tmp_path, cohort, monkeypatch, error
):
    from meganorm.API import pipeline

    def fail(*args, **kwargs):
        raise error("stop")

    monkeypatch.setattr(pipeline, "process_participant", fail)
    with pytest.raises(error):
        pipeline.Pipeline(config=Config(), output_dir=tmp_path / "out").run(cohort)
    report = json.loads((tmp_path / "out" / "run_summary.json").read_text())
    assert report["attempted"] == 1 and report["succeeded"] == 0
    assert (tmp_path / "out" / "processing.csv").exists()


@pytest.mark.unit
def test_pipeline_all_failed_raises_even_with_continue(tmp_path, cohort, monkeypatch):
    from meganorm.API import pipeline

    def fail(*args, **kwargs):
        raise RuntimeError("bad")

    monkeypatch.setattr(pipeline, "process_participant", fail)
    with pytest.raises(RuntimeError, match="All"):
        pipeline.Pipeline(
            config=Config(), output_dir=tmp_path / "out", on_error="continue"
        ).run(cohort)


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        "missing_demo",
        "duplicate_datasets",
        "duplicate_ids",
        "bad_session",
        "input_output",
        "surfaces",
    ],
)
def test_pipeline_preflights_before_processing(tmp_path, cohort, monkeypatch, case):
    from meganorm.API import pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("Processing started before preflight")

    monkeypatch.setattr(pipeline, "process_participant", forbidden)
    data = cohort
    output = tmp_path / "out"
    config = Config()
    if case == "missing_demo":
        (cohort.root / "participants.tsv").unlink()
    if case == "duplicate_datasets":
        data = [cohort, cohort]
    if case == "duplicate_ids":
        data = [
            cohort,
            Dataset(name="other", root=cohort.root, task="rest", extension=".fif"),
        ]
    if case == "bad_session":
        config.which_meg_session = 20
    if case == "input_output":
        output = cohort.root / "out"
    if case == "surfaces":
        config.apply_source_localization = True
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.Pipeline(config=config, output_dir=output).run(data)
    assert not (output / "config.json").exists()


@pytest.mark.unit
def test_pipeline_creates_preprocessed_output_directory(tmp_path, cohort, monkeypatch):
    from meganorm.API import pipeline

    def process(info, **kwargs):
        expected = kwargs["temp_dir"].parent / "Saved_outputs" / "Preprocessed_data"
        assert (
            expected.is_dir()
        ), "Preprocessed output parent must exist before processing"
        write_features(info, **kwargs)

    monkeypatch.setattr(pipeline, "process_participant", process)
    pipeline.Pipeline(
        config=Config(save_preprocessed_data=True), output_dir=tmp_path / "out"
    ).run(cohort)


@pytest.mark.unit
def test_multiple_datasets_keep_optional_line_frequencies(
    tmp_path, cohort, monkeypatch
):
    from meganorm.API import pipeline

    root = tmp_path / "second"
    (root / "003").mkdir(parents=True)
    (root / "003" / "rest.fif").touch()
    other = Dataset(
        name="other", root=root, task="rest", extension=".fif", line_freq=None
    )

    def process(info, **kwargs):
        expected = 50 if info["dataset"] == "cohort" else None
        assert (
            info["line_freq"] is None
            if expected is None
            else type(info["line_freq"]) is int
        )
        write_features(info, **kwargs)

    monkeypatch.setattr(pipeline, "process_participant", process)
    result = pipeline.Pipeline(config=Config(), output_dir=tmp_path / "out").run(
        [cohort, other]
    )
    assert result.data.index.tolist() == ["001", "002", "003"]
    assert result.summary["succeeded"] == 3


@pytest.mark.unit
@pytest.mark.parametrize(
    "family,options",
    [
        (
            "event",
            {
                "event_file_path": "events",
                "event_file_task": "event",
                "event_file_ending": ".tsv",
                "event_of_interest": 16,
            },
        ),
        (
            "annotation",
            {
                "annotation_path": "annotations",
                "annotaion_task_name": "bad",
                "annotation_ending": ".txt",
            },
        ),
        (
            "empty-room",
            {
                "empty_room_path": "empty",
                "empty_room_task": "empty",
                "empty_room_ending": ".fif",
            },
        ),
        ("position", {"pos_path": "headshape", "pos_file_ending": ".pos"}),
        ("transform", {"trans_path": "transforms"}),
    ],
)
def test_requested_auxiliary_matches_cannot_disappear(
    tmp_path, cohort, monkeypatch, family, options
):
    from meganorm.API import pipeline

    ds = Dataset(
        name="cohort", root=cohort.root, task="rest", extension=".fif", options=options
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Requested input vanished before processing")

    monkeypatch.setattr(pipeline, "process_participant", forbidden)
    with pytest.raises(FileNotFoundError, match=f"{family}.*cohort.*001"):
        pipeline.Pipeline(config=Config(), output_dir=tmp_path / "out").run(ds)
    assert not (tmp_path / "out").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        {"event_file_path": "events"},
        {"event_of_interest": 16},
        {"annotation_path": "annotations"},
        {"empty_room_task": "empty"},
        {"pos_path": "headshape"},
    ],
)
def test_incomplete_auxiliary_options_fail_before_processing(tmp_path, cohort, options):
    from meganorm.API import pipeline

    ds = Dataset(
        name="cohort", root=cohort.root, task="rest", extension=".fif", options=options
    )
    with pytest.raises(ValueError, match="Incomplete.*cohort"):
        pipeline.Pipeline(config=Config(), output_dir=tmp_path / "out").run(ds)
    assert not (tmp_path / "out").exists()
