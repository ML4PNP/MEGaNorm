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
    with pytest.warns(UserWarning, match="removed before rerun"):
        repeated = run.run(cohort)
    assert repeated.summary["succeeded"] == 2


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


@pytest.mark.unit
def test_rerun_clears_residual_outputs_and_uses_changed_cohort(
    tmp_path, cohort, monkeypatch
):
    from meganorm.API import pipeline

    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "process_participant", write_features)
    pipeline.Pipeline(config=Config(), output_dir=out).run(cohort)
    residual = out / "Features" / "Saved_outputs" / "Preprocessed_data" / "old.fif"
    residual.parent.mkdir(parents=True, exist_ok=True)
    residual.write_text("old preprocessing")
    unrelated = out / "research-notes.txt"
    unrelated.write_text("keep me")
    (cohort.root / "002" / "002_rest.fif").unlink()

    def new_features(info, **kwargs):
        # Cleanup and warning must precede processing.
        assert not residual.exists()
        assert not (out / "Features" / "temp" / "002.csv").exists()
        assert not (out / "Features" / "all_features.csv").exists()
        assert not (out / "features_with_demographics.csv").exists()
        assert Config.load(str(kwargs["config_path"])).save_preprocessed_data is False
        pd.DataFrame(
            {"feature__alpha": [0.8]}, index=[kwargs["participant_id"]]
        ).to_csv(kwargs["temp_dir"] / f'{kwargs["participant_id"]}.csv')

    monkeypatch.setattr(pipeline, "process_participant", new_features)
    with pytest.warns(UserWarning, match="removed before rerun"):
        result = pipeline.Pipeline(
            config=Config(save_preprocessed_data=False), output_dir=out
        ).run(cohort)
    assert result.data.index.tolist() == ["001"]
    assert result.data["feature__alpha"].tolist() == [0.8]
    assert result.manifest.participant_id.tolist() == ["001"]
    assert result.processing.index.tolist() == ["001"]
    assert unrelated.read_text() == "keep me"


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["raise", "continue"])
def test_failed_rerun_removes_previous_aggregate_tables(
    tmp_path, cohort, monkeypatch, mode
):
    from meganorm.API import pipeline

    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "process_participant", write_features)
    pipeline.Pipeline(config=Config(), output_dir=out).run(cohort)

    def fail(*args, **kwargs):
        raise RuntimeError("rerun failure")

    monkeypatch.setattr(pipeline, "process_participant", fail)
    with pytest.warns(UserWarning, match="removed before rerun"):
        with pytest.raises(RuntimeError):
            pipeline.Pipeline(config=Config(), output_dir=out, on_error=mode).run(
                cohort
            )
    assert not (out / "Features" / "all_features.csv").exists()
    assert not (out / "features_with_demographics.csv").exists()
    assert not list((out / "Features" / "temp").glob("*.csv"))
    assert json.loads((out / "run_summary.json").read_text())["succeeded"] == 0


@pytest.mark.unit
def test_partial_rerun_does_not_collect_previous_successes(
    tmp_path, cohort, monkeypatch
):
    from meganorm.API import pipeline

    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "process_participant", write_features)
    pipeline.Pipeline(config=Config(), output_dir=out).run(cohort)

    def process(info, **kwargs):
        if kwargs["participant_id"] == "002":
            raise RuntimeError("failed on rerun")
        write_features(info, **kwargs)

    monkeypatch.setattr(pipeline, "process_participant", process)
    with pytest.warns(UserWarning, match="removed before rerun"):
        result = pipeline.Pipeline(
            config=Config(), output_dir=out, on_error="continue"
        ).run(cohort)
    assert result.data.index.tolist() == ["001"]
    assert result.processing.loc["002", "status"] == "failed"
    assert not (out / "Features" / "temp" / "002.csv").exists()


@pytest.mark.unit
def test_rerun_requires_new_participant_csv(tmp_path, cohort, monkeypatch):
    from meganorm.API import pipeline

    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "process_participant", write_features)
    pipeline.Pipeline(config=Config(), output_dir=out).run(cohort)
    monkeypatch.setattr(pipeline, "process_participant", lambda *args, **kwargs: None)
    with pytest.warns(UserWarning, match="removed before rerun"):
        with pytest.raises(RuntimeError, match="All participants failed"):
            pipeline.Pipeline(config=Config(), output_dir=out, on_error="continue").run(
                cohort
            )
    assert json.loads((out / "run_summary.json").read_text())["succeeded"] == 0


@pytest.mark.unit
def test_preflight_failure_preserves_existing_run(tmp_path, cohort, monkeypatch):
    import warnings
    from meganorm.API import pipeline

    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "process_participant", write_features)
    pipeline.Pipeline(config=Config(), output_dir=out).run(cohort)
    paths = [
        out / "config.json",
        out / "Features" / "all_features.csv",
        out / "run_summary.json",
    ]
    before = {p: p.read_bytes() for p in paths}
    cohort.demographics.unlink()
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        with pytest.raises(FileNotFoundError):
            pipeline.Pipeline(config=Config(), output_dir=out).run(cohort)
    assert not any(
        "removed before rerun" in str(w.message) or "overwritten" in str(w.message)
        for w in messages
    )
    assert {p: p.read_bytes() for p in paths} == before


@pytest.mark.unit
def test_warning_as_error_preserves_previous_run(tmp_path, cohort, monkeypatch):
    import warnings
    from meganorm.API import pipeline

    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "process_participant", write_features)
    pipeline.Pipeline(config=Config(), output_dir=out).run(cohort)
    before = (out / "Features" / "all_features.csv").read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="removed before rerun"):
            pipeline.Pipeline(config=Config(), output_dir=out).run(cohort)
    assert (out / "Features" / "all_features.csv").read_bytes() == before


@pytest.mark.unit
def test_cleanup_refuses_managed_directory_containing_inputs(
    tmp_path, cohort, monkeypatch
):
    import shutil
    from meganorm.API import pipeline

    out = tmp_path / "out"
    nested = out / "Features" / "input"
    shutil.copytree(cohort.root, nested)
    dataset = Dataset(name="nested", root=nested, task="rest", extension=".fif")
    monkeypatch.setattr(pipeline, "process_participant", write_features)
    with pytest.raises(ValueError, match="managed output.*input"):
        pipeline.Pipeline(config=Config(), output_dir=out).run(dataset)
    assert (nested / "001" / "001_rest.fif").exists()


@pytest.mark.unit
def test_cleanup_unlinks_features_symlink_without_deleting_target(
    tmp_path, cohort, monkeypatch
):
    from meganorm.API import pipeline

    out = tmp_path / "out"
    out.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "keep.txt"
    sentinel.write_text("keep")
    try:
        (out / "Features").symlink_to(external, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip(
                "Windows symlink creation requires Developer Mode or privileges"
            )
        raise
    monkeypatch.setattr(pipeline, "process_participant", write_features)
    with pytest.warns(UserWarning, match="removed before rerun"):
        pipeline.Pipeline(config=Config(), output_dir=out).run(cohort)
    assert sentinel.read_text() == "keep"
    assert not (out / "Features").is_symlink()


@pytest.mark.unit
def test_each_participant_must_write_its_own_csv(tmp_path, cohort, monkeypatch):
    from meganorm.API import pipeline

    def process(info, **kwargs):
        if kwargs["participant_id"] == "001":
            write_features(info, **kwargs)
            premature = dict(kwargs, participant_id="002")
            write_features(info, **premature)

    monkeypatch.setattr(pipeline, "process_participant", process)
    result = pipeline.Pipeline(
        config=Config(), output_dir=tmp_path / "out", on_error="continue"
    ).run(cohort)
    assert result.data.index.tolist() == ["001"]
    assert result.processing.loc["002", "status"] == "failed"


@pytest.mark.unit
def test_cleanup_preserves_demographics_in_managed_directory(
    tmp_path, cohort, monkeypatch
):
    from meganorm.API import pipeline

    out = tmp_path / "out"
    features = out / "Features"
    features.mkdir(parents=True)
    demo = features / "participants.tsv"
    demo.write_bytes(cohort.demographics.read_bytes())
    cohort.demographics = demo
    monkeypatch.setattr(pipeline, "process_participant", write_features)
    with pytest.raises(ValueError, match="managed output.*input"):
        pipeline.Pipeline(config=Config(), output_dir=out).run(cohort)
    assert demo.exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    [
        "freesurfer_template_path",
        "freesurfer_home",
        "freesurfer_license",
        "parcellation_annot_fname",
    ],
)
def test_cleanup_preserves_configured_input_paths(tmp_path, cohort, monkeypatch, field):
    from meganorm.API import pipeline

    out = tmp_path / "out"
    features = out / "Features"
    features.mkdir(parents=True)
    source = features / field
    if field in {"freesurfer_template_path", "freesurfer_home"}:
        source.mkdir()
        sentinel = source / "input.txt"
    else:
        sentinel = source
    sentinel.write_text("input data")
    settings = {field: source if field == "parcellation_annot_fname" else str(source)}
    if field == "freesurfer_template_path":
        settings.update(apply_source_localization=True, apply_mri_template=True)
    monkeypatch.setattr(pipeline, "process_participant", write_features)
    with pytest.raises(ValueError, match="managed output.*input"):
        pipeline.Pipeline(config=Config(**settings), output_dir=out).run(cohort)
    assert sentinel.read_text() == "input data"
