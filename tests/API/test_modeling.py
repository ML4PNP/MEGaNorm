from meganorm.API.modeling import NormativeModel
import numpy as np
import pandas as pd
import pytest
from pcntoolkit import BLR


@pytest.fixture
def model_frame():
    rng = np.random.default_rng(42)
    age = np.linspace(18, 75, 40)
    return pd.DataFrame(
        {
            "age": age,
            "site": ["A"] * 40,
            "sex": ["F", "M"] * 20,
            "large": 100 + 2 * age + rng.normal(0, 2, 40),
            "small": 2 + age / 10 + rng.normal(0, 0.2, 40),
            "unused": [np.nan] * 40,
        },
        index=pd.Index([f"{i:03}" for i in range(40)], name="participant_id"),
    )


def analysis(tmp_path, **kwargs):

    return NormativeModel(
        estimator=BLR(),
        covariates=["age"],
        batch_effects=["site"],
        output_dir=tmp_path / "model",
        save_plots=False,
        **kwargs,
    )


@pytest.mark.integration
def test_real_fit_returns_aligned_tables_and_copies_inputs(tmp_path, model_frame):
    original = model_frame.copy(deep=True)
    runner = analysis(tmp_path)
    results = runner.fit(model_frame, responses=["small", "large"])
    assert results.predictions.columns.tolist() == ["small", "large"]
    assert results.predictions.index.tolist() == list(results.test.subject_ids.values)
    assert results.z_scores.shape == (8, 2)
    assert results.predictions.large.mean() > 100
    assert results.metrics.index.tolist() == ["small", "large"]
    assert all(path.exists() for path in results.paths.values())
    pd.testing.assert_frame_equal(model_frame, original)
    assert not runner.estimator.is_fitted
    assert set(results.train.subject_ids.values).isdisjoint(
        results.test.subject_ids.values
    )
    with pytest.raises(RuntimeError, match="fitted"):
        runner.fit(model_frame, responses="large")


@pytest.mark.integration
def test_train_only_and_explicit_test_cohorts(tmp_path, model_frame):
    runner = analysis(tmp_path)
    results = runner.fit(model_frame, responses="large", train_fraction=None)
    assert (
        results.test is None and results.predictions is None and results.metrics is None
    )
    other = analysis(tmp_path / "other")
    results = other.fit(
        model_frame.iloc[:30],
        test_data=model_frame.iloc[30:],
        responses="large",
        train_fraction=None,
    )
    assert results.predictions.index.tolist() == model_frame.index[30:].tolist()


@pytest.mark.integration
def test_explicit_test_cohort_preserves_native_training_results(tmp_path, model_frame):
    results = analysis(tmp_path).fit(
        model_frame.iloc[:30],
        test_data=model_frame.iloc[30:],
        responses="large",
        train_fraction=None,
    )

    for cohort, expected_ids in [
        (results.train, model_frame.index[:30].tolist()),
        (results.test, model_frame.index[30:].tolist()),
    ]:
        table = pd.read_csv(
            results.paths["results"] / f"Z_{cohort.name}.csv",
            dtype={"subject_ids": str},
        )
        assert table.subject_ids.tolist() == expected_ids


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        "missing_col",
        "nonnumeric",
        "missing",
        "infinite",
        "duplicate",
        "overlap",
        "fraction",
        "test_fraction",
        "test_overlap",
        "new_site",
    ],
)
def test_model_input_errors_precede_fitting(tmp_path, model_frame, case):
    runner = analysis(tmp_path)
    data = model_frame.copy()
    args = {"responses": ["large"]}
    if case == "missing_col":
        args["responses"] = ["absent"]
    if case == "nonnumeric":
        data["age"] = "old"
    if case == "missing":
        data.iloc[0, data.columns.get_loc("large")] = np.nan
    if case == "infinite":
        data.iloc[0, data.columns.get_loc("age")] = np.inf
    if case == "duplicate":
        data.index = pd.Index(["same"] * 40, name="participant_id")
    if case == "overlap":
        args["responses"] = ["age"]
    if case == "fraction":
        args["train_fraction"] = 80
    if case == "test_fraction":
        args["test_data"] = data.iloc[30:]
        data = data.iloc[:30]
    if case == "test_overlap":
        args.update(test_data=data.iloc[30:], train_fraction=None)
    if case == "new_site":
        test = data.iloc[30:].copy()
        test["site"] = "B"
        args.update(test_data=test, train_fraction=None)
        data = data.iloc[:30]
    with pytest.raises(ValueError):
        runner.fit(data, **args)
    assert not (tmp_path / "model" / "Normative_models").exists()


@pytest.mark.integration
def test_drop_records_only_selected_model_missingness(tmp_path, model_frame):
    model_frame.loc["001", "large"] = np.nan
    results = analysis(tmp_path, missing="drop", evaluate=False).fit(
        model_frame, responses="large"
    )
    assert results.summary["eligible"] == 39
    assert results.participants.loc["001", "split"] == "excluded"
    assert results.metrics is None
    assert results.z_scores is not None


@pytest.mark.integration
def test_existing_model_outputs_are_overwritten(tmp_path, model_frame):
    output = tmp_path / "model" / "Normative_models"
    output.mkdir(parents=True)

    sentinel = output / "sentinel"
    sentinel.write_text("keep")

    model = NormativeModel(
        estimator=BLR(),
        covariates=["age"],
        batch_effects=["site"],
        output_dir=tmp_path / "model",
        name="test_model",
        save_plots=False,
    )

    with pytest.warns(UserWarning, match="removed"):
        model.fit(
            model_frame,
            responses=["small"],
            train_fraction=0.5,
            random_state=42,
        )

    assert not sentinel.exists()


@pytest.mark.integration
def test_model_cleanup_unlinks_directory_symlink_without_touching_target(
    tmp_path, model_frame
):
    import os

    runner = analysis(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("keep")
    runner.output_dir.mkdir()
    link = runner.output_dir / "Normative_models"
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privileges unavailable")
        raise
    with pytest.warns(UserWarning, match="removed"):
        runner.fit(model_frame, responses="large", train_fraction=None)
    assert sentinel.read_text() == "keep"
    assert not link.is_symlink()


@pytest.mark.integration
def test_model_cleanup_removes_dangling_output_link(tmp_path, model_frame):
    import os

    runner = analysis(tmp_path, evaluate=False)
    runner.output_dir.mkdir()
    target = tmp_path / "external-predictions.csv"
    link = runner.output_dir / "predictions.csv"
    try:
        link.symlink_to(target)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privileges unavailable")
        raise
    with pytest.warns(UserWarning, match="removed"):
        runner.fit(model_frame, responses="large", train_fraction=None)
    assert not link.is_symlink()
    assert not target.exists()


@pytest.mark.unit
def test_model_cleanup_protects_feature_dataset_artifacts(tmp_path, model_frame):
    from meganorm.API.results import FeatureDataset

    runner = analysis(tmp_path)
    source = runner.output_dir / "Normative_models" / "source"
    source.mkdir(parents=True)
    path = source / "features.csv"
    path.write_text("input data")
    data = FeatureDataset(
        model_frame,
        ("large",),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        source,
        {"data": path},
    )
    with pytest.raises(ValueError, match="input"):
        runner.fit(data, responses="large")
    assert path.read_text() == "input data"


@pytest.mark.unit
def test_invalid_model_input_preserves_previous_results(tmp_path, model_frame):
    runner = analysis(tmp_path)
    runner.output_dir.mkdir()
    previous = runner.output_dir / "predictions.csv"
    previous.write_text("previous results")
    with pytest.raises(ValueError, match="Missing model columns"):
        runner.fit(model_frame, responses="absent")
    assert previous.read_text() == "previous results"


@pytest.mark.unit
def test_failed_model_rerun_does_not_leave_old_predictions(
    tmp_path, model_frame, monkeypatch
):
    from meganorm.API import modeling

    runner = analysis(tmp_path)
    runner.output_dir.mkdir()
    previous = runner.output_dir / "predictions.csv"
    previous.write_text("previous results")
    unrelated = runner.output_dir / "notes.txt"
    unrelated.write_text("keep")

    def fail(**kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(modeling, "nm_model_train", fail)
    with (
        pytest.warns(UserWarning, match="removed"),
        pytest.raises(RuntimeError, match="fit failed"),
    ):
        runner.fit(model_frame, responses="large")
    assert not previous.exists()
    assert unrelated.read_text() == "keep"


@pytest.mark.unit
def test_model_cleanup_warning_can_prevent_deletion(tmp_path, model_frame):
    import warnings

    runner = analysis(tmp_path)
    runner.output_dir.mkdir()
    previous = runner.output_dir / "predictions.csv"
    previous.write_text("previous results")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="removed"):
            runner.fit(model_frame, responses="large")
    assert previous.read_text() == "previous results"


@pytest.mark.unit
@pytest.mark.parametrize("field", ["input_scaler", "output_scaler"])
def test_invalid_scaler_is_rejected_when_constructing_the_model(tmp_path, field):
    with pytest.raises(ValueError, match=field):
        analysis(tmp_path, **{field: "standarize"})


@pytest.mark.unit
@pytest.mark.parametrize("field", ["input_scaler", "output_scaler"])
def test_changed_invalid_scaler_preserves_previous_results(
    tmp_path, model_frame, field
):
    runner = analysis(tmp_path)
    runner.output_dir.mkdir()
    previous = runner.output_dir / "predictions.csv"
    previous.write_text("previous results")
    setattr(runner, field, "standarize")

    with pytest.raises(ValueError):
        runner.fit(model_frame, responses="large", train_fraction=None)

    assert previous.read_text() == "previous results"


@pytest.mark.unit
@pytest.mark.parametrize(
    "response",
    [
        "absolute",
        "../outside",
        r"..\outside",
        "nested/response",
        r"nested\response",
        r"C:\outside",
        "C:outside",
        ".",
        "..",
        "response\0name",
        "Alpha:power",
        "Alpha*power",
        "Alpha?power",
        "Alpha<power",
        "Alpha>power",
        'Alpha"power',
        "Alpha|power",
        "Alpha\x01power",
        "Alpha\npower",
        "Alpha.",
        "Alpha ",
        "NUL",
        "com1.txt",
        "LPT9",
        "normative_model.json",
    ],
)
def test_unsafe_response_paths_fail_before_cleanup_or_fitting(
    tmp_path, model_frame, monkeypatch, response
):
    from meganorm.API import modeling

    if response == "absolute":
        response = str(tmp_path / "outside")
    runner = analysis(tmp_path)
    runner.output_dir.mkdir()
    previous = runner.output_dir / "predictions.csv"
    previous.write_text("previous results")
    model_frame = model_frame.rename(columns={"large": response})

    def forbidden(**kwargs):
        pytest.fail("An unsafe response label reached the model fitter")

    monkeypatch.setattr(modeling, "nm_model_train", forbidden)
    with pytest.raises(ValueError, match="[Rr]esponse"):
        runner.fit(model_frame, responses=response, train_fraction=None)

    assert previous.read_text() == "previous results"
    assert not (tmp_path / "outside").exists()


@pytest.mark.unit
def test_response_case_collisions_fail_before_cleanup_or_fitting(
    tmp_path, model_frame, monkeypatch
):
    from meganorm.API import modeling

    runner = analysis(tmp_path)
    runner.output_dir.mkdir()
    previous = runner.output_dir / "predictions.csv"
    previous.write_text("previous results")
    model_frame = model_frame.rename(columns={"large": "Alpha", "small": "alpha"})

    def forbidden(**kwargs):
        pytest.fail("Response filenames that differ only in case reached the fitter")

    monkeypatch.setattr(modeling, "nm_model_train", forbidden)
    with pytest.raises(ValueError, match="[Rr]esponse"):
        runner.fit(model_frame, responses=["Alpha", "alpha"], train_fraction=None)

    assert previous.read_text() == "previous results"


@pytest.mark.integration
def test_portable_scientific_response_name_can_be_saved(tmp_path, model_frame):
    response = "Alpha__MEG001 [fT^2 Hz^-1]"
    results = analysis(tmp_path).fit(
        model_frame.rename(columns={"large": response}),
        responses=response,
        train_fraction=None,
    )

    assert (results.paths["model"] / response / "regression_model.json").is_file()
