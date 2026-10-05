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
    from meganorm.API.modeling import NormativeModel

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


@pytest.mark.unit
def test_existing_model_outputs_are_not_overwritten(tmp_path, model_frame):
    output = tmp_path / "model" / "Normative_models"
    output.mkdir(parents=True)
    (output / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        analysis(tmp_path).fit(model_frame, responses="large")
    assert (output / "sentinel").read_text() == "keep"
