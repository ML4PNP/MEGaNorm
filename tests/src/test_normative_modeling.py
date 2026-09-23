import json
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest

from meganorm.src import normative_modeling as nm


class RecordingNormData:
    """Small boundary double for inspecting data passed to PCNtoolkit."""

    calls = []

    @classmethod
    def from_dataframe(cls, **kwargs):
        instance = cls()
        instance.from_dataframe_kwargs = kwargs
        instance.response_vars = kwargs["response_vars"]
        instance.split_call = None
        cls.calls.append(instance)
        return instance

    def train_test_split(self, *, splits, split_names, random_state):
        self.split_call = {
            "splits": splits,
            "split_names": split_names,
            "random_state": random_state,
        }
        return "train-data", "test-data"


@pytest.fixture
def recording_norm_data(monkeypatch):
    RecordingNormData.calls.clear()
    monkeypatch.setattr(nm, "NormData", RecordingNormData)
    return RecordingNormData


@pytest.mark.unit
def test_impute_by_subgroup_uses_same_group_and_age_window():
    data = pd.DataFrame(
        {
            "age": [10.0, 9.0, 30.0, 10.0],
            "site": ["A", "A", "A", "B"],
            "feature": [np.nan, 2.0, 100.0, 50.0],
        }
    )

    result = nm.impute_by_subgroup(
        data,
        group_cols=["site"],
        subject_removal_nan_thr=0.5,
        imputation_con_var_window=5,
    )

    assert result.loc[0, "feature"] == pytest.approx(2.0)


@pytest.mark.unit
def test_impute_by_subgroup_falls_back_to_same_group_at_any_age():
    data = pd.DataFrame(
        {
            "age": [10.0, 30.0, 10.0],
            "site": ["A", "A", "B"],
            "feature": [np.nan, 4.0, 100.0],
        }
    )

    result = nm.impute_by_subgroup(
        data,
        group_cols=["site"],
        subject_removal_nan_thr=0.5,
        imputation_con_var_window=5,
    )

    assert result.loc[0, "feature"] == pytest.approx(4.0)


@pytest.mark.unit
def test_impute_by_subgroup_falls_back_to_global_statistic():
    data = pd.DataFrame(
        {
            "age": [10.0, 10.0, 30.0],
            "site": ["C", "A", "B"],
            "feature": [np.nan, 6.0, 10.0],
        }
    )

    result = nm.impute_by_subgroup(
        data,
        group_cols=["site"],
        subject_removal_nan_thr=0.5,
        imputation_con_var_window=5,
    )

    assert result.loc[0, "feature"] == pytest.approx(8.0)


@pytest.mark.unit
def test_impute_by_subgroup_respects_median_strategy():
    data = pd.DataFrame(
        {
            "age": [10.0, 9.0, 10.0, 11.0],
            "site": ["A", "A", "A", "A"],
            "feature": [np.nan, 1.0, 2.0, 100.0],
        }
    )

    result = nm.impute_by_subgroup(
        data,
        group_cols=["site"],
        subject_removal_nan_thr=0.5,
        strategy="median",
    )

    assert result.loc[0, "feature"] == pytest.approx(2.0)


@pytest.mark.unit
def test_impute_by_subgroup_applies_site_specific_window_extension():
    data = pd.DataFrame(
        {
            "age": [10.0, 17.0, 30.0],
            "site": ["A", "A", "A"],
            "feature": [np.nan, 7.0, 30.0],
        }
    )

    result = nm.impute_by_subgroup(
        data,
        group_cols=["site"],
        subject_removal_nan_thr=0.5,
        imputation_con_var_window=5,
        customized_age_window={"A": 3},
    )

    assert result.loc[0, "feature"] == pytest.approx(7.0)


@pytest.mark.unit
def test_impute_by_subgroup_drops_columns_at_missingness_threshold():
    data = pd.DataFrame(
        {
            "age": [10.0, 11.0, 12.0, 13.0],
            "site": ["A", "A", "A", "A"],
            "feature": [np.nan, 1.0, 2.0, 3.0],
        }
    )

    result = nm.impute_by_subgroup(
        data,
        group_cols=["site"],
        subject_removal_nan_thr=0.25,
    )

    assert "feature" not in result.columns


@pytest.mark.unit
def test_impute_by_subgroup_rejects_unknown_strategy():
    data = pd.DataFrame(
        {"age": [10.0, 11.0], "site": ["A", "A"], "feature": [np.nan, 2.0]}
    )

    with pytest.raises(ValueError, match="strategy"):
        nm.impute_by_subgroup(
            data,
            group_cols=["site"],
            subject_removal_nan_thr=0.75,
            strategy="mode",
        )


@pytest.mark.unit
def test_prepare_nm_data_filters_with_configured_subject_id_column(
    recording_norm_data,
):
    data = pd.DataFrame(
        {
            "subject_code": ["s1", "s2", "s3"],
            "age": [10.0, 11.0, 12.0],
            "site": ["A", "A", "B"],
            "roi": [1.0, 2.0, 3.0],
        }
    )

    result = nm.prepare_nm_data(
        data,
        response_vars=["roi"],
        covariate_list=["age"],
        batch_effect_list=["site"],
        subject_id_col_name="subject_code",
        which_subjects=["s1", "s3"],
        train_split_size=None,
    )

    passed = result.from_dataframe_kwargs["dataframe"]
    assert passed["subject_code"].tolist() == ["s1", "s3"]


@pytest.mark.unit
def test_prepare_nm_data_filters_cohorts_and_required_complete_rows(
    recording_norm_data,
):
    data = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3", "s4"],
            "age": [10.0, np.nan, 12.0, 13.0],
            "site": ["A", "A", None, "B"],
            "diagnosis": ["control", "control", "control", "case"],
            "roi": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = nm.prepare_nm_data(
        data,
        response_vars=["roi"],
        covariate_list=["age"],
        batch_effect_list=["site"],
        subject_id_col_name="subject",
        which_cohorts=["control"],
        train_split_size=None,
    )

    passed = result.from_dataframe_kwargs["dataframe"]
    assert passed["subject"].tolist() == ["s1"]
    assert result.from_dataframe_kwargs["remove_Nan"] is True


@pytest.mark.unit
def test_prepare_nm_data_excludes_rois_from_columns_and_response_vars(
    recording_norm_data,
):
    data = pd.DataFrame(
        {
            "subject": ["s1", "s2"],
            "age": [10.0, 11.0],
            "site": ["A", "B"],
            "frontal-lh": [1.0, 2.0],
            "occipital-lh": [3.0, 4.0],
        }
    )

    result = nm.prepare_nm_data(
        data,
        response_vars=["frontal-lh", "occipital-lh"],
        covariate_list=["age"],
        batch_effect_list=["site"],
        subject_id_col_name="subject",
        excluding_ROIs=["occipital"],
        train_split_size=None,
    )

    passed = result.from_dataframe_kwargs["dataframe"]
    assert "occipital-lh" not in passed.columns
    assert result.response_vars == ["frontal-lh"]


@pytest.mark.unit
def test_prepare_nm_data_includes_requested_bilateral_roi_and_metadata(
    recording_norm_data,
):
    data = pd.DataFrame(
        {
            "subject": ["s1", "s2"],
            "age": [10.0, 11.0],
            "site": ["A", "B"],
            "frontal-lh": [1.0, 2.0],
            "frontal-rh": [3.0, 4.0],
            "occipital-lh": [5.0, 6.0],
        }
    )

    result = nm.prepare_nm_data(
        data,
        response_vars=["frontal-lh", "frontal-rh", "occipital-lh"],
        covariate_list=["age"],
        batch_effect_list=["site"],
        subject_id_col_name="subject",
        including_ROIs=["frontal-lh"],
        train_split_size=None,
    )

    passed = result.from_dataframe_kwargs["dataframe"]
    assert passed.columns.tolist() == [
        "subject",
        "age",
        "site",
        "frontal-lh",
        "frontal-rh",
    ]
    assert result.response_vars == ["frontal-lh", "frontal-rh"]


@pytest.mark.unit
def test_prepare_nm_data_imputes_infinite_response_and_updates_boundary_flags(
    recording_norm_data,
):
    data = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3"],
            "age": [10.0, 11.0, 12.0],
            "site": ["A", "A", "A"],
            "roi": [1.0, np.inf, 3.0],
        }
    )

    result = nm.prepare_nm_data(
        data,
        response_vars=["roi"],
        covariate_list=["age"],
        batch_effect_list=["site"],
        subject_id_col_name="subject",
        subject_removal_nan_thr=0.5,
        missing_value_handling_method="mean",
        train_split_size=None,
    )

    passed = result.from_dataframe_kwargs["dataframe"]
    assert passed.loc[1, "roi"] == pytest.approx(2.0)
    assert result.from_dataframe_kwargs["remove_Nan"] is False
    assert result.response_vars == ["roi"]


@pytest.mark.unit
def test_prepare_nm_data_rejects_unknown_missing_value_method(recording_norm_data):
    data = pd.DataFrame(
        {
            "subject": ["s1", "s2"],
            "age": [10.0, 11.0],
            "site": ["A", "A"],
            "roi": [1.0, np.nan],
        }
    )

    with pytest.raises(ValueError, match="missing_value_handling_method"):
        nm.prepare_nm_data(
            data,
            response_vars=["roi"],
            covariate_list=["age"],
            batch_effect_list=["site"],
            subject_id_col_name="subject",
            missing_value_handling_method="mode",
            train_split_size=None,
        )


@pytest.mark.unit
def test_prepare_nm_data_converts_percentage_split_and_forwards_random_state(
    recording_norm_data,
):
    data = pd.DataFrame(
        {
            "subject": ["s1", "s2"],
            "age": [10.0, 11.0],
            "site": ["A", "B"],
            "roi": [1.0, 2.0],
        }
    )

    train, test = nm.prepare_nm_data(
        data,
        response_vars=["roi"],
        covariate_list=["age"],
        batch_effect_list=["site"],
        subject_id_col_name="subject",
        train_split_size=60,
        random_state=7,
    )

    created = recording_norm_data.calls[-1]
    assert (train, test) == ("train-data", "test-data")
    assert created.split_call == {
        "splits": pytest.approx((0.6, 0.4)),
        "split_names": ["train", "test"],
        "random_state": 7,
    }


@pytest.mark.unit
def test_prepare_nm_data_supports_multiple_covariates(recording_norm_data):
    data = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3"],
            "age": [10.0, 11.0, 12.0],
            "motion": [0.1, 0.2, 0.3],
            "site": ["A", "A", "B"],
            "roi": [1.0, 2.0, 3.0],
        }
    )

    result = nm.prepare_nm_data(
        data,
        response_vars=["roi"],
        covariate_list=["age", "motion"],
        batch_effect_list=["site"],
        subject_id_col_name="subject",
        train_split_size=None,
    )

    assert result.from_dataframe_kwargs["covariates"] == ["age", "motion"]


@pytest.mark.unit
def test_prepare_nm_data_removes_rows_missing_any_model_covariate(
    recording_norm_data,
):
    data = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3"],
            "age": [10.0, 11.0, 12.0],
            "motion": [0.1, np.nan, 0.3],
            "site": ["A", "A", "B"],
            "roi": [1.0, 2.0, 3.0],
        }
    )

    result = nm.prepare_nm_data(
        data,
        response_vars=["roi"],
        covariate_list=["age", "motion"],
        batch_effect_list=["site"],
        subject_id_col_name="subject",
        train_split_size=None,
    )

    passed = result.from_dataframe_kwargs["dataframe"]
    assert passed["subject"].tolist() == ["s1", "s3"]


@pytest.mark.unit
def test_prepare_nm_data_uses_selected_imputation_covariate(recording_norm_data):
    data = pd.DataFrame(
        {
            "subject": ["s0", "s1", "s2"],
            "age": [10.0, 11.0, 50.0],
            "motion": [100.0, 0.0, 101.0],
            "site": ["A", "A", "A"],
            "roi": [np.nan, 2.0, 8.0],
        }
    )

    result = nm.prepare_nm_data(
        data,
        response_vars=["roi"],
        covariate_list=["age", "motion"],
        batch_effect_list=["site"],
        subject_id_col_name="subject",
        impute_by_column_name="motion",
        subject_removal_nan_thr=0.5,
        missing_value_handling_method="mean",
        train_split_size=None,
    )

    passed = result.from_dataframe_kwargs["dataframe"]
    assert passed.loc[0, "roi"] == pytest.approx(8.0)


@pytest.mark.unit
def test_prepare_nm_data_forwards_outlier_configuration(recording_norm_data):
    data = pd.DataFrame(
        {
            "subject": ["s1", "s2"],
            "age": [10.0, 11.0],
            "site": ["A", "B"],
            "roi": [1.0, 2.0],
        }
    )

    result = nm.prepare_nm_data(
        data,
        response_vars=["roi"],
        covariate_list=["age"],
        batch_effect_list=["site"],
        subject_id_col_name="subject",
        remove_outliers=True,
        remove_outliers_approach="iqr",
        remove_outliers_group_by="site",
        iqr_factor=2.5,
        train_split_size=None,
    )

    boundary = result.from_dataframe_kwargs
    assert boundary["remove_outliers"] is True
    assert boundary["remove_outliers_approach"] == "iqr"
    assert boundary["remove_outliers_group_by"] == "site"
    assert boundary["iqr_factor"] == pytest.approx(2.5)


@pytest.mark.unit
def test_prepare_nm_data_rejects_empty_covariate_list():
    data = pd.DataFrame({"subject": ["s1"], "site": ["A"], "roi": [1.0]})

    with pytest.raises(ValueError, match="at least one covariate"):
        nm.prepare_nm_data(
            data,
            response_vars=["roi"],
            covariate_list=[],
            batch_effect_list=["site"],
            subject_id_col_name="subject",
            train_split_size=None,
        )


@pytest.mark.unit
@pytest.mark.parametrize("split", [-0.1, 1.0, 100, 150])
def test_prepare_nm_data_rejects_split_outside_open_unit_interval(
    split, recording_norm_data
):
    data = pd.DataFrame(
        {
            "subject": ["s1", "s2"],
            "age": [10.0, 11.0],
            "site": ["A", "B"],
            "roi": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="train_split_size"):
        nm.prepare_nm_data(
            data,
            response_vars=["roi"],
            covariate_list=["age"],
            batch_effect_list=["site"],
            subject_id_col_name="subject",
            train_split_size=split,
        )


@pytest.mark.integration
def test_model_diagnostics_collects_and_saves_real_arviz_summaries(tmp_path):
    models_path = tmp_path / "models"
    models_path.mkdir()
    posterior = np.vstack(
        [
            np.linspace(-1.0, 1.0, 100),
            np.linspace(-0.9, 1.1, 100),
        ]
    )
    for model_name in ["roi_alpha", "roi_beta"]:
        model_path = models_path / model_name
        model_path.mkdir()
        az.from_dict(posterior={"theta": posterior}).to_netcdf(model_path / "idata.nc")
    (models_path / "normative_model.json").write_text("{}")
    save_path = tmp_path / "diagnostics"

    result = nm.model_diagnostics(models_path, save_path)

    assert set(result["model"]) == {"roi_alpha", "roi_beta"}
    assert set(result["parameter"]) == {"theta"}
    assert result[["r_hat", "ess_bulk", "ess_tail", "mcse_sd"]].notna().all().all()
    saved = pd.read_csv(save_path / "models_diagnosis.csv", index_col=0)
    pd.testing.assert_frame_equal(saved, result, check_dtype=False)


class RecordingNormativeModel:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_call = None
        self.fit_predict_call = None
        type(self).instances.append(self)

    def fit(self, train):
        self.fit_call = train

    def fit_predict(self, train, test):
        self.fit_predict_call = (train, test)


class RecordingRunner:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_call = None
        self.fit_predict_call = None
        type(self).instances.append(self)

    def fit(self, model, train, *, observe):
        self.fit_call = (model, train, observe)

    def fit_predict(self, model, train, test, *, observe):
        self.fit_predict_call = (model, train, test, observe)


@pytest.fixture
def recording_training_boundaries(monkeypatch):
    RecordingNormativeModel.instances.clear()
    RecordingRunner.instances.clear()
    monkeypatch.setattr(nm, "NormativeModel", RecordingNormativeModel)
    monkeypatch.setattr(nm, "Runner", RecordingRunner)
    return RecordingNormativeModel, RecordingRunner


class AmbiguousNormData:
    def __init__(self, response_vars):
        self.response_vars = response_vars

    def __bool__(self):
        raise ValueError("NormData truth value is ambiguous")


@pytest.mark.unit
def test_nm_model_train_fits_training_data_without_test_set(
    tmp_path, recording_training_boundaries
):
    train = SimpleNamespace(response_vars=["roi"])

    nm.nm_model_train(
        train=train,
        test=None,
        project_dir=tmp_path,
        experiment_name="experiment",
        template_regression_model="template",
        model_name="model",
    )

    model = RecordingNormativeModel.instances[-1]
    assert model.fit_call is train
    assert model.fit_predict_call is None
    assert model.kwargs == {
        "template_regression_model": "template",
        "savemodel": True,
        "evaluate_model": True,
        "saveresults": True,
        "saveplots": True,
        "save_dir": str(tmp_path / "Normative_models"),
        "inscaler": "standardize",
        "outscaler": "standardize",
        "name": "model",
    }


@pytest.mark.unit
def test_nm_model_train_accepts_xarray_like_test_data_without_truth_check(
    tmp_path, recording_training_boundaries
):
    train = SimpleNamespace(response_vars=["roi"])
    test = AmbiguousNormData(response_vars=["roi"])

    nm.nm_model_train(
        train=train,
        test=test,
        project_dir=tmp_path,
        experiment_name="experiment",
        template_regression_model="template",
        model_name="model",
    )

    model = RecordingNormativeModel.instances[-1]
    assert model.fit_predict_call == (train, test)
    assert model.fit_call is None


@pytest.mark.unit
def test_nm_model_train_parallel_requires_job_configuration(
    tmp_path, recording_training_boundaries
):
    train = SimpleNamespace(response_vars=["roi"])

    with pytest.raises(ValueError, match="configuration"):
        nm.nm_model_train(
            train=train,
            test=None,
            project_dir=tmp_path,
            experiment_name="experiment",
            template_regression_model="template",
            model_name="model",
            if_parallel=True,
            job_configs=None,
        )


@pytest.mark.unit
def test_nm_model_train_configures_parallel_fit_predict(
    tmp_path, recording_training_boundaries
):
    train = SimpleNamespace(response_vars=["roi_a", "roi_b"])
    test = AmbiguousNormData(response_vars=["roi_a", "roi_b"])
    jobs = {
        "env_path": "/envs/meganorm",
        "job_type": "slurm",
        "time_limit": "01:00:00",
        "memory": "8G",
        "n_cores": 4,
        "preamble": ["module load python"],
        "max_retries": 2,
    }

    nm.nm_model_train(
        train=train,
        test=test,
        project_dir=tmp_path,
        experiment_name="experiment",
        template_regression_model="template",
        model_name="model",
        if_cross_validate=True,
        if_parallel=True,
        job_configs=jobs,
    )

    model = RecordingNormativeModel.instances[-1]
    runner = RecordingRunner.instances[-1]
    assert runner.kwargs == {
        "cross_validate": True,
        "parallelize": True,
        "n_batches": 2,
        "environment": "/envs/meganorm",
        "job_type": "slurm",
        "time_limit": "01:00:00",
        "memory": "8G",
        "n_cores": 4,
        "preamble": ["module load python"],
        "log_dir": str(tmp_path / "Normative_models" / "nm_parallel_logs"),
        "temp_dir": str(tmp_path / "Normative_models" / "nm_temp"),
        "max_retries": 2,
    }
    assert runner.fit_predict_call == (model, train, test, False)
    assert runner.fit_call is None


@pytest.mark.integration
def test_anova_group_level_effect_creates_output_directory_and_saves_json(
    tmp_path,
):
    data = pd.DataFrame(
        {
            "site": ["A", "A", "B", "B"],
            "roi": [1.0, 2.0, 4.0, 5.0],
        }
    )
    output_dir = tmp_path / "nested" / "group_effects"

    result = nm.anova_group_level_effect(
        data,
        batch_effect="site",
        save_tag="raw",
        save_output_path=output_dir,
    )

    output_file = output_dir / "site_group_effect_raw.json"
    assert json.loads(output_file.read_text()) == result


@pytest.mark.unit
def test_anova_group_level_effect_does_not_swallow_user_interrupt(monkeypatch):
    data = pd.DataFrame(
        {
            "site": ["A", "A", "B", "B"],
            "roi": [1.0, 2.0, 4.0, 5.0],
        }
    )

    def interrupt_anova(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(nm.pg, "anova", interrupt_anova)

    with pytest.raises(KeyboardInterrupt):
        nm.anova_group_level_effect(data, batch_effect="site")


@pytest.mark.integration
def test_anova_group_level_effect_filters_each_response_independently():
    data = pd.DataFrame(
        {
            "site": ["A", "A", "A", "B", "B", "B"],
            "roi": [1.0, 2.0, np.inf, 4.0, 5.0, np.nan],
            "second_roi": [10.0, -np.inf, 12.0, np.nan, 20.0, 22.0],
        }
    )

    result = nm.anova_group_level_effect(data, batch_effect="site")

    assert result["roi"]["p_val"] == pytest.approx(0.0513167019)
    assert result["roi"]["np2"] == pytest.approx(0.9)
    assert result["second_roi"]["p_val"] == pytest.approx(0.0194193243)
    assert result["second_roi"]["np2"] == pytest.approx(0.9615384615)


@pytest.mark.integration
def test_anova_group_level_effect_skips_metadata_columns():
    data = pd.DataFrame(
        {
            "age": [10.0, 11.0, 12.0, 13.0],
            "sex": ["F", "M", "F", "M"],
            "site": ["A", "A", "B", "B"],
            "eyes": ["open", "closed", "open", "closed"],
            "diagnosis": ["control", "control", "case", "case"],
            "roi": [1.0, 2.0, 4.0, 5.0],
        }
    )

    result = nm.anova_group_level_effect(data, batch_effect="site")

    assert set(result) == {"roi"}


@pytest.mark.integration
def test_anova_group_level_effect_isolates_invalid_response_column():
    data = pd.DataFrame(
        {
            "site": ["A", "A", "B", "B"],
            "roi": [1.0, 2.0, 4.0, 5.0],
            "label": ["low", "medium", "high", "very high"],
        }
    )

    result = nm.anova_group_level_effect(data, batch_effect="site")

    assert result["roi"]["p_val"] is not None
    assert result["roi"]["np2"] is not None
    assert result["label"] == {"p_val": None, "np2": None}
