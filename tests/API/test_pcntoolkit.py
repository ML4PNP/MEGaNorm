import numpy as np
import pandas as pd
import pytest
from pcntoolkit import BLR, NormData, NormativeModel


@pytest.mark.integration
def test_real_outputs_keep_ids_response_order_and_original_scale(tmp_path):
    from meganorm.API._pcntoolkit import extract_test_outputs

    rng = np.random.default_rng(42)
    age = np.linspace(20, 70, 40)
    frame = pd.DataFrame(
        {
            "id": [f"{i:03}" for i in range(40)],
            "age": age,
            "large": 100 + age * 2 + rng.normal(0, 2, 40),
            "small": 3 + age / 10 + rng.normal(0, 0.2, 40),
        }
    )

    def prepare(part):
        return NormData.from_dataframe(
            name="test",
            dataframe=part,
            covariates=["age"],
            batch_effects=[],
            response_vars=["large", "small"],
            subject_ids="id",
        )

    train, test = prepare(frame.iloc[:30]), prepare(frame.iloc[30:])
    model = NormativeModel(
        template_regression_model=BLR(),
        save_dir=str(tmp_path),
        savemodel=False,
        saveresults=False,
        saveplots=False,
        evaluate_model=True,
    )
    model.fit_predict(train, test)
    prediction, z, metrics = extract_test_outputs(
        model, test, ("small", "large"), evaluate=True
    )
    assert prediction.index.tolist() == frame.id.iloc[30:].tolist()
    assert prediction.columns.tolist() == ["small", "large"]
    assert prediction["large"].mean() > 100
    np.testing.assert_allclose(
        prediction.to_numpy(), test.Yhat.sel(response_vars=["small", "large"]).values
    )
    np.testing.assert_allclose(
        z.to_numpy(), test.Z.sel(response_vars=["small", "large"]).values
    )
    assert metrics.index.tolist() == ["small", "large"]
    assert "RMSE" in metrics
    assert test.attrs["is_scaled"] is False
    assert extract_test_outputs(model, test, ("large",), evaluate=False)[2] is None


@pytest.mark.unit
def test_unknown_results_schema_fails_clearly():
    from meganorm.API._pcntoolkit import extract_test_outputs

    with pytest.raises(RuntimeError, match="PCNtoolkit"):
        extract_test_outputs(object(), object(), ("roi",), evaluate=True)


@pytest.mark.integration
def test_preparation_supports_released_pcntoolkit_without_outlier_extensions():
    from meganorm.API._pcntoolkit import prepare_data

    frame = pd.DataFrame(
        {
            "id": ["001", "002", "003", "004", "005"],
            "age": [20, 30, 40, 50, 60],
            "y": [1, 2, 3, 4, 5],
        }
    )
    train, test = prepare_data(
        frame,
        responses=["y"],
        covariates=["age"],
        batch_effects=[],
        participant_id="id",
        train_fraction=0.8,
        random_state=42,
    )
    assert set(train.subject_ids.values) | set(test.subject_ids.values) == set(frame.id)
    assert len(train.subject_ids) == 4 and len(test.subject_ids) == 1
