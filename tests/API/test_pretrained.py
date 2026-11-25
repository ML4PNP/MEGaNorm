"""Local bundle inference: real HBR first, with the same checks for BLR."""

import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pcntoolkit import NormData

from sharing.conftest import data
from sharing.test_pcn_roundtrip import RESPONSES, multi_fitted, multi_metadata
from sharing.test_schema import manifest

UNITS = {"age": "years", **dict.fromkeys(RESPONSES, "V")}
ORDER = ["theta", "alpha", "beta"]


def frame():
    """Reordered participants and columns, with IDs whose zeroes must survive."""
    source = data("test", responses=RESPONSES)
    table = pd.DataFrame(source.Y.values, columns=RESPONSES)
    table["age"] = source.X.values[:, 0]
    table["site"] = source.batch_effects.values[:, 0]
    table["participant_id"] = [f"{i:03}" for i in range(len(table))]
    return table.iloc[[7, 0, 23, 1, 18, 5]][
        ["beta", "site", "participant_id", "age", "theta", "alpha"]
    ].copy()


@pytest.fixture
def bundle(multi_fitted, multi_metadata, tmp_path):
    from meganorm.sharing import export_model

    return export_model(
        multi_fitted[0],
        metadata=multi_metadata,
        output_dir=tmp_path / "bundle",
        metadata_reviewed=True,
    )


def test_prediction_matches_native_without_changing_inputs(
    bundle, multi_fitted, tmp_path, monkeypatch
):
    from meganorm.API import SharedNormativeModel, FeatureDataset
    from meganorm.sharing import PredictionResults

    source = frame()
    original = source.copy(deep=True)
    raw = NormData.from_dataframe(
        "test",
        source,
        covariates=["age"],
        batch_effects=["site"],
        response_vars=RESPONSES,
        subject_ids="participant_id",
    )
    multi_fitted[0].predict(raw)
    shared = SharedNormativeModel.load(
        bundle.path, expected_sha256=bundle.bundle_sha256
    )
    archive = bundle.path.read_bytes()
    scalers = copy.deepcopy(
        [
            s.to_dict()
            for s in [
                *shared._backend.inscalers.values(),
                *shared._backend.outscalers.values(),
            ]
        ]
    )
    original_directory = shared._backend.save_dir

    def no_fit(*args, **kwargs):
        raise AssertionError("Prediction must not fit models or scalers.")

    monkeypatch.setattr(type(shared._backend), "fit", no_fit)
    for cls in {
        type(s)
        for s in [
            *shared._backend.inscalers.values(),
            *shared._backend.outscalers.values(),
        ]
    }:
        monkeypatch.setattr(cls, "fit", no_fit)
    monkeypatch.chdir(tmp_path)
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    result = shared.predict(source, units=UNITS)
    assert isinstance(result, PredictionResults)
    assert before == {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    assert shared._backend.save_dir == original_directory
    assert result.participant_index == tuple(source.participant_id)
    for actual, field in [(result.predictions, "Yhat"), (result.z_scores, "Z")]:
        assert actual.index.tolist() == source.participant_id.tolist()
        assert actual.columns.tolist() == ORDER
        assert np.isfinite(actual.values).all()
        np.testing.assert_allclose(
            actual.values,
            raw[field].sel(response_vars=ORDER).values,
            rtol=1e-8,
            atol=1e-8,
        )
    assert scalers == [
        s.to_dict()
        for s in [
            *shared._backend.inscalers.values(),
            *shared._backend.outscalers.values(),
        ]
    ]
    pd.testing.assert_frame_equal(source, original)
    assert bundle.path.read_bytes() == archive

    # Reuse the existing feature container and opt in to local CSV output.
    indexed = source.set_index("participant_id").rename_axis("subject")
    features = FeatureDataset(
        indexed, tuple(RESPONSES), pd.DataFrame(), pd.DataFrame(), {}, tmp_path, {}
    )
    output = tmp_path / "local-results"
    second = shared.predict(
        features, units=UNITS, participant_id="subject", output_dir=output
    )
    pd.testing.assert_frame_equal(second.predictions, result.predictions)
    assert {p.name for p in output.iterdir()} == {"predictions.csv", "z_scores.csv"}
    saved = pd.read_csv(
        output / "predictions.csv",
        dtype={"participant_id": str},
        index_col="participant_id",
    )
    pd.testing.assert_frame_equal(saved, result.predictions, check_exact=False)
    with pytest.raises(FileExistsError):
        shared.predict(source, units=UNITS, output_dir=output)
    assert bundle.path.read_bytes() == archive


@pytest.mark.parametrize(
    "problem",
    [
        "missing_response",
        "duplicate_response",
        "missing_covariate",
        "unknown_site",
        "missing_measurement",
        "nonfinite_covariate",
        "nonnumeric_response",
        "duplicate_id",
        "empty",
        "wrong_units",
        "missing_units",
    ],
)
def test_invalid_input_is_rejected_before_prediction(
    bundle, tmp_path, monkeypatch, problem
):
    from meganorm.API import SharedNormativeModel

    shared = SharedNormativeModel.load(bundle.path)
    source, units = frame(), dict(UNITS)
    if problem == "missing_response":
        source = source.drop(columns="alpha")
    elif problem == "duplicate_response":
        source = pd.concat([source, source[["alpha"]]], axis=1)
    elif problem == "missing_covariate":
        source = source.drop(columns="age")
    elif problem == "unknown_site":
        source.loc[source.index[0], "site"] = "PRIVATE_UNKNOWN_SITE"
    elif problem == "missing_measurement":
        source.loc[source.index[0], "theta"] = np.nan
    elif problem == "nonfinite_covariate":
        source.loc[source.index[0], "age"] = np.inf
    elif problem == "nonnumeric_response":
        source["beta"] = "PRIVATE_MEASUREMENT"
    elif problem == "duplicate_id":
        source["participant_id"] = "PRIVATE_ID"
    elif problem == "empty":
        source = source.iloc[:0]
    elif problem == "wrong_units":
        units["alpha"] = "uV"
    elif problem == "missing_units":
        units.pop("age")

    def unexpected_prediction(*args, **kwargs):
        pytest.fail("Invalid input reached native prediction.")

    monkeypatch.setattr(shared._backend, "predict", unexpected_prediction)
    output = tmp_path / "invalid-results"
    with pytest.raises(ValueError) as error:
        shared.predict(source, units=units, output_dir=output)
    assert "PRIVATE_" not in str(error.value)
    assert not output.exists()


def test_fresh_process_prediction(bundle, tmp_path):
    from meganorm.API import SharedNormativeModel

    source = frame()
    expected = SharedNormativeModel.load(bundle.path).predict(source, units=UNITS)
    source.to_csv(tmp_path / "input.csv", index=False)
    np.savez(
        tmp_path / "expected.npz",
        predictions=expected.predictions.values,
        z_scores=expected.z_scores.values,
    )
    code = """
import numpy as np
import pandas as pd
from meganorm.API import SharedNormativeModel
source = pd.read_csv('input.csv', dtype={'participant_id': str})
model = SharedNormativeModel.load('bundle/model.zip')
result = model.predict(source, units={'age': 'years', 'alpha': 'V', 'beta': 'V', 'theta': 'V'})
assert result.participant_index == tuple(source.participant_id)
expected = np.load('expected.npz')
for name in ('predictions', 'z_scores'):
    np.testing.assert_allclose(getattr(result, name).values, expected[name], rtol=1e-8, atol=1e-8)
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


def test_native_failure_is_sanitized_and_cleans_work_directory(
    bundle, multi_fitted, tmp_path, monkeypatch
):
    from meganorm.API import SharedNormativeModel

    model = SharedNormativeModel.load(bundle.path)
    source = frame().iloc[:1] if multi_fitted[1] == "hbr" else frame()
    original_directory = model._backend.save_dir
    work_paths = []

    def fail_native(data):
        work_paths.append(Path(model._backend.save_dir))
        raise IndexError("PRIVATE_INPUT_VALUES")

    monkeypatch.setattr(model._backend, "predict", fail_native)
    output = tmp_path / "failed-results"
    with pytest.raises(RuntimeError) as error:
        model.predict(source, units=UNITS, output_dir=output)
    assert "PRIVATE_INPUT_VALUES" not in str(error.value)
    if multi_fitted[1] == "hbr":
        assert "single observation" in str(error.value)
    assert model._backend.save_dir == original_directory
    assert not output.exists()
    assert all(not path.exists() for path in work_paths)
