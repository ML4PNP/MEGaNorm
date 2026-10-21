import numpy as np
import pytest
from pcntoolkit import BLR
from meganorm.API import NormativeModel

pytestmark = pytest.mark.integration


def test_complete_real_local_workflow(api_features, tmp_path):
    features = api_features
    results = NormativeModel(
        estimator=BLR(),
        covariates=["age"],
        batch_effects=["sex", "site"],
        output_dir=tmp_path / "results",
        name="MEGaNorm_quickstart",
        save_plots=False,
    ).fit(
        features,
        responses=[features.feature_names[0]],
        train_fraction=0.8,
        random_state=42,
    )
    assert len(features.data) == 24
    assert features.data.index.is_unique
    assert len(features.feature_names) == 24
    assert results.predictions.index.tolist() == list(results.test.subject_ids.values)
    assert np.isfinite(results.predictions.values).all()
    assert np.isfinite(results.z_scores.values).all()
    assert all(p.exists() for p in results.paths.values())
    assert set(results.train.subject_ids.values).isdisjoint(
        results.test.subject_ids.values
    )
