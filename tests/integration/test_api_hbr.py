import numpy as np
import pytest
from pcntoolkit import HBR
from meganorm.API import NormativeModel

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_published_hbr_workflow(api_features, tmp_path):
    features = api_features
    results = NormativeModel(
        estimator=HBR(
            name="quickstart_hbr",
            cores=1,
            chains=2,
            draws=500,
            tune=500,
            progressbar=False,
        ),
        covariates=["age"],
        batch_effects=["sex", "site"],
        output_dir=tmp_path / "hbr",
        name="MEGaNorm_quickstart",
        save_plots=False,
    ).fit(
        features,
        responses=[features.feature_names[0]],
        train_fraction=0.8,
        random_state=42,
    )
    assert len(features.data) == 24
    assert np.isfinite(results.predictions.values).all()
    assert np.isfinite(results.z_scores.values).all()
    assert results.predictions.index.tolist() == list(results.test.subject_ids.values)
