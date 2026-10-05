import pytest


@pytest.mark.unit
def test_public_imports_preserve_existing_aliases():
    import meganorm
    from meganorm.API import (
        Config,
        Dataset,
        Pipeline,
        NormativeModel,
        FeatureDataset,
        NormativeResults,
    )
    from meganorm.utils.IO import Config as LegacyConfig

    assert Config is LegacyConfig
    assert meganorm.Dataset is Dataset
    assert meganorm.Pipeline is Pipeline
    assert meganorm.NormativeModel is NormativeModel
    assert meganorm.featureExtraction is not None
