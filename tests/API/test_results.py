import pandas as pd
import pytest


@pytest.mark.unit
def test_feature_dataframe_copy_does_not_change_result(tmp_path):
    from meganorm.API.results import FeatureDataset

    data = pd.DataFrame({"f__x": [1.0]}, index=pd.Index(["001"], name="participant_id"))
    result = FeatureDataset(
        data=data,
        feature_names=("f__x",),
        manifest=pd.DataFrame(),
        processing=pd.DataFrame(),
        summary={},
        output_dir=tmp_path,
        paths={},
    )
    copied = result.to_dataframe()
    copied.iloc[0, 0] = 3.0
    assert result.data.iloc[0, 0] == 1.0
