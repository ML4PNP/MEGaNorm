import pickle
from itertools import product

import numpy as np
import pandas as pd
import pytest

from meganorm.utils import nm

pytestmark = pytest.mark.unit


def _split_data():
    return pd.DataFrame(
        {
            "age": range(40),
            "site": [0] * 40,
            "sex": [0, 1] * 20,
            "diagnosis": ["HC"] * 40,
            "feature": np.arange(40) / 10,
        }
    )


def test_haddbr_data_split_default_seed_is_usable(tmp_path):
    assert nm.haddbr_data_split(
        _split_data(), str(tmp_path), batch_effects=["site", "sex"]
    ) == ["feature"]
    assert (tmp_path / "x_train.pkl").is_file()


def test_haddbr_data_split_stratifies_validation_subset_and_preserves_input(tmp_path):
    data = _split_data()
    original = data.copy(deep=True)

    biomarkers = nm.haddbr_data_split(
        data,
        str(tmp_path),
        batch_effects=["site", "sex"],
        random_seed=23,
        validation_split=0.2,
    )

    assert biomarkers == ["feature"]
    partitions = [
        pd.read_pickle(tmp_path / f"x_{name}.pkl") for name in ["train", "val", "test"]
    ]
    assert [len(partition) for partition in partitions] == [16, 4, 20]
    assert len(set().union(*(set(partition.index) for partition in partitions))) == 40
    pd.testing.assert_frame_equal(data, original)


class _ZeroQuantileModel:
    def get_mcmc_quantiles(self, x, batches, z_scores):
        assert batches.ndim == 2
        return np.zeros((len(z_scores), len(x)))


@pytest.mark.parametrize("sites", [[0, 0, 1, 1], [0, 0, 0, 0]])
def test_evaluate_mace_accepts_one_batch_dimension_and_one_level(tmp_path, sites):
    pd.DataFrame({"age": [20, 30, 40, 50]}).to_pickle(tmp_path / "x.pkl")
    pd.DataFrame({"feature": [0, 1, 0, 1]}).to_pickle(tmp_path / "y.pkl")
    pd.DataFrame({"site": sites}).to_pickle(tmp_path / "b.pkl")
    with (tmp_path / "NM_0_0_ms.pkl").open("wb") as file:
        pickle.dump(_ZeroQuantileModel(), file)
    with (tmp_path / "meta_data.md").open("wb") as file:
        pickle.dump({"scaler_cov": [], "scaler_resp": []}, file)

    result = nm.evaluate_mace(
        str(tmp_path),
        str(tmp_path / "x.pkl"),
        str(tmp_path / "y.pkl"),
        str(tmp_path / "b.pkl"),
    )

    assert result == pytest.approx(
        np.abs(np.array([0.05, 0.25, 0.5, 0.75, 0.95]) - 0.5).mean()
    )


@pytest.mark.parametrize(("site_id", "expected"), [(0, 10.0), (1, 30.0), (None, 20.0)])
def test_cal_stats_for_inocs_respects_site_zero_and_num_points(
    tmp_path, site_id, expected
):
    points = 4
    batches = np.array(list(product(range(2), range(2)))).repeat(points, axis=0)
    quantiles = np.array([10 if site == 0 else 30 for _, site in batches])[
        :, None, None
    ].repeat(5, axis=1)
    path = tmp_path / "quantiles.pkl"
    with path.open("wb") as file:
        pickle.dump(
            {
                "quantiles": quantiles,
                "batch_effects": batches,
                "synthetic_X": np.tile(np.linspace(0, 100, points), (4, 1)).reshape(
                    -1, 1
                ),
            },
            file,
        )

    result = nm.cal_stats_for_INOCs(
        str(path),
        ["feature"],
        site_id,
        sex_id=0,
        age=25,
        num_of_datasets=2,
        num_points=points,
    )

    assert result == {"feature": [expected] * 5}
