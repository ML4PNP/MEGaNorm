"""Site-specific legacy growth charts must use the requested site's curves."""

import pickle
import numpy as np
import pytest
import meganorm.plots.plots as plots


@pytest.mark.parametrize(
    ("site", "expected"), [(0, [10, 20]), (1, [100, 200]), (None, [55, 110])]
)
def test_growthchart_selects_site_zero_or_averages_only_when_omitted(
    tmp_path, monkeypatch, site, expected
):
    batch = np.array(
        [[sex, group] for sex in (0, 1) for group in (0, 1) for _ in range(2)]
    )
    values = np.array([10, 10, 100, 100, 20, 20, 200, 200])[:, None, None]
    with (tmp_path / "Quantiles_estimate.pkl").open("wb") as stream:
        pickle.dump(
            {
                "quantiles": values,
                "synthetic_X": np.tile([20, 30], 4),
                "batch_effects": batch,
            },
            stream,
        )
    observed = []
    # Keep file loading, site selection, and reshape real; omit figure rendering.
    monkeypatch.setattr(
        plots, "plot_growthchart", lambda x, y, **kwargs: observed.append(y.copy())
    )
    plots.plot_growthcharts(
        str(tmp_path),
        [0],
        ["Alpha"],
        site=site,
        point_num=2,
        num_of_sites=2,
        centiles_name=["50th"],
        suffix="estimate",
    )
    assert observed[0][:, 0, :].tolist() == [expected, expected]
