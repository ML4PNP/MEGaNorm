import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from meganorm.plots.plots import (
    box_plot_auc,
    convert_region_name,
    define_lut,
    parse_aparc2009_name,
    plot_age_hist,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("region_name", "expected"),
    [
        ("ctx_lh_G_front_middle", ("G_front_middle", ["left"])),
        ("ctx_rh_G_front_middle", ("G_front_middle", ["right"])),
        ("ctx_G_front_middle", ("G_front_middle", ["left", "right"])),
        ("Left-Hippocampus", ("Left-Hippocampus", ["left", "right"])),
    ],
)
def test_parse_aparc2009_name_extracts_region_and_hemisphere(region_name, expected):
    assert parse_aparc2009_name(region_name) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("region_name", "expected"),
    [
        ("ctx_lh_G_front_middle", "ctx-lh-G_front_middle"),
        ("ctx_rh_G_front_middle", "ctx-rh-G_front_middle"),
        ("Left-Hippocampus", "Left-Hippocampus"),
    ],
)
def test_convert_region_name_converts_only_cortical_prefixes(region_name, expected):
    assert convert_region_name(region_name) == expected


@pytest.mark.unit
def test_define_lut_reads_entries_and_skips_comments_and_blank_lines(tmp_path):
    lut_path = tmp_path / "FreeSurferColorLUT.txt"
    lut_path.write_text(
        "# index name R G B A\n"
        "\n"
        "17 Left-Hippocampus 220 216 20 0\n"
        "53 Right-Hippocampus 220 216 20 0\n"
    )

    assert define_lut(lut_path) == {
        "Left-Hippocampus": 17,
        "Right-Hippocampus": 53,
    }


@pytest.mark.unit
def test_plot_age_hist_saves_png_and_svg(tmp_path):
    participants = pd.DataFrame(
        {
            "site": [0, 0, 1, 1],
            "age": [10, 20, 30, 40],
        }
    )

    plot_age_hist(
        participants,
        site_names=["Site A", "Site B"],
        save_path=str(tmp_path),
        lower_age_range=5,
        upper_age_range=50,
        step_size=5,
        colors=["navy", "orange"],
    )

    assert (tmp_path / "age_hist.svg").is_file()
    assert (tmp_path / "age_hist.png").is_file()


@pytest.mark.unit
def test_plot_age_hist_rejects_too_few_colors(tmp_path):
    participants = pd.DataFrame({"site": [0, 1], "age": [20, 30]})

    with pytest.raises(Exception, match="number of colors"):
        plot_age_hist(
            participants,
            site_names=["Site A", "Site B"],
            save_path=str(tmp_path),
            colors=["navy"],
        )


@pytest.mark.unit
def test_box_plot_auc_saves_png_and_svg(tmp_path):
    auc = pd.DataFrame(
        {
            "alpha": [0.70, 0.75, 0.80],
            "beta": [0.65, 0.72, 0.77],
        }
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="(?s).*Passing `palette` without assigning `hue`.*",
            category=FutureWarning,
        )
        box_plot_auc(auc, save_path=str(tmp_path), color=["teal", "gold"])

    assert (tmp_path / "AUCs.svg").is_file()
    assert (tmp_path / "AUCs.png").is_file()


@pytest.mark.unit
def test_box_plot_auc_rejects_mismatched_labels(tmp_path):
    auc = pd.DataFrame(
        {
            "alpha": [0.70, 0.75],
            "beta": [0.65, 0.72],
        }
    )

    with pytest.raises(ValueError, match="biomarkers_new_name"):
        box_plot_auc(
            auc,
            save_path=str(tmp_path),
            biomarkers_new_name=["Only one label"],
        )
