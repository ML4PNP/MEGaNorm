import numpy as np
import pandas as pd
import pytest
from meganorm.API.datasets import Dataset


def dataset(tmp_path, text):
    (tmp_path / "participants.tsv").write_text(text)
    return Dataset(
        name="siteA",
        root=tmp_path,
        task="rest",
        extension=".fif",
        demographics="participants.tsv",
        participant_id="ID",
    )


@pytest.mark.unit
def test_named_ids_and_metadata_are_joined_without_silent_losses(tmp_path):
    from meganorm.API._metadata import attach_metadata

    ds = dataset(
        tmp_path,
        "ID\tage\teyes\tnote__meta\n001\t20\topen\tx\n002\t30\tclosed\ty\n003\t40\topen\tz\n",
    )
    features = pd.DataFrame(
        {"feature__Alpha": [2.0, 3.0], "subject": ["001", "002"]}, index=["001", "002"]
    )
    manifest = pd.DataFrame(
        {"dataset": ["siteA", "siteA"], "participant_id": ["001", "002"]}
    )
    data, counts = attach_metadata(features, [ds], manifest)
    assert data.index.tolist() == ["001", "002"]
    assert data.index.name == "participant_id"
    assert data.eyes.tolist() == ["open", "closed"]
    assert data.site.tolist() == ["siteA", "siteA"]
    assert data.dataset.tolist() == ["siteA", "siteA"]
    assert "subject" not in data
    assert counts["unused_demographics"] == {"siteA": 1}
    assert counts["matched_demographics"] == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "text, match",
    [
        ("ID\tage\n001\t20\n001\t30\n", "duplicate"),
        ("ID\tage\n\t20\n", "missing"),
        ("ID\tage\n 001\t20\n", "whitespace"),
        ("wrong\tage\n001\t20\n", "ID"),
        ("ID\tage\n002\t20\n", "001"),
    ],
)
def test_invalid_demographics_are_rejected(tmp_path, text, match):
    from meganorm.API._metadata import attach_metadata

    ds = dataset(tmp_path, text)
    with pytest.raises(ValueError, match=match):
        attach_metadata(
            pd.DataFrame({"f__x": [1]}, index=["001"]),
            [ds],
            pd.DataFrame({"dataset": ["siteA"], "participant_id": ["001"]}),
        )


@pytest.mark.unit
def test_existing_missing_site_is_not_filled_and_overlaps_are_rejected(tmp_path):
    from meganorm.API._metadata import attach_metadata

    ds = dataset(tmp_path, "ID\tage\tsite\n001\t20\t\n")
    manifest = pd.DataFrame({"dataset": ["siteA"], "participant_id": ["001"]})
    data, _ = attach_metadata(
        pd.DataFrame({"f__x": [1]}, index=["001"]), [ds], manifest
    )
    assert pd.isna(data.site.iloc[0])
    with pytest.raises(ValueError, match="overlap"):
        attach_metadata(pd.DataFrame({"age": [1]}, index=["001"]), [ds], manifest)


@pytest.mark.unit
def test_numeric_string_id_collision_and_conflicting_index_fail():
    from meganorm.API._metadata import normalize_ids

    with pytest.raises(ValueError, match="duplicate"):
        normalize_ids(pd.DataFrame({"ID": [1, "1"]}), "ID")
    with pytest.raises(ValueError, match="disagree"):
        normalize_ids(
            pd.DataFrame({"ID": ["001"]}, index=pd.Index(["002"], name="ID")), "ID"
        )
