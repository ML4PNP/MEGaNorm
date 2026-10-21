"""Small result containers; mutating a table does not update saved files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd


@dataclass
class FeatureDataset:
    """Extracted features, participant metadata, processing outcomes and paths."""

    data: pd.DataFrame
    feature_names: tuple[str, ...]
    manifest: pd.DataFrame
    processing: pd.DataFrame
    summary: dict
    output_dir: Path
    paths: dict[str, Path]

    def to_dataframe(self):
        """Return an independent copy for analysis or filtering."""
        return self.data.copy(deep=True)


@dataclass
class NormativeResults:
    """Fitted PCNtoolkit model and label-aligned held-out results."""

    model: Any
    train: Any
    test: Any
    responses: tuple[str, ...]
    predictions: pd.DataFrame | None
    z_scores: pd.DataFrame | None
    metrics: pd.DataFrame | None
    participants: pd.DataFrame
    output_dir: Path
    paths: dict[str, Path]
    summary: dict
