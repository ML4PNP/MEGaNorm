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

    def load_erp(self, participant_id):
        """Load saved averages using an exact string participant identifier."""
        from meganorm.utils.erp import validate_erp_subject_id
        from meganorm.utils.erp_outputs import load_erp_result

        validate_erp_subject_id(participant_id)
        if self.summary.get("analysis_type") != "erp":
            raise ValueError("This result is not an ERP run.")
        if participant_id not in set(self.manifest.participant_id):
            raise KeyError(participant_id)
        return load_erp_result(self.output_dir / "participants" / participant_id)

    def plot_erp(self, participant_id, *, channels=None, show=False):
        """Load and plot a participant's saved averages and measurements."""
        return self.load_erp(participant_id).plot(channels=channels, show=show)


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
