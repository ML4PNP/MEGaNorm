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
        """Load a participant's saved ERP without repeating processing.

        Parameters
        ----------
        participant_id : str
            Exact identifier in this result's participant manifest.

        Returns
        -------
        ERPResult
            Saved features, QC, averages and optional bootstrap summaries.
            Individual epochs are not stored and ``epochs`` is None.

        Raises
        ------
        ValueError
            If this is not an ERP run, the identifier is invalid, or saved
            metadata are inconsistent.
        KeyError
            If the identifier is absent from the manifest.
        FileNotFoundError
            If required files are absent, including when ``save_evokeds=False``
            or participant processing failed.
        """
        from meganorm.utils.erp import validate_erp_subject_id
        from meganorm.utils.erp_outputs import load_erp_result

        validate_erp_subject_id(participant_id)
        if self.summary.get("analysis_type") != "erp":
            raise ValueError("This result is not an ERP run.")
        if participant_id not in set(self.manifest.participant_id):
            raise KeyError(participant_id)
        return load_erp_result(self.output_dir / "participants" / participant_id)

    def plot_erp(self, participant_id, *, channels=None, show=False):
        """Load and plot a participant's saved ERP and measurements.

        Parameters
        ----------
        participant_id : str
            Exact participant identifier in the manifest.
        channels : sequence of str or None, optional
            Unique sensor names in display order. By default use component
            sensors, falling back to sensors in the available averages.
        show : bool, default=False
            Display the figure through Matplotlib.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure owned by the caller; close it when finished.
        axes : numpy.ndarray
            One-dimensional array with one axis per requested sensor.

        Notes
        -----
        Uses saved original IDPs for annotations and saved bootstrap ranges
        for uncertainty shading. No measurements are recalculated. Requires
        saved evoked files; see :meth:`load_erp` for loading errors.
        """
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
