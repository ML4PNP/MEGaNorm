"""Local prediction from validated bundles using PCNToolkit's fitted models."""

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from meganorm.sharing.bundle import DEFAULT_LIMITS, _load_bundle
from meganorm.sharing.schema import PredictionResults
from ._metadata import normalize_ids
from ._pcntoolkit import extract_test_outputs, prepare_data
from .results import FeatureDataset


class SharedNormativeModel:
    """Use a validated shared model with local participant measurements.

    Construct with :meth:`load`. Prediction reuses PCNToolkit's native model,
    scalers and output extraction. Input tables and the bundle remain unchanged.
    This interface currently provides prediction; new-site calibration and
    model extension are not exposed here yet.
    """

    def __init__(self, backend, manifest, bundle_sha256):
        """Retain the native backend and metadata from one validated load."""
        self._backend = backend
        self._manifest = manifest
        self._bundle_sha256 = bundle_sha256

    @classmethod
    def load(cls, path, *, expected_sha256=None, limits=DEFAULT_LIMITS):
        """Validate a local bundle and load all its fitted responses.

        Parameters
        ----------
        path : str or pathlib.Path
            Local ``model.zip`` produced by the sharing exporter.
        expected_sha256 : str or None, optional
            Independently retained digest of the expected archive.
        limits : BundleLimits, optional
            Archive, file and posterior-array resource limits.

        Returns
        -------
        SharedNormativeModel
            Loaded model ready for local prediction; no network access is used.
        """
        return cls(*_load_bundle(path, expected_sha256=expected_sha256, limits=limits))

    @property
    def manifest(self):
        """Return an independent copy of the validated publication metadata."""
        return self._manifest.model_copy(deep=True)

    @property
    def bundle_sha256(self):
        """Return the digest of the archive validated during loading."""
        return self._bundle_sha256

    @property
    def capabilities(self):
        """Return the operations currently exposed by this local interface."""
        return ("predict",)

    def _prepare(self, data, units, participant_id):
        """Validate complete measurements, units, IDs and known site labels."""
        if isinstance(data, FeatureDataset):
            data = data.data
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a DataFrame or FeatureDataset.")
        specs = [*self._manifest.covariates, *self._manifest.responses]
        numerical = [spec.name for spec in specs]
        batches = [spec.name for spec in self._manifest.batch_effects]
        required = [*numerical, *batches]
        if (
            not isinstance(participant_id, str)
            or not participant_id.strip()
            or participant_id in required
        ):
            raise ValueError(
                "Choose a participant ID name distinct from model features."
            )
        if not data.columns.is_unique:
            raise ValueError("Input contains duplicate column names.")
        try:
            frame = normalize_ids(data, participant_id)
        except ValueError:
            raise ValueError(
                "Supply unique, nonempty participant IDs in the declared column or index."
            ) from None
        if frame.empty:
            raise ValueError("Supply at least one participant.")
        if not set(required) <= set(frame.columns):
            raise ValueError(
                "Supply every declared covariate, site and response measurement; "
                "z-scores require measured responses."
            )
        if not isinstance(units, Mapping) or any(
            units.get(spec.name) != spec.unit for spec in specs
        ):
            raise ValueError(
                "Declare units for every covariate and response exactly as in the manifest; "
                "convert the input values before prediction if needed."
            )
        frame = frame[required].copy()
        for name in numerical:
            series = frame[name]
            if (
                not pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_bool_dtype(series)
                or pd.api.types.is_complex_dtype(series)
                or not np.isfinite(series.to_numpy(dtype=float, na_value=np.nan)).all()
            ):
                raise ValueError(
                    "Covariates and responses must contain finite real numbers."
                )
            frame[name] = series.astype(float)
        for spec in self._manifest.batch_effects:
            if not frame[spec.name].isin([c.label for c in spec.categories]).all():
                raise ValueError(
                    "Site labels must match the model's declared categories; "
                    "unknown sites require calibration before prediction."
                )
        return frame

    def predict(self, data, *, units, participant_id="participant_id", output_dir=None):
        """Compute predictions and z-scores for every bundled response.

        Parameters
        ----------
        data : pandas.DataFrame or FeatureDataset
            Original-scale covariates, measured responses and known site labels.
            Columns may be reordered; unrelated columns are ignored. Missing
            measurements, nonfinite values and duplicate columns/IDs are rejected.
        units : Mapping[str, str]
            Units of the supplied covariates and responses. Every declaration
            must equal its manifest unit; no automatic unit conversion is done.
        participant_id : str, default="participant_id"
            Identifier column or named index. IDs are normalized to strings by
            the existing API rules and retain their input row order.
        output_dir : str or pathlib.Path or None, optional
            New local directory for ``predictions.csv`` and ``z_scores.csv``.
            Existing directories are never overwritten. With None, no outputs
            persist on disk.

        Returns
        -------
        PredictionResults
            Original-scale predictions and dimensionless z-scores, with input
            participant order and manifest response order. No refitting occurs.

        Notes
        -----
        All response measurements are required because this method returns
        z-scores too. PCNToolkit runs on a new data object in a temporary work
        directory. The optional CSV files contain participant-level results;
        they are local analysis outputs and are never added to the bundle.
        """
        frame = self._prepare(data, units, participant_id)
        output = Path(output_dir).expanduser() if output_dir is not None else None
        if output is not None and (output.exists() or output.is_symlink()):
            raise FileExistsError("Prediction output directory already exists.")
        responses = [r.name for r in self._manifest.responses]
        original_directory = self._backend.save_dir
        try:
            # Native predict creates working directories even with output flags off.
            with TemporaryDirectory(prefix="meganorm-predict-") as work:
                self._backend.save_dir = work
                prepared = prepare_data(
                    frame.reset_index(),
                    responses=responses,
                    covariates=[c.name for c in self._manifest.covariates],
                    batch_effects=[b.name for b in self._manifest.batch_effects],
                    participant_id=participant_id,
                    train_fraction=None,
                    random_state=0,
                )
                self._backend.predict(prepared)
                predictions, z_scores, _ = extract_test_outputs(
                    self._backend,
                    prepared,
                    responses,
                    evaluate=False,
                )
        except Exception as error:
            message = "PCNToolkit could not compute predictions for the supplied data."
            if (
                len(frame) == 1
                and self._manifest.adapter.profile == "hbr-shashb-linear-site"
                and isinstance(error, (IndexError, ValueError))
            ):
                message += (
                    " For a single observation, this HBR build may need the "
                    "PCNToolkit observation-axis fix."
                )
            raise RuntimeError(message) from None
        finally:
            self._backend.save_dir = original_directory
        for table in (predictions, z_scores):
            if (
                tuple(table.index) != tuple(frame.index)
                or list(table.columns) != responses
                or not np.isfinite(table.to_numpy()).all()
            ):
                raise RuntimeError(
                    "PCNToolkit returned nonfinite or misaligned outputs."
                )
        result = PredictionResults(predictions, z_scores, tuple(frame.index))
        if output is not None:
            output.mkdir(parents=True, mode=0o700)
            predictions.to_csv(output / "predictions.csv")
            z_scores.to_csv(output / "z_scores.csv")
        return result
