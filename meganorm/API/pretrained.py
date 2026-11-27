"""Local prediction from validated bundles using PCNToolkit's fitted models."""

from collections.abc import Mapping
import copy
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from meganorm.sharing.bundle import DEFAULT_LIMITS, _load_bundle, export_model
from meganorm.sharing.schema import PredictionResults, parse_manifest, _key
from meganorm.sharing.versions import capture_runtime_versions
from ._metadata import normalize_ids
from ._pcntoolkit import extract_test_outputs, prepare_data
from .results import FeatureDataset


class SharedNormativeModel:
    """Use a validated shared model with local participant measurements.

    Construct with :meth:`load`. Prediction reuses PCNToolkit's native model,
    scalers and output extraction. Input tables and the bundle remain unchanged.
    Original fitted bundles support new-site calibration and extension. Their
    children support prediction and export; repeated adaptation is not audited.
    """

    def __init__(self, backend, manifest, bundle_sha256):
        """Retain the native backend and metadata from one validated load."""
        self._backend = backend
        self._manifest = manifest
        self._bundle_sha256 = bundle_sha256
        self._adaptation_ids = frozenset()
        self._bundle_path = None
        self._limits = DEFAULT_LIMITS

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
        model = cls(*_load_bundle(path, expected_sha256=expected_sha256, limits=limits))
        model._bundle_path = Path(path).resolve()
        model._limits = limits
        return model

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
        return tuple(
            c
            for c in ("predict", "calibrate", "extend")
            if c in self._manifest.capabilities
        )

    def _prepare(self, data, units, participant_id, *, known_sites=True):
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
            if (
                known_sites
                and not frame[spec.name].isin([c.label for c in spec.categories]).all()
            ):
                raise ValueError(
                    "Site labels must match the model's declared categories; "
                    "unknown sites require calibration before prediction."
                )
        return frame

    def calibrate(
        self,
        data,
        *,
        units,
        output_dir,
        model_id,
        version,
        metadata=None,
        metadata_reviewed=False,
        participant_id="participant_id",
        freedom=1.0,
    ):
        """Calibrate to new sites using native PCNToolkit transfer.

        Supply original-scale local measurements and explicit units, a new output
        directory, and a distinct child model_id/version. Review inherited public
        metadata (or replace it with the publication metadata dictionary supplied
        as ``metadata``) and affirm ``metadata_reviewed=True``. The directory
        receives only a validated model.zip. The returned child supports the new
        sites only. HBR ``freedom`` must be finite and positive; BLR accepts its
        default value only. All other native fitting settings come from the parent.

        Calibration participant IDs are retained in memory to reject overlapping
        evaluation rows. They are never exported, so this protection ends after
        reloading. Keep calibration and evaluation participants separate locally.
        """
        if (
            type(freedom) not in (int, float)
            or not np.isfinite(freedom)
            or freedom <= 0
        ):
            raise ValueError("freedom must be a finite positive number.")
        if self._manifest.adapter.profile == "blr-default-site" and freedom != 1:
            raise ValueError("BLR calibration does not support a freedom option.")
        return self._adapt(
            data,
            operation="calibrate",
            units=units,
            output_dir=output_dir,
            model_id=model_id,
            version=version,
            metadata=metadata,
            metadata_reviewed=metadata_reviewed,
            participant_id=participant_id,
            options=(
                {"freedom": float(freedom)}
                if self._manifest.adapter.profile == "hbr-shashb-linear-site"
                else {}
            ),
        )

    def extend(
        self,
        data,
        *,
        units,
        output_dir,
        model_id,
        version,
        metadata=None,
        metadata_reviewed=False,
        participant_id="participant_id",
        n_synth_samples=None,
    ):
        """Fit a child on new-site data plus native synthetic parent observations.

        Identity, units, metadata review and output rules match :meth:`calibrate`.
        ``n_synth_samples`` is a positive integer, or None to use the parent's
        aggregate training count. The child supports old and new sites, but is
        refitted: old-site predictions need not equal the parent's predictions.
        Native synthesis and fitting are stochastic. Calibration rows, synthetic
        rows and native reports are discarded after validated model export.
        """
        if n_synth_samples is not None and (
            type(n_synth_samples) is not int or not 0 < n_synth_samples <= 10**9
        ):
            raise ValueError("n_synth_samples must be a positive integer or None.")
        return self._adapt(
            data,
            operation="extend",
            units=units,
            output_dir=output_dir,
            model_id=model_id,
            version=version,
            metadata=metadata,
            metadata_reviewed=metadata_reviewed,
            participant_id=participant_id,
            options={"n_synth_samples": n_synth_samples},
        )

    def export(self, *, output_dir, metadata_reviewed=False):
        """Re-export this model with unchanged identity and lineage to a new directory."""
        document = self._manifest.model_dump(mode="json", exclude={"files"})
        return export_model(
            self._backend,
            metadata=document,
            output_dir=output_dir,
            metadata_reviewed=metadata_reviewed,
            limits=self._limits,
        )

    def _adapt(
        self,
        data,
        *,
        operation,
        units,
        output_dir,
        model_id,
        version,
        metadata,
        metadata_reviewed,
        participant_id,
        options,
    ):
        """Validate the request, isolate native side effects, and retain only a bundle."""
        from .modeling import _validate_responses

        if operation not in self.capabilities:
            raise ValueError("Repeated adaptation of child bundles is not supported.")
        if metadata_reviewed is not True:
            raise ValueError(
                "Review child public labels and metadata before adaptation."
            )
        try:
            _validate_responses(
                [response.name for response in self._manifest.responses]
            )
        except ValueError:
            raise ValueError(
                "Adaptation requires response names that are safe native model directory names."
            ) from None
        output = Path(output_dir).expanduser()
        if output.is_symlink():
            raise ValueError("Adaptation output must not be a symbolic link.")
        output = output.resolve()
        if output.exists() or output.is_symlink():
            raise FileExistsError("Adaptation output directory already exists.")
        if self._bundle_path is not None and output.is_relative_to(
            self._bundle_path.parent
        ):
            raise ValueError(
                "Choose a child output directory outside the parent bundle directory."
            )
        frame = self._prepare(data, units, participant_id, known_sites=False)
        batch = self._manifest.batch_effects[0]
        labels = frame[batch.name].unique().tolist()
        parent_keys = {_key(c.label) for c in batch.categories}
        if any(
            type(label) is not str or _key(label) in parent_keys for label in labels
        ):
            raise ValueError(
                "Adaptation requires distinct new site labels without parent-category collisions."
            )
        participant_keys = {_key(identifier) for identifier in frame.index}
        if any(_key(label) in participant_keys for label in labels):
            raise ValueError(
                "Public site labels must not match calibration participant IDs."
            )
        if (frame.groupby(batch.name).size() < 2).any():
            raise ValueError(
                "Supply at least two calibration observations per new site."
            )
        document = self._manifest.model_dump(mode="json")
        document.update(model_id=model_id, version=version, capabilities=["predict"])
        runtime = capture_runtime_versions()
        document["software"] = dict(
            meganorm_version=runtime.meganorm_version,
            python_version=runtime.python_version,
        )
        document["adapter"]["pcntoolkit_version"] = runtime.pcntoolkit_version
        document["lineage"] = dict(
            operation=operation,
            parent=dict(
                model_id=self._manifest.model_id,
                version=self._manifest.version,
                bundle_sha256=self._bundle_sha256,
            ),
            settings=dict(
                parent_categories=[c.model_dump(mode="json") for c in batch.categories],
                **options,
            ),
        )
        if metadata is not None:
            document["metadata"] = copy.deepcopy(metadata)
        expected = labels + (
            [c.label for c in batch.categories] if operation == "extend" else []
        )
        document["batch_effects"][0]["categories"] = [
            dict(label=label, code=i) for i, label in enumerate(expected)
        ]
        # Validate identity, category normalization and metadata before fitting.
        parse_manifest(document)
        try:
            with TemporaryDirectory(prefix="meganorm-adapt-") as work:
                parent = copy.deepcopy(self._backend)
                parent.save_dir = str(Path(work) / "parent")
                prepared = prepare_data(
                    frame.reset_index(),
                    responses=[r.name for r in self._manifest.responses],
                    covariates=[c.name for c in self._manifest.covariates],
                    batch_effects=[batch.name],
                    participant_id=participant_id,
                    train_fraction=None,
                    random_state=0,
                )
                method = parent.transfer if operation == "calibrate" else parent.extend
                native = method(
                    prepared, save_dir=str(Path(work) / "native"), **options
                )
                actual = native.unique_batch_effects[batch.name]
                if set(actual) != set(expected):
                    raise ValueError(
                        "Native adaptation did not retain the required site coverage."
                    )
                document["batch_effects"][0]["categories"] = [
                    dict(label=label, code=i) for i, label in enumerate(actual)
                ]
                document.pop("files")
                bundle = export_model(
                    native,
                    metadata=document,
                    output_dir=Path(work) / "bundle",
                    metadata_reviewed=True,
                    limits=self._limits,
                )
                child = type(self).load(
                    bundle.path,
                    expected_sha256=bundle.bundle_sha256,
                    limits=self._limits,
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.mkdir(mode=0o700)
                try:
                    with (
                        bundle.path.open("rb") as src,
                        (output / "model.zip").open("xb") as dst,
                    ):
                        shutil.copyfileobj(src, dst)
                except BaseException:
                    (output / "model.zip").unlink(missing_ok=True)
                    output.rmdir()
                    raise
                child._bundle_path = output / "model.zip"
                child._adaptation_ids = frozenset(frame.index)
                return child
        except Exception:
            raise RuntimeError(
                "PCNToolkit adaptation or validated child export failed; the parent is unchanged."
            ) from None

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
        if self._adaptation_ids.intersection(frame.index):
            raise ValueError(
                "Evaluation participant IDs overlap the local calibration data."
            )
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
