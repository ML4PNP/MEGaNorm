"""Sequential orchestration over the existing feature extraction engine."""

import json
import logging
import shutil
import warnings
from pathlib import Path
from collections.abc import Sequence
from typing import Literal
import pandas as pd
from meganorm.utils.IO import Config, select_session_path
from meganorm.utils.parallel import collect_results
from .datasets import Dataset, _path
from ._metadata import attach_metadata
from ._processing import process_participant
from .results import FeatureDataset

_LOG = logging.getLogger("meganorm.API.pipeline")
_METADATA = {"subject", "scanner", "if_eroom", "number_of_epochs"}
_AUXILIARY = {
    "empty-room": (
        "empty_room_record",
        ("empty_room_path", "empty_room_task", "empty_room_ending"),
    ),
    "event": (
        "event_record",
        (
            "event_file_path",
            "event_file_task",
            "event_file_ending",
            "event_of_interest",
        ),
    ),
    "annotation": (
        "annotation_path",
        ("annotation_path", "annotaion_task_name", "annotation_ending"),
    ),
    "position": ("pos_path", ("pos_path", "pos_file_ending")),
    "transform": ("trans_path", ("trans_path",)),
}


class Pipeline:
    """Extract features locally, preserving processing outcomes and metadata.

    Reruns warn and clear managed outputs after input validation. By default,
    participant failures are reported and processing continues; only valid
    current-run feature files are collected. on_error='raise' stops at the
    first processing failure. Interrupts always stop the run.
    """

    def __init__(
        self,
        *,
        config: Config,
        output_dir: str | Path,
        on_error: Literal["raise", "continue"] = "continue",
        progress: bool = True,
    ):
        if not isinstance(config, Config):
            raise TypeError("config must be an existing MEGaNorm Config.")
        if on_error not in {"raise", "continue"}:
            raise ValueError("on_error must be 'raise' or 'continue'.")
        self.config = config.model_copy(deep=True)
        self.output_dir = _path(output_dir)
        self.on_error, self.progress = on_error, progress

    def _preflight(self, datasets, manifest):
        root = self.output_dir
        requested = {}

        for dataset in datasets:
            if root.is_relative_to(dataset.root):
                raise ValueError("output_dir must be outside every input dataset root.")

            requested[dataset.name] = []

            for family, (field, keys) in _AUXILIARY.items():
                if any(dataset.options.get(key) is not None for key in keys):
                    missing = [
                        key
                        for key in keys
                        if dataset.options.get(key) is None
                        or dataset.options.get(key) == ""
                    ]

                    if missing:
                        raise ValueError(
                            f"Incomplete {family} options for dataset "
                            f"{dataset.name}: missing {missing}."
                        )

                    requested[dataset.name].append((family, field, keys))

        # Validate metadata before any costly processing or output creation.
        _, counts = attach_metadata(
            pd.DataFrame(index=manifest.participant_id),
            datasets,
            manifest,
        )

        selected = []

        for row in manifest.to_dict("records"):
            for family, field, keys in requested[row["dataset"]]:
                if not row.get(field):
                    raise FileNotFoundError(
                        f"Requested {family} files missing for dataset "
                        f"{row['dataset']} participant "
                        f"{row['participant_id']}; options: {keys}."
                    )

            if self.config.which_sensor == "eeg" and row.get("device") is not None:
                raise ValueError(
                    "EEG readers are inferred from extension; "
                    "omit the MEG device setting."
                )

            try:
                path = row["recording_paths"][self.config.which_meg_session]
            except IndexError as error:
                raise ValueError(
                    f'Invalid session index for {row["participant_id"]}: '
                    f"{self.config.which_meg_session}"
                ) from error

            if not path.exists():
                raise FileNotFoundError(f"Selected recording missing: {path}")

            selected.append(path)

            if (
                self.config.apply_source_localization
                and not self.config.apply_mri_template
            ):
                surface = row.get("mri_surface")

                if not surface or not (Path(surface) / row["participant_id"]).is_dir():
                    raise FileNotFoundError(
                        f"FreeSurfer derivatives missing for "
                        f'{row["participant_id"]}.'
                    )

            if self.config.apply_source_localization and self.config.apply_mri_template:
                template = self.config.freesurfer_template_path

                if not template or not Path(template).exists():
                    raise FileNotFoundError(
                        "Source localization requires a valid "
                        "freesurfer_template_path."
                    )

            for key, index in [
                ("empty_room_record", -1),
                ("event_record", self.config.which_meg_session),
                ("pos_path", -1),
                ("trans_path", self.config.which_meg_session),
                ("annotation_path", self.config.which_meg_session),
            ]:
                if row.get(key):
                    try:
                        auxiliary = select_session_path(
                            row[key],
                            index,
                        )
                    except IndexError as error:
                        raise ValueError(
                            f"Invalid auxiliary session for {key}."
                        ) from error

                    if not Path(auxiliary).exists():
                        raise FileNotFoundError(f"Missing {key}: {auxiliary}")

            if row.get("event_record"):
                try:
                    int(row["event_of_interest"])
                except (ValueError, TypeError) as error:
                    raise ValueError(
                        "event_of_interest must be an integer "
                        "when event files are used."
                    ) from error

            if row.get("layout_path") and not Path(row["layout_path"]).is_file():
                raise FileNotFoundError(f'Missing layout_path: {row["layout_path"]}')

        manifest["selected_recording"] = selected

        return counts

    def _reset_outputs(self, datasets, manifest):
        """Remove this pipeline's previous outputs without following symlinks."""
        existing = [
            self.output_dir / name
            for name in (
                "config.json",
                "Features",
                "run_summary.json",
                "manifest.csv",
                "processing.csv",
                "features_with_demographics.csv",
            )
            if (self.output_dir / name).exists()
            or (self.output_dir / name).is_symlink()
        ]
        protected = []
        for field in (
            "freesurfer_template_path",
            "freesurfer_home",
            "freesurfer_license",
            "parcellation_annot_fname",
        ):
            if value := getattr(self.config, field):
                protected.append(Path(value).expanduser())
        for dataset in datasets:
            protected.append(dataset.root)
            if dataset.demographics is not None:
                protected.append(dataset.demographics)
            protected.extend(
                value for value in dataset.options.values() if isinstance(value, Path)
            )
        for row in manifest.to_dict("records"):
            protected.extend(row["recording_paths"])
            for field in {field for field, _ in _AUXILIARY.values()} | {
                "layout_path",
                "mri_surface",
            }:
                value = row.get(field)
                if isinstance(value, str) and value and value != "None":
                    protected.extend(Path(part) for part in value.split("*") if part)
        protected = [Path(path).resolve() for path in protected]
        # Check every deletion before changing anything. A managed directory
        # may be an ancestor of an input even when output_dir is outside root.
        for path in existing:
            if path.is_symlink():
                continue
            resolved = path.resolve()
            if any(source.is_relative_to(resolved) for source in protected):
                raise ValueError(
                    f"Cannot remove managed output {path}: it contains input data."
                )
        if existing:
            message = (
                "Existing managed output(s) will be removed before rerun: "
                + ", ".join(str(path) for path in existing)
            )
            warnings.warn(message, UserWarning, stacklevel=2)
            _LOG.warning(message)
        for path in existing:
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                shutil.rmtree(path)

    def run(self, dataset: Dataset | Sequence[Dataset]) -> FeatureDataset:
        """Process one or several datasets, returning a FeatureDataset."""
        datasets = [dataset] if isinstance(dataset, Dataset) else list(dataset)
        if not datasets or not all(isinstance(d, Dataset) for d in datasets):
            raise ValueError(
                "Provide one Dataset or a nonempty sequence of Dataset objects."
            )
        if len({d.name for d in datasets}) != len(datasets):
            raise ValueError("Dataset names must be unique.")
        manifest = pd.concat([d.discover() for d in datasets], ignore_index=True)
        if manifest.participant_id.duplicated().any():
            raise ValueError(
                "Duplicate participant IDs across datasets; Phase 1 requires globally unique IDs."
            )
        counts = self._preflight(datasets, manifest)
        self._reset_outputs(datasets, manifest)
        root = self.output_dir
        temp = root / "Features" / "temp"
        temp.mkdir(parents=True, exist_ok=True)
        if self.config.save_preprocessed_data:
            (root / "Features" / "Saved_outputs" / "Preprocessed_data").mkdir(
                parents=True, exist_ok=True
            )
        config_path = root / "config.json"
        self.config.save(str(config_path), overwrite=True)
        rows = []
        successful = {}
        feature_names = None
        processing_columns = [
            "dataset",
            "status",
            "error_type",
            "error_message",
            "feature_path",
            "log_path",
        ]
        paths = {
            "config": config_path,
            "manifest": root / "manifest.csv",
            "processing": root / "processing.csv",
            "summary": root / "run_summary.json",
        }
        serial_manifest = manifest.copy()
        for column in serial_manifest:
            serial_manifest[column] = serial_manifest[column].map(
                lambda value: (
                    json.dumps([str(p) for p in value])
                    if isinstance(value, tuple)
                    else str(value) if isinstance(value, Path) else value
                )
            )
        serial_manifest.to_csv(paths["manifest"], index=False)

        def report():
            table = pd.DataFrame(
                rows, columns=["participant_id", *processing_columns]
            ).set_index("participant_id")
            summary = dict(
                counts,
                discovered=len(manifest),
                attempted=len(rows),
                succeeded=len(successful),
                failed=sum(r["status"] == "failed" for r in rows),
                interrupted=sum(r["status"] == "interrupted" for r in rows),
            )
            demo_names = {d.name for d in datasets if d.demographics is not None}
            summary["matched_demographics"] = sum(
                r["status"] == "success" and r["dataset"] in demo_names for r in rows
            )
            table.to_csv(paths["processing"])
            paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return table, summary

        if self.progress:
            _LOG.info("Found %d participants", len(manifest))
        try:
            for info in manifest.to_dict("records"):
                participant = info["participant_id"]
                path = temp / f"{participant}.csv"
                log = (
                    root
                    / "Features"
                    / "Saved_outputs"
                    / "log_summary"
                    / f"subject_{participant}_report.log"
                )
                row = dict(
                    participant_id=participant,
                    dataset=info["dataset"],
                    status="interrupted",
                    error_type=None,
                    error_message=None,
                    feature_path=str(path),
                    log_path=str(log),
                )
                rows.append(row)
                try:
                    if self.progress:
                        _LOG.info(
                            "Processing %s (%d/%d)",
                            participant,
                            len(rows),
                            len(manifest),
                        )
                    # A successful participant must produce its own new CSV.
                    path.unlink(missing_ok=True)
                    process_participant(
                        info,
                        participant_id=participant,
                        config_path=config_path,
                        temp_dir=temp,
                    )
                    header = pd.read_csv(path, nrows=0)
                    id_column = header.columns[0]
                    frame = pd.read_csv(path, dtype={id_column: str}).set_index(
                        id_column
                    )
                    if (
                        len(frame) != 1
                        or str(frame.index[0]) != participant
                        or not frame.columns.is_unique
                    ):
                        raise ValueError(
                            f"Invalid feature table for {participant}; expected one correctly indexed row."
                        )
                    names = tuple(c for c in frame if c not in _METADATA and "__" in c)
                    if not names:
                        raise ValueError(f"No extracted features for {participant}.")
                    if feature_names is not None and names != feature_names:
                        raise ValueError(
                            f"Inconsistent extracted feature columns for {participant}."
                        )
                    feature_names = names
                    row["status"] = "success"
                    successful[participant] = info
                except Exception as error:
                    row.update(
                        status="failed",
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                    if self.on_error == "raise":
                        raise
                finally:
                    report()
            if not successful:
                raise RuntimeError(
                    "All participants failed; inspect processing.csv and per-participant logs."
                )
            collect_results(
                str(root / "Features"),
                successful,
                str(temp),
                file_name="all_features",
                append=False,
                clean=False,
            )
            features_path = root / "Features" / "all_features.csv"
            features = pd.read_csv(
                features_path, index_col=0, dtype={0: str, "subject": str}
            )
            if features.subject.tolist() != list(successful):
                raise RuntimeError(
                    "Collected features do not match current-run successful participants."
                )
            features.index = pd.Index(
                features.subject.to_numpy(), name="participant_id"
            )
            features.to_csv(features_path)
            data, counts = attach_metadata(features, datasets, manifest)
            data_path = root / "features_with_demographics.csv"
            data.to_csv(data_path)
            paths.update(features=features_path, data=data_path)
            processing, summary = report()
            if self.progress:
                _LOG.info(
                    "%d processed successfully; %d failed",
                    summary["succeeded"],
                    summary["failed"],
                )
            return FeatureDataset(
                data, feature_names, manifest, processing, summary, root, paths
            )
        finally:
            report()
