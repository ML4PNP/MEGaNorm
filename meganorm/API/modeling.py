"""Data validation and local normative-model orchestration."""

import copy
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Literal
import numpy as np
import pandas as pd
from pcntoolkit.regression_model.regression_model import RegressionModel
from meganorm.src.normative_modeling import nm_model_train
from .datasets import _path
from ._metadata import normalize_ids
from ._pcntoolkit import extract_test_outputs, prepare_data
from .results import FeatureDataset, NormativeResults


def _names(values, label, *, allow_empty=False):
    if isinstance(values, str):
        raise ValueError(f"{label} must be a sequence of column names, not a string.")
    values = tuple(values)
    if (
        (not values and not allow_empty)
        or len(set(values)) != len(values)
        or any(not isinstance(v, str) or not v for v in values)
    ):
        raise ValueError(f"{label} must contain unique nonempty column names.")
    return values


class NormativeModel:
    """Fit an explicitly configured PCNtoolkit regression template locally.

    Responses and covariates must be numeric. missing='drop' performs complete
    case filtering only over selected model columns. No imputation is applied.
    """

    def __init__(
        self,
        *,
        estimator: RegressionModel,
        covariates: Sequence[str],
        batch_effects: Sequence[str] = (),
        output_dir: str | Path,
        name: str = "meganorm",
        participant_id: str = "participant_id",
        input_scaler: str = "standardize",
        output_scaler: str = "standardize",
        evaluate: bool = True,
        save_models: bool = True,
        save_plots: bool = True,
        missing: Literal["raise", "drop"] = "raise",
    ):
        if not isinstance(estimator, RegressionModel) or estimator.is_fitted:
            raise ValueError(
                "estimator must be an unfitted PCNtoolkit regression template (HBR or BLR)."
            )
        self.estimator = copy.deepcopy(estimator)
        self.covariates = _names(covariates, "covariates")
        self.batch_effects = _names(batch_effects, "batch_effects", allow_empty=True)
        if set(self.covariates) & set(self.batch_effects):
            raise ValueError("Covariates and batch effects must not overlap.")
        if missing not in {"raise", "drop"}:
            raise ValueError("missing must be 'raise' or 'drop'.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be nonempty.")
        if not isinstance(participant_id, str) or not participant_id.strip():
            raise ValueError("participant_id must be nonempty.")
        self.output_dir = _path(output_dir)
        self.name, self.participant_id = name, participant_id
        self.input_scaler, self.output_scaler = input_scaler, output_scaler
        self.evaluate, self.save_models, self.save_plots = (
            evaluate,
            save_models,
            save_plots,
        )
        self.missing = missing
        self._fitted = False

    def _normalize(self, data, responses):
        if isinstance(data, FeatureDataset):
            data = data.data
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a FeatureDataset or DataFrame.")
        frame = normalize_ids(data, self.participant_id)
        required = [*self.covariates, *self.batch_effects, *responses]
        if missing := set(required) - set(frame.columns):
            raise ValueError(f"Missing model columns: {sorted(missing)}")
        for col in (*self.covariates, *responses):
            if not pd.api.types.is_numeric_dtype(frame[col]):
                raise ValueError(
                    f"Model column {col!r} must be numeric; explicitly encode categorical covariates."
                )
        selected = frame[required].copy()
        bad = selected.isna().any(axis=1)
        for col in (*self.covariates, *responses):
            bad |= ~np.isfinite(selected[col].to_numpy(dtype=float, na_value=np.nan))
        for col in self.batch_effects:
            bad |= selected[col].astype(str).str.strip().eq("")
        excluded = selected.index[bad].tolist()
        if excluded and self.missing == "raise":
            columns = selected.columns[selected.isna().any()].tolist()
            raise ValueError(
                f"Missing/nonfinite model values in {required}; affected IDs: {excluded}; missing columns: {columns}"
            )
        return selected.loc[~bad], excluded

    def fit(
        self, data, *, responses, test_data=None, train_fraction=0.8, random_state=42
    ):
        """Fit, optionally splitting data or using an explicit held-out cohort.

        Explicit test_data requires train_fraction=None. With neither a test
        cohort nor a split, fit all eligible participants and return no holdout
        predictions. A wrapper cannot silently refit after a successful fit.
        """
        if self._fitted:
            raise RuntimeError(
                "This wrapper is already fitted; create another analysis."
            )
        responses = _names(
            [responses] if isinstance(responses, str) else responses, "responses"
        )
        if set(responses) & set((*self.covariates, *self.batch_effects)):
            raise ValueError(
                "Responses, covariates, and batch effects must not overlap."
            )
        if train_fraction is not None and (
            isinstance(train_fraction, bool)
            or not isinstance(train_fraction, (float, int))
            or not 0 < train_fraction < 1
        ):
            raise ValueError(
                "train_fraction must be strictly between 0 and 1, or None."
            )
        if test_data is not None and train_fraction is not None:
            raise ValueError("Explicit test_data requires train_fraction=None.")
        frame, excluded = self._normalize(data, responses)
        test_frame, test_excluded = (
            self._normalize(test_data, responses)
            if test_data is not None
            else (None, [])
        )
        if test_frame is not None and (set(frame.index) | set(excluded)) & (
            set(test_frame.index) | set(test_excluded)
        ):
            raise ValueError("Training and test participant IDs overlap.")
        if not len(frame) or (test_frame is not None and not len(test_frame)):
            raise ValueError(
                "Training and test cohorts must contain eligible participants."
            )

        def prepare(table, fraction):
            return prepare_data(
                table.reset_index(),
                responses=responses,
                covariates=self.covariates,
                batch_effects=self.batch_effects,
                participant_id=self.participant_id,
                train_fraction=fraction,
                random_state=random_state,
            )

        prepared = prepare(frame, train_fraction)
        if train_fraction is not None:
            train, test = prepared
        else:
            train = prepared
            test = prepare(test_frame, None) if test_frame is not None else None
        train_ids = train.subject_ids.values.astype(str).tolist()
        test_ids = (
            test.subject_ids.values.astype(str).tolist() if test is not None else []
        )
        if not train_ids or (test is not None and not test_ids):
            raise ValueError("The split produced an empty cohort.")
        if set(train_ids) & set(test_ids):
            raise ValueError("Prepared training and test participant IDs overlap.")
        if set(train_ids + test_ids) != set(frame.index) | (
            set(test_frame.index) if test_frame is not None else set()
        ):
            raise RuntimeError(
                "PCNtoolkit preparation unexpectedly omitted participants."
            )
        if test is not None:
            for col in self.batch_effects:
                train_labels = set(
                    train.batch_effects.sel(batch_effect_dims=col).values.astype(str)
                )
                test_labels = set(
                    test.batch_effects.sel(batch_effect_dims=col).values.astype(str)
                )
                if unseen := test_labels - train_labels:
                    raise ValueError(
                        f"Unseen test batch labels for {col}: {sorted(unseen)}. Choose a supported split or use the advanced transfer workflow."
                    )
        root = self.output_dir
        managed = [
            "Normative_models",
            "predictions.csv",
            "z_scores.csv",
            "metrics.csv",
            "participants.csv",
            "analysis_summary.json",
        ]
        for name in managed:
            if (root / name).exists():
                raise FileExistsError(
                    f"Existing model output: {root/name}; use a fresh output_dir."
                )
        root.mkdir(parents=True, exist_ok=True)
        model = nm_model_train(
            train=train,
            test=test,
            project_dir=str(root),
            experiment_name=self.name,
            template_regression_model=copy.deepcopy(self.estimator),
            model_name=self.name,
            inscaler_method=self.input_scaler,
            outscaler_method=self.output_scaler,
            if_parallel=False,
            if_cross_validate=False,
            if_evaluate_models=self.evaluate,
            if_save_models=self.save_models,
            if_save_plots=self.save_plots,
            return_model=True,
            save_results=self.evaluate,
        )
        prediction = z_scores = metrics = None
        if test is not None:
            prediction, z_scores, metrics = extract_test_outputs(
                model, test, responses, evaluate=self.evaluate
            )
        participants = pd.DataFrame(
            [(i, "train", None) for i in train_ids]
            + [(i, "test", None) for i in test_ids]
            + [
                (i, "excluded", "missing/nonfinite selected model values")
                for i in excluded + test_excluded
            ],
            columns=["participant_id", "split", "exclusion_reason"],
        ).set_index("participant_id")
        summary = dict(
            input=len(frame)
            + len(excluded)
            + (len(test_frame) + len(test_excluded) if test_frame is not None else 0),
            eligible=len(train_ids) + len(test_ids),
            excluded=len(excluded) + len(test_excluded),
            train=len(train_ids),
            test=len(test_ids),
            covariates=list(self.covariates),
            batch_effects=list(self.batch_effects),
            responses=list(responses),
            train_fraction=train_fraction,
            random_state=random_state,
            missing=self.missing,
            input_scaler=self.input_scaler,
            output_scaler=self.output_scaler,
            evaluate=self.evaluate,
        )
        paths = {"model_outputs": root / "Normative_models"}
        for label, table in [
            ("predictions", prediction),
            ("z_scores", z_scores),
            ("metrics", metrics),
            ("participants", participants),
        ]:
            if table is not None:
                path = root / f"{label}.csv"
                table.to_csv(path)
                paths[label] = path
        paths["summary"] = root / "analysis_summary.json"
        paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for folder in ["model", "results", "plots"]:
            path = root / "Normative_models" / folder
            if path.exists():
                paths[folder] = path
        self._fitted = True
        return NormativeResults(
            model,
            train,
            test,
            responses,
            prediction,
            z_scores,
            metrics,
            participants,
            root,
            paths,
            summary,
        )
