"""Result boundary verified against PCNtoolkit 1.3.0.

PCNtoolkit.predict mutates NormData with Yhat/Z/statistics and postprocesses
Yhat back to response units. Do not rescale it a second time.
"""

import pandas as pd


def extract_test_outputs(model, test, responses, *, evaluate):
    """Return label-aligned original-scale predictions, deviations, and metrics."""
    try:
        if test.attrs.get("is_scaled") is not False:
            raise ValueError("test data are still scaled")
        ids = pd.Index(test.subject_ids.values.astype(str), name="participant_id")
        if ids.has_duplicates:
            raise ValueError("duplicate test participant IDs")
        tables = []
        for field in ("Yhat", "Z"):
            array = (
                test[field]
                .sel(response_vars=list(responses))
                .transpose("observations", "response_vars")
            )
            tables.append(
                pd.DataFrame(array.values.copy(), index=ids, columns=list(responses))
            )
        metrics = None
        if evaluate:
            array = (
                test["statistics"]
                .sel(response_vars=list(responses))
                .transpose("response_vars", "statistic")
            )
            metrics = pd.DataFrame(
                array.values.copy(),
                index=pd.Index(responses, name="response"),
                columns=array.statistic.values,
            )
        return *tables, metrics
    except (AttributeError, KeyError, ValueError, TypeError) as error:
        raise RuntimeError(f"Unsupported PCNtoolkit results schema: {error}") from error


def prepare_data(
    frame,
    *,
    responses,
    covariates,
    batch_effects,
    participant_id,
    train_fraction,
    random_state,
):
    """Prepare without imputation/outlier removal on release and fork schemas.

    Some PCNtoolkit development versions accept the extended outlier settings
    used by prepare_nm_data. Released 1.3.0 does not. Both paths use NormData's
    own preparation and split, without altering the legacy advanced helper.
    """
    import inspect
    from pcntoolkit import NormData
    from meganorm.src.normative_modeling import prepare_nm_data

    parameters = inspect.signature(NormData.from_dataframe).parameters
    if "remove_outliers_approach" in parameters or any(
        p.kind == p.VAR_KEYWORD for p in parameters.values()
    ):
        return prepare_nm_data(
            df=frame,
            response_vars=list(responses),
            covariate_list=list(covariates),
            batch_effect_list=list(batch_effects),
            subject_id_col_name=participant_id,
            train_split_size=train_fraction,
            random_state=random_state,
            missing_value_handling_method=None,
            remove_outliers=False,
        )
    data = NormData.from_dataframe(
        name="reference_data",
        dataframe=frame,
        covariates=list(covariates),
        batch_effects=list(batch_effects),
        response_vars=list(responses),
        subject_ids=participant_id,
        remove_Nan=False,
        remove_outliers=False,
    )
    if train_fraction is not None:
        return data.train_test_split(
            splits=(train_fraction, 1 - train_fraction),
            split_names=["train", "test"],
            random_state=random_state,
        )
    return data
