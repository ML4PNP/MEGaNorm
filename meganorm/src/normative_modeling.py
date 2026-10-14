from pcntoolkit import NormativeModel, HBR, BLR, Runner
from pcntoolkit import NormData
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az
import pingouin as pg
import xarray as xr
import pymc as pm
import json
import sys
import os
import re
import inspect
from sklearn.model_selection import train_test_split


def impute_by_subgroup(
    df,
    group_cols,
    subject_removal_nan_thr=0.2,
    continous_cov_col="age",
    imputation_con_var_window=5,
    strategy="mean",
    customized_age_window=None,
    *,
    reference_df=None,
):
    """
    Impute missing values in numeric columns using subgroup- and
    age-window-based neighbor statistics.

    Columns with a missing-value fraction at or above
    `subject_removal_nan_thr` are dropped entirely. For each remaining
    missing value, imputation
    is attempted using rows in the same `group_cols` subgroup that fall
    within an age window around the subject's value of
    `continous_cov_col`; if no such neighbors exist, the function falls
    back to the full subgroup, and finally to the global column
    statistic.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing numeric columns with missing values.
    group_cols : list of str
        Column names defining the subgroup (e.g., batch effects) used
        to find comparable neighbors.
    subject_removal_nan_thr : float, optional
        Missing-value fraction at or above which a column is dropped.
        Default is 0.2.
    continous_cov_col : str, optional
        Name of the continuous covariate column (e.g., "age") used to
        define the neighbor window. Only a single column is supported.
        Default is "age".
    imputation_con_var_window : float, optional
        Half-width of the window around a subject's covariate value
        used to select neighbors. Default is 5.
    strategy : {"mean", "median"}, optional
        Aggregation strategy applied to neighbor values. Default is
        "mean".
    customized_age_window : dict or None, optional
        Mapping from site name to an additional window width added to
        `imputation_con_var_window` for that site. Default is None.
    reference_df : pandas.DataFrame or None, optional
        Observed donor data used for column eligibility and imputation. Supply
        the un-imputed training subset when transforming held-out data. If
        None, the input dataframe supplies its own donors.

    Returns
    -------
    pandas.DataFrame
        Copy of `df`, with high-missingness columns dropped and
        remaining missing numeric values imputed.

    Raises
    ------
    ValueError
        If `continous_cov_col` is not a string or `strategy` is not
        ``"mean"`` or ``"median"``.
    """
    if not isinstance(continous_cov_col, str):
        err_msg = "continous_cov_col should be a string. Multiple covriates are not supported yet."
        raise ValueError(err_msg)

    if strategy not in {"mean", "median"}:
        raise ValueError("strategy should be either 'mean' or 'median'.")

    donors = df if reference_df is None else reference_df
    eligible = donors.columns[donors.isna().mean(axis=0) < subject_removal_nan_thr]
    df = df.loc[:, df.columns.isin(eligible)]
    donors = donors.loc[:, df.columns]

    df_imputed = df.copy()
    agg_fn = np.nanmean if strategy == "mean" else np.nanmedian

    # Numeric columns with NaNs, excluding the age col itself
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols_to_impute = [
        c for c in numeric_cols if c != continous_cov_col and df[c].isna().any()
    ]

    for idx, row in tqdm(df_imputed.iterrows(), total=len(df_imputed)):
        for col in cols_to_impute:
            if pd.isna(row[col]):

                # Build mask: same subgroup + within age window
                group_mask = pd.Series(True, index=donors.index)
                for g in group_cols:
                    group_mask &= donors[g] == row[g]

                window = imputation_con_var_window
                if customized_age_window and row["site"] in customized_age_window:
                    window += customized_age_window[row["site"]]

                age_mask = donors[continous_cov_col].between(
                    row[continous_cov_col] - window, row[continous_cov_col] + window
                )

                neighbors = donors.loc[group_mask & age_mask, col].dropna()

                if len(neighbors) > 0:
                    df_imputed.at[idx, col] = agg_fn(neighbors)
                else:
                    # Fallback 1: same group, any age
                    fallback_group = donors.loc[group_mask, col].dropna()
                    if len(fallback_group) > 0:
                        df_imputed.at[idx, col] = agg_fn(fallback_group)
                    else:
                        # Fallback 2: global statistic
                        df_imputed.at[idx, col] = agg_fn(donors[col].dropna())

    return df_imputed


def _filter_response_outliers(
    frame, reference, responses, approach, group_by, iqr_factor
):
    """Filter responses using group bounds estimated only from reference rows.

    IQR bounds are Q1/Q3 plus ``iqr_factor`` times their difference. Z-score
    bounds use three population standard deviations from the reference mean.
    Missing values and groups absent from the reference do not supply evidence
    of an outlier and are left for the caller's missing-value policy.
    """
    if approach not in {"iqr", "zscore"}:
        raise ValueError("remove_outliers_approach must be 'iqr' or 'zscore'.")
    if approach == "iqr" and (not np.isfinite(iqr_factor) or iqr_factor <= 0):
        raise ValueError("iqr_factor must be a positive finite number.")
    keep = pd.Series(True, index=frame.index)
    groups = [group_by] if isinstance(group_by, str) else list(group_by or [])
    grouped = reference.groupby(groups, dropna=False) if groups else [(None, reference)]
    for key, observed in grouped:
        mask = pd.Series(True, index=frame.index)
        if groups:
            keys = key if isinstance(key, tuple) else (key,)
            for column, value in zip(groups, keys):
                mask &= (
                    frame[column].isna() if pd.isna(value) else frame[column] == value
                )
        for response in responses:
            values = observed[response].dropna()
            if values.empty:
                continue
            if approach == "iqr":
                q1, q3 = values.quantile([0.25, 0.75])
                margin = iqr_factor * (q3 - q1)
                lower, upper = q1 - margin, q3 + margin
            else:
                mean, margin = values.mean(), 3 * values.std(ddof=0)
                lower, upper = mean - margin, mean + margin
            keep &= ~(
                mask & frame[response].notna() & ~frame[response].between(lower, upper)
            )
    return frame.loc[keep].copy()


def _normalize_split_size(train_split_size):
    """Return the training fraction (0-1), or None if no split is wanted.

    Values above 1 are treated as percentages (e.g. 70 -> 0.7).
    """
    if not train_split_size:
        return None

    if train_split_size > 1:
        train_split_size /= 100

    if not 0 < train_split_size < 1:
        raise ValueError(
            "train_split_size should be between 0 and 1, or "
            "between 0 and 100 when expressed as a percentage."
        )
    return train_split_size


def _filter_rois(df, response_vars, including_ROIs, excluding_ROIs, always_keep):
    """Drop excluded ROIs, then keep only included ROIs (plus `always_keep`).

    Returns the filtered dataframe and the matching list of response variables.
    """
    if excluding_ROIs:
        excluded_columns = df.filter(regex="|".join(excluding_ROIs)).columns
        df = df.drop(columns=excluded_columns)
        response_vars = [
            var for var in response_vars
            if not any(roi in var for roi in excluding_ROIs)
        ]

    if including_ROIs:
        # "insula-lh" and "insula-rh" both become "insula"
        rois = [re.sub(r"-(lh|rh)$", "", roi) for roi in including_ROIs]

        pattern = "|".join(map(re.escape, rois + always_keep))
        df = df.loc[:, df.columns.str.contains(pattern)]
        response_vars = [
            var for var in response_vars
            if any(roi in var for roi in rois)
        ]

    return df, response_vars


def _split_by_batch_effects(
    df, batch_effect_list, subject_id_col_name, train_size, random_state
):
    """Train/test split by subject, stratified by the combination of batch effects.

    All rows of a subject end up in the same partition. Each subject is
    stratified by the batch-effect values of its first row.
    """
    # One row per subject
    subjects = df.drop_duplicates(subset=subject_id_col_name)

    if batch_effect_list:
        strata = subjects[batch_effect_list].astype(str).agg("".join, axis=1)
    else:
        strata = None

    train_ids, test_ids = train_test_split(
        subjects[subject_id_col_name],
        test_size=1 - train_size,
        random_state=random_state,
        stratify=strata,
    )

    # All rows of each subject follow its ID
    train_df = df[df[subject_id_col_name].isin(train_ids)]
    test_df = df[df[subject_id_col_name].isin(test_ids)]
    return train_df, test_df


def prepare_nm_data(
    df,
    response_vars,
    covariate_list,
    batch_effect_list,
    subject_id_col_name,
    impute_by_column_name="age",
    impute_group_by=None,
    which_cohorts=None,
    subject_removal_nan_thr=0.2,
    including_ROIs=None,
    excluding_ROIs=None,
    name_data="reference_data",
    missing_value_handling_method=None,
    customized_con_var_imputation_window=None,
    remove_outliers=False,
    remove_outliers_group_by="site",
    iqr_factor=3,
    train_split_size=0.5,
    which_subjects=None,
    random_state=42,
):
    """
    Filter, clean, and package a dataframe into PCNtoolkit NormData.

    Steps
    -----
    1. Validate the inputs.
    2. Remove rows with missing covariates/batch effects; filter by cohort.
    3. Keep/drop ROI columns; filter by subject.
    4. Split into train/test by subject (optional).
    5. Remove response outliers with the IQR rule (optional).
    6. Impute remaining missing values (optional).
    7. Package as NormData.

    Outliers are removed BEFORE imputation, so extreme values cannot
    leak into imputed values. When splitting, all rows of a subject go
    to the same partition, and outlier bounds and imputation donors come
    from the training set only and are applied to both sets, so the test
    set does not influence preprocessing.

    Parameters
    ----------
    df : pandas.DataFrame
        Covariates, batch effects, responses and subject IDs.
    response_vars : list of str
        Candidate response-variable columns.
    covariate_list : list of str
        Model covariates (at least one).
    batch_effect_list : list of str
        Batch-effect columns, e.g. ["site", "sex"].
    subject_id_col_name : str
        Subject identifier column.
    impute_by_column_name : str, optional
        Continuous column defining the imputation window. Default "age".
    impute_group_by : list of str or None, optional
        Categorical columns to group by during imputation, e.g.
        ["sex", "eyes"]. Default None means batch_effect_list.
    which_cohorts : list of str or None, optional
        Keep only rows whose "diagnosis" is in this list.
    subject_removal_nan_thr : float, optional
        Columns with at least this fraction missing are dropped when
        imputing. Default 0.2.
    including_ROIs, excluding_ROIs : list of str or None, optional
        ROI name patterns to keep / drop.
    name_data : str, optional
        Name of the NormData object when no split is requested.
    missing_value_handling_method : {"mean", "median", None}, optional
        Imputation strategy. None means rows with NaNs are removed instead.
    customized_con_var_imputation_window : dict or None, optional
        Extra imputation window width per site.
    remove_outliers : bool, optional
        Remove subjects with any response outside the IQR bounds.
        Default False.
    remove_outliers_group_by : str, list of str, or None, optional
        Grouping for the IQR bounds. Default "site".
    iqr_factor : float, optional
        Bounds are Q1 - iqr_factor*IQR and Q3 + iqr_factor*IQR. Default 3.
    train_split_size : float or None, optional
        Fraction of subjects in the training set (or percentage if > 1).
        None/0 means no split.
    which_subjects : list or None, optional
        Keep only these subject IDs.
    random_state : int, optional
        Seed for the train/test split.

    Returns
    -------
    NormData or (NormData, NormData)
        The full dataset, or (train, test) if a split was requested.
    """
    imputing = missing_value_handling_method is not None

    if impute_group_by is None:
        impute_group_by = batch_effect_list

    # 1. Validate inputs 
    if not covariate_list:
        raise ValueError("covariate_list should contain at least one covariate.")

    if missing_value_handling_method not in {"mean", "median", None}:
        raise ValueError(
            "missing_value_handling_method should be 'mean', 'median', or None."
        )

    if imputing:
        if not isinstance(impute_by_column_name, str):
            raise ValueError("impute_by_column_name should be a string.")
        missing = [
            col for col in [impute_by_column_name] + impute_group_by
            if col not in df.columns
        ]
        if missing:
            raise ValueError(f"Imputation columns not found in the dataframe: {missing}")

    train_split_size = _normalize_split_size(train_split_size)

    # Columns that must always be present (and never be NaN)
    required_columns = covariate_list + batch_effect_list
    if imputing:
        required_columns = required_columns + [impute_by_column_name] + impute_group_by
    required_columns = list(dict.fromkeys(required_columns))  # remove duplicates

    # 2. Clean rows 
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_columns)

    if which_cohorts:
        df = df[df["diagnosis"].isin(which_cohorts)]

    #  3. Select columns and subjects 
    df, response_vars = _filter_rois(
        df,
        response_vars,
        including_ROIs,
        excluding_ROIs,
        always_keep=required_columns + [subject_id_col_name],
    )
    print("Number of response variables: ", len(response_vars))

    if which_subjects:
        df = df[df[subject_id_col_name].isin(which_subjects)]

    #  Local shortcuts for the processing steps 
    def drop_outliers(frame, reference):
        return _filter_response_outliers(
            frame,
            reference,
            response_vars,
            approach="iqr",
            group_by=remove_outliers_group_by,
            iqr_factor=iqr_factor,
        )

    def impute(frame, donors):
        return impute_by_subgroup(
            frame,
            group_cols=impute_group_by,
            subject_removal_nan_thr=subject_removal_nan_thr,
            continous_cov_col=impute_by_column_name,
            imputation_con_var_window=5,
            strategy=missing_value_handling_method,
            customized_age_window=customized_con_var_imputation_window,
            reference_df=donors,
        )

    def to_normdata(frame, name):
        return NormData.from_dataframe(
            name=name,
            dataframe=frame,
            covariates=covariate_list,
            batch_effects=batch_effect_list,
            response_vars=response_vars,
            subject_ids=subject_id_col_name,
            remove_Nan=not imputing,  # without imputation, NaN rows are dropped
            remove_outliers=False,    # already done above, before imputation
        )

    if train_split_size:
        train_df, test_df = _split_by_batch_effects(
            df, batch_effect_list, subject_id_col_name, train_split_size, random_state
        )
    else:
        # No split: the whole dataset plays the role of the training set.
        train_df, test_df = df, None

    # 5. Remove outliers (bounds from the training set only) 
    if remove_outliers:
        reference = train_df.copy()
        train_df = drop_outliers(train_df, reference)
        if test_df is not None:
            test_df = drop_outliers(test_df, reference)

    # 6. Impute (donors from the cleaned training set only) 
    if imputing:
        donors = train_df.copy()
        train_df = impute(train_df, donors)
        if test_df is not None:
            test_df = impute(test_df, donors)
        # Imputation may drop columns with too many NaNs
        response_vars = [v for v in response_vars if v in train_df.columns]

    # . Package 
    if test_df is None:
        return to_normdata(train_df, name_data)

    return to_normdata(train_df, "train"), to_normdata(test_df, "test")


def model_diagnostics(
    models_path,
    save_path,
    if_loo_cv=False,
):
    """
    Collect MCMC convergence diagnostics across fitted normative models.

    Loads the ArviZ inference data (`idata.nc`) for each model
    subdirectory under `models_path`, computes a parameter-level
    summary (r_hat, ESS, MCSE), and aggregates the results into a
    single dataframe, optionally saving it to disk as CSV.

    Parameters
    ----------
    models_path : str
        Path to the directory containing one subfolder per fitted
        model, each with an `idata.nc` file.
    save_path : str or None
        Directory in which to save the aggregated diagnostics CSV. If
        the directory does not exist, it is created. If falsy, results
        are not saved to disk.
    if_loo_cv : bool, optional
        Reserved for future leave-one-out cross-validation diagnostics;
        currently unused. Default is False.

    Returns
    -------
    pandas.DataFrame
        Diagnostics table with one row per (model, parameter) pair,
        containing r_hat, ess_bulk, ess_tail, and mcse_sd.
    """
    if save_path and not os.path.isdir(save_path):
        os.mkdir(save_path)

    records = []
    models = os.listdir(models_path)

    for model in tqdm(models):
        if "normative_model.json" in model:
            continue

        idata_path = os.path.join(models_path, model, "idata.nc")
        idata = az.from_netcdf(idata_path)
        summary = az.summary(idata)

        for param, row in summary.iterrows():
            records.append(
                {
                    "model": model,
                    "parameter": param,
                    "r_hat": row.get("r_hat"),
                    "ess_bulk": row.get("ess_bulk"),
                    "ess_tail": row.get("ess_tail"),
                    "mcse_sd": row.get("mcse_sd"),
                }
            )

    df_result = pd.DataFrame(records)
    if save_path:
        save_path = os.path.join(save_path, "models_diagnosis.csv")
        df_result.to_csv(save_path)

    return df_result


def nm_model_train(
    train,
    test,
    project_dir,
    experiment_name,
    template_regression_model,
    model_name,
    inscaler_method="standardize",
    outscaler_method="standardize",
    if_cross_validate=False,
    if_parallel=False,
    if_evaluate_models=True,
    if_model_diagnosis=True,
    if_save_models=True,
    if_save_plots=True,
    colors=None,
    job_configs=None,
    *,
    return_model=False,
    save_results=True,
):
    """
    Fit a PCNtoolkit normative model, optionally in parallel across
    response variables.

    Constructs a `NormativeModel` with the given regression template
    and I/O scalers, then either fits (and predicts, if `test` is
    provided) directly, or submits the job via a `Runner` for parallel
    execution across compute batches.

    Parameters
    ----------
    train : NormData
        Training data for the normative model.
    test : NormData or None
        Test data used for prediction after fitting. If None, only
        fitting is performed.
    project_dir : str
        Root project directory; a "Normative_models" subdirectory is
        created here to store outputs.
    experiment_name : str
        Name of the experiment (currently unused within the function
        body but provided for context/labeling).
    template_regression_model :
        Regression model template (e.g., HBR or BLR configuration) used
        to build the normative model.
    model_name : str
        Name assigned to the normative model instance.
    inscaler_method : str, optional
        Input covariate scaling method. Default is "standardize".
    outscaler_method : str, optional
        Output response variable scaling method. Default is
        "standardize".
    if_cross_validate : bool, optional
        Whether to run cross-validation when using the parallel
        `Runner`. Default is False.
    if_parallel : bool, optional
        Whether to submit fitting as parallel jobs via `Runner` instead
        of fitting in-process. Default is False.
    if_evaluate_models : bool, optional
        Whether the `NormativeModel` should evaluate fitted models.
        Default is True.
    if_model_diagnosis : bool, optional
        Reserved flag for running post-fit diagnostics (currently
        unused in the function body). Default is True.
    if_save_models : bool, optional
        Whether to save the fitted model to disk. Default is True.
    if_save_plots : bool, optional
        Whether to save diagnostic/result plots. Default is True.
    colors : optional
        Reserved for customizing plot colors; currently unused.
    job_configs : dict or None, optional
        Configuration for parallel job submission when `if_parallel` is
        True. Expected keys include "env_path", "job_type",
        "time_limit", "memory", "n_cores", "preamble", and
        "max_retries".

    return_model : bool, optional
        Return the fitted PCNtoolkit model after a local fit. Default False
        preserves the historical None return. Cannot be used with parallel jobs.

    save_results : bool, optional
        Save native PCNtoolkit result files. Default True preserves historical
        behavior. Disable when evaluation is off on PCNtoolkit versions whose
        native result writer requires evaluation statistics.

    Returns
    -------
    NormativeModel or None
        The fitted model when return_model=True; otherwise None.

    Raises
    ------
    ValueError
        If `if_parallel` is True and `job_configs` is not provided.
    """

    if return_model and if_parallel:
        raise ValueError("return_model=True requires local execution.")

    nm_dir = os.path.join(project_dir, "Normative_models")
    if not os.path.isdir(nm_dir):
        os.mkdir(nm_dir)

    model = NormativeModel(
        template_regression_model=template_regression_model,
        savemodel=if_save_models,
        evaluate_model=if_evaluate_models,
        saveresults=save_results,
        saveplots=if_save_plots,
        save_dir=nm_dir,
        inscaler=inscaler_method,
        outscaler=outscaler_method,
        name=model_name,
    )

    if not if_parallel:
        if test is not None:
            model.fit_predict(train, test)
        else:
            model.fit(train)

    else:
        if not job_configs:
            err = "jobs configuration should be passed."
            raise ValueError(err)

        runner = Runner(
            cross_validate=if_cross_validate,
            parallelize=True,
            n_batches=len(train.response_vars),
            environment=job_configs["env_path"],
            job_type=job_configs["job_type"],
            time_limit=job_configs["time_limit"],
            memory=job_configs["memory"],
            n_cores=job_configs["n_cores"],
            preamble=job_configs["preamble"],
            log_dir=os.path.join(nm_dir, "nm_parallel_logs"),
            temp_dir=os.path.join(nm_dir, "nm_temp"),
            max_retries=job_configs["max_retries"],
        )

        if test is not None:
            runner.fit_predict(model, train, test, observe=False)
        else:
            runner.fit(model, train, observe=False)

    if return_model:
        return model


def prior_predictive_check(
    idata_path,
    regression_model_path,
    model,
    X,
    Y,
    be,
    be_maps,
    n_samples=500,
    random_seed=None,
):
    """
    Perform a prior predictive check for a fitted normative model.

    Loads a fitted model and its inference data, rebuilds the PyMC model
    with the provided data, samples from the prior predictive distribution,
    and plots the resulting prior predictive check.

    Parameters
    ----------
    idata_path : str
        Path to the NetCDF file containing the fitted inference data
        (e.g., ``"path/to/idata.nc"``).
    regression_model_path : str
        Path to the JSON file containing the serialized regression model
        (e.g., ``"path/to/regression_model.json"``).
    model : {"hbr"}
        Name of the normative model to use. Currently only ``"hbr"``
        (Hierarchical Bayesian Regression) is supported.
    X : numpy.ndarray of shape (n_samples, n_covariates)
        Covariate matrix (e.g., age, brain volume).
    Y : numpy.ndarray of shape (n_samples,)
        Response variable (e.g., neuroimaging measure).
    be : numpy.ndarray of shape (n_samples, 2)
        Batch effects array with columns corresponding to ``["sex", "site"]``.
    be_maps : dict of {str : dict}
        Mapping of batch effect names to label-to-integer encodings. Expected
        keys are ``"sex"`` and ``"site"``. For example::

            {
                "sex":  {"Female": 0, "Male": 1},
                "site": {"BTNRH": 0, "CAMCAN": 1}
            }

    n_samples : int, optional
        Number of prior predictive samples to draw. Default is ``500``.
    random_seed : int or None, optional
        Random seed for reproducibility. If ``None``, no seed is set.
        Default is ``None``.

    Returns
    -------
    pymc_model : pymc.Model
        The compiled PyMC model used for prior predictive sampling.

    Raises
    ------
    ValueError
        If ``model`` is not ``"hbr"``.

    Notes
    -----
    The function prints the PyMC model string representation before sampling.
    The prior predictive plot is displayed inline via ``plt.show()``.
    """
    idata = az.from_netcdf(idata_path)
    with open(regression_model_path, "rb") as file:
        m = json.load(file)

    if model == "hbr":
        hbr = HBR()
        hbr.is_fitted = True
    else:
        err_msg = "prior_predictive_check currently only supports model='hbr'."
        raise ValueError(err_msg)

    hbr.from_dict(my_dict=m["model"])
    hbr.load_idata(path=idata_path)

    n = len(Y)
    X = xr.DataArray(
        X, dims=["observations", "covariates"], coords={"observations": np.arange(n)}
    )

    Y = xr.DataArray(Y, dims=["observations"], coords={"observations": np.arange(n)})
    be = xr.DataArray(
        be,
        dims=["observations", "batch_effect_dims"],
        coords={"batch_effect_dims": ["sex", "site"]},  # named coordinate
    )

    # Rebuild the PyMC model with dummy data
    pymc_model = hbr.likelihood.compile(X, be, be_maps, Y)

    print("The model:\n", pymc_model.str_repr())

    with pymc_model:
        prior_idata = pm.sample_prior_predictive(
            draws=n_samples,
            random_seed=random_seed,
        )

    az.plot_ppc(prior_idata, group="prior", observed=True)
    plt.show()

    return pymc_model


def compute_idp_centile(model, IDP, upper_limit=80, scale_centiles=True):
    """
    Compute the median centile curve for an IDP across the age range
    and characterize its peak, minimum, and slope sign changes.

    Builds a synthetic covariate grid spanning the model's fitted age
    range (using the most common batch effect category for each batch
    effect), computes the 0.5 centile curve for `IDP`, restricts the
    curve to ages up to `upper_limit`, optionally rescales the curve to
    a 0-100 percent range, and identifies the peak, minimum, and
    locations where the slope changes sign.

    Parameters
    ----------
    model : NormativeModel
        A fitted PCNtoolkit normative model with an "age" covariate.
    IDP : str
        Name of the response variable (imaging-derived phenotype) to
        evaluate.
    upper_limit : float, optional
        Maximum age (in the same units as the model's original age
        covariate) to include in the
        curve. Default is 80.
    scale_centiles : bool, optional
        If True, rescale the computed centile values to a 0-100 range
        based on their min and max. Default is True.

    Returns
    -------
    x_vals_real : ndarray
        Age values (inverse-transformed to original scale) for the
        curve, restricted to `upper_limit`.
    y_vals_pct : ndarray
        Centile values corresponding to `x_vals_real`, scaled to
        percent if `scale_centiles` is True.
    peak_x : float
        Age at the curve's maximum value.
    peak_y : float
        Curve value at the peak.
    min_x : float
        Age at the curve's minimum value.
    min_y : float
        Curve value at the minimum.
    slope_change_x : ndarray
        Age values where the slope of the curve changes sign.
    slope_change_y : ndarray
        Curve values at the slope sign-change points.

    Raises
    ------
    Exception
        If `scale_centiles` is True and the centile curve has zero
        range (constant values).
    """
    covariate = "age"
    cov_min = model.covariate_ranges[covariate]["min"]
    cov_max = model.covariate_ranges[covariate]["max"]
    centile_covariates = np.linspace(cov_min, cov_max, 150)
    centile_df = pd.DataFrame({covariate: centile_covariates})
    batch_effects = {
        k: max(v.items(), key=lambda x: x[1])[0]
        for k, v in model.batch_effect_counts.items()
    }
    for be, v in batch_effects.items():
        centile_df[be] = v
    centile_df[IDP] = 1e-6
    centile_data = NormData.from_dataframe(
        "centile",
        dataframe=centile_df,
        covariates=model.covariates,
        response_vars=[IDP],
        batch_effects=list(batch_effects.keys()),
    )
    model.compute_centiles(centile_data, centiles=[0.5], recompute=True)
    # compute_centiles postprocesses its NormData back to the original units.
    x_vals_real = centile_data.X.sel(covariates="age").values
    y_vals = centile_data.centiles.sel(centile=0.5, response_vars=IDP).values

    # Limit to upper_limit BEFORE scaling
    mask = x_vals_real <= upper_limit
    x_vals_real = x_vals_real[mask]
    y_vals = y_vals[mask]
    if len(x_vals_real) == 0:
        raise ValueError("upper_limit excludes every age in the fitted model range.")

    def scale_to_percent(values):
        lo = min(values)
        hi = max(values)
        span = hi - lo

        if span == 0:
            raise Exception

        return [(x - lo) / span * 100 for x in values]

    if scale_centiles:
        y_vals_pct = np.array(scale_to_percent(y_vals))
    else:
        y_vals_pct = y_vals.copy()

    # Peak
    peak_idx = np.argmax(y_vals_pct)
    peak_x = x_vals_real[peak_idx]
    peak_y = y_vals_pct[peak_idx]

    # Minimum
    min_idx = np.argmin(y_vals_pct)
    min_x = x_vals_real[min_idx]
    min_y = y_vals_pct[min_idx]

    # Slope sign changes — all computed on the already-masked arrays
    dy = np.diff(y_vals_pct)
    sign_changes = np.where(np.diff(np.sign(dy)))[0] + 1
    slope_change_x = x_vals_real[sign_changes]
    slope_change_y = y_vals_pct[sign_changes]

    return (
        x_vals_real,
        y_vals_pct,
        peak_x,
        peak_y,
        min_x,
        min_y,
        slope_change_x,
        slope_change_y,
    )


def anova_group_level_effect(
    data,
    batch_effect,
    ignore_columns=["age", "sex", "site", "eyes", "diagnosis"],
    save_tag="",
    save_output_path=False,
):
    """Estimate one-way ANOVA group effects for each eligible column.

    Non-finite observations are removed independently for each response
    variable before running the ANOVA. Columns listed in ``ignore_columns``
    and the batch-effect column itself are skipped. If an individual response
    cannot be analysed, its p-value and partial eta-squared are recorded as
    ``None`` without preventing the remaining responses from being processed.

    Parameters
    ----------
    data : pandas.DataFrame
        Table containing the grouping variable and candidate responses.
    batch_effect : str
        Column used as the between-subject grouping variable.
    ignore_columns : collection of str, optional
        Columns excluded from analysis. By default, common demographic and
        acquisition metadata are ignored.
    save_tag : str, optional
        Suffix included in the JSON output filename.
    save_output_path : str or os.PathLike or False, optional
        Output directory for the JSON results. Missing directories are
        created. If falsy, the results are printed instead.

    Returns
    -------
    dict
        Mapping from response name to ``p_val`` and partial eta-squared
        (``np2``). Both values are ``None`` when that response cannot be
        analysed.
    """

    res = {}

    for name in data.columns:

        if name in ignore_columns or name == batch_effect:
            continue

        sub = pd.DataFrame(
            {
                name: data[name],
                batch_effect: data[batch_effect],
            }
        )
        sub = sub.replace([np.inf, -np.inf], np.nan)
        sub = sub.dropna()

        try:
            aov = pg.anova(data=sub, dv=name, between=batch_effect, detailed=False)
            p = float(aov["p_unc"].iloc[0])
            np2 = float(aov["np2"].iloc[0])
        except Exception:
            p = np2 = None

        res[name] = {"p_val": p, "np2": np2}

    if save_output_path:
        os.makedirs(save_output_path, exist_ok=True)
        save_path = os.path.join(
            save_output_path, f"{batch_effect}_group_effect_{save_tag}.json"
        )
        with open(save_path, "w") as file:
            json.dump(res, file, indent=2)
    else:
        print(res)

    return res
