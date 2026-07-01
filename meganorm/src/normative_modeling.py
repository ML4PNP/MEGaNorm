from pcntoolkit import NormativeModel, HBR, BLR, Runner
from pcntoolkit import NormData
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az
import xarray as xr
import pymc as pm
import json
import sys
import os
import re


def impute_by_subgroup(
    df,
    group_cols,
    subject_removal_nan_thr=0.2,
    continous_cov_col="age",
    imputation_con_var_window=5,
    strategy="mean",
    customized_age_window=None,
):
    """
    Impute missing values in numeric columns using subgroup- and
    age-window-based neighbor statistics.

    Columns with a missing-value fraction above `subject_removal_nan_thr`
    are dropped entirely. For each remaining missing value, imputation
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
        Maximum allowed fraction of missing values for a column to be
        retained. Default is 0.2.
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

    Returns
    -------
    pandas.DataFrame
        Copy of `df`, with high-missingness columns dropped and
        remaining missing numeric values imputed.

    Raises
    ------
    ValueError
        If `continous_cov_col` is not a string.
    """
    if not isinstance(continous_cov_col, str):
        err_msg = "continous_cov_col should be a string. Multiple covriates are not supported yet."
        raise ValueError(err_msg)

    df = df.loc[:, df.isna().mean(axis=0) < subject_removal_nan_thr]

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
                group_mask = pd.Series([True] * len(df), index=df.index)
                for g in group_cols:
                    group_mask &= df[g] == row[g]

                window = imputation_con_var_window
                if customized_age_window and row["site"] in customized_age_window:
                    window += customized_age_window[row["site"]]

                age_mask = df[continous_cov_col].between(
                    row[continous_cov_col] - window, row[continous_cov_col] + window
                )

                neighbors = df.loc[group_mask & age_mask, col].dropna()

                if len(neighbors) > 0:
                    df_imputed.at[idx, col] = agg_fn(neighbors)
                else:
                    # Fallback 1: same group, any age
                    fallback_group = df.loc[group_mask, col].dropna()
                    if len(fallback_group) > 0:
                        df_imputed.at[idx, col] = agg_fn(fallback_group)
                    else:
                        # Fallback 2: global statistic
                        df_imputed.at[idx, col] = agg_fn(df[col].dropna())

    return df_imputed


def prepare_nm_data(
    df,
    response_vars,
    covariate_list,
    batch_effect_list,
    subject_id_col_name,
    which_cohorts=None,
    subject_removal_nan_thr=0.2,
    including_ROIs=None,
    excluding_ROIs=None,
    name_data="reference_data",
    missing_value_handling_method=None,
    customized_con_var_imputation_window=None,
    removing_outliers_thr=None,
    train_split_size=0.5,
    which_subjects=None,
    random_state=42,
):
    """
    Filter, clean, and package a dataframe into PCNtoolkit NormData for
    normative modeling.

    Optionally filters rows by cohort or subject list, includes/excludes
    ROI columns by name pattern, replaces infinities with NaN, imputes
    or drops missing values, removes outliers, and constructs a
    `NormData` object. If `train_split_size` is given, the data is
    additionally split into train/test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing covariates, batch effects, and
        response variables.
    response_vars : list of str
        Column names of candidate response variables (e.g., ROI
        measures).
    covariate_list : list of str
        Column name(s) to use as model covariates. Only a single
        covariate is currently supported.
    batch_effect_list : list of str
        Column names representing batch effects (e.g., site, sex).
    subject_id_col_name : str
        Column name containing subject identifiers.
    which_cohorts : list or None, optional
        If provided, restrict `df` to rows whose "diagnosis" column is
        in this list. Default is None.
    subject_removal_nan_thr : float, optional
        Missing-value fraction threshold passed to `impute_by_subgroup`
        when imputation is used. Default is 0.2.
    including_ROIs : list of str or None, optional
        ROI name patterns to keep; only matching response variables and
        columns are retained. Default is None.
    excluding_ROIs : list of str or None, optional
        ROI name patterns to exclude from columns and response
        variables. Default is None.
    name_data : str, optional
        Name assigned to the resulting `NormData` object. Default is
        "reference_data".
    missing_value_handling_method : {"mean", "median", None}, optional
        If "mean" or "median", missing values are imputed via
        `impute_by_subgroup` using that strategy; otherwise rows with
        NaNs are removed by `NormData`. Default is None.
    customized_con_var_imputation_window : dict or None, optional
        Site-specific imputation window overrides passed to
        `impute_by_subgroup`. Default is None.
    removing_outliers_thr : float or None, optional
        Z-score threshold for outlier removal. If None, outlier removal
        is skipped. Default is None.
    train_split_size : float or None, optional
        Fraction (or percentage, if >1) of data assigned to the
        training split. If falsy, no split is performed and the full
        `NormData` object is returned. Default is 0.5.
    which_subjects : list or None, optional
        If provided, restrict `df` to rows whose "participants_id" is
        in this list. Default is None.
    random_state : int, optional
        Random seed used for the train/test split. Default is 42.

    Returns
    -------
    NormData or tuple of NormData
        If `train_split_size` is set, returns `(train, test)` NormData
        splits; otherwise returns a single `NormData` object.

    Raises
    ------
    ValueError
        If `covariate_list` contains more than one covariate.
    """
    if len(covariate_list) > 1:
        err_msg = "continous_cov_col should be a single string. Multiple covriates are not supported yet."
        raise ValueError(err_msg)

    if which_cohorts:
        df = df[df["diagnosis"].isin(which_cohorts)]

    if excluding_ROIs:
        df = df.drop(columns=df.filter(regex="|".join(excluding_ROIs)).columns)
        response_vars = [
            var
            for var in response_vars
            if not any(excl in var for excl in excluding_ROIs)
        ]

    if including_ROIs:
        stripped_ROIs = [re.sub(r"-(lh|rh)$", "", roi) for roi in including_ROIs]
        pattern = "|".join(
            map(
                re.escape,
                stripped_ROIs
                + batch_effect_list
                + [subject_id_col_name]
                + covariate_list,
            )
        )
        df = df.loc[:, df.columns.str.contains(pattern)]

        response_vars = [
            var for var in response_vars if any(incl in var for incl in stripped_ROIs)
        ]
    print("Number of response variables: ", len(response_vars))

    if which_subjects:
        df = df[df["participants_id"].isin(which_subjects)]

    df = df.replace([np.inf, -np.inf], np.nan)
    if missing_value_handling_method in ["mean", "median"]:
        df = impute_by_subgroup(
            df=df,
            group_cols=batch_effect_list,
            subject_removal_nan_thr=subject_removal_nan_thr,
            continous_cov_col=covariate_list[0],
            imputation_con_var_window=5,
            strategy=missing_value_handling_method,
            customized_age_window=customized_con_var_imputation_window,
        )
        response_vars = [var for var in response_vars if var in df.columns.to_list()]
        remove_nan = False
    else:
        remove_nan = True

    if removing_outliers_thr:
        remove_outlier = True
    else:
        remove_outlier = False

    reference_data = NormData.from_dataframe(
        name=name_data,
        dataframe=df,
        covariates=covariate_list,
        batch_effects=batch_effect_list,
        response_vars=response_vars,
        subject_ids=subject_id_col_name,
        remove_Nan=remove_nan,
        remove_outliers=remove_outlier,
        z_threshold=removing_outliers_thr,
    )

    if train_split_size:
        if train_split_size > 1:
            train_split_size /= 100

        train, test = reference_data.train_test_split(
            splits=(train_split_size, 1 - train_split_size),
            split_names=["train", "test"],
            random_state=random_state,
        )
        return train, test

    else:
        return reference_data


def model_diagnostics(
    models_path,
    save_path,
    if_loo_cv=False,  # TODO
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
    colors=None,  # TODO
    job_configs=None,
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

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `if_parallel` is True and `job_configs` is not provided.
    """

    nm_dir = os.path.join(project_dir, "Normative_models")
    if not os.path.isdir(nm_dir):
        os.mkdir(nm_dir)

    model = NormativeModel(
        template_regression_model=template_regression_model,
        savemodel=if_save_models,
        evaluate_model=if_evaluate_models,
        saveresults=True,
        saveplots=if_save_plots,
        save_dir=nm_dir,
        inscaler=inscaler_method,
        outscaler=outscaler_method,
        name=model_name,
    )

    if not if_parallel:
        if test:
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

        if test:
            runner.fit_predict(model, train, test, observe=False)
        else:
            runner.fit(model, train, observe=False)

    # if if_model_diagnosis:
    #     model_diagnostics(
    #         models_path=f"{nm_dir}/model",
    #         save_path=os.path.join(nm_dir, "results"),
    #     )


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
        err_msg = "Did not have time to implement for all models!"
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
        Maximum age (in the same percentage/decade scale as the
        model's inverse-transformed covariate) to include in the
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
    x_vals = centile_data.X.sel(covariates="age").values
    x_vals_real = model.inscalers["age"].inverse_transform(x_vals)
    y_vals = centile_data.centiles.sel(centile=0.5, response_vars=IDP).values

    # Limit to upper_limit BEFORE scaling
    mask = x_vals_real * 100 <= upper_limit
    x_vals_real = x_vals_real[mask]
    y_vals = y_vals[mask]

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
