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
    impute_by_column_name="age",
    which_cohorts=None,
    subject_removal_nan_thr=0.2,
    including_ROIs=None,
    excluding_ROIs=None,
    name_data="reference_data",
    missing_value_handling_method=None,
    customized_con_var_imputation_window=None,
    remove_outliers=False,
    remove_outliers_approach="iqr",
    remove_outliers_group_by="site",
    iqr_factor=3,
    train_split_size=0.5,
    which_subjects=None,
    random_state=42,
):
    """
    Filter, clean, and package a dataframe into PCNtoolkit NormData.

    The function supports one or more model covariates. When imputation
    is requested, `impute_by_column_name` specifies the single continuous
    variable used to define the imputation neighbourhood.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing covariates, batch effects, response
        variables, and subject identifiers.
    response_vars : list of str
        Candidate response-variable columns.
    covariate_list : list of str
        One or more columns to use as model covariates.
    batch_effect_list : list of str
        Columns representing batch effects, such as site or sex.
    subject_id_col_name : str
        Column containing subject identifiers.
    impute_by_column_name : str, optional
        Continuous column used to define neighbourhoods during
        imputation. Default is "age".
    which_cohorts : list of str or None, optional
        Restrict the data to rows whose "diagnosis" value is included
        in this list.
    subject_removal_nan_thr : float, optional
        Missing-value fraction at or above which a column is dropped
        during imputation. Default is 0.2.
    including_ROIs : list of str or None, optional
        ROI name patterns to retain.
    excluding_ROIs : list of str or None, optional
        ROI name patterns to exclude.
    name_data : str, optional
        Name assigned to the resulting NormData object.
    missing_value_handling_method : {"mean", "median", None}, optional
        Imputation strategy. If None, NormData removes rows containing
        missing values.
    customized_con_var_imputation_window : dict or None, optional
        Site-specific additions to the default imputation window.
    remove_outliers : bool, optional
        Whether NormData should remove response-variable outliers.
    remove_outliers_approach : str, optional
        Outlier-removal method forwarded to NormData. Default is "iqr".
    remove_outliers_group_by : str or None, optional
        Column used to group observations during outlier removal.
        Default is "site".
    iqr_factor : float, optional
        IQR multiplier used for outlier detection. Default is 3.
    train_split_size : float or None, optional
        Training proportion. Values greater than 1 are interpreted as
        percentages. If None or 0, the full NormData object is returned.
    which_subjects : list or None, optional
        Restrict the data to identifiers in `subject_id_col_name`.
    random_state : int, optional
        Random seed used for the train/test split.

    Returns
    -------
    NormData or tuple of NormData
        Full NormData object, or `(train, test)` when splitting is
        requested.

    Raises
    ------
    ValueError
        If no covariate is provided, the imputation method is invalid,
        the imputation column is invalid, or the split size is outside
        its valid range.
    """
    if not covariate_list:
        raise ValueError(
            "covariate_list should contain at least one covariate."
        )

    valid_missing_methods = {"mean", "median", None}
    if missing_value_handling_method not in valid_missing_methods:
        raise ValueError(
            "missing_value_handling_method should be 'mean', "
            "'median', or None."
        )

    if missing_value_handling_method is not None:
        if not isinstance(impute_by_column_name, str):
            raise ValueError("impute_by_column_name should be a string.")

        if impute_by_column_name not in df.columns:
            raise ValueError(
                f"Imputation column '{impute_by_column_name}' "
                "was not found in the dataframe."
            )

    if train_split_size:
        if train_split_size > 1:
            train_split_size /= 100

        if not 0 < train_split_size < 1:
            raise ValueError(
                "train_split_size should be between 0 and 1, or "
                "between 0 and 100 when expressed as a percentage."
            )

    # Convert infinities before removing rows with invalid required data.
    df = df.replace([np.inf, -np.inf], np.nan)

    required_columns = list(
        dict.fromkeys(covariate_list + batch_effect_list)
    )

    if (
        missing_value_handling_method is not None
        and impute_by_column_name not in required_columns
    ):
        required_columns.append(impute_by_column_name)

    df = df.dropna(subset=required_columns)

    if which_cohorts:
        df = df[df["diagnosis"].isin(which_cohorts)]

    if excluding_ROIs:
        excluded_columns = df.filter(
            regex="|".join(excluding_ROIs)
        ).columns
        df = df.drop(columns=excluded_columns)

        response_vars = [
            variable
            for variable in response_vars
            if not any(
                excluded_roi in variable
                for excluded_roi in excluding_ROIs
            )
        ]

    if including_ROIs:
        stripped_rois = [
            re.sub(r"-(lh|rh)$", "", roi)
            for roi in including_ROIs
        ]

        required_patterns = (
            stripped_rois
            + batch_effect_list
            + [subject_id_col_name]
            + covariate_list
        )

        if (
            missing_value_handling_method is not None
            and impute_by_column_name not in required_patterns
        ):
            required_patterns.append(impute_by_column_name)

        pattern = "|".join(map(re.escape, required_patterns))
        df = df.loc[:, df.columns.str.contains(pattern)]

        response_vars = [
            variable
            for variable in response_vars
            if any(roi in variable for roi in stripped_rois)
        ]

    print("Number of response variables: ", len(response_vars))

    if which_subjects:
        df = df[df[subject_id_col_name].isin(which_subjects)]

    if missing_value_handling_method in {"mean", "median"}:
        df = impute_by_subgroup(
            df=df,
            group_cols=batch_effect_list,
            subject_removal_nan_thr=subject_removal_nan_thr,
            continous_cov_col=impute_by_column_name,
            imputation_con_var_window=5,
            strategy=missing_value_handling_method,
            customized_age_window=(
                customized_con_var_imputation_window
            ),
        )

        response_vars = [
            variable
            for variable in response_vars
            if variable in df.columns
        ]
        remove_nan = False
    else:
        remove_nan = True

    reference_data = NormData.from_dataframe(
        name=name_data,
        dataframe=df,
        covariates=covariate_list,
        batch_effects=batch_effect_list,
        response_vars=response_vars,
        subject_ids=subject_id_col_name,
        remove_Nan=remove_nan,
        remove_outliers=remove_outliers,
        remove_outliers_approach=remove_outliers_approach,
        remove_outliers_group_by=remove_outliers_group_by,
        iqr_factor=iqr_factor,
    )

    if train_split_size:
        train, test = reference_data.train_test_split(
            splits=(train_split_size, 1 - train_split_size),
            split_names=["train", "test"],
            random_state=random_state,
        )
        return train, test

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


def anova_group_level_effect(
    data,
    batch_effect,
    ignore_columns=["age", "sex", "site", "eyes", "diagnosis"],
    save_tag="",
    save_output_path=False,
):

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
        except:
            p = np2 = None

        res[name] = {"p_val": p, "np2": np2}

    if save_output_path:
        save_path = os.path.join(
            save_output_path, f"{batch_effect}_group_effect_{save_tag}.json"
        )
        with open(save_path, "w") as file:
            json.dump(res, file, indent=2)
    else:
        print(res)

    return res
