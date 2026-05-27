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



def impute_by_subgroup(df, 
    group_cols, 
    subject_removal_nan_thr=0.2, 
    continous_cov_col='age', 
    imputation_con_var_window=5, 
    strategy='mean', 
    customized_age_window=None
    ):

    if not isinstance(continous_cov_col, str):
        err_msg = "continous_cov_col should be a string. Multiple covriates are not supported yet."
        raise ValueError(err_msg)

    df = df.loc[:, df.isna().mean(axis=0) < subject_removal_nan_thr]

    df_imputed = df.copy()
    agg_fn = np.nanmean if strategy == 'mean' else np.nanmedian

    # Numeric columns with NaNs, excluding the age col itself
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols_to_impute = [c for c in numeric_cols if c != continous_cov_col and df[c].isna().any()]

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
                    row[continous_cov_col] - window,
                    row[continous_cov_col] + window
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
    customized_con_var_imputation_window = None,
    removing_outliers_thr=None,
    train_split_size=0.5,
    which_subjects=None,
    random_state=42
):

    if len(covariate_list) > 1:
        err_msg = "continous_cov_col should be a single string. Multiple covriates are not supported yet."
        raise ValueError(err_msg)
    
    if which_cohorts:
        df = df[df["diagnosis"].isin(which_cohorts)]

    if excluding_ROIs:
        df = df.drop(columns=df.filter(regex="|".join(excluding_ROIs)).columns)
        response_vars = [var for var in response_vars if not any(excl in var for excl in excluding_ROIs)]

    if including_ROIs:
        stripped_ROIs = [re.sub(r'-(lh|rh)$', '', roi) for roi in including_ROIs]
        pattern = "|".join(map(re.escape, stripped_ROIs + batch_effect_list + [subject_id_col_name]  + covariate_list))
        df= df.loc[:, df.columns.str.contains(pattern)]

        response_vars = [var for var in response_vars if any(incl in var for incl in stripped_ROIs)]
    print("Number of response variables: ", len(response_vars))
        
        
    if which_subjects:
        df = df[df["participants_id"].isin(which_subjects)]

    if missing_value_handling_method in ["mean", "median"]:
        df = impute_by_subgroup(
            df=df,
            group_cols=batch_effect_list,
            subject_removal_nan_thr=subject_removal_nan_thr,
            continous_cov_col=covariate_list[0],
            imputation_con_var_window=5,
            strategy=missing_value_handling_method,
            customized_age_window=customized_con_var_imputation_window
        )
        response_vars = [var for var in response_vars if var in df.columns.to_list()]
        remove_nan = False
    else : 
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
        z_threshold=removing_outliers_thr
    )

    if train_split_size:
        if train_split_size > 1: 
            train_split_size/=100

        train, test = reference_data.train_test_split(
            splits=(train_split_size, 1-train_split_size), 
            split_names=["train", "test"], 
            random_state=random_state
            )
        return train, test
    
    else: 
        return reference_data


def model_diagnostics(
    models_path,
    save_path,
    if_loo_cv=False, # TODO 
):
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
            records.append({
                "model":      model,
                "parameter":  param,
                "r_hat":      row.get("r_hat"),
                "ess_bulk":   row.get("ess_bulk"),
                "ess_tail":   row.get("ess_tail"),
                "mcse_sd":    row.get("mcse_sd")
            })

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
        colors=None, # TODO
        job_configs=None,
        
):
    
    nm_dir = os.path.join(project_dir, "Normative_models")
    if not os.path.isdir(nm_dir):
        os.mkdir(nm_dir)

    model = NormativeModel(
            template_regression_model=template_regression_model,
            savemodel = if_save_models,
            evaluate_model=if_evaluate_models,
            saveresults=True,
            saveplots=if_save_plots,
            save_dir=nm_dir,
            inscaler=inscaler_method,
            outscaler=outscaler_method,
            name=model_name
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
            max_retries=job_configs["max_retries"]
        )

        if test:
            runner.fit_predict(model, 
                train, 
                test, 
                observe=False
            )
        else:
            runner.fit(model, 
                train,  
                observe=False
            )

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
    random_seed=None
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

    Examples
    --------
    >>> import numpy as np
    >>> n = 50
    >>> X = np.random.uniform(-1.5, 2.3, (n, 1))
    >>> Y = np.random.randn(n)
    >>> be = np.column_stack([
    ...     np.random.randint(0, 2, n),
    ...     np.random.randint(0, 6, n),
    ... ])
    >>> be_maps = {
    ...     "sex":  {"Female": 0, "Male": 1},
    ...     "site": {"BTNRH": 0, "CAMCAN": 1, "MOUS": 2,
    ...              "NIMH": 3, "OMEGA": 4, "WAND": 5}
    ... }
    >>> pymc_model = prior_predictive_check(
    ...     idata_path="path/to/idata.nc",
    ...     regression_model_path="path/to/regression_model.json",
    ...     model="hbr",
    ...     X=X,
    ...     Y=Y,
    ...     be=be,
    ...     be_maps=be_maps,
    ...     n_samples=500,
    ...     random_seed=42,
    ... )
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
        X,
        dims=["observations", "covariates"],
        coords={"observations": np.arange(n)}
    )

    Y = xr.DataArray(
        Y,
        dims=["observations"],
        coords={"observations": np.arange(n)}
    )
    be = xr.DataArray(
        be,
        dims=["observations", "batch_effect_dims"],
        coords={"batch_effect_dims": ["sex", "site"]}  # named coordinate
    )

    # Rebuild the PyMC model with dummy data
    pymc_model = hbr.likelihood.compile(X, be, be_maps, Y)

    print("The model:\n" , pymc_model.str_repr())

    with pymc_model:
        prior_idata = pm.sample_prior_predictive(
            draws=n_samples,
            random_seed=random_seed,
        )
    
    az.plot_ppc(prior_idata, group="prior", observed=True)
    plt.show()

    return pymc_model