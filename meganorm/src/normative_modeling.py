from pcntoolkit import NormativeModel, HBR, BLR, Runner
from pcntoolkit import NormData
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import arviz as az
import sys
import os



def impute_by_subgroup(df, group_cols, continous_cov_col='age', imputation_con_var_window=5, strategy='mean', customized_age_window=None):

    if not isinstance(continous_cov_col, str):
        err_msg = "continous_cov_col should be a string. Multiple covriates are not supported yet."
        raise ValueError(err_msg)

    df = df.loc[:, df.isna().mean(axis=0) < 0.2]

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
    which_cohorts,
    response_vars,
    covariate_list,
    batch_effect_list,
    subject_id_col_name,
    including_brain_regions=None,
    excluding_brain_regions=None,
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

    if excluding_brain_regions:
        df = df.drop(columns=df.filter(regex="|".join(excluding_brain_regions)).columns)
        response_vars = [var for var in response_vars if not any(excl in var for excl in excluding_brain_regions)]

    if including_brain_regions:
        df = df[df.filter(regex="|".join(including_brain_regions)).columns]
        response_vars = [var for var in response_vars if any(incl in var for incl in including_brain_regions)]
        
        
    if which_subjects:
        df = df[df["participants_id"].isin(which_subjects)]

    if missing_value_handling_method in ["mean", "median"]:
        df = impute_by_subgroup(
            df,
            group_cols=batch_effect_list,
            continous_cov_col=covariate_list[0],
            imputation_con_var_window=5,
            strategy=missing_value_handling_method,
            customized_age_window=customized_con_var_imputation_window
        )
        response_vars = [var for var in response_vars if var in df.columns.to_list()]
        remove_nan = False
    else : 
        remove_nan = True

    
    if removing_outliers_thr: remove_outlier = True
    else: remove_outlier = False

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
        job_configs=None
):
    
    nm_dir = os.path.join(project_dir, "Normative_models", experiment_name)
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