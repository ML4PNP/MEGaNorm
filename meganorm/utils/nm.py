import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy.stats import shapiro
import itertools
from sklearn.model_selection import StratifiedKFold
import shutil
from scipy.stats import skew, kurtosis
from pcntoolkit.util.utils import z_to_abnormal_p, anomaly_detection_auc
from scipy.stats import false_discovery_control
from meganorm.plots.plots import KDE_plot


def hbr_data_split(
    data,
    save_path,
    covariates=["age"],
    batch_effects=None,
    train_split=0.5,
    validation_split=None,
    drop_nans=False,
    random_seed="23d",
    prefix="",
):
    """Utility function for splitting data into training, validation and test sets
    before HBR normative modeling. The sets are save as picke file in the specified
    save path.

    Args:
        data (DataFrame): Panda dataframe of data created for example using "load_camcan_data"
        function.
        save_path (str): Path to save the results.
        covariates (list, optional): List of covariates. Defaults to ['age'].
        batch_effects (list, optional): Used for deciding batch effects in HBR model.
        Defaults to None for no batch effect.
        train_split (float, optional): train set split ratio. Defaults to 0.5.
        validation_split (_type_, optional): If not None will be used for validation set
        split ratio. Defaults to None.
        drop_nans (boolean, optional): If True, drops columns with missing values. Defaults to False.
        random_seed (int, optional): Random seed in the splitting data. Defaults to 42.

    Returns:
        The number of biomarkers.
    """
    np.random.seed(random_seed)
    os.makedirs(save_path, exist_ok=True)

    if drop_nans:
        data = data.dropna(axis=0)

    x_train_all, y_train_all, b_train_all = [], [], []
    x_val_all, y_val_all, b_val_all = [], [], []
    x_test_all, y_test_all, b_test_all = [], [], []

    ## Looping through sites to split the data in a stratified way
    for uniques_site in data.site.unique():

        data_site = data[data.site == uniques_site]

        rand_idx = np.random.permutation(data_site.shape[0])

        train_num = np.round(train_split * data_site.shape[0]).astype(int)

        train_set = data_site.iloc[rand_idx[0:train_num]]
        x_train = train_set.loc[:, covariates]
        b_train = (
            train_set.loc[:, batch_effects]
            if batch_effects is not None
            else pd.DataFrame(
                np.zeros([x_train.shape[0], 1], dtype=int),
                index=x_train.index,
                columns=["site"],
            )
        )
        y_train = (
            train_set.drop(columns=covariates + batch_effects)
            if batch_effects is not None
            else train_set.drop(columns=covariates)
        )
        x_train_all.append(x_train), y_train_all.append(y_train), b_train_all.append(
            b_train
        )

        if validation_split is not None:
            validation_num = np.round(validation_split * data_site.shape[0]).astype(int)
            validation_set = data_site.iloc[
                rand_idx[train_num : train_num + validation_num]
            ]
            x_val = validation_set.loc[:, covariates]
            b_val = (
                validation_set.loc[:, batch_effects]
                if batch_effects is not None
                else pd.DataFrame(
                    np.zeros([x_val.shape[0], 1], dtype=int),
                    index=x_val.index,
                    columns=["site"],
                )
            )
            y_val = (
                validation_set.drop(columns=covariates + batch_effects)
                if batch_effects is not None
                else validation_set.drop(columns=covariates)
            )
            x_val_all.append(x_val), y_val_all.append(y_val), b_val_all.append(b_val)

            test_set = data_site.iloc[rand_idx[train_num + validation_num :]]
        else:
            validation_set = None
            test_set = data_site.iloc[rand_idx[train_num:]]

        x_test = test_set.loc[:, covariates]
        b_test = (
            test_set.loc[:, batch_effects]
            if batch_effects is not None
            else pd.DataFrame(
                np.zeros([x_test.shape[0], 1], dtype=int),
                index=x_test.index,
                columns=["site"],
            )
        )
        y_test = (
            test_set.drop(columns=covariates + batch_effects)
            if batch_effects is not None
            else test_set.drop(columns=covariates)
        )
        x_test_all.append(x_test), y_test_all.append(y_test), b_test_all.append(b_test)

    # train
    pd.concat(x_train_all, axis=0).to_pickle(
        os.path.join(save_path, prefix + "x_train.pkl")
    )
    pd.concat(y_train_all, axis=0).to_pickle(
        os.path.join(save_path, prefix + "y_train.pkl")
    )
    pd.concat(b_train_all, axis=0).to_pickle(
        os.path.join(save_path, prefix + "b_train.pkl")
    )
    # validation
    if validation_split is not None:
        pd.concat(x_val_all, axis=0).to_pickle(
            os.path.join(save_path, prefix + "x_val.pkl")
        )
        pd.concat(y_val_all, axis=0).to_pickle(
            os.path.join(save_path, prefix + "y_val.pkl")
        )
        pd.concat(b_val_all, axis=0).to_pickle(
            os.path.join(save_path, prefix + "b_val.pkl")
        )
    # test
    pd.concat(x_test_all, axis=0).to_pickle(
        os.path.join(save_path, prefix + "x_test.pkl")
    )
    pd.concat(y_test_all, axis=0).to_pickle(
        os.path.join(save_path, prefix + "y_test.pkl")
    )
    pd.concat(b_test_all, axis=0).to_pickle(
        os.path.join(save_path, prefix + "b_test.pkl")
    )

    with open(os.path.join(save_path, prefix + "random_seed.pkl"), "wb") as file:
        pickle.dump({"random_seed": random_seed}, file)

    return y_test.shape[1]


def mace(
    nm, x_test, y_test, be_test, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], plot=False
):

    z_scores = st.norm.ppf(quantiles)
    batch_ids = np.unique(be_test)
    batch_mace = np.zeros(
        [
            len(batch_ids),
        ]
    )
    empirical_quantiles = []

    for b, batch_id in enumerate(batch_ids):
        model_be = np.repeat(
            np.array([[batch_id]]), x_test[be_test == batch_id, :].shape[0]
        )
        mcmc_quantiles = nm.get_mcmc_quantiles(
            x_test[be_test == batch_id, :], model_be, z_scores=z_scores
        ).T
        empirical_quantiles.append(
            (mcmc_quantiles >= y_test[be_test == batch_id, 0:1]).mean(axis=0)
        )
        batch_mace[b] = np.abs((np.array(quantiles) - empirical_quantiles[b])).mean()

    if plot:
        plt.figure(figsize=(10, 6))
        sns.set_context("notebook", font_scale=2)
        sns.lineplot(
            x=quantiles,
            y=quantiles,
            color="magenta",
            linestyle="--",
            linewidth=3,
            label="ideal",
        )
        for b, batch_id in enumerate(batch_ids):
            sns.lineplot(
                x=quantiles,
                y=empirical_quantiles[b],
                color="black",
                linestyle="dashdot",
                linewidth=3,
                label=f"observed {b}",
            )
            sns.scatterplot(
                x=quantiles, y=empirical_quantiles[b], marker="o", s=150, alpha=0.5
            )
        plt.legend()
        plt.xlabel("True Quantile")
        plt.ylabel("Empirical Quantile")
        _ = plt.title("Reliability diagram")

    return batch_mace.mean()


def evaluate_mace(
    model_path,
    X_path,
    y_path,
    be_path,
    save_path=None,
    model_id=0,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    plot=False,
    outputsuffix="ms",
):

    nm = pickle.load(
        open(
            os.path.join(
                model_path, "NM_0_" + str(model_id) + "_" + outputsuffix + ".pkl"
            ),
            "rb",
        )
    )
    x_test = pickle.load(open(X_path, "rb")).to_numpy()
    be_test = pickle.load(open(be_path, "rb")).to_numpy().squeeze()
    y_test = pickle.load(open(y_path, "rb")).to_numpy()[:, model_id : model_id + 1]

    meta_data = pickle.load(open(os.path.join(model_path, "meta_data.md"), "rb"))

    cov_scaler = meta_data["scaler_cov"]
    res_scaler = meta_data["scaler_resp"]

    if len(cov_scaler) > 0:
        x_test = cov_scaler[model_id][0].transform(x_test)
    if len(res_scaler) > 0:
        y_test = res_scaler[model_id][0].transform(y_test)

    z_scores = st.norm.ppf(quantiles)
    batch_num = be_test.shape[1]

    batch_mace = []
    empirical_quantiles = []

    b = 0

    mcmc_quantiles = nm.get_mcmc_quantiles(x_test, be_test, z_scores=z_scores).T

    for i in range(batch_num):
        batch_ids = list(np.unique(be_test[:, i]))
        if len(batch_ids) > 1:
            for batch_id in batch_ids:
                empirical_quantiles.append(
                    (
                        mcmc_quantiles[be_test[:, i] == batch_id, :]
                        >= y_test[be_test[:, i] == batch_id, :]
                    ).mean(axis=0)
                )
                batch_mace.append(
                    np.abs(np.array(quantiles) - empirical_quantiles[b]).mean()
                )
                b += 1

    batch_mace = np.array(batch_mace)

    if plot:
        plt.figure(figsize=(10, 6))
        sns.set_context("notebook", font_scale=2)
        sns.lineplot(
            x=quantiles,
            y=quantiles,
            color="magenta",
            linestyle="--",
            linewidth=3,
            label="ideal",
        )
        b = 0
        for i in range(batch_num):
            batch_ids = list(np.unique(be_test[:, i]))
            for batch_id in batch_ids:
                sns.lineplot(
                    x=quantiles,
                    y=empirical_quantiles[b],
                    color="black",
                    linestyle="dashdot",
                    linewidth=3,
                    label=f"observed {b}",
                )
                sns.scatterplot(
                    x=quantiles, y=empirical_quantiles[b], marker="o", s=150, alpha=0.5
                )
                b += 1
        plt.legend()
        plt.xlabel("True Quantile")
        plt.ylabel("Empirical Quantile")
        _ = plt.title("Reliability diagram")
        plt.savefig(os.path.join(save_path, "MACE_" + str(model_id) + ".png"), dpi=300)

    return batch_mace.mean()


def model_quantile_evaluation(
    configs,
    save_path,
    valcovfile_path,
    valrespfile_path,
    valbefile,
    bio_num,
    plot=True,
    outputsuffix="ms",
    quantiles=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
):

    mace = np.zeros([len(configs.keys()), bio_num])
    best_models = []

    for c, config in enumerate(configs.keys()):
        for ind in range(bio_num):
            mace[c, ind] = evaluate_mace(
                os.path.join(save_path, config, "Models"),
                valcovfile_path,
                valrespfile_path,
                valbefile,
                model_id=ind,
                quantiles=quantiles,
                outputsuffix=outputsuffix,
            )
            print(f"Config:{config}, id:{ind}")

        with open(
            os.path.join(save_path, config, "MACE_" + outputsuffix + ".pkl"), "wb"
        ) as file:
            pickle.dump(mace[c, :].T, file)

    for ind in range(bio_num):
        best_models.append(list(configs.keys())[np.argmin(mace[:, ind])])

    bio_ids = dict()
    for model in np.unique(best_models):
        bio_ids[model] = np.where(np.array(best_models) == model)[0]

    with open(os.path.join(save_path, "model_selection_results.pkl"), "wb") as file:
        pickle.dump(
            {"best_models": best_models, "bio_ids": bio_ids, "mace": mace}, file
        )

    if plot:
        KDE_plot(mace, list(configs.keys()), "MACE")
        plt.savefig(os.path.join(save_path, "model_comparison_mace.png"), dpi=600)

    return mace, best_models, bio_ids


def calculate_oscilochart(
    quantiles_path,
    gender_ids,
    frequency_band_model_ids,
    quantile_id=2,
    site_id=None,
    point_num=100,
    age_slices=None,
):

    if age_slices is None:
        age_slices = np.arange(5, 80, 5)

    oscilogram = {
        gender: dict.fromkeys(frequency_band_model_ids.keys())
        for gender in gender_ids.keys()
    }

    temp = pickle.load(open(os.path.join(quantiles_path), "rb"))
    q = temp["quantiles"]
    x = temp["synthetic_X"][0:point_num].squeeze()
    b = temp["batch_effects"]

    for fb in frequency_band_model_ids.keys():
        model_id = frequency_band_model_ids[fb]

        if site_id is None:
            data = np.concatenate(
                [
                    q[b[:, 0] == 0, quantile_id, model_id : model_id + 1],
                    q[b[:, 0] == 1, quantile_id, model_id : model_id + 1],
                ],
                axis=1,
            )
            data = data.reshape(5, 100, 2)
            data = data.mean(axis=0)
        else:
            data = np.concatenate(
                [
                    q[
                        np.logical_and(b[:, 0] == 0, b[:, 1] == site_id),
                        quantile_id,
                        model_id : model_id + 1,
                    ],
                    q[
                        np.logical_and(b[:, 0] == 1, b[:, 1] == site_id),
                        quantile_id,
                        model_id : model_id + 1,
                    ],
                ],
                axis=1,
            )

        for gender in gender_ids.keys():
            batch_id = gender_ids[gender]
            oscilogram[gender][fb] = []
            for slice in age_slices:
                d = data[np.logical_and(x >= slice, x < slice + 5), batch_id]
                m = np.mean(d)
                s = np.std(d)
                oscilogram[gender][fb].append([m, s])

    return oscilogram, age_slices


def shapiro_stat(z_scores, covariates, n_bins=10):
    """Perform Shapiro-Wilk test for normality on z-scores in bins of covariates.

    Args:
        z_scores (numpy.ndarray): n by p matrix of z-scores (n subjects, p measures)
        covariates (numpy.ndarray): n by 1 matrix of covariates (n subjects)
        n_bins (int, optional): Number of bins to slice the covariates into (default is 10)

    Returns:
        numpy.ndarray: a p vector of Shapiro-Wilk test statistics for eaxh measure
    """

    z_scores = np.asarray(z_scores)
    covariates = np.asarray(covariates).flatten()

    test_statistics = np.zeros((n_bins, z_scores.shape[1]))

    # Get the bin edges and digitize the covariates into bins
    bin_edges = np.linspace(np.min(covariates), np.max(covariates), n_bins + 1)
    bin_indices = np.digitize(covariates, bins=bin_edges) - 1

    # Perform the Shapiro-Wilk test for each bin and for each measure
    for bin_idx in range(n_bins):
        for measure_idx in range(z_scores.shape[1]):

            z_in_bin = z_scores[bin_indices == bin_idx, measure_idx]

            if len(z_in_bin) > 2:  ## Check if there are enough data points for the test
                test_statistics[bin_idx, measure_idx], _ = shapiro(z_in_bin)
            else:  # If not set the statistic to NaN
                test_statistics[bin_idx, measure_idx] = np.nan

    return test_statistics.mean(axis=0)


def estimate_centiles(
    processing_dir,
    bio_num,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    batch_map={0: {"Male": 0, "Female": 1}, 1: {"CAMCAN": 0, "BTNRH": 1}},
    age_range=[0, 100],
    point_num=100,
    outputsuffix="estimate",
    save=True,
):

    z_scores = st.norm.ppf(quantiles)

    group_sizes = [len(batch_map[key].items()) for key in batch_map.keys()]
    ranges = [range(size) for size in group_sizes]
    combinations = list(itertools.product(*ranges))
    batch_effects = np.array(
        [combination for combination in combinations for _ in range(point_num)]
    )
    synthetic_X = np.vstack(
        [
            np.linspace(age_range[0], age_range[1], point_num)[:, np.newaxis]
            for i in range(np.product(group_sizes))
        ]
    )

    meta_data = pickle.load(
        open(
            os.path.join(processing_dir, "batch_" + str(1), "Models/meta_data.md"), "rb"
        )
    )

    if len(meta_data["scaler_cov"]) > 0:
        in_scaler = meta_data["scaler_cov"][0]
        scaled_synthetic_X = in_scaler.transform(synthetic_X)
    else:
        in_scaler = None
        scaled_synthetic_X = synthetic_X / 100

    q = np.zeros([scaled_synthetic_X.shape[0], len(quantiles), bio_num])

    for model_id in range(bio_num):

        meta_data = pickle.load(
            open(
                os.path.join(
                    processing_dir, "batch_" + str(model_id + 1), "Models/meta_data.md"
                ),
                "rb",
            )
        )
        nm = pickle.load(
            open(
                os.path.join(
                    processing_dir,
                    "batch_" + str(model_id + 1),
                    "Models/NM_0_0_" + outputsuffix + ".pkl",
                ),
                "rb",
            )
        )
        q[:, :, model_id] = nm.get_mcmc_quantiles(
            scaled_synthetic_X, batch_effects, z_scores=z_scores
        ).T

        if len(meta_data["scaler_resp"]) > 0:
            out_scaler = meta_data["scaler_resp"][0]
            for i in range(len(z_scores)):
                q[:, i, model_id] = out_scaler.inverse_transform(q[:, i, model_id])

        print(f"Quantiles for model {model_id} are estimated.")

    if save:
        with open(
            os.path.join(processing_dir, "Quantiles_" + outputsuffix + ".pkl"), "wb"
        ) as file:
            pickle.dump(
                {
                    "quantiles": q,
                    "synthetic_X": synthetic_X,
                    "batch_effects": batch_effects,
                },
                file,
            )

    return q


def saving(data, path, counter, tag, split):
    fold_path = os.path.join(path, f"{tag}_fold_{counter}_{split}.pkl")
    data.to_pickle(fold_path)


def kfold_split(
    data_path: str,
    save_dir: str,
    n_folds: int,
    sub_file="folds",
    prefix="",
    random_state=42,
):
    """
    This function creates stratified k-fold cross-validation splits and saves them as separate batch, x, and y files.
    Both 'sex' and 'site' are used to stratify the data, preserving their distribution across the folds.

    Parameters:
        data_path (str): The path to the directory containing the input data files (e.g., x_train.pkl, b_train.pkl, y_train.pkl).
        save_dir (str): The directory where the generated fold data will be saved.
        n_folds (int): The number of folds to split the data into for cross-validation.
        sub_file (str): The subdirectory name within `save_dir` where fold data will be saved (default is "folds").
        prefix (str): Prefix to add to the file names when loading data files (default is an empty string).
        random_state (int): Random seed for reproducibility (default is 42).

    returns:
        folds_path (str): where the folds are saved
    """

    folds_path = os.path.join(save_dir, sub_file)

    if os.path.exists(folds_path):
        shutil.rmtree(folds_path)
    os.makedirs(folds_path)

    x_all = pickle.load(open(os.path.join(data_path, prefix + "x_train.pkl"), "rb"))
    b_all = pickle.load(open(os.path.join(data_path, prefix + "b_train.pkl"), "rb"))
    y_all = pickle.load(open(os.path.join(data_path, prefix + "y_train.pkl"), "rb"))

    numpy_batch = b_all.apply(
        lambda row: row["site"] * 2 + row["sex"], axis=1
    ).to_numpy()

    skf = StratifiedKFold(n_splits=n_folds)
    skf.get_n_splits(x_all, numpy_batch)

    for counter, (train_ind, test_ind) in enumerate(skf.split(x_all, numpy_batch)):
        saving(
            data=x_all.iloc[train_ind, :],
            path=folds_path,
            counter=counter,
            tag="x",
            split="tr",
        )
        saving(
            data=b_all.iloc[train_ind, :],
            path=folds_path,
            counter=counter,
            tag="b",
            split="tr",
        )
        saving(
            data=y_all.iloc[train_ind, :],
            path=folds_path,
            counter=counter,
            tag="y",
            split="tr",
        )

        saving(
            data=x_all.iloc[test_ind, :],
            path=folds_path,
            counter=counter,
            tag="x",
            split="te",
        )
        saving(
            data=b_all.iloc[test_ind, :],
            path=folds_path,
            counter=counter,
            tag="b",
            split="te",
        )
        saving(
            data=y_all.iloc[test_ind, :],
            path=folds_path,
            counter=counter,
            tag="y",
            split="te",
        )

    return folds_path


def prepare_prediction_data(
    data, save_path, covariates=["age"], batch_effects=None, drop_nans=False, prefix=""
):

    os.makedirs(save_path, exist_ok=True)

    if drop_nans:
        data = data.dropna(axis=0)

    x_test = data.loc[:, covariates]
    b_test = (
        data.loc[:, batch_effects]
        if batch_effects is not None
        else pd.DataFrame(
            np.zeros([x_test.shape[0], 1], dtype=int),
            index=x_test.index,
            columns=["site"],
        )
    )
    y_test = (
        data.drop(columns=covariates + batch_effects)
        if batch_effects is not None
        else data.drop(columns=covariates)
    )

    x_test.to_pickle(os.path.join(save_path, prefix + "x_test.pkl"))
    y_test.to_pickle(os.path.join(save_path, prefix + "y_test.pkl"))
    b_test.to_pickle(os.path.join(save_path, prefix + "b_test.pkl"))

    return None


def cal_stats_for_gauge(q_path, features, site_id, gender_id, age):

    q = pickle.load(open(q_path, "rb"))
    quantiles = q["quantiles"]
    synthetic_X = (
        q["synthetic_X"].reshape(10, 100).mean(axis=0)
    )  # since Xs are repeated !
    b = q["batch_effects"]

    statistics = {feature: [] for feature in features}
    for ind in range(len(features)):

        biomarker_stats = []
        for quantile_id in range(quantiles.shape[1]):

            if (
                not site_id
            ):  # if not any specific site, average between all sites (batch effect)
                data = quantiles[b[:, 0] == gender_id, quantile_id, ind : ind + 1]
                data = data.reshape(5, 100, 1)
                data = data.mean(axis=0)
            if site_id:
                data = quantiles[
                    np.logical_and(b[:, 0] == gender_id, b[:, 1] == site_id),
                    quantile_id,
                    ind : ind + 1,
                ]

            data = data.squeeze()

            closest_x = min(synthetic_X, key=lambda x: abs(x - age))
            age_bin_ind = np.where(synthetic_X == closest_x)[0][0]

            biomarker_stats.append(data[age_bin_ind])

        statistics[features[ind]].extend(biomarker_stats)
    return statistics


def abnormal_probability(
    processing_dir, nm_processing_dir, site_id, n_permutation=1000
):

    with open(os.path.join(processing_dir, "Z_clinicalpredict.pkl"), "rb") as file:
        z_patient = pickle.load(file)

    with open(os.path.join(processing_dir, "Z_estimate.pkl"), "rb") as file:
        z_healthy = pickle.load(file)

    with open(os.path.join(nm_processing_dir, "b_test.pkl"), "rb") as file:
        b_healthy = pickle.load(file)

    z_healthy = z_healthy.iloc[np.where(b_healthy["site"] == site_id)[0], :]

    # z_patient = pd.concat([z_patient, np.sqrt((z_patient.iloc[:, [0, 1, 2, 3]]**2).mean(axis=1))], axis=1)
    # z_healthy = pd.concat([z_healthy, np.sqrt((z_healthy.iloc[:, [0, 1, 2, 3]]**2).mean(axis=1))], axis=1)

    p_patient = z_to_abnormal_p(z_patient)
    p_healthy = z_to_abnormal_p(z_healthy)

    p_patient = np.hstack(
        [p_patient, p_patient[:, [0, 2, 3]].mean(axis=1).reshape(-1, 1)]
    )
    p_healthy = np.hstack(
        [p_healthy, p_healthy[:, [0, 2, 3]].mean(axis=1).reshape(-1, 1)]
    )

    p = np.concatenate([p_patient, p_healthy])
    labels = np.concatenate([np.ones(p_patient.shape[0]), np.zeros(p_healthy.shape[0])])

    auc, p_val = anomaly_detection_auc(p, labels, n_permutation=n_permutation)

    p_val = false_discovery_control(p_val)

    return p_val, auc


def aggregate_metrics_across_runs(
    path,
    method_name,
    biomarker_names,
    valcovfile_path,
    valrespfile_path,
    valbefile,
    metrics=["skewness", "kurtosis", "W", "MACE"],
    num_runs=10,
    quantiles=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
    outputsuffix="estimate",
    zscore_clipping_value=8,
):

    # index_labels = [metric + "_" + biomarker_name for metric in metrics for biomarker_name in biomarker_names]
    # df = pd.DataFrame(index=index_labels, columns=list(range(10)))
    data = {
        metric: {biomarker_name: [] for biomarker_name in biomarker_names}
        for metric in metrics
    }

    for run in range(num_runs):

        run_path = path.replace("Run_0", f"Run_{run}")
        with open(os.path.join(run_path, method_name, "Z_estimate.pkl"), "rb") as file:
            z_scores = pickle.load(file)

            # clipping
            z_scores = z_scores.applymap(
                lambda x: zscore_clipping_value if abs(x) > zscore_clipping_value else x
            )

            for metric in metrics:
                values = []

                if metric == "MACE":
                    for ind in range(len(biomarker_names)):
                        values.append(
                            evaluate_mace(
                                os.path.join(run_path, method_name, "Models"),
                                valcovfile_path,
                                valrespfile_path,
                                valbefile,
                                model_id=ind,
                                quantiles=quantiles,
                                outputsuffix=outputsuffix,
                            )
                        )

                if metric == "W":
                    with open(os.path.join(run_path, "x_test.pkl"), "rb") as file:
                        cov = pickle.load(file)
                    values.extend(shapiro_stat(z_scores, cov))

                if metric == "skewness":
                    values.extend(skew(z_scores))

                if metric == "kurtosis":
                    values.extend(kurtosis(z_scores))

                for counter, name in enumerate(biomarker_names):
                    data[metric][name].append(values[counter])

    return data
