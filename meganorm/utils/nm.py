import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy.stats import shapiro
import itertools
from scipy.stats import skew, kurtosis
from pcntoolkit.util.utils import z_to_abnormal_p, anomaly_detection_auc
from scipy.stats import false_discovery_control
from scipy.stats import ranksums
from sklearn.model_selection import train_test_split


# **
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
    stratification_columns=["site", "sex"],
):
    """
    Splits a given DataFrame into training, validation, and test sets for normative modeling,
    while considering stratification based on specified categorical columns. The data is saved as
    pickled files for normative modeling (PCNToolkit requires paths to the files).

    Parameters
    ----------
    data : pd.DataFrame
        A Pandas DataFrame containing the data to be split. Created using functions like "load_camcan_data".

    save_path : str
        Path where the resulting training, validation, and test sets will be saved as pickled files.

    covariates : list of str, optional, default=["age"]
        List of covariates to be used in the analysis (default is `["age"]`).

    batch_effects : list of str, optional, default=None
        List of batch effects to be accounted for in the HBR model. Default is `None`.

    train_split : float, optional, default=0.5
        Proportion of the data to be used for training (default is 0.5).

    validation_split : float, optional, default=None
        Proportion of the training data to be used for validation (default is `None`, meaning no validation set is created).

    drop_nans : bool, optional, default=False
        If `True`, rows with missing values are dropped (default is `False`).

    random_seed : int or str, optional, default="23d"
        Seed for random number generation to ensure reproducibility (default is `23d`).

    prefix : str, optional, default=""
        Prefix to be added to the filenames when saving the pickled data (default is `""`).

    stratification_columns : list of str, optional, default=["site", "sex"]
        List of categorical columns used for stratification during splitting (default is `["site", "sex"]`).

    Returns
    -------
    list of str
        A list of biomarker names (columns in the target `y` DataFrame), which represent the dependent
        variables for the HBR normative modeling.

    Notes
    -----
    The function performs the following steps:
        - Drops any rows with missing values if `drop_nans=True`.
        - Creates a new column "combination" based on the specified stratification columns.
        - Splits the data into training, validation (optional), and test sets while preserving the stratification.
        - Saves the resulting splits (`x_train`, `y_train`, `b_train`, etc.) as pickled files in the specified `save_path`.
        - Saves the random seed used for splitting into a separate pickled file.
        - Returns the names of the biomarkers (columns in `y_train`).

    Example
    -------
    biomarker_names = hbr_data_split(
        data=df,
        save_path="./data_split/",
        covariates=["age", "sex"],
        batch_effects=["site"],
        train_split=0.7,
        validation_split=0.2,
        random_seed=42
    )
    """
    os.makedirs(save_path, exist_ok=True)

    if drop_nans:
        data = data.dropna(axis=0)

    data["combination"] = data[stratification_columns].astype(str).agg("_".join, axis=1)
    train_df, test_df = train_test_split(
        data,
        stratify=data["combination"],
        test_size=(1 - train_split),
        random_state=random_seed,
    )

    if validation_split:
        train_df, val_df = train_test_split(
            train_df,
            stratify=data["combination"],
            test_size=validation_split,
            random_state=random_seed,
        )

    # train ********
    x_train = train_df.loc[:, covariates]
    b_train = (
        train_df.loc[:, batch_effects]
        if batch_effects is not None
        else pd.DataFrame(
            np.zeros([x_train.shape[0], 1], dtype=int),
            index=x_train.index,
            columns=["site"],
        )
    )
    y_train = (
        train_df.drop(
            columns=covariates + batch_effects + ["combination", "diganosis"],
            errors="ignore",
        )
        if batch_effects is not None
        else train_df.drop(
            columns=covariates + ["combination", "diganosis"], errors="ignore"
        )
    )

    # test ********
    x_test = test_df.loc[:, covariates]
    b_test = (
        test_df.loc[:, batch_effects]
        if batch_effects is not None
        else pd.DataFrame(
            np.zeros([x_test.shape[0], 1], dtype=int),
            index=x_test.index,
            columns=["site"],
        )
    )
    y_test = (
        test_df.drop(
            columns=covariates + batch_effects + ["combination", "diganosis"],
            errors="ignore",
        )
        if batch_effects is not None
        else test_df.drop(
            columns=covariates + ["combination", "diganosis"], errors="ignore"
        )
    )

    # validation ********
    if validation_split:
        x_val = val_df.loc[:, covariates]
        b_val = (
            val_df.loc[:, batch_effects]
            if batch_effects is not None
            else pd.DataFrame(
                np.zeros([x_val.shape[0], 1], dtype=int),
                index=x_val.index,
                columns=["site"],
            )
        )
        y_val = (
            val_df.drop(
                columns=covariates + batch_effects + ["combination", "diganosis"],
                errors="ignore",
            )
            if batch_effects is not None
            else val_df.drop(
                columns=covariates + ["combination", "diganosis"], errors="ignore"
            )
        )

    # train
    x_train.to_pickle(os.path.join(save_path, prefix + "x_train.pkl"))
    y_train.to_pickle(os.path.join(save_path, prefix + "y_train.pkl"))
    b_train.to_pickle(os.path.join(save_path, prefix + "b_train.pkl"))
    # validation
    if validation_split:
        x_val.to_pickle(os.path.join(save_path, prefix + "x_val.pkl"))
        y_val.to_pickle(os.path.join(save_path, prefix + "y_val.pkl"))
        b_val.to_pickle(os.path.join(save_path, prefix + "b_val.pkl"))
    # test
    x_test.to_pickle(os.path.join(save_path, prefix + "x_test.pkl"))
    y_test.to_pickle(os.path.join(save_path, prefix + "y_test.pkl"))
    b_test.to_pickle(os.path.join(save_path, prefix + "b_test.pkl"))

    with open(os.path.join(save_path, prefix + "random_seed.pkl"), "wb") as file:
        pickle.dump({"random_seed": random_seed}, file)

    biomarker_name = y_train.columns
    return biomarker_name.tolist()


# **
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
    """
    Evaluate model calibration using the Mean Absolute Calibration Error (MACE) metric.

    This function computes MACE by comparing model-predicted quantiles with the
    empirical distribution of outcomes across batch groups. Optionally, it plots a
    reliability diagram to visually assess calibration performance.

    Parameters
    ----------
    model_path : str
        Path to the directory containing the saved model and its metadata.
    X_path : str
        Path to the test covariates (.pkl file), expected as a pandas DataFrame.
    y_path : str
        Path to the true test responses (.pkl file), expected as a pandas DataFrame.
    be_path : str
        Path to the batch effect file (.pkl file), with each column as a batch dimension.
    save_path : str, optional
        Directory to save the reliability diagram if `plot` is True. Required when plotting.
    model_id : int, optional
        Index of the model (biomarker) to evaluate. Corresponds to index X in 'NM_0_X_<suffix>.pkl'.
    quantiles : list of float, optional
        Quantiles to use for computing calibration (default: [0.05, 0.25, 0.5, 0.75, 0.95]).
    plot : bool, optional
        Whether to generate and save a reliability diagram (default: False).
    outputsuffix : str, optional
        Suffix of the saved model filename (default: "ms").

    Returns
    -------
    float
        Mean absolute calibration error (MACE) across all batches and batch IDs.

    Notes
    -----
    - This function assumes all inputs are pickled files in the expected format.
    - Empirical quantiles are computed within each batch group and compared to the target quantiles.
    - Plotting requires `matplotlib` and `seaborn`.
    - Input file formats:
        - `X_path`: shape (n_samples, n_features)
        - `y_path`: shape (n_samples, n_outputs)
        - `be_path`: shape (n_samples, n_batch_dims)
    """
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


# **
def calculate_PNOCs(
    quantiles_path,
    gender_ids,
    frequency_band_model_ids,
    quantile_id=2,
    site_id=None,
    point_num=100,
    sex_batch_ind=0,
    site_batch_ind=1,
    num_of_sexs=2,
    num_of_datasets=None,
    age_slices=None,
):
    """
    Prepares the data required for the `plot_PNOCs` function.

    This function slices the covariate into multiple bins and calculates the mean and
    standard deviation of each frequency band across the population for both sexes.

    Parameters
    ----------
    quantiles_path : str
        Path to a pickle file containing the keys: 'quantiles', 'synthetic_X', and 'batch_effects'.
    gender_ids : dict
        Dictionary mapping gender labels (e.g., {"male": 0, "female": 1}) to their batch indices.
    frequency_band_model_ids : dict
        Dictionary mapping frequency band names (e.g., {"alpha": 0, "beta": 1}) to model indices.
    quantile_id : int, optional
        Index of the quantile to use from the loaded quantiles array (default is 2). This number
        corresponds to the ith element of the computed percentiles. If the computed percentiles
        were [0.05, 0.25, 0.5, 0.75, 0.95], then 'quantile_id=2' corresponds to 0.5.
    site_id : int, optional
        Site ID to condition the P-NOCs on. If None, PNOCs from all sites are averaged (default is None).
    point_num : int, optional
        Number of synthetic data points used in deriving quantiles (default is 100).
    sex_batch_ind : int, optional
        Index in the batch array corresponding to sex (default is 0).
    site_batch_ind : int, optional
        Index in the batch array corresponding to site (default is 1).
    num_of_sexs : int, optional
        Number of sex categories (default is 2).
    num_of_datasets : int, optional
        Number of datasets used in data aggregation (required if `site_id` is None).
    age_slices : array-like of int, optional
        Array of starting ages to define age bins. If None, defaults to `np.arange(5, 80, 5)`.

    Returns
    -------
    oscilogram : dict
        Nested dictionary with structure: oscilogram[gender][frequency_band] = list of [mean, std]
        values for each age slice.
    age_slices : numpy.ndarray
        Array of age slice start values used for binning.

    Notes
    -----
    - The input pickle file must contain:
        - 'quantiles': array of shape (n_samples, n_quantiles, n_models)
        - 'synthetic_X': array of age values of shape (n_samples, 1)
        - 'batch_effects': array of shape (n_samples, n_batch_dims)
    """

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
                    q[b[:, sex_batch_ind] == 0, quantile_id, model_id : model_id + 1],
                    q[b[:, sex_batch_ind] == 1, quantile_id, model_id : model_id + 1],
                ],
                axis=1,
            )
            data = data.reshape(num_of_datasets, point_num, num_of_sexs)
            data = data.mean(axis=0)
        else:
            data = np.concatenate(
                [
                    q[
                        np.logical_and(
                            b[:, sex_batch_ind] == 0, b[:, site_batch_ind] == site_id
                        ),
                        quantile_id,
                        model_id : model_id + 1,
                    ],
                    q[
                        np.logical_and(
                            b[:, sex_batch_ind] == 1, b[:, site_batch_ind] == site_id
                        ),
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
                d = data[
                    np.logical_and(
                        x >= slice, x < slice + int(age_slices[1] - age_slices[0])
                    ),
                    batch_id,
                ]
                m = np.mean(d)
                s = np.std(d)
                oscilogram[gender][fb].append([m, s])

    return oscilogram, age_slices


# **
def shapiro_stat(z_scores, covariates, n_bins=10):
    """
    Computes Shapiro-Wilk test statistics for z-scores stratified by covariate bins.

    The z-scores are grouped into bins based on the values of the covariate, and the
    Shapiro-Wilk test for normality is applied within each bin for every feature.
    The function returns the average Shapiro-Wilk statistic across all bins for each biomarker.

    Parameters
    ----------
    z_scores : numpy.ndarray
        A 2D array of shape (n_samples, n_features) containing the z-scores
        for each subject and feature.
    covariates : numpy.ndarray
        A 1D or 2D array of shape (n_samples,) or (n_samples, 1) containing the covariate
        values used for binning.
    n_bins : int, optional
        The number of equal-width bins to divide the covariate range into. Default is 10.

    Returns
    -------
    numpy.ndarray
        A 1D array of length `n_features`, where each element is the mean
        Shapiro-Wilk test statistic across bins for the corresponding feature.
        NaN is returned for bins with fewer than 3 samples.

    Notes
    -----
    - The Shapiro-Wilk test is only performed for bins with at least 3 samples.
      Bins with fewer samples contribute NaN to the average.
    - The output values range from 0 to 1, where values closer to 1 suggest better
      adherence to a normal distribution.
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


# **
def estimate_centiles(
    processing_dir,
    bio_num,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    batch_sizes=[2, 6],  # e.g., 2 sexes, 6 sites
    age_range=(0, 100),
    point_num=100,
    outputsuffix="estimate",
    save=True,
):
    """
    Estimate centile curves using a normative model for synthetic subjects across batch combinations.

    Parameters
    ----------
    processing_dir : str
        Path to the normative modeling output directory (Models, log, and batch files).
    bio_num : int
        Number of biomarkers or target variables (i.e., number of models to load).
    quantiles : list of float, optional
        List of quantiles to estimate (default is [0.05, 0.25, 0.5, 0.75, 0.95]).
    batch_sizes : list of int, optional
        List indicating number of levels for each batch variable.
        Example: [2, 2] for two binary batch variables (e.g., sex and site).
    age_range : tuple of float, optional
        Age range over which to generate synthetic samples (default is (0, 100)).
    point_num : int, optional
        Number of age points per batch combination (default is 100).
    outputsuffix : str, optional
        Suffix used when loading model output files (default is 'estimate').
    save : bool, optional
        If True, saves the estimated quantiles and synthetic inputs to disk (default is True).

    Returns
    -------
    q : np.ndarray
        Estimated quantile array of shape (N, Q, B) where:
        - N is the number of synthetic points,
        - Q is the number of quantiles,
        - B is the number of biomarkers.
    """
    z_scores = st.norm.ppf(quantiles)

    # Generate all combinations of batch levels
    combinations = list(itertools.product(*[range(size) for size in batch_sizes]))

    # Construct synthetic inputs
    batch_effects = np.repeat(combinations, point_num, axis=0)
    synthetic_X = np.vstack(
        [
            np.linspace(age_range[0], age_range[1], point_num)[:, np.newaxis]
            for _ in range(len(combinations))
        ]
    )

    # Load input scaler from first model
    meta_path = os.path.join(processing_dir, "batch_1", "Models", "meta_data.md")
    with open(meta_path, "rb") as f:
        meta_data = pickle.load(f)

    if meta_data.get("scaler_cov"):
        in_scaler = meta_data["scaler_cov"][0]
        scaled_synthetic_X = in_scaler.transform(synthetic_X)
    else:
        scaled_synthetic_X = synthetic_X / 100

    q = np.zeros((scaled_synthetic_X.shape[0], len(quantiles), bio_num))

    for model_id in range(bio_num):
        model_path = os.path.join(
            processing_dir, f"batch_{model_id + 1}", "Models"
        )  # TODO: it should not go to the batch files, it should go to the Models

        with open(os.path.join(model_path, "meta_data.md"), "rb") as f:
            meta_data = pickle.load(f)

        with open(os.path.join(model_path, f"NM_0_0_{outputsuffix}.pkl"), "rb") as f:
            nm = pickle.load(f)

        q[:, :, model_id] = nm.get_mcmc_quantiles(
            scaled_synthetic_X, batch_effects, z_scores=z_scores
        ).T

        if meta_data.get("scaler_resp"):
            out_scaler = meta_data["scaler_resp"][0]
            for i in range(len(z_scores)):
                q[:, i, model_id] = out_scaler.inverse_transform(q[:, i, model_id])

        print(f"Quantiles for model {model_id} estimated.")

    if save:
        out_path = os.path.join(processing_dir, f"Quantiles_{outputsuffix}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "quantiles": q,
                    "synthetic_X": synthetic_X,
                    "batch_effects": np.array(batch_effects),
                },
                f,
            )

    return q


# **
def prepare_prediction_data(
    data: pd.DataFrame,
    save_path: str,
    covariates: list[str] = ["age"],
    batch_effects: list[str] = None,
    drop_nans: bool = False,
    prefix: str = "",
) -> None:
    """
    Prepares and saves test data (covariates, batch effects, and targets)
    for normative model prediction.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing covariates, batch effects, and target biomarkers.
    save_path : str
        Directory to save the output .pkl files.
    covariates : list of str, optional
        List of column names to be used as covariates (default is ["age"]).
    batch_effects : list of str, optional
        List of column names to be treated as batch effects. If None, a dummy batch column is used.
    drop_nans : bool, optional
        Whether to drop rows containing NaN values (default is False).
    prefix : str, optional
        Prefix for the saved .pkl file names (default is "").

    Saves
    -----
    - {prefix}x_test.pkl : Covariates.
    - {prefix}y_test.pkl : Target values (biomarkers).
    - {prefix}b_test.pkl : Batch effects or dummy batch variable.

    Returns
    -------
    None
    """

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


# **
def cal_stats_for_INOCs(
    q_path: str,
    features: list,
    site_id: int,
    sex_id: int,
    age: float,
    num_of_datasets: int,
    num_points: int = 100,
) -> dict:
    """
    Calculates population statistics (centiles of variation) give a subject age, sex and site.

    Parameters
    ----------
    q_path : str
        Path to the pickled file containing 'quantiles', 'synthetic_X', and 'batch_effects'.
        This is the output of 'estimate_centiles()' function.
    features : list of str
        List of biomarker feature names.
    site_id : int
        Index representing the participant's site. If None, averages across all sites.
    sex_id : int
        Index representing the participant's sex.
    age : float
        Age of the participant.
    num_of_datasets : int
        Number of datasets used to generate quantiles.
    num_points : int, optional
        Number of points for synthetic X axis (default is 100).

    Returns
    -------
    dict
        Dictionary mapping each feature to a list of statistics across quantiles at the given age.
    """
    q = pickle.load(open(q_path, "rb"))
    quantiles = q["quantiles"]
    synthetic_X = (
        q["synthetic_X"].reshape(num_of_datasets * 2, 100).mean(axis=0)
    )  # since Xs are repeated !
    b = q["batch_effects"]

    statistics = {feature: [] for feature in features}
    for ind in range(len(features)):

        biomarker_stats = []
        for quantile_id in range(quantiles.shape[1]):

            if (
                not site_id
            ):  # if not any specific site, average between all sites (batch effect)
                data = quantiles[b[:, 0] == sex_id, quantile_id, ind : ind + 1]
                data = data.reshape(num_of_datasets, num_points, 1)
                data = data.mean(axis=0)
            if site_id:
                data = quantiles[
                    np.logical_and(b[:, 0] == sex_id, b[:, 1] == site_id),
                    quantile_id,
                    ind : ind + 1,
                ]

            data = data.squeeze()

            closest_x = min(synthetic_X, key=lambda x: abs(x - age))
            age_bin_ind = np.where(synthetic_X == closest_x)[0][0]

            biomarker_stats.append(data[age_bin_ind])

        statistics[features[ind]].extend(biomarker_stats)
    return statistics


# **
def abnormal_probability(
    processing_dir: str,
    nm_processing_dir: str,
    n_permutation: int = 1000,
    site_id: int = None,
    healthy_data_prefix: str = "",
    patient_data_prefix: str = "",
):
    """
    Computes the abnormality probability index for both control and patient groups
    based on z-scores from normative modeling. Then calculates the AUC between
    these two groups and estimates the statistical significance of AUC values using
    permutation testing. Finally, it applies false discovery rate (FDR) correction
    to the p-values.

    Parameters
    ----------
    processing_dir : str
        Path to the directory containing z-score files.
    nm_processing_dir : str
        Path to normative modeling directory containing batch info.
    n_permutation : int, optional
        Number of permutations for statistical testing (default is 1000).
    site_id : int, optional
        If provided, filters both healthy and patient data by this site ID.
    healthy_data_prefix : str, optional
        Prefix used for healthy subject files (e.g., 'control').
    patient_data_prefix : str, optional
        Prefix used for patient subject files (e.g., 'patient').

    Returns
    -------
    p_val : np.ndarray
        Adjusted p-values for each biomarker based on FDR correction.
    auc : np.ndarray
        AUC values comparing abnormal probability between groups.
    """

    # Load z-scores
    with open(
        os.path.join(processing_dir, f"Z_{patient_data_prefix}.pkl"), "rb"
    ) as file:
        z_patient = pickle.load(file)
    with open(
        os.path.join(processing_dir, f"Z_{healthy_data_prefix}.pkl"), "rb"
    ) as file:
        z_healthy = pickle.load(file)

    # Filter by site if specified
    if site_id is not None:
        # Control group
        with open(os.path.join(nm_processing_dir, "b_test.pkl"), "rb") as file:
            b_healthy = pickle.load(file)
        z_healthy = z_healthy.iloc[np.where(b_healthy["site"] == site_id)[0], :]

        # Patient group
        with open(
            os.path.join(nm_processing_dir, f"{patient_data_prefix}_b_test.pkl"), "rb"
        ) as file:
            b_patient = pickle.load(file)
        z_patient = z_patient.iloc[np.where(b_patient["site"] == site_id)[0], :]

    # Convert z-scores to abnormal probabilities
    p_patient = z_to_abnormal_p(z_patient)
    p_healthy = z_to_abnormal_p(z_healthy)

    # Combine for AUC analysis
    p = np.concatenate([p_patient, p_healthy])
    # Assign 0 to control group and 1 to patient group as label
    labels = np.concatenate([np.ones(p_patient.shape[0]), np.zeros(p_healthy.shape[0])])

    # Compute AUC and p-values
    auc, p_val = anomaly_detection_auc(p, labels, n_permutation=n_permutation)

    # FDR correction
    p_val = false_discovery_control(p_val)

    return p_val, auc


# **
def aggregate_metrics_across_runs(
    path: str,
    method_name: str,
    biomarker_names: list,
    valcovfile_path: str,
    valrespfile_path: str,  # Corrected semicolon to colon
    valbefile: str,
    metrics: list = ["skewness", "kurtosis", "W", "MACE", "SMSE"],
    num_runs: int = 10,
    quantiles: list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
    outputsuffix: str = "estimate",
    zscore_clipping_value: float = 8.0,
):
    """
    Aggregates statistical metrics across multiple runs for given biomarkers.

    This function evaluates and aggregates 4 statistical metrics, namely skewness, kurtosis, mean absolute
    centiles error (MACE), and W, for a set of biomarkers across multiple runs. The resulting data can be
    used later for plotting. See also: `plot_metrics()`.

    Parameters
    ----------
    path : str
        The directory path containing the individual run folders.
    method_name : str
        The name of the method folder within each run's directory. Since different HBR configurations can
        be saved in each run directory, method_name should be specified.
    biomarker_names : list of str
        A list of biomarker names for which metrics are to be calculated.
    valcovfile_path : str
        The file path to the validation covariance matrix.
    valrespfile_path : str
        The file path to the validation response file.
    valbefile : str
        The file path to the validation bivariate evaluation file.
    metrics : list of str, optional
        A list of metrics to compute for each biomarker. Options include "skewness", "kurtosis", "W", and "MACE".
        Default is ["skewness", "kurtosis", "W", "MACE"].
    num_runs : int, optional
        The number of runs to aggregate metrics across. Default is 10.
    quantiles : list of float, optional
        A list of quantiles to use for MACE evaluation. Default is [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99].
        This tells the function to calculate MACE for these centiles.
    outputsuffix : str, optional
        The suffix to append to output files (e.g., for naming model outputs). Default is "estimate".
    zscore_clipping_value : float, optional
        The maximum z-score value for clipping. Any z-score above this threshold will be clipped to this value. Default is 8.0.
        This is due to the sensitivity of kurtosis to noise. Given that |z| > 8 is almost as equal as |z| = 8, we clip them to 8.

    Returns
    -------
    data : dict
        A dictionary where keys are the metric names (e.g., "skewness", "kurtosis", "W", "MACE") and values
        are dictionaries with biomarker names as keys and lists of aggregated metric values across runs as values.

    Notes
    -----
    The function performs z-score clipping to limit extreme values, applies the skewness and kurtosis calculations,
    evaluates MACE using the provided validation data, and computes the W statistic for the test data.

    Example
    -------
    data = aggregate_metrics_across_runs(
        path='/path/to/runs',
        method_name='method_A',
        biomarker_names=['biomarker_1', 'biomarker_2'],
        valcovfile_path='/path/to/valcovfile',
        valrespfile_path='/path/to/valrespfile',
        valbefile='/path/to/valbefile',
        metrics=['MACE', 'W'],
        num_runs=5
    )
    """
    # Check if all requested metrics are supported
    for elem in metrics:
        if elem not in ["skewness", "kurtosis", "W", "MACE", "SMSE"]:
            raise ValueError(
                f"{elem} is not supported. Supported metrics include 'skewness', 'kurtosis', 'W', 'MACE'."
            )

    data = {
        metric: {biomarker_name: [] for biomarker_name in biomarker_names}
        for metric in metrics
    }

    # Loop through each run
    for run in range(num_runs):
        run_path = path.replace("Run_0", f"Run_{run}")
        valcovfile_path = valcovfile_path.replace("Run_0", f"Run_{run}")
        valrespfile_path = valrespfile_path.replace("Run_0", f"Run_{run}")
        valbefile = valbefile.replace("Run_0", f"Run_{run}")

        # Load z-scores for the current run
        temp_path = os.path.join(run_path, method_name, f"Z_{outputsuffix}.pkl")
        with open(temp_path, "rb") as file:
            z_scores = pickle.load(file)

        # Apply z-score clipping
        z_scores = z_scores.applymap(
            lambda x: zscore_clipping_value if abs(x) > zscore_clipping_value else x
        )

        # Evaluate metrics for the current run
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

            if metric == "SMSE":
                temp_path = os.path.join(
                    run_path, method_name, f"SMSE_{outputsuffix}.pkl"
                )
                with open(temp_path, "rb") as file:
                    smse = pickle.load(file)
                values.extend(smse.iloc[:, 0].tolist())

            if metric == "W":
                with open(os.path.join(run_path, "x_test.pkl"), "rb") as file:
                    cov = pickle.load(file)
                values.extend(shapiro_stat(z_scores, cov))

            if metric == "skewness":
                values.extend(skew(z_scores))

            if metric == "kurtosis":
                values.extend(kurtosis(z_scores))

            # Store values in the data dictionary for each biomarker
            for counter, name in enumerate(biomarker_names):
                data[metric][name].append(values[counter])

    return data


def wilcoxon_rank_test(proposed_dict, baseline_dict):
    """
    Applies the Wilcoxon rank-sum test to compare metric distributions between two model
    configurations across multiple biomarkers. Applies FDR correction
    (Benjamini-Hochberg) to the resulting p-values.

    Parameters
    ----------
    proposed_dict : dict
        Dictionary of metrics for the proposed model configuration.
        Expected format: {metric: {biomarker: list of values}}.

    baseline_dict : dict
        Dictionary of metrics for the baseline model configuration.
        Same format as proposed_dict.

    Returns
    -------
    stat_df : pandas.DataFrame
        DataFrame of Wilcoxon rank-sum test statistics. Rows = metrics, Columns = biomarkers.

    pval_df : pandas.DataFrame
        DataFrame of uncorrected p-values.

    fdr_corrected_df : pandas.DataFrame
        DataFrame of Benjamini-Hochberg FDR-corrected p-values.
    """
    # Dynamically extract metrics and biomarkers
    metrics = list(proposed_dict.keys())
    biomarkers = list(proposed_dict.get(metrics[0]).keys())

    stat_df = pd.DataFrame(index=metrics, columns=biomarkers)
    pval_df = pd.DataFrame(index=metrics, columns=biomarkers)
    raw_pvals = []

    # Compute statistics and collect p-values
    for metric in metrics:
        for biomarker in biomarkers:
            proposed_vals = [
                float(x) for x in proposed_dict.get(metric, {}).get(biomarker, [])
            ]
            baseline_vals = [
                float(x) for x in baseline_dict.get(metric, {}).get(biomarker, [])
            ]

            if proposed_vals and baseline_vals:
                stat, pval = ranksums(proposed_vals, baseline_vals)
                stat_df.at[metric, biomarker] = round(stat, 5)
                pval_df.at[metric, biomarker] = round(pval, 5)
                raw_pvals.append(pval)
            else:
                stat_df.at[metric, biomarker] = np.nan
                pval_df.at[metric, biomarker] = np.nan
                raw_pvals.append(np.nan)

    # Apply FDR correction (Benjamini-Hochberg), ignoring NaNs
    raw_pvals_array = np.array(raw_pvals, dtype=float)
    valid_mask = ~np.isnan(raw_pvals_array)
    corrected = np.full_like(raw_pvals_array, np.nan)
    corrected[valid_mask] = false_discovery_control(raw_pvals_array[valid_mask])

    fdr_corrected_df = pd.DataFrame(
        corrected.reshape(pval_df.shape), index=metrics, columns=biomarkers
    )
    fdr_corrected_df = fdr_corrected_df.round(5)

    return stat_df, pval_df, fdr_corrected_df
