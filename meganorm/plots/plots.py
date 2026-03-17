import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as grid_spec
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm
import nibabel as nib
import plotly.graph_objects as go

from typing import Union
from scipy.stats import chi2
from sklearn.neighbors import KernelDensity
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from sklearn.neighbors import KernelDensity
from typing import Optional
from nilearn import surface, plotting, datasets
from nilearn.surface import PolyMesh, SurfaceImage
from nilearn.datasets import (
    fetch_atlas_surf_destrieux,
    load_fsaverage,
    load_fsaverage_data,
    load_nki,
)
from nilearn.plotting import plot_surf_roi, plot_surf_contours 


# ***
def plot_age_hist(
    df: pd.DataFrame,
    site_names: list,
    save_path: str,
    lower_age_range: int = 5,
    upper_age_range: int = 90,
    step_size: int = 5,
    colors: list = ["#006685", "#591154", "#E84653", "black", "#E6B213", "slategrey"],
) -> None:
    """
    Plot and save a stacked histogram of age distributions across sites.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing participant information. Must include at least two columns:
        - 'site': numeric site identifier (int or category)
        - 'age': age of each participant (in years)
    site_names : list of str
        List of site names to be used as legend labels. The order should correspond to the
        numeric site identifiers in 'df.site'.
    save_path : str
        Path to the directory where the generated plot images will be saved.
    lower_age_range : int, optional
        Minimum age to include in the histogram bins. Default is 5.
    upper_age_range : int, optional
        Maximum age to include in the histogram bins. Default is 90.
    step_size : int, optional
        Width of the histogram bins in years. Default is 5.
    colors : list of str, optional
        List of colors to use for each site's histogram bar. Must be at least as long as `site_names`.
        Default uses a predefined color palette.

    Raises
    ------
    Exception
        If the number of colors is less than the number of site names.

    Returns
    -------
    None

    Saves
    -----
    age_hist.svg : SVG format plot saved in `save_path`.
    age_hist.png : PNG format plot saved in `save_path`.
    """
    if len(site_names) > len(colors):
        raise Exception(
            "The number of colors is less than site_names, please specify a longer list of colors."
        )

    bins = list(range(lower_age_range, upper_age_range, step_size))
    ages = []

    ages = list(
        map(lambda i: df[df["site"] == i]["age"].to_numpy(), range(len(site_names)))
    )

    plt.figure(figsize=(12, 7))
    plt.hist(
        ages,
        bins=bins,
        color=colors,
        edgecolor="black",
        alpha=0.6,
        histtype="barstacked",
        rwidth=0.9,
    )

    # Remove the top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Offset the bottom and left spines
    plt.gca().spines["bottom"].set_position(("outward", 15))
    plt.gca().spines["left"].set_position(("outward", 15))

    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("left")
    plt.grid(axis="y", color="black", linestyle="--")
    plt.grid(axis="x", linestyle="")

    plt.xlabel("Age (years)", fontsize=25)
    plt.legend(site_names, prop={"size": 23}, loc="upper right")
    plt.tick_params(axis="both", labelsize=19)
    plt.xticks(list(range(lower_age_range, upper_age_range, step_size * 2)))
    plt.ylabel("Count", fontsize=25)
    plt.savefig(
        os.path.join(save_path, "age_hist.svg"),
        format="svg",
        dpi=600,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(save_path, "age_hist.png"),
        format="png",
        dpi=600,
        bbox_inches="tight",
    )


# ***
def plot_PNOCs(data: dict, age_slices: list, save_path: str | None) -> None:
    """
    Generate and save population-level NeuroOscilloCharts (PNOCs) showing age-related changes
    in frequency band contributions to overall neural power for males and females.

    This function creates stacked bar plots for each gender, visualizing the relative
    contributions (as percentages) of four frequency bands across defined age bins.
    Mean values and 95% confidence intervals (approximated as 1.96 * std) are displayed.
    The resulting figure is saved in SVG and PNG formats if a `save_path` is provided.

    Parameters
    ----------
    data : dict
        A nested dictionary of the form:
        {
            'Male': {
                'delta': list of [mean, std],
                'theta': list of [mean, std],
                ...
            },
            'Female': {
                ...
            }
        }
        Each list should have one [mean, std] pair per age slice. This can be calculated using
        meganorm.nm.calculate_PNOCs function.

    age_slices : list of int
        List of starting ages for each 5-year bin used as x-axis labels (e.g., [5, 10, 15, ..., 75]).

    save_path : str or None
        Directory path to save the generated figure. If None, the plot is not saved to disk.

    Returns
    -------
    None
        Displays the plot and, if `save_path` is not None, saves the following files:
        - 'Chrono-NeuroOscilloChart.svg'
        - 'Chrono-NeuroOscilloChart.png'
    """

    # Age ranges
    ages = [f"{i}-{i+5}" for i in age_slices]

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    def plot_gender_data(ax, gender_data, title, legend=False, colors=None):

        means = {k: [item[0] * 100 for item in v] for k, v in gender_data.items()}
        stds = {k: [item[1] * 100 * 1.96 for item in v] for k, v in gender_data.items()}

        df_means = pd.DataFrame(means, index=ages)
        df_stds = pd.DataFrame(stds, index=ages)

        my_cmap = ListedColormap(colors, name="my_cmap")

        bar_plot = df_means.plot(
            kind="bar",
            yerr=df_stds,
            capsize=4,
            stacked=True,
            ax=ax,
            alpha=0.6,
            colormap=my_cmap,
        )
        for p in bar_plot.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            bar_plot.text(
                x + width / 2,
                y + height / 2 + 2,
                f"{height:.0f}%",
                ha="center",
                va="center",
                fontsize=14,
            )
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Age ranges (years)", fontsize=16)
        if legend:
            ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
        else:
            ax.get_legend().remove()

        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", labelsize=20)
        ax.set_yticklabels([])

    plot_gender_data(
        axes[0],
        data["Male"],
        "Males' Chrono-NeuroOscilloChart",
        colors=["orange", "teal", "olive", "tomato"],
    )

    plot_gender_data(
        axes[1],
        data["Female"],
        "Females' Chrono-NeuroOscilloChart",
        legend=False,
        colors=["orange", "teal", "olive", "tomato"],
    )

    axes[1].set_xlabel("Age ranges (years)", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "Chrono-NeuroOscilloChart.svg"), dpi=600)
        plt.savefig(os.path.join(save_path, "Chrono-NeuroOscilloChart.png"), dpi=600)
    plt.show()


# ***
def plot_growthchart(
    age_vector: list,
    centiles_matrix: list,
    cut: int = 0,
    idp: str = "",
    save_path: str | None = None,
    colors: dict | None = None,
    centiles_name: list = ["5th", "25th", "50th", "75th", "95th"],
) -> None:
    """
    Plot a growth chart for a given f-IDP, showing centile trajectories by sex across age.

    This function generates a two-panel plot (males and females), visualizing centile curves
    over age for a given imaging-derived phenotype (IDP). Optional age compression above a
    specified cutoff is applied for better visualization. Shaded regions denote inter-centile
    ranges (5th–95th and 25th–75th).

    Parameters
    ----------
    age_vector : numpy.ndarray
        1D array of age values corresponding to the centile data.
    centiles_matrix : numpy.ndarray
        3D array of shape (n_ages, n_centiles, 2), where axis 2 indexes sex
        (0 = male, 1 = female), and axis 1 contains ordered centiles.
    cut : int, optional
        Age cutoff after which ages are compressed for visualization (default is 0, no compression).
    idp : str, optional
        Name of the biomarker or phenotype to be displayed as the y-axis label and used in the filename.
    save_path : str or None, optional
        If specified, saves the figure as an SVG file in the given directory. If None, only displays the plot.
    colors : dict, optional
        Dictionary with 'male' and 'female' keys, each mapping to a list of color hex codes
        for the centile lines. If None, a default color scheme is used.
    centiles_name : list of str, optional
        List of string labels for each centile (e.g., ['5th', '25th', '50th', '75th', '95th']).

    Returns
    -------
    None
        Displays the plot and, if `save_path` is provided, saves the following file:
        - '<idp>_growthchart.svg'
    """
    if colors is None:
        colors = {
            "male": ["#4c061d", "#662333", "#803449", "#993d5e", "#b34e74"],
            "female": ["#FF6F00", "#FF8C1A", "#FFA726", "#FFB74D", "#FFD54F"],
        }

    min_age = age_vector.min()
    max_age = age_vector.max()

    def age_transform(age):  # Age transformation based on cut
        if age < cut:
            return age
        else:
            return cut + (age - cut) / 3

    transformed_age = np.array([age_transform(age) for age in age_vector])

    fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)

    genders = ["male", "female"]

    for j, gender in enumerate(genders):
        for i in range(len(centiles_name)):
            linestyle = "-" if i in [1, 2, 3] else "--"
            linewidth = 5 if i == 2 else (3 if i in [1, 3] else 2)
            axes[j].plot(
                transformed_age,
                centiles_matrix[:, i, j],
                label=f"{centiles_name[i]} Percentile",
                linestyle=linestyle,
                color=colors[gender][i],
                linewidth=linewidth,
                alpha=1 if i == 2 else 0.8,
            )

        axes[j].fill_between(
            transformed_age,
            centiles_matrix[:, 0, j],
            centiles_matrix[:, 4, j],
            color=colors[gender][2],
            alpha=0.2,
        )
        axes[j].fill_between(
            transformed_age,
            centiles_matrix[:, 1, j],
            centiles_matrix[:, 3, j],
            color=colors[gender][2],
            alpha=0.2,
        )

        transformed_ticks = [
            age_transform(age)
            for age in np.concatenate(
                (
                    np.arange(min_age, cut + 1, 2, dtype=int),
                    np.arange(np.ceil((cut + 1) / 10) * 10, max_age + 1, 10, dtype=int),
                )
            )
        ]
        axes[j].set_xticks(transformed_ticks)
        axes[j].set_xticklabels(
            np.concatenate(
                (
                    np.arange(min_age, cut + 1, 2, dtype=int),
                    np.arange(np.ceil((cut + 1) / 10) * 10, max_age + 1, 10, dtype=int),
                )
            ),
            fontsize=37,
        )
        axes[j].tick_params(axis="both", labelsize=45)
        axes[j].grid(True, which="both", linestyle="--", linewidth=2, alpha=0.95)
        axes[j].spines["top"].set_visible(False)
        axes[j].spines["right"].set_visible(False)
        # axes[j].set_xlabel('Age (years)', fontsize=28)

        for i, label in enumerate(centiles_name):
            axes[j].annotate(
                label,
                xy=(transformed_age[-1], centiles_matrix[-1, i, j]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=46,
                color=colors[gender][i],
                fontweight="bold",
            )

    axes[0].set_ylabel(idp, fontsize=28)
    # axes[0].set_title("Males", fontsize=28)
    # axes[1].set_title("Females", fontsize=28)

    plt.tight_layout(pad=2)

    if save_path is not None:
        plt.savefig(
            os.path.join(save_path, idp.replace(" ", "_") + "_growthchart.svg"), dpi=600
        )


# ***
def plot_growthcharts(
    path,
    model_indices: list,
    biomarker_names: list,
    site: int = None,
    point_num: int = 100,
    number_of_sexs: int = 2,
    num_of_sites: int = None,
    centiles_name: list = ["5th", "25th", "50th", "75th", "95th"],
    colors: dict = None,
    suffix: str = "",
    save_path: str | None = None,
):
    """
    Generate and save growth charts for multiple biomarkers using precomputed quantile estimates.

    Parameters
    ----------
    path : str
        Directory containing 'Quantiles_estimate.pkl'.
    model_indices : list of int
        Indices of the models/biomarkers to plot.
    biomarker_names : list of str
        Descriptive names matching model_indices.
    site : int, optional
        If specified, selects only data from this site. Not yet implemented.
    point_num : int, optional
        Number of synthetic X points to use (default 100).
    number_of_sexs : int, optional
        Number of sexes (default 2).
    num_of_sites : int, optional
        If averaging across sites, specify how many.
    centiles_name : list of str
        Labels for centiles (e.g., ['5th', ..., '95th']).
    colors : dict, optional
        Color dictionary with 'male' and 'female' keys.
    suffix : str, optional
        Suffix of the saved quantile identifying which output file to use. Default is 'estimate'.
    save_path:
        Where to save figures, if not None.
    Returns
    -------
    None

    """
    temp = pickle.load(open(os.path.join(path, f"Quantiles_{suffix}.pkl"), "rb"))

    q = temp["quantiles"]
    x = temp["synthetic_X"]
    b = temp["batch_effects"]

    for i, idp in enumerate(model_indices):

        if not site:
            data = np.concatenate(
                [q[b[:, 0] == 0, :, idp : idp + 1], q[b[:, 0] == 1, :, idp : idp + 1]],
                axis=2,
            )
            data = data.reshape(
                num_of_sites, point_num, len(centiles_name), number_of_sexs
            )
            data = data.mean(axis=0)
        if site:
            raise ValueError(f"still not implmented")
            # TODO

        plot_growthchart(
            x[0:point_num].squeeze(),
            data,
            cut=0,
            idp=biomarker_names[i],
            save_path=save_path,
            centiles_name=centiles_name,
            colors=colors,
        )


# ***
def plot_INOCs(
    sub_index: int | str,
    observed_value: float,
    q1: float,
    q3: float,
    percentile_5: float,
    percentile_95: float,
    percentile_50: float,
    title: str = "Quantile-Based Gauge",
    min_value: float = 0,
    max_value: float = 1,
    show_legend: bool = False,
    bio_name: str | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plots individual-level NeuroOscilloChart (INOCs) showing where an individual's
    measurements stands in relation to the corresponding population.

    Parameters
    ----------
    sub_index : int or str
        Unique identifier for the subject (used for file naming).
    observed_value : float
        Observed biomarker value.
    q1 : float
        25th percentile value.
    q3 : float
        75th percentile value.
    percentile_5 : float
        5th percentile value.
    percentile_95 : float
        95th percentile value.
    percentile_50 : float
        50th percentile (median) value. Used as reference for delta.
    title : str, optional
        Title for the gauge plot. Default is "Quantile-Based Gauge".
    min_value : float, optional
        Minimum value of the gauge range. Default is 0.
    max_value : float, optional
        Maximum value of the gauge range. Default is 1.
    show_legend : bool, optional
        Whether to display the legend. Default is False.
    bio_name : str or None, optional
        Name of the biomarker (used in title and file naming).
    save_path : str or None, optional
        Directory to save the output images. If None, the plot is not saved.

    Returns
    -------
    None
        Displays and optionally saves a gauge chart indicating where the current
        biomarker value lies within the distribution.
    """
    observed_value = round(observed_value, 3)
    if bio_name == "Gamma":
        max_value = 0.1

    # Determine color based on value position
    if observed_value < percentile_5:
        value_color = "rgb(8, 65, 92)"  # Extremely low
    elif observed_value < q1:
        value_color = "rgb(0, 191, 255)"  # Below normal
    elif observed_value <= q3:
        value_color = "rgb(129, 193, 75)"  # Normal
    elif observed_value <= percentile_95:
        value_color = "rgb(255, 201, 20)"  # Above normal
    else:
        value_color = "rgb(188, 44, 26)"  # Extremely high

    number_font_size = 75 if show_legend else 120
    delta_font_size = 30 if show_legend else 90

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=observed_value,
            number={
                "font": {
                    "size": number_font_size,
                    "family": "Arial",
                    "color": value_color,
                }
            },
            delta={
                "reference": percentile_50,
                "position": "top",
                "font": {"size": delta_font_size},
                "valueformat": ".3f",
            },
            gauge={
                "axis": {
                    "range": [min_value, max_value],
                    "tickfont": {"size": 60, "family": "Arial", "color": "black"},
                    "showticklabels": True,
                    "tickwidth": 12,
                    "tickcolor": "black",
                    "tickvals": [
                        round(min_value + i * (max_value - min_value) / 10, 2)
                        for i in range(11)
                    ],
                },
                "bar": {
                    "color": "rgb(255, 255, 255)",
                    "line": {"color": "black", "width": 3},
                },
                "steps": [
                    {"range": [min_value, percentile_5], "color": "rgb(8, 65, 92)"},
                    {"range": [percentile_5, q1], "color": "rgb(0, 191, 255)"},
                    {"range": [q1, q3], "color": "rgb(129, 193, 75)"},
                    {"range": [q3, percentile_95], "color": "rgb(255, 201, 20)"},
                    {"range": [percentile_95, max_value], "color": "rgb(188, 44, 26)"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 6},
                    "thickness": 0.75,
                    "value": percentile_50,
                },
            },
            title={
                "text": bio_name,
                "font": {"size": 70, "family": "Arial", "color": "black"},
            },
        )
    )

    # Legend (as fake traces)
    if show_legend:
        colors = [
            ("rgb(8, 65, 92)", "0-5th Percentile (Extremely Low)"),
            ("rgb(0, 191, 255)", "5th-25th Percentile (Below Normal)"),
            ("rgb(129, 193, 75)", "25th-75th Percentile (Normal)"),
            ("rgb(255, 201, 20)", "75th-95th Percentile (Above Normal)"),
            ("rgb(188, 44, 26)", "95th-100th Percentile (Extremely High)"),
        ]
        for color, label in colors:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=color),
                    name=label,
                )
            )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=50, b=100 if show_legend else 30, l=100, r=130),
        showlegend=show_legend,
        width=1000,
        height=800,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    plt.tight_layout()

    # Save if a valid path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.write_image(os.path.join(save_path, f"{sub_index}_{bio_name}.svg"))
        fig.write_image(os.path.join(save_path, f"{sub_index}_{bio_name}.png"))

    plt.show()


# ***
def plot_nm_ranges_with_marker(
    processing_dir: str,
    data_dir: str,
    quantiles: list = [0.05, 0.25, 0.5, 0.75, 0.95],
    save_plot: bool = True,
    suffix: str = "",
    batch_curve: dict = {0: ["Male", "Female"]},
    batch_marker: dict = {1: ["BTH", "Cam-Can", "NIMH", "OMEGA", "HCP", "MOUS"]},
    new_names: list = ["Theta", "Alpha", "Beta", "Gamma"],
    colors: list = ["#006685", "#591154", "#E84653", "black", "#E6B213", "Slategrey"],
) -> None:
    """
    Plot normative centile curves and overlay test sample data with marker-based group labels.

    This function visualizes normative model quantiles (e.g., 5th, 25th, 50th, 75th, 95th) across age
    for multiple biomarkers, averaged across sites and stratified by a curve batch (e.g., sex).
    Test data is overlaid as scatter points, grouped by an independent batch (e.g., acquisition site),
    with customizable colors and markers. One plot is produced per biomarker.

    Parameters
    ----------
    processing_dir : str
        Path to the directory containing normative model outputs (e.g., Quantiles_<suffix>.pkl).
    data_dir : str
        Path to the directory containing test data files (`x_test.pkl`, `y_test.pkl`, and `b_test.pkl`).
    quantiles : list of float, optional
        List of quantile levels to plot (default is [0.05, 0.25, 0.5, 0.75, 0.95]).
    save_plot : bool, optional
        Whether to save the generated plots to disk (default is True).
    suffix : str, optional
        Suffix to identify which Quantiles_<suffix>.pkl file to load in `processing_dir`.
    batch_curve : dict
        Dictionary mapping the index used for curve stratification (e.g., sex) to a list of labels.
        Example: {0: ["Male", "Female"]}.
    batch_marker : dict
        Dictionary mapping the index used for marker-based grouping (e.g., site) to a list of labels.
        Example: {1: ['BTH', 'Cam-Can', "NIMH", ...]}.
    new_names : list of str, optional
        List of strings used as display names for each biomarker plotted (default is ['Theta', 'Alpha', ...]).
    colors : list of str, optional
        List of hex color codes used for marker groups defined in `batch_marker`.

    Returns
    -------
    None
        Displays the plots for each biomarker. If `save_plot` is True, saves each plot to:
        - <processing_dir>/Figures_experiment/{biomarker}.svg
        - <processing_dir>/Figures_experiment/{biomarker}.png
    """
    matplotlib.rcParams["pdf.fonttype"] = 42

    # Load data
    x_test = pickle.load(open(os.path.join(data_dir, "x_test.pkl"), "rb")).to_numpy(
        float
    )
    y_test_df = pickle.load(open(os.path.join(data_dir, "y_test.pkl"), "rb"))
    b_test = pickle.load(open(os.path.join(data_dir, "b_test.pkl"), "rb")).to_numpy(
        float
    )

    temp = pickle.load(
        open(os.path.join(processing_dir, f"Quantiles_{suffix}.pkl"), "rb")
    )
    q = temp["quantiles"]
    synthetic_X = temp["synthetic_X"]
    quantiles_be = temp["batch_effects"]

    # Convert age values to original space
    x_test *= 100

    # Identify batch indices
    curve_idx = list(batch_curve.keys())[0]
    marker_idx = list(batch_marker.keys())[0]
    curve_labels = batch_curve[curve_idx]
    marker_labels = batch_marker[marker_idx]

    z_scores = st.norm.ppf(quantiles)
    markers = ["o", "^"]
    curve_colors = ["#6E750E", "#A9561E"]

    num_biomarkers = q.shape[2]

    for ind in range(num_biomarkers):
        bio_name = y_test_df.columns[ind]
        y_test = y_test_df[[bio_name]].to_numpy(float)

        fig, ax = plt.subplots(figsize=(8, 6), sharex=True, sharey=True)

        # Scatter test data
        for m in np.unique(b_test[:, marker_idx]):
            for c in np.unique(b_test[:, curve_idx]):
                ts_idx = np.logical_and(
                    b_test[:, marker_idx] == m, b_test[:, curve_idx] == c
                )

                ax.scatter(
                    x_test[ts_idx],
                    y_test[ts_idx],
                    s=35,
                    alpha=0.6,
                    label=f"{curve_labels[int(c)]} {marker_labels[int(m)]}",
                    color=colors[int(m)],
                    marker=markers[int(c)],
                    edgecolors="none",
                )

        # Plot quantile curves
        for c in np.unique(b_test[:, curve_idx]):
            q_idx = np.where(quantiles_be[:, curve_idx] == c)[0]

            for i, v in enumerate(z_scores):
                linestyle = "-" if v == 0 else "--"
                thickness = 3 if v == 0 else 1

                x = synthetic_X[q_idx].reshape(-1, 100).mean(axis=0)
                y = q[q_idx, i : i + 1, ind].reshape(-1, 100).mean(axis=0)

                ax.plot(
                    x.tolist(),
                    y.tolist(),
                    linewidth=thickness,
                    linestyle=linestyle,
                    color=curve_colors[int(c)],
                    alpha=1,
                )

        # Formatting
        ax.grid(True, linewidth=0.5, alpha=0.5, linestyle="--")
        ax.set_ylabel(new_names[ind], fontsize=25)
        ax.set_xlabel("Age (years)", fontsize=25)
        ax.tick_params(axis="both", which="major", labelsize=22)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if ind + 1 == num_biomarkers:
            ax.legend(loc="upper right", prop={"size": 17}, ncol=2)

        plt.tight_layout()

        if save_plot:
            save_path = os.path.join(processing_dir, "Figures_experiment")
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"{ind}_{bio_name}.svg"), dpi=300)
            plt.savefig(os.path.join(save_path, f"{ind}_{bio_name}.png"), dpi=300)


# ***
def box_plot_auc(
    df: pd.DataFrame,
    save_path: str,
    color: Union[str, list] = "teal",
    alpha: float = 0.7,
    biomarkers_new_name: list = None,
) -> None:
    """
    Creates and saves a boxplot with stripplot overlay showing AUC values for different biomarkers.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame where each column corresponds to a biomarker, and each row is an AUC value
        from a model run.
    save_path : str
        Directory path where the resulting plots will be saved. The function creates the
        directory if it doesn't exist.
    color : str or list of str, optional
        Color(s) for the boxplots. If a single string, the same color is used for all boxes.
        If a list, its length must match the number of biomarkers.
    alpha : float, optional
        Transparency of the box colors (0.0 to 1.0). Default is 0.7.
    biomarkers_new_name : list of str, optional
        If provided, replaces the default column names (biomarker names) with these labels
        for display on the x-axis. Must be the same length as the number of DataFrame columns.

    Returns
    -------
    None
        The function saves the following plots in `save_path`:
        - AUCs.svg (SVG format)
        - AUCs.png (SVG format)
    """
    if biomarkers_new_name:
        if len(biomarkers_new_name) != len(df.columns):
            raise ValueError(
                "Length of 'biomarkers_new_name' must match number of columns in df."
            )
        df.columns = biomarkers_new_name

    data_long = pd.melt(df)

    if isinstance(color, str):
        palette = [color] * len(df.columns)
    elif isinstance(color, list):
        if len(color) != len(df.columns):
            raise ValueError(
                "If 'color' is a list, it must match the number of biomarkers."
            )
        palette = color
    else:
        raise TypeError("'color' must be a string or a list of strings.")

    sns.set_theme(style="ticks")
    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(
        x="variable", y="value", data=data_long, palette=palette, showfliers=False
    )

    # Apply alpha to each PathPatch (box area)
    num_boxes = len(df.columns)
    for i, patch in enumerate(ax.patches[:num_boxes]):
        patch.set_facecolor(palette[i])
        patch.set_alpha(alpha)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)

    # Add stripplot
    sns.stripplot(
        x="variable",
        y="value",
        data=data_long,
        color="black",
        size=6,
        alpha=0.6,
        jitter=True,
    )

    sns.despine(offset=0, trim=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.ylabel("AUC", fontsize=20)
    plt.xlabel("")
    plt.grid(True, linestyle="--", linewidth=1, alpha=1)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "AUCs.svg"), dpi=600, format="svg")
        plt.savefig(os.path.join(save_path, "AUCs.png"), dpi=600, format="png")


# ***
def z_scores_scatter_plot(
    X,
    Y,
    bands_name=["Theta", "Beta"],
    lower_lim: float = -4.0,
    upper_lim: float = 4.0,
    ticks: list = None,
    box_z_values: list = [0.674, 1.645],
    box_colors: list = ["#a0a0a0", "#202020"],
    save_path: str = None,
):
    """
    Generates a 2D scatter plot of z-scores between two frequency bands, with optional
    contour boxes highlighting specific z-score thresholds.

    Parameters
    ----------
    X : array-like
        Z-score values for the x-axis (e.g., corresponding to the first frequency band).
    Y : array-like
        Z-score values for the y-axis (e.g., corresponding to the second frequency band).
    bands_name : list of str, optional
        Names of the two frequency bands to label the x and y axes. Default is ['Theta', 'Beta'].
    lower_lim : float, optional
        Lower limit for both axes. Default is -4.0.
    upper_lim : float, optional
        Upper limit for both axes. Default is 4.0.
    ticks : list of float, optional
        Custom tick positions for both axes. If None, ticks are automatically determined.
    box_z_values : list of float, optional
        Z-score thresholds at which to draw square boundary boxes. Each value defines a
        centered square extending from -z to +z on both axes. Must be the same length as `box_colors`.
    box_colors : list of str, optional
        Colors of the square boundary boxes. Must match the length of `box_z_values`.
    save_path : str, optional
        Directory path to save the plot. If provided, saves the figure as
        'z_scores_plot.svg' and 'z_scores_plot.png'.

    Returns
    -------
    None
        The plot is shown or saved depending on the `save_path` argument.

    Raises
    ------
    ValueError
        If `box_z_values` and `box_colors` are not the same length.
    """
    if len(box_z_values) != len(box_colors):
        raise ValueError("Length of 'box_z_values' and 'box_colors' must be equal.")

    X = np.asarray(X)
    Y = np.asarray(Y)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)

    # Compute point sizes relative to X values
    sizes = 20 + (X - np.min(X)) / (np.max(X) - np.min(X)) * 500

    # Scatter plot (size ~ X, color ~ Y)
    scatter = ax.scatter(
        X,
        Y,
        s=sizes,
        c=Y,
        cmap="inferno_r",
        edgecolor="black",
        alpha=0.8,
        vmin=np.min(Y),
        vmax=np.max(Y),
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap="inferno_r", norm=plt.Normalize(vmin=np.min(Y), vmax=np.max(Y))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label(f"{bands_name[1].capitalize()} z-scores", fontsize=20)
    cbar.ax.tick_params(labelsize=0, length=0)

    # Axis labels
    ax.set_xlabel(f"{bands_name[0].capitalize()} z-scores", fontsize=22)
    ax.set_ylabel(f"{bands_name[1].capitalize()} z-scores", fontsize=22)

    # Clean up spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))

    # Draw filled boxes and borders
    for i in range(len(box_z_values)):
        bound = list(reversed(box_z_values))[i]
        ax.add_patch(
            plt.Rectangle(
                (-bound, -bound),
                2 * bound,
                2 * bound,
                color=box_colors[i],
                alpha=0.4,
                ec=None,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (-bound, -bound),
                2 * bound,
                2 * bound,
                fill=False,
                edgecolor="black",
                linewidth=3,
            )
        )

    # Axis ticks
    if ticks:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    else:
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
    ax.tick_params(axis="both", labelsize=18)

    plt.tight_layout()

    # Save if path is specified
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        base_name = f"{bands_name[0].capitalize()}_{bands_name[1].capitalize()}_z_scores_scatter"
        plt.savefig(os.path.join(save_path, f"{base_name}.svg"), dpi=600, format="svg")
        plt.savefig(os.path.join(save_path, f"{base_name}.png"), dpi=600, format="png")


def z_scores_contour_plot(
    X, Y, bands_name, percentiles=[0.05, 0.25, 0.50, 0.75, 0.95], save_path=None
):
    "scatterplot of patient Z-scores with 75th, and 95th percentile"

    # define range from -4 to 4
    delta = 0.025
    x = np.arange(-4.0, 4.0, delta)
    y = np.arange(-4.0, 4.0, delta)
    xx, yy = np.meshgrid(x, y)

    Z_magnitude = np.sqrt(xx**2 + yy**2)

    # Compute contour levels
    # Compute Mahalanobis distances for each percentile, bc multivariate normal distribution
    # bivariate normal distribution, the Mahalanobis distance follows a chi-squared distribution with 2 degrees of freedom
    thr = [np.sqrt(chi2.ppf(p, df=2)) for p in percentiles]
    levels = thr

    # contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contour(
        xx,
        yy,
        Z_magnitude,
        levels=levels,
        colors="black",
        linewidths=2,
        linestyles="dashed",
    )

    # Labels
    percentile_labels = [f"{int(p * 100)}th" for p in percentiles]
    fmt = {
        thr[i]: percentile_labels[i] for i in range(len(thr))
    }  # Map contour levels to labels
    ax.clabel(contour, fmt=fmt, fontsize=10)

    # Scatter plot of clinical data
    color_values = np.sqrt(
        np.array(X) ** 2 + np.array(Y) ** 2
    )  # Euclidean distance of Z-scores
    norm = mcolors.Normalize(vmin=0, vmax=3.5)
    scatter = plt.scatter(
        X,
        Y,
        c=color_values,
        cmap="coolwarm",
        norm=norm,
        edgecolors="black",
        linewidth=0.2,
    )
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ticks = [-3, -2, -1, 0, 1, 2, 3]
    plt.xticks(ticks)
    plt.yticks(ticks)

    # Style the plot
    ax.grid(alpha=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Magnitude of Z-scores")

    # Labels & title
    plt.xlabel(f"{bands_name[0]} z-scores", fontsize=16)
    plt.ylabel(f"{bands_name[1]} z-scores", fontsize=16)
    ax.set_title("Z-scores contour plot")

    plt.tight_layout()
    if save_path:
        plt.savefig(
            os.path.join(save_path, "z_scores_contour.svg"), dpi=600, format="svg"
        )

    plt.show()


# ***
def plot_metrics(
    metrics_path: list,
    which_biomarkers: list,
    biomarkers_new_name: list = None,
    colors: list = None,
    save_path: str = None,
    which_metrics: list = ["skewness", "kurtosis", "W", "SMSE", "MACE"],
):
    """
    Plots KDE distributions of metrics across biomarkers and models.

    Parameters
    ----------
    metrics_path : list of str
        List of file paths to pickle files containing aggregated metrics.
        If you want to compare multiple models, you can pass multiple paths in a list
        to plot them together. Otherwise, one single path in a list is enough.
    which_biomarkers : list of str
        Biomarker names to include in the plots.
    biomarkers_new_name : list of str, optional
        New names for the biomarkers (for labeling). Must match length of which_biomarkers.
    colors : list of str, optional
        Colors for each metrics_path (used in KDE plots).
    save_path : str, optional
        Directory to save the plots. If None, plots will not be saved.
    whic_metrics: list, optional
        Which metrics to be shown. Default: ['SMSE', 'skewness', 'kurtosis', "W", "MACE"]

    Returns
    -------
    None
    """

    sns.set_theme(style="ticks", palette="pastel")

    n_rows = len(which_biomarkers)
    n_cols = len(which_metrics)
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )

    for i, path in enumerate(metrics_path):
        with open(path, "rb") as file:
            metrics_dic = pickle.load(file)

        # Keep only the ordered metrics
        metrics_dic = {k: metrics_dic[k] for k in which_metrics if k in metrics_dic}

        for col_idx, metric in enumerate(which_metrics):
            if metric not in metrics_dic:
                continue

            # Subset and rename biomarker columns
            df_temp = pd.DataFrame(metrics_dic[metric])[which_biomarkers]
            if biomarkers_new_name:
                df_temp.columns = biomarkers_new_name
            else:
                biomarkers_new_name = which_biomarkers

            for row_idx, biomarker in enumerate(df_temp.columns):
                values = df_temp[biomarker]
                current_ax = ax[row_idx, col_idx]

                if colors:
                    sns.kdeplot(
                        values, ax=current_ax, fill=True, color=colors[i], alpha=0.6
                    )
                else:
                    sns.kdeplot(values, ax=current_ax, fill=True, alpha=0.6)

                sns.rugplot(values, ax=current_ax, color="black", height=0.05)

                current_ax.set_yticks([])
                current_ax.spines["top"].set_visible(False)
                current_ax.spines["right"].set_visible(False)
                current_ax.spines["left"].set_visible(False)

                # Set titles and labels only on first row/col
                if row_idx == 0:
                    current_ax.set_title(metric, fontsize=14)
                else:
                    current_ax.set_title("")

                if col_idx == 0:
                    current_ax.set_ylabel(biomarker.capitalize(), fontsize=14)
                else:
                    current_ax.set_ylabel("")

                current_ax.set_xlabel("")
                current_ax.tick_params(axis="both", labelsize=25)

    plt.tight_layout(h_pad=1.5, w_pad=1.0)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "metric.svg"), dpi=600, format="svg")
        plt.savefig(os.path.join(save_path, "metric.png"), dpi=600, format="png")

    plt.show()


# ***
def qq_plot(
    processing_dir: str,
    save_fig: str,
    label_dict: dict,
    colors: list,
    markersize: int = 8,
    alpha: float = 0.6,
    lower_lim: float = -4.0,
    upper_lim: float = 4.0,
    prefix: str = "estimate",
):
    """
    Generate QQ plots for estimated Z-scores across multiple biomarkers.

    This function loads Z-score data from multiple directories and creates QQ plots
    comparing the sample quantiles to the theoretical quantiles. Each directory should
    contain a pickle file with Z-score data, named using the provided prefix.
    One QQ plot is generated per biomarker (as specified in `label_dict`).

    Parameters
    ----------
    processing_dir : list of str
        List of directories containing Z-score pickle files (`Z_<prefix>.pkl`).
        To compare multiple models, provide multiple file paths in the list.
        To plot results from a single model, provide a list containing one path.
    save_fig : str or None
        Directory where generated plots will be saved. If None, plots are not saved.
    label_dict : dict
        Dictionary mapping biomarker names (str) to column indices (int) in the Z-score DataFrames.
    colors : list of str
        List of color codes for each dataset, used for plotting.
        Must be at least as long as the number of entries in `processing_dir`.
    markersize : int, optional
        Size of the plot markers. Default is 8.
    alpha : float, optional
        Transparency level of the markers (0.0 to 1.0). Default is 0.6.
    lower_lim : float, optional
        Lower bound for both x and y axes in the QQ plot. Default is -4.0.
    upper_lim : float, optional
        Upper bound for both x and y axes in the QQ plot. Default is 4.0.
    prefix : str, optional
        Prefix used in the Z-score pickle file name (e.g., "Z_<prefix>.pkl"). Default is "estimate".

    Returns
    -------
    None
        This function does not return any values. It generates and optionally saves plots.

    Raises
    ------
    ValueError
        If `processing_dir` is not a list of strings or if `colors` has fewer elements than `processing_dir`.
    FileNotFoundError
        If a required Z-score pickle file is missing in any specified directory.
    """
    if not isinstance(processing_dir, list) or not all(
        isinstance(p, str) for p in processing_dir
    ):
        raise ValueError("processing_dir must be a list of strings (paths).")

    if len(colors) < len(processing_dir):
        raise ValueError("Not enough colors provided for the number of directories.")

    z_scores_dict = {}
    for path in processing_dir:
        file_path = os.path.join(path, f"Z_{prefix}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        with open(file_path, "rb") as f:
            z_scores_dict[path] = pickle.load(f)

    for label_name, col_index in label_dict.items():
        plt.figure(figsize=(5, 5))
        ax = plt.gca()

        # Plot theoretical line once per figure
        x = np.linspace(lower_lim, upper_lim, 100)
        ax.plot(x, x, color="black", linewidth=2, linestyle="--", alpha=0.9)

        # Plot each dataset's QQ points
        for idx, (path, z_scores) in enumerate(z_scores_dict.items()):
            plotkwargs = {
                "markerfacecolor": colors[idx],
                "markeredgecolor": colors[idx],
                "markersize": markersize,
                "alpha": alpha,
            }

            sm.qqplot(
                z_scores.iloc[:, col_index].to_numpy(), line=None, ax=ax, **plotkwargs
            )

        # Styling
        ax.set_xlim(lower_lim, upper_lim)
        ax.set_ylim(lower_lim, upper_lim)
        ax.set_xlabel("Theoretical quantiles", fontsize=25)
        ax.set_ylabel("Sample quantiles", fontsize=25)
        ax.set_title(label_name.capitalize(), fontsize=22)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_position(("outward", 10))
        ax.spines["left"].set_position(("outward", 10))

        plt.xticks(np.linspace(lower_lim, upper_lim, 5), fontsize=25)
        plt.yticks(np.linspace(lower_lim, upper_lim, 5), fontsize=25)
        plt.grid(True, linestyle="--", color="gray", alpha=0.4)

        # Save if needed
        if save_fig:
            os.makedirs(save_fig, exist_ok=True)
            plt.savefig(
                os.path.join(save_fig, f"{label_name}_qqplot.png"),
                dpi=600,
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(save_fig, f"{label_name}_qqplot.svg"),
                dpi=600,
                bbox_inches="tight",
            )

        plt.show()


# ***
def plot_extreme_deviation(
    base_path: str,
    len_runs: int,
    save_path: str,
    healthy_prefix: str,
    patient_prefix: str,
    legend: list,
    method: str,
    site_id: list = None,
    new_col_name: list = None,
    y_upper_lim: float = 15,
    y_lower_lim: float = 0,
):
    """
    Computes and plots extreme deviation statistics (|Z|>2) across biomarkers.
    This is done by averaging statistics across multiple runs and comparing
    patient and healthy groups. Two separate plots are generated: one for
    positive deviations and one for negative deviations.

    Parameters
    ----------
    base_path : str
        Path template pointing to the directory containing run subfolders. This should include
        the placeholder "Run_0", which will be programmatically replaced with the actual run indices.
    len_runs : int
        Number of runs (iterations) to include in the averaging process.
    save_path : str
        Directory path where the resulting plots will be saved.
    healthy_prefix : str
        Filename prefix identifying the Z-score files for healthy participants.
    patient_prefix : str
        Filename prefix identifying the Z-score files for patient participants.
    legend : list of str
        List of legend labels for the healthy and patient groups, respectively.
    method : str
        Name of the subfolder within each run directory where Z-score files are located.
    site_id : list of str, optional
        List of site IDs used to filter participants. If None, data from all sites is included.
    new_col_name : list of str, optional
        List of new column names for the Z-score DataFrames, used for relabeling features.
    y_upper_lim : float, optional
        Upper limit for the y-axis in the generated plots. Default is 15.
    y_lower_lim : float, optional
        Lower limit for the y-axis in the generated plots. Default is 0.

    Returns
    -------
    df_c_pos : pandas.DataFrame
        DataFrame of mean positive extreme deviation proportions (|Z| > 2) for healthy participants across runs.
    df_p_pos : pandas.DataFrame
        DataFrame of mean positive extreme deviation proportions for patient participants across runs.
    df_c_neg : pandas.DataFrame
        DataFrame of mean negative extreme deviation proportions (Z < -2) for healthy participants across runs.
    df_p_neg : pandas.DataFrame
        DataFrame of mean negative extreme deviation proportions for patient participants across runs.
    """
    df_c_pos, df_p_pos = pd.DataFrame(), pd.DataFrame()
    df_c_neg, df_p_neg = pd.DataFrame(), pd.DataFrame()

    for run_num in range(len_runs):
        run_dir = base_path.replace("Run_0", f"Run_{run_num}")

        # Load Z-score data
        with open(os.path.join(run_dir, method, f"Z_{patient_prefix}.pkl"), "rb") as f:
            z_p = pickle.load(f)
        with open(os.path.join(run_dir, method, f"Z_{healthy_prefix}.pkl"), "rb") as f:
            z_c = pickle.load(f)

        if new_col_name:
            z_p.columns = new_col_name
            z_c.columns = new_col_name

        # Load metadata
        with open(os.path.join(run_dir, "b_test.pkl"), "rb") as f:
            b_c = pickle.load(f)
        z_c.index = b_c.index

        with open(os.path.join(run_dir, f"{patient_prefix}_b_test.pkl"), "rb") as f:
            b_p = pickle.load(f)
        z_p.index = b_p.index

        if site_id:
            z_c = z_c[b_c["site"].isin(site_id)]
            z_p = z_p[b_p["site"].isin(site_id)]

        # Compute extremes
        for col in z_c.columns:
            pos_c = z_c[z_c[col] > 2]
            neg_c = z_c[z_c[col] < -2]
            df_c_pos.loc[run_num, col] = (pos_c.shape[0] / z_c.shape[0]) * 100
            df_c_neg.loc[run_num, col] = (neg_c.shape[0] / z_c.shape[0]) * 100

        for col in z_p.columns:
            pos_p = z_p[z_p[col] > 2]
            neg_p = z_p[z_p[col] < -2]
            df_p_pos.loc[run_num, col] = (pos_p.shape[0] / z_p.shape[0]) * 100
            df_p_neg.loc[run_num, col] = (neg_p.shape[0] / z_p.shape[0]) * 100

    def plot_bar(df_h, df_p, mode, ylabel, legend_labels):
        means_h = df_h.mean()
        means_p = df_p.mean()
        ci_h = df_h.std() / np.sqrt(len(df_h)) * 1.96
        ci_p = df_p.std() / np.sqrt(len(df_p)) * 1.96

        x = np.arange(len(means_h))
        width = 0.3

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            x - width / 2,
            means_h,
            yerr=ci_h,
            label=legend_labels[0],
            color="tomato",
            width=width,
            capsize=4,
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            means_p,
            yerr=ci_p,
            label=legend_labels[1],
            color="darkslategray",
            width=width,
            capsize=4,
            alpha=0.8,
        )

        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(means_h.index, fontsize=12, rotation=45, ha="right")
        if mode == "positive":
            ax.legend(fontsize=12)
        ax.set_ylim(y_lower_lim, y_upper_lim)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_position(("outward", 5))
        ax.spines["left"].set_position(("outward", 5))
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"extreme_{mode}_dev.svg"), dpi=600)
        plt.savefig(os.path.join(save_path, f"extreme_{mode}_dev.png"), dpi=600)
        plt.show()

    # Plot and save
    plot_bar(df_c_pos, df_p_pos, "positive", "Percentage", legend)
    plot_bar(df_c_neg, df_p_neg, "negative", "Percentage", legend)

    return df_c_pos, df_p_pos, df_c_neg, df_p_neg


def plot_site_diff(
    processing_dir,
    data_dir,
    save_dir,
    which_quantile: int = 2,
    suffix: str = "estimate",
    batch_curve: dict = {"sex": ["Male", "Female"]},
    batch_marker: dict = {"site": ["BTH", "Cam-Can", "NIMH", "OMEGA", "HCP", "MOUS"]},
    new_names: list = ["Theta", "Alpha", "Beta", "Gamma"],
    colors: list = ["#006685", "#591154", "#E84653", "black", "#E6B213", "Slategrey"],
    num_point: int = 100,
    age_normalizer: float = 100,
) -> None:
    """
    Plot particular centile (determined by 'which_quantile') across sites for each biomarker.
    This function shows site differences at 'which_quantile' centile.

    Parameters
    ----------
    processing_dir : str
        Path to the directory containing normative model output files (e.g., quantile predictions).
    data_dir : str
        Path to the directory containing test set covariates, responses, and batch effect files.
    save_dir : bool, optional
        If not none, saves the generated plots to the this directory.
    which_quantile : int, optional
        Index of the quantile to plot (e.g., 2 for 50th centile in a list like [0.05, 0.25, 0.5, 0.75, 0.95]).
    suffix : str, optional
        Suffix of the saved quantile identifying which output file to use. Default is 'estimate'.
    experiment_id : int, optional
        An integer identifier used to separate outputs from different experiments. Default is 0.
    batch_curve : dict, optional
        Dictionary specifying the batch effect (e.g., {"sex": ["Male", "Female"]}) used to filter data
        before plotting. Only the first listed group is used (e.g., "Male").
    batch_marker : dict, optional
        Dictionary specifying the batch variable to distinguish site-specific curves
        (e.g., {"site": ["BTH", "Cam-Can", ...]}).
    new_names : list of str, optional
        Descriptive names of biomarkers to use for axis labeling. Length must match number of biomarkers.
    colors : list of str, optional
        List of color values corresponding to the number of sites being plotted. Must match `batch_marker`.
    num_point : int, optional
        Number of synthetic age points per site used in the model predictions. Default is 100.
    age_normalizer : float, optional
        Value used to rescale normalized age data back to the original scale. Default is 100.

    Returns
    -------
    None
        This function produces matplotlib figures showing site-specific trajectories of model-predicted
        quantiles for each biomarker. If `save_plot` is True, plots are saved as both SVG and PNG files
        in the specified output directory.
    """

    matplotlib.rcParams["pdf.fonttype"] = 42

    # Load required data
    X_test = (
        pickle.load(open(os.path.join(data_dir, "x_test.pkl"), "rb")).to_numpy(float)
        * age_normalizer
    )
    Y_test = pickle.load(open(os.path.join(data_dir, "y_test.pkl"), "rb"))
    be_test_df = pickle.load(open(os.path.join(data_dir, "b_test.pkl"), "rb"))
    quantiles_data = pickle.load(
        open(os.path.join(processing_dir, f"Quantiles_{suffix}.pkl"), "rb")
    )

    q = quantiles_data["quantiles"]
    synthetic_X = quantiles_data["synthetic_X"]
    quantiles_be = quantiles_data["batch_effects"]

    be_test = be_test_df.to_numpy(float)
    curve_col = list(batch_curve.keys())[0]
    marker_col = list(batch_marker.keys())[0]

    curve_indx = be_test_df.columns.get_loc(curve_col)
    marker_values = batch_marker[marker_col]

    num_biomarkers = q.shape[2]
    num_sites = len(marker_values)

    if num_sites > len(colors):
        raise ValueError("Not enough colors provided for the number of sites.")

    for biomarker_idx in range(num_biomarkers):
        biomarker_name = Y_test.columns[biomarker_idx]
        fig, ax = plt.subplots(figsize=(10, 5))

        # Filter by selected batch effect (e.g., only males)
        selected_indices = np.where(quantiles_be[:, curve_indx] == 0)[0]

        # Reshape data
        x_vals = np.asarray(synthetic_X[selected_indices]).reshape(-1, num_point)
        y_vals = q[selected_indices, which_quantile, biomarker_idx].reshape(
            -1, num_point
        )

        y_mean = np.mean(y_vals, axis=0)

        for site_idx in range(num_sites):
            ax.plot(
                x_vals[site_idx],
                y_vals[site_idx],
                linewidth=3,
                linestyle="-",
                alpha=0.8,
                color=colors[site_idx],
                label=marker_values[site_idx],
            )

        ax.plot(
            x_vals[0],
            y_mean,
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            color="blue",
            label="Site mean",
        )

        ax.set_xlabel("Age (years)", fontsize=20)
        ax.set_ylabel(new_names[biomarker_idx], fontsize=20)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, linewidth=0.5, alpha=0.5, linestyle="--")
        ax.legend(fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        if save_dir:
            os.makedirs(os.path.join(save_dir, "site_diff"), exist_ok=True)
            fname_base = f"{biomarker_idx}_{biomarker_name}_site_diff"
            fig.savefig(os.path.join(save_dir, fname_base + ".svg"), dpi=600)
            fig.savefig(os.path.join(save_dir, fname_base + ".png"), dpi=600)



def parse_aparc2009_name(region_name):
    """
    Convert aparc.a2009s+aseg region name to Destrieux surface label.
    
    Examples
    --------
    ctx_lh_G_front_sup  → G_front_sup  (left hemisphere)
    ctx_rh_G_front_sup  → G_front_sup  (right hemisphere)
    """
    if region_name.startswith("ctx_lh_"):
        return region_name[len("ctx_lh_"):], ["left"]
    elif region_name.startswith("ctx_rh_"):
        return region_name[len("ctx_rh_"):], ["right"]
    elif region_name.startswith("ctx_"):
        return region_name[len("ctx_"):], ["left", "right"]
    else:
        return region_name, ["left", "right"]
    
    
def plot_roi(
    region_name,
    fsaverage="fsaverage5",
    mesh_type="pial",
    hemispheres=["left", "right"],
    fsaverage_sulcal_type="curvature",
    views=["lateral"],
    plot_contour=True,
    colorbar=False,
    contour_color=None,
    cmap="Oranges",
    **kwargs
):
    """
    Plot a Destrieux atlas ROI on a cortical surface mesh.

    Displays a named region from the Destrieux parcellation on an fsaverage
    surface, with optional sulcal background shading and ROI contour overlay.
    Supports multiple hemispheres and views arranged in a grid layout.

    Parameters
    ----------
    region_name : str
        Name of the Destrieux atlas region to plot (e.g. "G_cingul-Post-dorsal").
        Must exactly match a label in ``fetch_atlas_surf_destrieux().labels``.
    fsaverage : str, optional
        Name of the fsaverage template mesh to use. Default is "fsaverage5".
        Common options: "fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6",
        "fsaverage".
    mesh_type : str, optional
        Surface mesh type to use for plotting. Default is "pial".
        Options: "pial", "white_matter", "inflated", "sphere", "flat".
    hemispheres : list of str, optional
        Hemispheres to plot. Default is ["left", "right"].
        Each entry becomes a row in the output figure.
    fsaverage_sulcal_type : str, optional
        Type of sulcal background data to use. Default is "curvature".
        Options: "curvature", "sulcal".
    views : list of str, optional
        Camera views to render. Default is ["lateral"].
        Each entry becomes a column in the output figure.
        Options: "lateral", "medial", "dorsal", "ventral", "anterior",
        "posterior".
    plot_contour : bool, optional
        Whether to draw a contour outline around the ROI boundary.
        Default is True.
    colorbar : bool, optional
        Whether to display a colorbar on each subplot. Default is False.
    contour_color : str, tuple, or None, optional
        Color for the ROI contour. Accepts any matplotlib-compatible color
        (e.g. "black", "#ff0000", (0, 0, 0)). If None, defaults to black.
    cmap : str, optional
        Colormap for the ROI fill. Default is "Oranges".
        Accepts any matplotlib colormap name.
    **kwargs
        Additional keyword arguments passed to ``plot_surf_roi``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing all subplots.
    axes : numpy.ndarray of matplotlib.axes.Axes
        2D array of axes with shape (n_hemispheres, n_views).

    Examples
    --------
    Plot a single hemisphere and view:

    >>> plot_roi(
    ...     region_name="G_cingul-Post-dorsal",
    ...     hemispheres=["left"],
    ...     views=["medial"],
    ... )

    Plot both hemispheres with multiple views and no contour:

    >>> plot_roi(
    ...     region_name="G_cingul-Post-dorsal",
    ...     hemispheres=["left", "right"],
    ...     views=["medial", "lateral"],
    ...     plot_contour=False,
    ...     cmap="Blues",
    ... )

    Notes
    -----
    The figure layout is a grid of shape (n_hemispheres x n_views), where each
    row corresponds to a hemisphere and each column to a view.

    The Destrieux atlas is fetched via ``fetch_atlas_surf_destrieux()`` and the
    background sulcal map via ``load_fsaverage_data()``. Both are filtered to
    the requested hemispheres to ensure mesh compatibility.
    """
    fsaverage_meshes = load_fsaverage(mesh=fsaverage)
    destrieux = fetch_atlas_surf_destrieux(verbose=False)

    destrieux_label, inferred_hemis = parse_aparc2009_name(region_name)
    
    if region_name.startswith("ctx_lh_"):
        hemispheres = [h for h in hemispheres if h == "left"] or ["left"]
    elif region_name.startswith("ctx_rh_"):
        hemispheres = [h for h in hemispheres if h == "right"] or ["right"]

    # Validate label exists
    if destrieux_label not in destrieux.labels:
        close = [l for l in destrieux.labels if destrieux_label.lower() in l.lower()]
        raise ValueError(
            f"'{destrieux_label}' not found in Destrieux labels.\n"
            f"Did you mean one of: {close}"
        )

    map_destrieux = {}
    if "left" in hemispheres:
        map_destrieux["left"] = destrieux.map_left
    if "right" in hemispheres:
        map_destrieux["right"] = destrieux.map_right

    full_mesh = fsaverage_meshes[mesh_type]
    filtered_mesh = PolyMesh(**{hemi: full_mesh.parts[hemi] for hemi in hemispheres})
    destrieux_atlas = SurfaceImage(mesh=filtered_mesh, data=map_destrieux)

    fsaverage_sulcal_full = load_fsaverage_data(data_type=fsaverage_sulcal_type)
    fsaverage_sulcal = SurfaceImage(
        mesh=filtered_mesh,
        data={hemi: fsaverage_sulcal_full.data.parts[hemi] for hemi in hemispheres}
    )

    label_region = destrieux.labels.index(destrieux_label) 
    mask = {
        hemi: data == label_region
        for hemi, data in destrieux_atlas.data.parts.items()
    }
    surface_mask = SurfaceImage(mesh=filtered_mesh, data=mask)
    
    n_rows = len(hemispheres)
    n_cols = len(views)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        subplot_kw={"projection": "3d"},
        figsize=(6 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for row, hemi in enumerate(hemispheres):
        for col, view in enumerate(views):
            fig_roi = plot_surf_roi(
                roi_map=surface_mask,
                hemi=hemi,
                view=view,
                bg_map=fsaverage_sulcal,
                bg_on_data=True,
                colorbar=colorbar,
                cmap=cmap,
                alpha=1,
                axes=axes[row, col],
                figure=fig,
                **kwargs
            )
            if plot_contour:
                _contour_color = contour_color if contour_color else ListedColormap(["black"])            
                plot_surf_contours(
                    roi_map=surface_mask,
                    figure=fig_roi, 
                    hemi=hemi,
                    axes=axes[row, col],  
                    cmap=_contour_color,
                    # **kwargs
                )

def define_lut(lut_path):
    """
    Load a FreeSurfer Look-Up Table (LUT) and return it as a dictionary.

    A LUT file maps region names to integer indices. This function reads
    the file, skips comments and empty lines, and builds a dict of
    {region_name: region_index}.

    Parameters
    ----------
    lut_path : str
        Path to the FreeSurfer LUT file
        (e.g. FreeSurferColorLUT.txt).

    Returns
    -------
    dict
        A dictionary mapping region name (str) to region index (int).
        Example: {"Left-Hippocampus": 17, "Right-Hippocampus": 53, ...}
    """
    lut = {}
    with open(lut_path) as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            lut[parts[1]] = int(parts[0])
    return lut


def plot_statistics_on_brain(
        parcellation_atlas_path,
        lut_path,
        stats,
        fsaverage="fsaverage5",
        surface_mesh_type="pial",
        bg_map_mesh_type="sulc",
        abs_threshold=None,
        cmap="Reds",
        alpha=0.7,
        symmetric_cbar="auto",
        bg_on_data=True,
        show_colorbar=False,
        title=None,
        vmin=None,
        vmax=None,
        save_fig_path=None,
        views=("lateral", "medial"),
        hemispheres=("left", "right"),
):
    """
    Project region-wise statistics onto the brain surface and plot them.

    Takes a parcellation atlas and a set of statistics (one value per brain
    region) and visualizes them on the cortical surface in three steps:

        1. Builds a 3D volume where each voxel takes the stat value of
           its region (matched by name via the LUT).
        2. Projects that volume onto the 2D surface mesh (vol_to_surf).
        3. Plots the result for each hemisphere × view combination.

    Parameters
    ----------
    parcellation_atlas_path : str
        Path to the parcellation atlas file (.mgz or .nii.gz). Each voxel
        holds an integer index identifying the brain region it belongs to.

    lut_path : str
        Path to the FreeSurfer LUT file (e.g. FreeSurferColorLUT.txt).
        Used to map region names to the integer indices in the atlas.

    stats : dict or pd.Series
        One scalar value per region name. Keys/index must match region
        names in the LUT.
        Example: {"Left-Hippocampus": 0.42, "Right-Hippocampus": 0.38}

    fsaverage : str, optional
        fsaverage resolution to use. Default is "fsaverage5" (10,242
        vertices per hemisphere). Options: "fsaverage3" through "fsaverage7".

    surface_mesh_type : str, optional
        Surface mesh to project the stats onto. Default is "pial" (outer
        gray matter surface). Options: "pial", "white", "infl", "flat",
        "sphere".

    bg_map_mesh_type : str, optional
        Background shading map. Default is "sulc" (sulcal depth), which
        gives the brain its natural light/dark folding pattern.
        Options: "sulc", "curv", "thick", "area".

    abs_threshold : float or None, optional
        Absolute threshold below which stat values are not shown
        (rendered transparent). If None, all values are shown.
        Default is None.

    cmap : str, optional
        Matplotlib colormap for the stat values. Default is "Reds".

    alpha : float, optional
        Opacity of the stat map overlay, between 0 (transparent) and 1
        (opaque). Default is 0.7.

    symmetric_cbar : bool or "auto", optional
        Whether to make the colorbar symmetric around zero. If "auto",
        nilearn decides based on the data. Default is "auto".

    bg_on_data : bool, optional
        If True, blends the background shading on top of the stat map,
        giving a more 3D appearance. Default is True.

    show_colorbar : bool, optional
        Whether to show a colorbar on each subplot. Default is False.

    title : str or None, optional
        Base title for each subplot. If None, titles are auto-generated
        as "hemi - view". If provided, becomes "title | hemi - view".

    vmin : float or None, optional
        Lower bound of the colormap scale. If None, uses the data minimum.

    vmax : float or None, optional
        Upper bound of the colormap scale. If None, uses the data maximum.

    save_fig_path : str or None, optional
        File path to save the figure (e.g. "brain.png"). If None, the
        figure is not saved. Default is None.

    views : tuple of str, optional
        Views to plot — each becomes one column in the figure.
        Default is ("lateral", "medial").

    hemispheres : tuple of str, optional
        Hemispheres to plot — each becomes one row in the figure.
        Default is ("left", "right").

    Returns
    -------
    None
        Displays the figure and optionally saves it to disk.

    Examples
    --------
    >>> plot_statistics_on_brain(
    ...     parcellation_atlas_path="aparc.a2009s+aseg.mgz",
    ...     lut_path="FreeSurferColorLUT.txt",
    ...     stats={"Left-Hippocampus": 0.42, "Right-Hippocampus": 0.38},
    ...     vmin=0, vmax=1,
    ...     cmap="Reds",
    ...     views=("lateral",),
    ...     hemispheres=("left",),
    ... )
    """
    parcell_atlas_img = nib.load(parcellation_atlas_path)
    parcell_atlas_data = parcell_atlas_img.get_fdata()

    if isinstance(stats, pd.DataFrame):
        stats = stats.to_dict()

    lut = define_lut(lut_path)

    missing_regions = []
    volume_img = np.zeros(parcell_atlas_data.shape)
    for region_name, region_ind in lut.items():
        roi_stats = stats.get(region_name)
        if roi_stats is not None:
            volume_img[parcell_atlas_data == region_ind] = roi_stats
        else:
            missing_regions.append(region_name)
    print(f"Statistics for the following regions are missing: {missing_regions}")

    img = nib.Nifti1Image(volume_img, parcell_atlas_img.affine)

    fsaverage_meshes = datasets.fetch_surf_fsaverage(fsaverage)

    n_rows = len(hemispheres)
    n_cols = len(views)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        subplot_kw={"projection": "3d"},
        figsize=(6 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for row, hemi in enumerate(hemispheres):
        surf_mesh = fsaverage_meshes[f"{surface_mesh_type}_{hemi}"]
        bg_map = fsaverage_meshes[f"{bg_map_mesh_type}_{hemi}"]
        texture = surface.vol_to_surf(img, surf_mesh)

        for col, view in enumerate(views):
            plotting.plot_surf_stat_map(
                surf_mesh=surf_mesh,
                stat_map=texture,
                hemi=hemi,
                view=view,
                bg_map=bg_map,
                bg_on_data=bg_on_data,
                colorbar=show_colorbar,
                threshold=abs_threshold,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                symmetric_cbar=symmetric_cbar,
                title=f"{hemi} - {view}" if title is None else f"{title} | {hemi} - {view}",
                axes=axes[row, col],
                figure=fig,
                cbar_tick_format="%.2g",
            )

    plt.tight_layout()
    if save_fig_path:
        fig.savefig(save_fig_path, dpi=150, bbox_inches="tight")
    plt.show()

    return None



def plot_mass_metrics(
    df,
    save_path: str,
    name: str,
    *,
    feature_categories: list[str],
    colors: list[str],
    figsize: tuple[int, int] = (9, 6),
    kernel: str = "epanechnikov",
    dpi: int = 600,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    bandwidth: float = 0.05,
    show_row_labels: bool = False,
    hspace: float = -0.8,
    new_names: Optional[list[str]] = None,
    row_label_fontsize=10,
    x_label_fontsize= 18,
    x_ticks_fontsize=15,
    kde_samples=4000,
) -> plt.Figure:
    """
    Plot stacked KDE ridge lines for a set of feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source data. Columns are matched with ``str.startswith(feature_category)``.
    save_path : str
        Directory where the PNG will be saved.
    name : str
        Filename stem (also used as the x-axis label on the bottom strip).
    feature_categories : list[str]
        Column prefixes to plot, one ridge per entry.
    colors : list[str]
        Fill colours, one per ridge (must be at least as long as ``feature_categories``).
        Pass an empty list to auto-generate a blue-to-red palette.
    figsize : tuple[int, int]
        Figure size in inches.
    kernel : str
        KDE kernel passed to ``sklearn.neighbors.KernelDensity``.
    dpi : int
        Resolution for the saved PNG.
    xlim : tuple[float, float] | None
        Optional shared x-axis limits.
    ylim : tuple[float, float] | None
        Optional shared y-axis limits.
    bandwidth : float
        KDE bandwidth.
    show_row_labels : bool
        When True, print the feature name to the right of each strip.
    hspace : float
        Vertical spacing between subplots (negative values create overlap).
    new_names : list[str] | None
        Optional display names for each ridge, replacing ``feature_categories``
        in row labels. Must match the length of ``feature_categories``.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure (also saved to disk).
    """

    def _generate_colors(n: int) -> list[str]:
        """Generate n colors interpolated from blue to red."""
        cmap = matplotlib.colormaps["RdYlBu_r"]  
        return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]

    if new_names is not None and len(new_names) != len(feature_categories):
        raise ValueError(
            f"new_names length ({len(new_names)}) must match "
            f"feature_categories length ({len(feature_categories)})."
        )

    if not colors:
        colors = _generate_colors(len(feature_categories))

    if len(colors) < len(feature_categories):
        raise ValueError(
            f"Need at least {len(feature_categories)} colours, got {len(colors)}."
        )

    n = len(feature_categories)
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = grid_spec.GridSpec(n, 1, figure=fig)  # attach GridSpec to figure

    for i, feature_category in enumerate(feature_categories):
        
        cols = df.loc[:, df.columns.str.startswith(feature_category)]
        values = cols.to_numpy().ravel() 

        if values.size == 0:
            raise ValueError(f"No columns found starting with '{feature_category}'.")

        if xlim:
            x_min, x_max = xlim
        else:
            data_min, data_max = values.min(), values.max()
            pad = (data_max - data_min) * 0.05  # 5% padding on each side
            x_min = data_min - pad
            x_max = data_max + pad

        x_grid = np.linspace(x_min, x_max, kde_samples)

        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(values[:, None])
        density = np.exp(kde.score_samples(x_grid[:, None]))
    
        ax = fig.add_subplot(gs[i : i + 1, 0:])
        ax.set_yticks([])
        ax.set_facecolor("none")

        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)

        is_last = i == n - 1

        if not is_last:
            ax.set_xticks([])
        else:
            ax.tick_params(axis="x", labelsize=x_ticks_fontsize)
            ax.set_xlabel(name, fontsize=x_label_fontsize, fontweight="bold")

        ax.plot(x_grid, density, color="#f0f0f0", lw=1)
        ax.fill_between(x_grid, density, alpha=1, color=colors[i])

        if show_row_labels:
            label = new_names[i] if new_names else feature_category
            x_pos = xlim[0] if xlim else x_min
            ax.text(
                x_pos,
                0,
                label,
                # fontweight="bold",
                fontsize=row_label_fontsize,
                ha="right",  
                va="bottom",
                transform=ax.transData,
                clip_on=False,
            )

    gs.update(hspace=hspace)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"{name}.png"), dpi=dpi, bbox_inches="tight")

    return fig