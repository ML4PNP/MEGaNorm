import os
import statsmodels.api as sm
import matplotlib
import pickle
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import matplotlib.ticker as mticker
from scipy.stats import chi2


def KDE_plot(data, experiments, metric, xlim="auto", fontsize=24):
    # Create the data
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = []
    x = data
    x_min = np.min(x)
    x_max = np.max(x)
    for i in range(len(experiments)):
        for j in range(data.shape[1]):
            g.append(experiments[i])
    df = pd.DataFrame(dict(x=x.ravel(), g=g))

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=0.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=0.2)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=0.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label_KDE(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=fontsize,
        )
        ax.plot([x.median(), x.median()], [0, 10], c="w")
        if xlim == "auto":
            ax.set_xlim([x_min - 0.05, x_max])
        else:
            ax.set_xlim([xlim[0], xlim[1]])
        plt.xticks(fontsize=fontsize)

    plt.yticks(fontsize=fontsize)
    g.map(label_KDE, "x")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.set_xlabels(metric.upper(), fontsize=fontsize)
    g.set_ylabels("")
    g.despine(bottom=True, left=True)


def plot_nm_range(
    processing_dir,
    data_dir,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    age_range=[15, 90],
    ind=0,
    parallel=True,
    save_plot=True,
    outputsuffix="estimate",
    ):
    """Function to plot notmative ranges. This function assumes only gender as batch effect
    stored in the first column of batch effect array.

    Args:
        processing_dir (str): Path to normative modeling processing directory.
        quantiles (list, optional): Plotted centiles. Defaults to [0.05, 0.25, 0.5, 0.75, 0.95].
        ind (int, optional): Index of target biomarker to plot. Defaults to 0.
        parallel (bool, optional): Is parallel NM used to estimate the model?. Defaults to True.
        save_plot (bool, optional): Save the plot?. Defaults to True.
        outputsuffix (str, optional): outputsuffix in normative modeling. Defaults to 'estimate'.
    """

    z_scores = st.norm.ppf(quantiles)
    testrespfile_path = os.path.join(data_dir, "y_test.pkl")
    testcovfile_path = os.path.join(data_dir, "x_test.pkl")
    tsbefile = os.path.join(data_dir, "b_test.pkl")

    if parallel:
        nm = pickle.load(
            open(
                os.path.join(
                    processing_dir,
                    "batch_" + str(ind + 1),
                    "Models/NM_0_0_" + outputsuffix + ".pkl",
                ),
                "rb",
            )
        )
        meta_data = pickle.load(
            open(
                os.path.join(
                    processing_dir, "batch_" + str(ind + 1), "Models/meta_data.md"
                ),
                "rb",
            )
        )
        in_scaler = meta_data["scaler_cov"][0]
        out_scaler = meta_data["scaler_resp"][0]
    else:
        nm = pickle.load(
            open(
                os.path.join(
                    processing_dir,
                    "Models/NM_0_" + str(ind) + "_" + outputsuffix + ".pkl",
                ),
                "rb",
            )
        )
        meta_data = pickle.load(
            open(os.path.join(processing_dir, "Models/meta_data.md"), "rb")
        )
        in_scaler = meta_data["scaler_cov"][ind][0]
        out_scaler = meta_data["scaler_resp"][ind][0]

    synthetic_X = np.linspace(age_range[0], age_range[1], 200)[
        :, np.newaxis
    ]  # Truncated

    X_test = pickle.load(open(testcovfile_path, "rb")).to_numpy(float)
    be_test = pickle.load(open(tsbefile, "rb")).to_numpy(float)
    Y_test = pickle.load(open(testrespfile_path, "rb"))
    bio_name = Y_test.columns[ind]
    Y_test = Y_test.to_numpy(float)[:, ind : ind + 1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, sharey=True)

    colors = ["#00BFFF", "#FF69B4"]
    labels = ["Males", "Females"]  # assumes 0 for males and 1 for females

    for be1 in list(np.unique(be_test[:, 0])):
        model_be = np.repeat(np.array([[be1]]), synthetic_X.shape[0], axis=0)
        q = nm.get_mcmc_quantiles(
            in_scaler.transform(synthetic_X), model_be, z_scores=z_scores
        )
        ts_idx = be_test[:, 0] == be1
        ax.scatter(
            X_test[ts_idx],
            Y_test[ts_idx],
            s=15,
            alpha=0.6,
            label=labels[int(be1)],
            color=colors[int(be1)],
        )
        for i, v in enumerate(z_scores):
            if v == 0:
                thickness = 3
                linestyle = "-"
            else:
                linestyle = "--"
                thickness = 1
            y = out_scaler.inverse_transform(q[i, :]).T
            ax.plot(
                synthetic_X,
                y,
                linewidth=thickness,
                linestyle=linestyle,
                alpha=0.9,
                color=colors[int(be1)],
            )
            if be1 == 0:
                plt.annotate(
                    str(int(quantiles[i] * 100)) + "%",
                    xy=(synthetic_X[-1], y[-1]),
                    xytext=(synthetic_X[-1] + 0.6, y[-1]),
                    ha="left",
                    va="center",
                    fontsize=14,
                )
        ax.grid(True, linewidth=0.5, alpha=0.5, linestyle="--")
        ax.set_ylabel(bio_name.replace("_", " "), fontsize=10)
        ax.set_xlabel("Age", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend()
    plt.tight_layout()

    if save_plot:
        save_path = os.path.join(processing_dir, "Figures")
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.savefig(
            os.path.join(save_path, str(ind) + "_" + bio_name + ".png"), dpi=300
        )


def plot_comparison(
    path,
    hbr_configs,
    biomarker_num,
    metrics=["Rho", "SMSE", "MSLL", "MACE"],
    plot_type="boxplot"):

    results = {
        metric: np.zeros([biomarker_num, len(hbr_configs.keys())]) for metric in metrics
    }

    for m, method in enumerate(hbr_configs.keys()):
        for metric in metrics:
            with open(
                os.path.join(path, method, metric + "_estimate.pkl"), "rb"
            ) as file:
                temp = pickle.load(file)
            results[metric][:, m] = temp.squeeze()

    methods = hbr_configs.keys()
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    index = np.arange(len(methods))

    for ax, (metric_name, values) in zip(axs, results.items()):

        if plot_type == "boxplot":
            ax.boxplot(values, positions=index, notch=True, showfliers=False)
        elif plot_type == "violin":
            violin_parts = ax.violinplot(
                values, index, showmedians=True, showextrema=False
            )

            for partname in ["cmedians"]:
                vp = violin_parts[partname]
                vp.set_edgecolor("black")
                vp.set_linewidth(1)

            # Make the violin body blue with a red border:
            for vp in violin_parts["bodies"]:
                vp.set_facecolor("#929591")
                vp.set_edgecolor("#000000")
                vp.set_alpha(1)

        ax.set_title(f"{metric_name} Comparison", fontsize=14)
        ax.set_xticks(index)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(path, plot_type + '_metric_comparison.png'), dpi=300)
    
    
# ***
def plot_age_hist(df, site_names, save_path, 
    lower_age_range=5, upper_age_range=90, step_size=5,
    colors=['#006685' ,'#591154' ,'#E84653' ,'black' ,'#E6B213', "Slategrey"]):
    """ 
    Plots and saves a stacked histogram showing the age distribution across multiple sites.

    Args:
        df (pandas.DataFrame): A dataframe containing at least two columns: "site" and "age".
                               Each row represents a participant. The "site" column should 
                               contain numeric identifiers corresponding to each site.
        site_names (list): A list of site names (str), used as labels in the legend. 
        save_path (str): Directory path where the resulting plots will be saved. 
        lower_age_range (int, optional): Minimum age to include in the histogram. Default is 5.
        upper_age_range (int, optional): Maximum age to include in the histogram. Default is 90.
        step_size (int, optional): Bin width (in years) for the histogram. Default is 5.
        colors (list, optional): List of colors for each site's histogram. Must be the same length
                                 or longer than `site_names`.

    Raises:
        Exception: If the number of provided colors is less than the number of sites.

    Saves:
        "age_dis.svg"
        "age_dis.png"
    """

    if len(site_names) > len(colors):
        raise Exception("The number of colors is less than site_names, please specify a longer list of colors.")

    
    bins = list(range(lower_age_range, upper_age_range, step_size))
    ages = []

    for counter in range(len(site_names)):
        ages.append(df[df["site"]==counter]["age"].to_numpy()*100)
    
    plt.figure(figsize=(12, 7))
    plt.hist(ages, bins=bins, color=colors, 
             edgecolor="black", 
             alpha=0.6, 
             histtype="barstacked", 
             rwidth=0.9)
    
    # Remove the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Offset the bottom and left spines
    plt.gca().spines['bottom'].set_position(('outward', 15))  
    plt.gca().spines['left'].set_position(('outward', 15))


    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.grid(axis="y", color = 'black', linestyle = '--')
    plt.grid(axis="x", linestyle = '')

    plt.xlabel("Age (years)", fontsize=25)
    plt.legend(site_names, prop={'size': 23}, loc='upper right')
    plt.tick_params(axis="both", labelsize=19)
    plt.xticks(list(range(lower_age_range, lower_age_range, step_size*2)))
    plt.ylabel("Count",  fontsize=25)
    plt.savefig(os.path.join(save_path, "age_hist.svg"), format="svg", dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(save_path, "age_hist.png"), format="png", dpi=600, bbox_inches="tight")

    

def plot_neurooscillochart(data, age_slices, save_path):

    # Age ranges
    ages = [f"{i}-{i+5}" for i in age_slices]

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    def plot_gender_data(ax, gender_data, title, legend=True, colors=None):

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
            alpha=0.7,
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
        ax.set_xlabel('Age ranges (years)', fontsize=16)
        if legend:
            ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
        else:
            ax.get_legend().remove()
            
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        ax.grid(False, axis='x')  
        ax.tick_params(axis='x', labelsize=14)
        ax.set_yticklabels([])  
    
    plot_gender_data(axes[0], data['Male'], "Males' Chrono-NeuroOscilloChart", 
                     colors= ['lightgrey', 'gray', 'dimgrey', 'lightslategray'])
    
    plot_gender_data(axes[1], data['Female'], "Females' Chrono-NeuroOscilloChart", legend=False, 
                     colors=['lightgrey', 'gray', 'dimgrey', 'lightslategray'])
    
    axes[1].set_xlabel('Age ranges (years)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "Chrono-NeuroOscilloChart.svg"), dpi=600)
    else:
        plt.show()
        


    

def plot_nm_range_site(
    processing_dir,
    data_dir,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    save_plot=True,
    outputsuffix="estimate",
    experiment_id=0,
    batch_curve={0: ["Male", "Female"]},
    batch_marker={1: ["CAMCAN", "BTNRH"]},
):
    """Function to plot notmative ranges. This function assumes only gender as batch effect
    stored in the first column of batch effect array.

    Args:
        processing_dir (str): Path to normative modeling processing directory.
        quantiles (list, optional): Plotted centiles. Defaults to [0.05, 0.25, 0.5, 0.75, 0.95].
        ind (int, optional): Index of target biomarker to plot. Defaults to 0.
        parallel (bool, optional): Is parallel NM used to estimate the model?. Defaults to True.
        save_plot (bool, optional): Save the plot?. Defaults to True.
        outputsuffix (str, optional): outputsuffix in normative modeling. Defaults to 'estimate'.
    """

    z_scores = st.norm.ppf(quantiles)
    testrespfile_path = os.path.join(data_dir, "y_test.pkl")
    testcovfile_path = os.path.join(data_dir, "x_test.pkl")
    tsbefile = os.path.join(data_dir, "b_test.pkl")
    quantiles_path = os.path.join(processing_dir, "Quantiles_" + outputsuffix + ".pkl")

    X_test = pickle.load(open(testcovfile_path, "rb")).to_numpy(float)
    be_test = pickle.load(open(tsbefile, "rb")).to_numpy(float)
    Y_test = pickle.load(open(testrespfile_path, "rb"))
    temp = pickle.load(open(quantiles_path, "rb"))
    q = temp["quantiles"]
    synthetic_X = temp["synthetic_X"]
    quantiles_be = temp["batch_effects"]

    X_test = X_test * 100

    colors = ["#00BFFF", "#FF69B4"]

    curve_indx = list(batch_curve.keys())[0]
    hue_indx = list(batch_marker.keys())[0]

    for ind in range(q.shape[2]):

        bio_name = Y_test.columns[ind]
        y_test = Y_test.to_numpy(float)[:, ind : ind + 1]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, sharey=True)

        for be1 in list(np.unique(be_test[:, curve_indx])):

            # Male
            ts_idx = np.logical_and(
                be_test[:, curve_indx] == be1, be_test[:, hue_indx] == 0
            )
            ax.scatter(
                X_test[ts_idx],
                y_test[ts_idx],
                s=35,
                alpha=0.6,
                label=[*batch_curve.values()][0][int(be1)]
                + " "
                + [*batch_marker.values()][0][0],
                color=colors[int(be1)],
                marker="o",
            )
            # Female
            ts_idx = np.logical_and(
                be_test[:, curve_indx] == be1, be_test[:, hue_indx] == 1
            )
            ax.scatter(
                X_test[ts_idx],
                y_test[ts_idx],
                s=35,
                alpha=0.6,
                label=[*batch_curve.values()][0][int(be1)]
                + " "
                + [*batch_marker.values()][0][1],
                color=colors[int(be1)],
                marker="^",
            )

            q_idx = np.logical_and(
                quantiles_be[:, curve_indx] == be1, quantiles_be[:, hue_indx] == 0
            )  # only for males
            for i, v in enumerate(z_scores):
                if v == 0:
                    thickness = 3
                    linestyle = "-"
                else:
                    linestyle = "--"
                    thickness = 1

                y = q[q_idx, i : i + 1, ind]

                ax.plot(
                    synthetic_X[q_idx],
                    y,
                    linewidth=thickness,
                    linestyle=linestyle,
                    alpha=0.9,
                    color=colors[int(be1)],
                )
                if be1 == 0:
                    plt.annotate(
                        str(int(quantiles[i] * 100)) + "%",
                        xy=(synthetic_X[-1], y[-1]),
                        xytext=(synthetic_X[-1] + 0.6, y[-1]),
                        ha="left",
                        va="center",
                        fontsize=14,
                    )
            ax.grid(True, linewidth=0.5, alpha=0.5, linestyle="--")
            ax.set_ylabel(bio_name.replace("_", " "), fontsize=10)
            ax.set_xlabel("Age", fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=14)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.legend()
        plt.tight_layout()

        if save_plot:
            save_path = os.path.join(
                processing_dir, f"Figures_experiment{experiment_id}"
            )
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            plt.savefig(
                os.path.join(save_path, str(ind) + "_" + bio_name + ".png"), dpi=300
            )


def plot_growthchart(age_vector, centiles_matrix, cut=0, idp="", save_path=None):
    """Plots growth chart for two sexes.

    Args:
        age_vector (numpy.ndarray): A 1D array representing the age values.
        centiles_matrix (numpy.ndarray): A 3D array of shape (age_vector.shape[0], 5, 2) where
                                    the second dimension represents the 5 centiles and
                                    the third dimension represents the 2 sexes (0 for male, 1 for female).
        cut (int, optional): The cutting age for the younger population. Defaults to 0.
        idp (str, optional): IDP name. Defaults to ''.
        save_path (str, optional): If not None saves the plot to the path. Defaults to None.
    """

    colors = {
        "male": ["#0a2f66", "#0d3a99", "#1350d0", "#4b71db", "#8195e6"],
        "female": ["#b30059", "#cc0066", "#e60073", "#ff3399", "#ff66b2"],
    }

    min_age = age_vector.min()
    max_age = age_vector.max()

    def age_transform(age):  # Age transformation based on cut
        if age < cut:
            return age
        else:
            return cut + (age - cut) / 3

    transformed_age = np.array([age_transform(age) for age in age_vector])

    fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=True)

    genders = ["male", "female"]

    for j, gender in enumerate(genders):
        for i in range(5):
            linestyle = "-" if i in [1, 2, 3] else "--"
            linewidth = 5 if i == 2 else (3 if i in [1, 3] else 2)
            axes[j].plot(
                transformed_age,
                centiles_matrix[:, i, j],
                label=f"{[5, 25, 50, 75, 95][i]}th Percentile",
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
            alpha=0.1,
        )
        axes[j].fill_between(
            transformed_age,
            centiles_matrix[:, 1, j],
            centiles_matrix[:, 3, j],
            color=colors[gender][2],
            alpha=0.1,
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
        axes[j].set_xticklabels(np.concatenate((np.arange(min_age, cut+1, 2, dtype=int), 
                                                np.arange(np.ceil((cut+1)/10)*10, max_age+1, 10, dtype=int))), fontsize=22)
        axes[j].tick_params(axis='y', labelsize=22)
        axes[j].grid(True, which='both', linestyle='--', linewidth=2, alpha=0.85)
        axes[j].spines['top'].set_visible(False)
        axes[j].spines['right'].set_visible(False)
        # axes[j].set_xlabel('Age (years)', fontsize=28)
        
        #axes[j].legend(loc='upper left', fontsize=20)

        # axes[j].legend(loc='upper left', fontsize=20)

        for i, label in enumerate(["5th", "25th", "50th", "75th", "95th"]):
            axes[j].annotate(
                label,
                xy=(transformed_age[-1], centiles_matrix[-1, i, j]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=18,
                color=colors[gender][i],
                fontweight="bold",
            )

        # axes[j].axvline(x=age_transform(cut), color='k', linestyle='--', linewidth=2, alpha=0.5)

    axes[0].set_ylabel(idp, fontsize=28)
    # axes[0].set_title("Males", fontsize=28)
    # axes[1].set_title("Females", fontsize=28)

    plt.tight_layout(pad=2)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, idp.replace(" ", "_") + '_growthchart.svg'), dpi=600)


def plot_growthcharts(path, idp_indices, idp_names, site=1, point_num=100, num_of_sites=None):
    """Plotting growth charts for multiple idps.

    Args:
        path (string): Path to processing directory in which the quantiles are saved.
        idp_indices (list): A list of IDP indices.
        idp_names (list): A list of IDP names corresponding to IDP indices.
        site (int, optional): The site id to plot. Defaults to 0.
        point_num (int, optional): Number of points used in creating the synthetic X. Defaults to 100.
        num_of_sites: number of sites (used for averaging)
    """

    temp = pickle.load(open(os.path.join(path, "Quantiles_estimate.pkl"), "rb"))

    q = temp["quantiles"]
    x = temp["synthetic_X"]
    b = temp["batch_effects"]

    for i, idp in enumerate(idp_indices):
        
        print(q.shape)
        data = np.concatenate([q[b[:,0]== 0,:,idp:idp+1], 
                            q[b[:,0]== 1,:,idp:idp+1]], axis=2)
        data = data.reshape(num_of_sites, 100, 5, 2) 
        data = data.mean(axis=0)

        plot_growthchart(
            x[0:point_num].squeeze(), data, cut=0, idp=idp_names[i], save_path=path
        )

def plot_quantile_gauge(sub_index, current_value, q1, q3, percentile_5, percentile_95, percentile_50, 
                        title="Quantile-Based Gauge", min_value=0, max_value=1, show_legend=False, bio_name=None, save_path=""):
    """
    Plots a gauge chart based on quantile ranges with a threshold marker for the 0.5 percentile.
    
    Parameters:
    - current_value (float): The current decimal value to display.
    - q1 (float): The 25th percentile value as a decimal.
    - q3 (float): The 75th percentile value as a decimal.
    - percentile_5 (float): The 5th percentile value as a decimal.
    - percentile_95 (float): The 95th percentile value as a decimal.
    - percentile_50 (float): The 0.5 percentile value as a decimal, marked by a threshold line.
    - title (str): The title of the gauge chart.
    - min_value (float): The minimum value for the gauge range (default is 0).
    - max_value (float): The maximum value for the gauge range (default is 1).
    - show_legend (bool): Whether to display the legend with color-coded ranges (default is False).
    """
    current_value = round(current_value, 3)
    
    if bio_name == "Gamma": max_value = 0.1

    if current_value < percentile_5:
        value_color = "rgb(8, 65, 92)"  # Purple 
    elif current_value < q1:
        value_color = "rgb(0, 191, 255)"  # Gold 
    elif current_value <= q3:
        value_color = "rgb(129, 193, 75)"  # Green 
    elif current_value <= percentile_95:
        value_color = "rgb(255, 201, 20)"  # Tomato red
    else:
        value_color = "rgb(188, 44, 26)"  # Purple

    if show_legend:
        number_font_size = 75
        delta_font_size = 30
    else:
        number_font_size = 120
        delta_font_size = 90
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_value,
        number={'font': {'size': number_font_size, 'family': 'Arial', 'color': value_color}},  
        delta={'reference': percentile_50, 'position': "top", 'font': {'size': delta_font_size}},
        gauge={
            'axis': {
                'range': [min_value, max_value],
                'tickfont': {'size': 60, 'family': 'Arial', 'color': 'black'},
                'showticklabels': True,
                'tickwidth': 12,
                'tickcolor': "black",
                'tickvals': [round(min_value + i * (max_value - min_value) / 10, 2) for i in range(11)],  
            },
            'bar': {'color': "rgb(255, 255, 255)", 'line': {'color': "black", 'width': 3}},  
            'steps': [
                {'range': [min_value, percentile_5], 'color': "rgb(8, 65, 92)"},  # Purple 
                {'range': [percentile_5, q1], 'color': "rgb(0, 191, 255)"},  # Warm gold 
                {'range': [q1, q3], 'color': "rgb(129, 193, 75)"},  # Forest green 
                {'range': [q3, percentile_95], 'color': "rgb(255, 201, 20)"},  # Soft tomato red
                {'range': [percentile_95, max_value], 'color': "rgb(188, 44, 26)"},  # dark Purple
            ],
            'threshold': {
                'line': {'color': "black", 'width': 6},  # Black line for the 0.5th percentile marker
                'thickness': 0.75,
                'value': percentile_50, 
            },
        },
        title={
            'text': bio_name,
            'font': {'size': 70, 'family': 'Arial', 'color': 'black'}
        }
    ))

    if show_legend:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color="rgb(8, 65, 92)"),
                                 name="0-5th Percentile (Extremely Low)"))

        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color="rgb(0, 191, 255)"),
                                 name="5th-25th Percentile (Below Normal)"))

        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color="rgb(129, 193, 75)"),
                                 name="25th-75th Percentile (Normal)"))

        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color="rgb(255, 201, 20)"),
                                 name="75th-95th Percentile (Above Normal)"))

        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color="rgb(188, 44, 26)"),
                                 name="95th-100th Percentile (Extremely High)"))
    
    # Update layout for better aesthetics
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=50, b=100 if show_legend else 30, l=100, r=130),  # Adjust bottom margin for legend
        showlegend=show_legend,
        width=1000,
        height=800,
        legend=dict(
            orientation="h",      
            yanchor="top",        
            y=-0.2,               
            xanchor="center",     
            x=0.5,                
            font=dict(size=14)    
        ),
        xaxis=dict(visible=False),  # Hide x-axis
        yaxis=dict(visible=False)   # Hide y-axis   
    )
    plt.tight_layout()
    fig.write_image(os.path.join(save_path, f"{sub_index}_{bio_name}.svg"))
    fig.write_image(os.path.join(save_path, f"{sub_index}_{bio_name}.png"))


def plot_nm_range_site2(
    processing_dir,
    data_dir,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    save_plot=True,
    outputsuffix="estimate",
    experiment_id=0,
    batch_curve={"sex": ["Male", "Female"]},
    batch_marker={"site": ["BTH", "Cam-Can", "NIMH", "OMEGA", "HCP", "MOUS"]},
    new_names=["Theta", "Alpha", "Beta", "Gamma"],
    colors=["#006685", "#591154", "#E84653", "black", "#E6B213", "Slategrey"],
):
    """Function to plot notmative ranges. This function assumes only gender as batch effect
    stored in the first column of batch effect array.

    Args:
        processing_dir (str): Path to normative modeling processing directory.
        quantiles (list, optional): Plotted centiles. Defaults to [0.05, 0.25, 0.5, 0.75, 0.95].
        ind (int, optional): Index of target biomarker to plot. Defaults to 0.
        parallel (bool, optional): Is parallel NM used to estimate the model?. Defaults to True.
        save_plot (bool, optional): Save the plot?. Defaults to True.
        outputsuffix (str, optional): outputsuffix in normative modeling. Defaults to 'estimate'.
    """
    matplotlib.rcParams["pdf.fonttype"] = 42

    z_scores = st.norm.ppf(quantiles)
    # paths
    testrespfile_path = os.path.join(data_dir, "y_test.pkl")
    testcovfile_path = os.path.join(data_dir, "x_test.pkl")
    tsbefile = os.path.join(data_dir, "b_test.pkl")
    quantiles_path = os.path.join(processing_dir, "Quantiles_" + outputsuffix + ".pkl")
    # reading the paths
    X_test = pickle.load(open(testcovfile_path, "rb")).to_numpy(float)
    be_test = pickle.load(open(tsbefile, "rb"))
    Y_test = pickle.load(open(testrespfile_path, "rb"))
    temp = pickle.load(open(quantiles_path, "rb"))

    q = temp["quantiles"]
    synthetic_X = temp["synthetic_X"]
    quantiles_be = temp["batch_effects"]
    # converting age values to original space
    X_test = X_test * 100

    markers = ["o", "^"]
    curves_colors = ["#6E750E", "#A9561E"]
    curve_indx = int(np.where(be_test.columns == list(batch_curve.keys())[0])[0])
    hue_indx = int(np.where(be_test.columns == list(batch_marker.keys())[0])[0])

    be_test = be_test.to_numpy(float)

    num_biomarkers = q.shape[2]
    for ind in range(num_biomarkers):
        bio_name = Y_test.columns[ind]
        y_test = Y_test[[bio_name]].to_numpy(float)

        # fig, ax = plt.subplots(1,1, figsize=(8,6), sharex=True, sharey=True)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, sharey=True)

        for unique_hue in np.unique(be_test[:, hue_indx]).tolist():

            # Extract all male and female subjects and use different
            # markers for them
            for unique_marker in np.unique(be_test[:, curve_indx]).tolist():

                ts_idx = np.logical_and(
                    be_test[:, hue_indx] == unique_hue,
                    be_test[:, curve_indx] == unique_marker,
                )

                ax.scatter(
                    X_test[ts_idx],
                    y_test[ts_idx],
                    s=35,
                    alpha=0.6,
                    label=(
                        [*batch_curve.values()][0][int(unique_marker)]
                        + " "
                        + [*batch_marker.values()][0][int(unique_hue)]
                    ),
                    color=colors[int(unique_hue)],
                    marker=markers[int(unique_marker)],
                )

        for unique_marker in np.unique(be_test[:, curve_indx]).tolist():
            q_idx = np.where(quantiles_be[:, curve_indx] == unique_marker)[
                0
            ]  # only for males

            for i, v in enumerate(z_scores):

                if v == 0:
                    thickness = 3
                    linestyle = "-"
                else:
                    linestyle = "--"
                    thickness = 1

                x = np.asarray(synthetic_X[q_idx]).flatten()
                x = np.mean(x.reshape(-1, 100), axis=0)

                y = q[q_idx, i : i + 1, ind]
                y = np.asarray(y).flatten()
                y = np.mean(y.reshape(-1, 100), axis=0)

                ax.plot(
                    x.tolist(),
                    y.tolist(),
                    linewidth=thickness,
                    linestyle=linestyle,
                    alpha=1,
                    color=curves_colors[int(unique_marker)],
                )

            ax.grid(True, linewidth=0.5, alpha=0.5, linestyle="--")
            ax.set_ylabel(f"{new_names[ind]} (proportion)", fontsize=16)
            ax.set_xlabel("Age (years)", fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=14)

            ax.set_xlim(0, 48)

            for spine in ax.spines.values():
                spine.set_visible(False)

            if ind + 1 == num_biomarkers:
                ax.legend(loc="upper right", prop={"size": 14})
            plt.tight_layout()

        if save_plot:
            save_path = os.path.join(
                processing_dir, f"Figures_experiment{experiment_id}"
            )
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            plt.savefig(
                os.path.join(save_path, str(ind) + "_" + bio_name + ".svg"),
                dpi=300,
                format="svg",
            )
            plt.savefig(
                os.path.join(save_path, str(ind) + "_" + bio_name + ".png"),
                dpi=300,
                format="png",
            )


# ***
def box_plot_auc(df_AUCs, save_path, color="teal", showfliers=False, jitter=True):
    """ 
    Creates a box plot with overlaid strip plot to visualize AUC distributions.

    Args:
        df_AUCs (pd.DataFrame): DataFrame where each column represents AUCs of a model/condition.
        save_path (str): Directory where the plot images will be saved.
        color (str): Color for boxplot fill.
        showfliers (bool): Whether to show outlier points in the boxplot.
        jitter (bool): Whether to jitter the individual data points in strip plot.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """

    sns.set_theme(style="ticks", palette="pastel")

    os.makedirs(save_path, exist_ok=True)

    data_long = pd.melt(df_AUCs)

    fig = plt.figure(figsize=(6, 5))
    sns.boxplot(x='variable', y='value', data=data_long, boxprops=dict(facecolor=color, alpha=0.7),  
                showfliers=showfliers)

    sns.stripplot(x='variable', y='value', data=data_long,             
                  color='black', 
                  marker='o', 
                  size=6, 
                  alpha=0.6, 
                  jitter=jitter)

    sns.despine(offset=0, trim=True)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.ylabel('AUC', fontsize=20)
    plt.xlabel("")
    plt.grid()

    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.tight_layout()

    fig_path_svg = os.path.join(save_path, "AUC_box_plot.svg")
    fig_path_png = os.path.join(save_path, "AUC_box_plot.png")
    fig.savefig(fig_path_svg, dpi=600, format="svg")
    fig.savefig(fig_path_png, dpi=600, format="png")

    return fig

# ***
def joint_z_scores_scatter_plot(X, Y, bands_name, z_values = [0.674, 1.645], colors = ['#a0a0a0', '#202020'], save_path=None):
    """
    Creates a joint scatter plot of two z-scored features (e.g., neural bands), with visual cues for effect size and confidence.

    Args:
        X (array-like): Z-scores for the x-axis (e.g., one frequency band).
        Y (array-like): Z-scores for the y-axis (e.g., another frequency band).
        bands_name (list[str]): Names of the bands being compared, e.g., ['alpha', 'beta'].
        save_path (str, optional): Directory to save the plot. Saves SVG and PNG if provided.

    Returns:
        None. Displays and optionally saves the plot.
    """
    if X.ndim != 1 or Y.ndim != 1:
        raise ValueError("X and Y must be 1-dimensional arrays.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must be the same length.")
    if len(z_values) != len(colors):
        raise ValueError("Length of 'z_values' and 'colors' must match.")

    X = np.array(X)  
    Y = np.array(Y)  

    fig, ax = plt.subplots(figsize=(7, 6))
    plt.xlim(-4.2, 4.)
    plt.ylim(-4.2, 4.)
    
    sizes = 20 + (X - np.min(X)) / (np.max(X) - np.min(X)) * 500
    scatter = ax.scatter(
        X, Y, s=sizes, c=Y, cmap="inferno_r", edgecolor="black", alpha=0.8,
        vmin=np.min(Y), vmax=np.max(Y)
    )

    # Color gradient bar to show Y (Beta) effect
    sm = plt.cm.ScalarMappable(cmap="inferno_r", norm=plt.Normalize(vmin=np.min(Y), vmax=np.max(Y)))
    sm.set_array([])
    cbar_scatter = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
    cbar_scatter.set_label(f"{bands_name[1]} z-scores", fontsize=20)
    cbar_scatter.ax.tick_params(labelsize=0, length=0)

    ax.set_xlabel(f'{bands_name[0].capitalize()} z-scores', fontsize=22)
    ax.set_ylabel(f'{bands_name[1].capitalize()} z-scores', fontsize=22)

    # Remove unnecessary spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position(('outward', 10))
    ax.spines["left"].set_position(('outward', 10))

    # Draw square contour regions with sharp edges to show
    # different centiles of variation
    for i in range(len(z_values)):
        bound = list(reversed(z_values))[i]
        ax.add_patch(plt.Rectangle(
            (-bound, -bound), 2*bound, 2*bound,
            color=colors[i], alpha=0.4, ec=None
        ))
        ax.add_patch(plt.Rectangle(
            (-bound, -bound), 2*bound, 2*bound,
            fill=False, edgecolor='black', linewidth=3
        ))
        
    plt.tight_layout()
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.tick_params(axis='both', labelsize=18)

    if save_path:
        plt.savefig(os.path.join(save_path, f"{bands_name[0]}_{bands_name[1]}_z_scores_scatter.svg"), dpi=600, format="svg")
        plt.savefig(os.path.join(save_path, f"{bands_name[0]}_{bands_name[1]}_z_scores_scatter.png"), dpi=600, format="png")

    plt.show()





def z_scores_scatter_plot_continuum(
    X, Y, bands_name=["theta", "beta"], thr=0.68, save_path=None
):
    plt.figure(figsize=(10, 8))

    plt.ylim((-4, 4))
    plt.xlim((-4, 4))

    # Define a continuous color scale using the magnitude of X and Y
    color_values = np.sqrt(
        np.array(X) ** 2 + np.array(Y) ** 2
    )  # Using Euclidean distance

    # Normalize the color scale from 0-3.5 to improve contrast
    norm = mcolors.Normalize(vmin=0, vmax=3.5)

    # Create scatter plot with a colormap
    scatter = plt.scatter(
        X,
        Y,
        c=color_values,
        cmap="coolwarm",
        norm=norm,
        edgecolors="black",
        linewidth=0.2,
    )

    # Add the gray region and lines
    plt.fill_betweenx(
        y=[-thr, thr], x1=-thr, x2=thr, color="gray", alpha=0.5, label=f"|z| < {thr}"
    )
    plt.hlines(
        y=[-thr, thr], xmin=-4, xmax=4, colors="black", linestyles="--", linewidth=1.5
    )
    plt.vlines(
        x=[-thr, thr], ymin=-4, ymax=4, colors="black", linestyles="--", linewidth=1.5
    )

    # Set axis ticks
    ticks = [-3, -thr, 0, thr, 3]
    plt.xticks(ticks)
    plt.yticks(ticks)

    # Labeling
    plt.xlabel(f"{bands_name[0]} z-scores", fontsize=16)
    plt.ylabel(f"{bands_name[1]} z-scores", fontsize=16)

    # Style the plot
    plt.grid(alpha=0.5)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Magnitude of z-scores")

    # Finalize and save the plot
    plt.tight_layout()
    if save_path:
        plt.savefig(
            os.path.join(save_path, "z_scores_scatter.svg"), dpi=600, format="svg"
        )

    plt.show()


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


def z_scores_quadrant_contour_plot(
    X, Y, bands_name, percentiles=[0.05, 0.25, 0.50, 0.75, 0.95], save_path=None
):
    # Convert data to a Pandas DataFrame
    data = pd.DataFrame({"X": X, "Y": Y})

    # Compute magnitude of Z-scores, the higher magnitude, the darker colour
    data["Magnitude"] = np.sqrt(data["X"] ** 2 + data["Y"] ** 2)

    # Assign quadrants
    conditions = [
        (data["X"] >= 0) & (data["Y"] >= 0),
        (data["X"] < 0) & (data["Y"] >= 0),
        (data["X"] < 0) & (data["Y"] < 0),
        (data["X"] >= 0) & (data["Y"] < 0),
    ]
    choices = ["Q1", "Q2", "Q3", "Q4"]
    data["Quadrant"] = np.select(conditions, choices)

    # Define quadrant colormaps
    quadrant_cmaps = {
        "Q1": plt.cm.Reds,
        "Q2": plt.cm.YlGn,
        "Q3": plt.cm.GnBu,
        "Q4": plt.cm.RdPu,
    }

    # Set Fixed Color Normalization Range (-3.5 to 3.5)
    norm = plt.Normalize(vmin=0, vmax=3.5)

    # figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Contour plot for threshold
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

    # Ensure each quadrant gets visible coolours depending on mgnitude
    for quadrant, cmap in quadrant_cmaps.items():
        subset = data[data["Quadrant"] == quadrant]

        if not subset.empty:
            colors = cmap(norm(subset["Magnitude"]))
            # Scatter plot
            ax.scatter(
                subset["X"],
                subset["Y"],
                color=colors,
                edgecolors="black",
                linewidth=0.3,
                alpha=0.9,
                s=50,
            )

    # Axis settings
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ticks = [-3, -2, -1, 0, 1, 2, 3]
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax.grid(alpha=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Labels & Title
    plt.xlabel(f"{bands_name[0]} z-scores", fontsize=16)
    plt.ylabel(f"{bands_name[1]} z-scores", fontsize=16)
    ax.set_title("Z-scores contour plot", fontsize=18)

    plt.tight_layout()
    if save_path:
        plt.savefig(
            os.path.join(save_path, "z_scores_contour_plot.svg"), dpi=600, format="svg"
        )

    plt.show()


def plot_metrics(metrics_path, which_features, feature_new_name=[], save_path=None):
    with open(metrics_path, "rb") as file:
        metrics_dic = pickle.load(file)

    for metric in metrics_dic.keys():
        df_temp = pd.DataFrame(metrics_dic.get(metric)).loc[:, which_features]
        df_temp.columns = feature_new_name
        df_temp = df_temp.melt(var_name="Variable", value_name="Value")

        sns.set_theme(style="ticks", palette="pastel")

        sns.boxplot(
            x="Variable",
            y="Value",
            data=df_temp,
            boxprops=dict(facecolor="dimgrey", alpha=0.7),
            showfliers=False,
        )

        sns.stripplot(
            x="Variable",
            y="Value",
            data=df_temp,
            color="black",
            marker="o",
            size=6,
            alpha=0.5,
            jitter=True,
        )

        # Set the y-axis to scientific notation
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # Move the scale to the top left
        ax.yaxis.get_offset_text().set_position((-0.1, 1.05))

        sns.despine(offset=0, trim=True)
        plt.ylabel(metric.title())

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(
                os.path.join(save_path, f"{metric}_metric.svg"), dpi=600, format="svg"
            )
            plt.savefig(
                os.path.join(save_path, f"{metric}_metric.png"), dpi=600, format="png"
            )

        plt.close()


def qq_plot(processing_dir, 
    save_fig, 
    label_dict, 
    colors,
    markersize: int = 8,
    alpha: float = 0.6,
    lower_lim: float = -4.0,
    upper_lim: float = 4.0
    ):
    """
    Generate QQ plots of Z-score estimates for different biomarkers.

    This function reads a pickle file containing Z-score estimates,
    generates QQ plots comparing sample quantiles to theoretical quantiles
    for each specified marker, and optionally saves the plots to disk.

    Parameters
    ----------
    processing_dir : str
        Path to the directory containing the 'Z_estimate.pkl' file.
    save_fig : str or None
        Directory to save generated plots. If None, plots are not saved.
    label_dict : dict
        Dictionary mapping column index to their corresponding column name (biomarker name).
    colors : list of str
        List of color codes (e.g., hex or named colors) to use for each QQ plot marker.
        Should be the same length as `label_dict`.
    markersize : int, optional
        Size of the plot markers. Default is 8.
    alpha : float, optional
        Transparency level of the markers, between 0 (transparent) and 1 (opaque). Default is 0.6.
    lower_lim : float, optional
        Lower limit for both axes in the QQ plot. Default is -4.0.
    upper_lim : float, optional
        Upper limit for both axes in the QQ plot. Default is 4.0.

    Returns
    -------
    None
        Displays and optionally saves the QQ plots for each dataset in `label_dict`.

    """
    with open(os.path.join(processing_dir, "Z_estimate.pkl"), "rb") as file:
        z_scores = pickle.load(file)

    for indx, (key, value) in enumerate(label_dict.items()):

        plotkwargs = {
        "markerfacecolor": colors[indx], 
        "markeredgecolor": colors[indx], 
        "markersize": markersize, 
        "alpha": alpha}
        
        plt.figure(figsize=(5, 5))
        ax = plt.gca()  

        # Generate QQ plot without the line
        sm.qqplot(
            z_scores.iloc[:, value].to_numpy(), 
            line=None, 
            ax=ax, 
            **plotkwargs
        )

        # Add a red 45-degree line manually
        x = np.linspace(lower_lim, upper_lim, 100)
        ax.plot(x, x, color='black', linewidth=4, linestyle='--', alpha=1)

        plt.ylabel("Sample quantiles", fontsize=25)
        plt.xlabel("Theoretical quantiles", fontsize=25)

        plt.ylim((lower_lim, upper_lim))
        plt.xlim((lower_lim, upper_lim))

        # Customize spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_position(('outward', 10))
        ax.spines["left"].set_position(('outward', 10))

        plt.xticks(np.linspace(lower_lim, upper_lim, 5))
        plt.yticks(np.linspace(lower_lim, upper_lim, 5))
        plt.tick_params(axis='both', labelsize=25)

        plt.title(key.capitalize(), fontsize=25)
        plt.grid(True, axis='both', linestyle="--", color="gray", alpha=0.3)
        # Save the figure
        if save_fig is not None:

            plt.savefig(os.path.join(save_fig, f"{key}_qqplot.png"), dpi=600, bbox_inches='tight')
            plt.savefig(os.path.join(save_fig, f"{key}_qqplot.svg"), dpi=600, bbox_inches='tight')
        

        plt.show()
