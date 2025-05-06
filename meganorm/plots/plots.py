import os
import statsmodels.api as sm
import matplotlib
import pickle
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import pandas as pd
from typing import Union
import plotly.graph_objects as go
from scipy.stats import chi2


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


# ***
def plot_age_hist(
    df,
    site_names,
    save_path,
    lower_age_range=5,
    upper_age_range=90,
    step_size=5,
    colors=['#006685', '#591154', '#E84653', 'black', '#E6B213', "slategrey"]
):
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

    Saves
    -----
    age_hist.svg : SVG format plot saved in `save_path`.
    age_hist.png : PNG format plot saved in `save_path`.
    """
    if len(site_names) > len(colors):
        raise Exception("The number of colors is less than site_names, please specify a longer list of colors.")

    bins = list(range(lower_age_range, upper_age_range, step_size))
    ages = []

    ages = list(map(lambda i: df[df["site"] == i]["age"].to_numpy(), range(len(site_names))))

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
    plt.xticks(list(range(lower_age_range, upper_age_range, step_size*2)))
    plt.ylabel("Count", fontsize=25)
    plt.savefig(os.path.join(save_path, "age_hist.svg"), format="svg", dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(save_path, "age_hist.png"), format="png", dpi=600, bbox_inches="tight")


# ***
def plot_PNOCs(data, age_slices, save_path):
    """    
    Plots the Chrono-NeuroOscilloChart to visualize the contribution of the i-th
    centiles of each frequency bands to the overal power across brain.

    This function generates two bar plots (for males and females) showing the mean 
    activity (in percentage) and corresponding 95% confidence intervals for multiple
    frequency bands across age bins. The result is either saved as an SVG or shown interactively.

    Parameters
    ----------
    data : dict
        Nested dictionary structured as data[gender][frequency_band] = list of [mean, std],
        where each inner list represents one age slice.
    age_slices : list or array-like of int
        Starting values of age bins used for labeling the x-axis (e.g., [5, 10, 15, ..., 75]).
    save_path : str or None
        Path to save the resulting SVG plot. If None, the plot is displayed instead.

    Returns
    -------
    None

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
        
        bar_plot = df_means.plot(kind='bar', yerr=df_stds, capsize=4, stacked=True, ax=ax, alpha=0.6, 
                                 colormap=my_cmap)
        for p in bar_plot.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            bar_plot.text(x + width / 2, 
                          y + height / 2 + 2, 
                          f'{height:.0f}%', 
                          ha='center', 
                          va='center', fontsize=14)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Age ranges (years)', fontsize=16)
        if legend:
            ax.legend(loc='upper right', bbox_to_anchor=(1.1,1))  
        else:    
            ax.get_legend().remove()
            
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        ax.grid(False, axis='x')  
        ax.tick_params(axis='x', labelsize=20)
        ax.set_yticklabels([])  
    
    plot_gender_data(axes[0], data['Male'], "Males' Chrono-NeuroOscilloChart", 
                     colors= ["orange", "teal", "olive", "tomato"])
    
    plot_gender_data(axes[1], data['Female'], "Females' Chrono-NeuroOscilloChart", legend=False, 
                     colors=["orange", "teal", "olive", "tomato"])
    
    axes[1].set_xlabel('Age ranges (years)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'Chrono-NeuroOscilloChart.svg'), dpi=600)
    else:
        plt.show()
 
# ***    
def plot_growthchart(age_vector, centiles_matrix, cut=0, idp='', save_path=None, colors=None, centiles_name=['5th', '25th', '50th', '75th', '95th']):
    """
    Plot a growth chart for a given IDP, visualizing centile curves for males and females.

    Parameters
    ----------
    age_vector : numpy.ndarray
        1D array of age values.
    centiles_matrix : numpy.ndarray
        3D array (n_ages, n_centiles, n_sexes), with centiles along axis 1.
    cut : int, optional
        Age cutoff to compress ages above using a linear transform.
    idp : str, optional
        Biomarker or phenotype name for labeling.
    save_path : str, optional
        Directory to save plot. If None, displays plot.
    colors : dict, optional
        Dictionary with 'male' and 'female' keys and color lists.
    centiles_name : list of str, optional
        Labels for each centile curve.

    Returns
    -------
    None
    """
    if colors is None:
        colors = {
            'male': ['#4c061d', '#662333', '#803449', '#993d5e', '#b34e74'],
            'female': ['#FF6F00', '#FF8C1A', '#FFA726', '#FFB74D', '#FFD54F']
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

    genders = ['male', 'female']

    for j, gender in enumerate(genders):
        for i in range(len(centiles_name)):
            linestyle = '-' if i in [1,2,3] else '--'
            linewidth = 5 if i == 2 else (3 if i in [1, 3] else 2)
            axes[j].plot(transformed_age, centiles_matrix[:, i, j], label=f'{centiles_name[i]} Percentile', 
                         linestyle=linestyle, color=colors[gender][i], linewidth=linewidth, alpha=1 if i == 2 else 0.8)
        
        axes[j].fill_between(transformed_age, centiles_matrix[:, 0, j], centiles_matrix[:, 4, j], 
                             color=colors[gender][2], alpha=0.2)
        axes[j].fill_between(transformed_age, centiles_matrix[:, 1, j], centiles_matrix[:, 3, j], 
                             color=colors[gender][2], alpha=0.2)
        
        transformed_ticks = [age_transform(age) for age in np.concatenate((np.arange(min_age, cut+1, 2, dtype=int), 
                                                                           np.arange(np.ceil((cut+1)/10)*10, max_age+1, 10, dtype=int)))]
        axes[j].set_xticks(transformed_ticks)
        axes[j].set_xticklabels(np.concatenate((np.arange(min_age, cut+1, 2, dtype=int), 
                                                np.arange(np.ceil((cut+1)/10)*10, max_age+1, 10, dtype=int))), fontsize=37)
        axes[j].tick_params(axis='both', labelsize=45)
        axes[j].grid(True, which='both', linestyle='--', linewidth=2, alpha=0.95)
        axes[j].spines['top'].set_visible(False)
        axes[j].spines['right'].set_visible(False)
        # axes[j].set_xlabel('Age (years)', fontsize=28)
        

        for i, label in enumerate(centiles_name):
            axes[j].annotate(label, xy=(transformed_age[-1], centiles_matrix[-1, i, j]),
                             xytext=(8, 0), textcoords='offset points', fontsize=46, color=colors[gender][i], fontweight='bold')

    axes[0].set_ylabel(idp, fontsize=28)
    # axes[0].set_title("Males", fontsize=28)
    # axes[1].set_title("Females", fontsize=28)

    plt.tight_layout(pad=2)
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, idp.replace(" ", "_") + '_growthchart.svg'), dpi=600)

# ***
def plot_growthcharts(path, 
                      model_indices: list, 
                      biomarker_names: list, 
                      site: int = None, 
                      point_num: int = 100, 
                      number_of_sexs: int = 2, 
                      num_of_sites: int = None,
                      centiles_name: list = ['5th', '25th', '50th', '75th', '95th'],
                      colors: dict = None):
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

    Returns
    -------
    None
    """
    
    temp = pickle.load(open(os.path.join(path, 'Quantiles_estimate.pkl'),'rb'))

    q = temp['quantiles']
    x = temp['synthetic_X']
    b = temp['batch_effects']

    for i, idp in enumerate(model_indices):
        
        if not site:
            data = np.concatenate([q[b[:,0]== 0,:,idp:idp+1], 
                                q[b[:,0]== 1,:,idp:idp+1]], axis=2)
            data = data.reshape(num_of_sites, point_num, len(centiles_name), number_of_sexs) 
            data = data.mean(axis=0)
        if site:
            raise ValueError(f"still not implmented")
            #TODO

        plot_growthchart(x[0:point_num].squeeze(), data, cut=0, idp=biomarker_names[i], save_path=path, centiles_name=centiles_name, colors=colors)


# ***
def plot_INOCs(
    sub_index, current_value, q1, q3, percentile_5, percentile_95, percentile_50,
    title="Quantile-Based Gauge", min_value=0, max_value=1,
    show_legend=False, bio_name=None, save_path=None
):
    """
    Plots INOCs showing where a biomarker value falls within population quantiles.

    Parameters
    ----------
    sub_index : int or str
        Unique identifier for the subject (used for file naming).
    current_value : float
        Current observed biomarker value.
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
    current_value = round(current_value, 3)
    if bio_name == "Gamma":
        max_value = 0.1

    # Determine color based on value position
    if current_value < percentile_5:
        value_color = "rgb(8, 65, 92)"  # Extremely low
    elif current_value < q1:
        value_color = "rgb(0, 191, 255)"  # Below normal
    elif current_value <= q3:
        value_color = "rgb(129, 193, 75)"  # Normal
    elif current_value <= percentile_95:
        value_color = "rgb(255, 201, 20)"  # Above normal
    else:
        value_color = "rgb(188, 44, 26)"  # Extremely high

    number_font_size = 75 if show_legend else 120
    delta_font_size = 30 if show_legend else 90

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_value,
        number={
            'font': {'size': number_font_size, 'family': 'Arial', 'color': value_color}
        },
        delta={
            'reference': percentile_50,
            'position': "top",
            'font': {'size': delta_font_size},
            'valueformat': ".3f"
        },
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
                {'range': [min_value, percentile_5], 'color': "rgb(8, 65, 92)"},
                {'range': [percentile_5, q1], 'color': "rgb(0, 191, 255)"},
                {'range': [q1, q3], 'color': "rgb(129, 193, 75)"},
                {'range': [q3, percentile_95], 'color': "rgb(255, 201, 20)"},
                {'range': [percentile_95, max_value], 'color': "rgb(188, 44, 26)"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 6},
                'thickness': 0.75,
                'value': percentile_50,
            },
        },
        title={
            'text': bio_name,
            'font': {'size': 70, 'family': 'Arial', 'color': 'black'}
        }
    ))

    # Legend (as fake traces)
    if show_legend:
        colors = [
            ("rgb(8, 65, 92)", "0-5th Percentile (Extremely Low)"),
            ("rgb(0, 191, 255)", "5th-25th Percentile (Below Normal)"),
            ("rgb(129, 193, 75)", "25th-75th Percentile (Normal)"),
            ("rgb(255, 201, 20)", "75th-95th Percentile (Above Normal)"),
            ("rgb(188, 44, 26)", "95th-100th Percentile (Extremely High)")
        ]
        for color, label in colors:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                     marker=dict(size=12, color=color), name=label))

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
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
            font=dict(size=14)
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    plt.tight_layout()

    # Save if a valid path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.write_image(os.path.join(save_path, f"{sub_index}_{bio_name}.svg"))
        fig.write_image(os.path.join(save_path, f"{sub_index}_{bio_name}.png"))

    plt.show()


# ***
def plot_nm_range_site(
    processing_dir,
    data_dir,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    save_plot=True,
    outputsuffix="estimate",
    experiment_id=0,
    batch_curve={0: ["Male", "Female"]},
    batch_marker={1: ['BTH', 'Cam-Can', "NIMH", "OMEGA", "HCP", "MOUS"]},
    new_names=['Theta', 'Alpha', 'Beta', 'Gamma'],
    colors=['#006685', '#591154', '#E84653', 'black', '#E6B213', "Slategrey"]
):
    """
    Plot normative centile ranges with batch curves and test data overlaid.

    Parameters
    ----------
    processing_dir : str
        Path to the normative modeling processing directory.
    data_dir : str
        Path to the directory containing test data (x_test, y_test, b_test).
    quantiles : list of float
        List of quantiles to plot.
    save_plot : bool
        Whether to save the output plots.
    outputsuffix : str
        Suffix used in quantile file naming.
    experiment_id : int
        ID for saving plots under specific experiment folder.
    batch_curve : dict
        Dictionary with key as curve batch index, and value as list of category labels (e.g., {0: ["Male", "Female"]}).
    batch_marker : dict
        Dictionary with key as marker batch index, and value as list of category labels.
    new_names : list of str
        Names of biomarkers for plot titles/labels.
    colors : list of str
        List of colors corresponding to batch_marker levels.
    """
    matplotlib.rcParams['pdf.fonttype'] = 42

    # Load data
    x_test = pickle.load(open(os.path.join(data_dir, 'x_test.pkl'), 'rb')).to_numpy(float)
    y_test_df = pickle.load(open(os.path.join(data_dir, 'y_test.pkl'), 'rb'))
    b_test = pickle.load(open(os.path.join(data_dir, 'b_test.pkl'), 'rb')).to_numpy(float)

    temp = pickle.load(open(os.path.join(processing_dir, f'Quantiles_{outputsuffix}.pkl'), 'rb'))
    q = temp['quantiles']
    synthetic_X = temp['synthetic_X']
    quantiles_be = temp['batch_effects']

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
                ts_idx = np.logical_and(b_test[:, marker_idx] == m, b_test[:, curve_idx] == c)

                ax.scatter(
                    x_test[ts_idx],
                    y_test[ts_idx],
                    s=35,
                    alpha=0.6,
                    label=f"{curve_labels[int(c)]} {marker_labels[int(m)]}",
                    color=colors[int(m)],
                    marker=markers[int(c)],
                    edgecolors='none'
                )

        # Plot quantile curves
        for c in np.unique(b_test[:, curve_idx]):
            q_idx = np.where(quantiles_be[:, curve_idx] == c)[0]

            for i, v in enumerate(z_scores):
                linestyle = "-" if v == 0 else "--"
                thickness = 3 if v == 0 else 1

                x = synthetic_X[q_idx].reshape(-1, 100).mean(axis=0)
                y = q[q_idx, i:i+1, ind].reshape(-1, 100).mean(axis=0)

                ax.plot(
                    x.tolist(), y.tolist(),
                    linewidth=thickness,
                    linestyle=linestyle,
                    color=curve_colors[int(c)],
                    alpha=1
                )

        # Formatting
        ax.grid(True, linewidth=0.5, alpha=0.5, linestyle='--')
        ax.set_ylabel(new_names[ind], fontsize=25)
        ax.set_xlabel('Age (years)', fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=22)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if ind + 1 == num_biomarkers:
            ax.legend(loc="upper right", prop={'size': 17}, ncol=2)

        plt.tight_layout()

        if save_plot:
            save_path = os.path.join(processing_dir, f'Figures_experiment{experiment_id}')
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"{ind}_{bio_name}.svg"), dpi=300)
            plt.savefig(os.path.join(save_path, f"{ind}_{bio_name}.png"), dpi=300)


# ***
def box_plot_auc(
    df: pd.DataFrame,
    save_path: str,
    color: Union[str, list] = "teal",
    alpha: float = 0.7,
    biomarkers_new_name: list = None
):
    """
    Creates and saves a boxplot with stripplot overlay showing AUC values for different biomarkers.
    Supports transparency (`alpha`) and individual colors per biomarker.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each column represents a biomarker and each row an AUC value.
    save_path : str
        Directory to save the output plots.
    color : str or list of str
        Color for the boxes. If a list, must match number of biomarkers.
    alpha : float
        Transparency of the box colors (0 to 1).
    biomarkers_new_name : list, optional
        New labels for the biomarkers (x-axis). Must match number of columns.
    """

    if biomarkers_new_name:
        if len(biomarkers_new_name) != len(df.columns):
            raise ValueError("Length of 'biomarkers_new_name' must match number of columns in df.")
        df.columns = biomarkers_new_name

    data_long = pd.melt(df)

    if isinstance(color, str):
        palette = [color] * len(df.columns)
    elif isinstance(color, list):
        if len(color) != len(df.columns):
            raise ValueError("If 'color' is a list, it must match the number of biomarkers.")
        palette = color
    else:
        raise TypeError("'color' must be a string or a list of strings.")

    sns.set_theme(style="ticks")
    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(x='variable', y='value', data=data_long, palette=palette, showfliers=False)

    # Apply alpha to each PathPatch (box area)
    num_boxes = len(df.columns)
    for i, patch in enumerate(ax.patches[:num_boxes]):
        patch.set_facecolor(palette[i])
        patch.set_alpha(alpha)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)

    # Add stripplot
    sns.stripplot(
        x='variable',
        y='value',
        data=data_long,
        color='black',
        size=6,
        alpha=0.6,
        jitter=True
    )

    sns.despine(offset=0, trim=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.ylabel("AUC", fontsize=20)
    plt.xlabel("")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
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
    box_colors: list = ['#a0a0a0', '#202020'],
    save_path: str = None):
    """
    Creates a 2D scatter plot of z-scores between two bands (e.g., Theta and Beta),
    with overlaid contour boxes representing specific z-score boundaries.

    Parameters
    ----------
    X : array-like
        Z-scores for the x-axis (e.g., Theta band).
    Y : array-like
        Z-scores for the y-axis (e.g., Beta band).
    bands_name : list of str, optional
        Names of the bands to label the axes and colorbar. Default is ['Theta', 'Beta'].
    lower_lim : float, optional
        Lower axis limit for both x and y. Default is -4.0.
    upper_lim : float, optional
        Upper axis limit for both x and y. Default is 4.0.
    ticks : list, optional
        Custom tick locations for both axes. If None, default ticks are used.
    box_z_values : list of float, optional
        List of z-score thresholds to draw square boundary boxes. Must match `box_colors`.
    box_colors : list of str, optional
        List of colors for each corresponding `box_z_value`. Must match in length.
    save_path : str, optional
        Directory to save the plot as SVG and PNG. If None, the plot is not saved.

    Raises
    ------
    ValueError
        If `box_z_values` and `box_colors` are not of equal length.
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
        X, Y, s=sizes, c=Y, cmap="inferno_r", edgecolor="black", alpha=0.8,
        vmin=np.min(Y), vmax=np.max(Y)
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="inferno_r", norm=plt.Normalize(vmin=np.min(Y), vmax=np.max(Y)))
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
    ax.spines["bottom"].set_position(('outward', 10))
    ax.spines["left"].set_position(('outward', 10))

    # Draw filled boxes and borders
    for i in range(len(box_z_values)):
        bound = list(reversed(box_z_values))[i]
        ax.add_patch(plt.Rectangle(
            (-bound, -bound), 2 * bound, 2 * bound,
            color=box_colors[i], alpha=0.4, ec=None
        ))
        ax.add_patch(plt.Rectangle(
            (-bound, -bound), 2 * bound, 2 * bound,
            fill=False, edgecolor='black', linewidth=3
        ))

    # Axis ticks
    if ticks:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    else:
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
    ax.tick_params(axis='both', labelsize=18)

    plt.tight_layout()

    # Save if path is specified
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        base_name = f"{bands_name[0].capitalize()}_{bands_name[1].capitalize()}_z_scores_scatter"
        plt.savefig(os.path.join(save_path, f"{base_name}.svg"), dpi=600, format="svg")
        plt.savefig(os.path.join(save_path, f"{base_name}.png"), dpi=600, format="png")


def z_scores_contour_plot(
    X, Y, bands_name, percentiles=[0.05, 0.25, 0.50, 0.75, 0.95], save_path=None):
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
    metrics_path: str,
    which_biomarkers: list,
    biomarkers_new_name: list = None,
    colors: list = None,
    save_path: str = None,
    x_limits: dict = None
):
    """
    Plots statistical distributions of metrics across multiple biomarkers and models.

    This function generates a series of kernel density estimate (KDE) plots for a specified set of biomarkers
    and their associated metrics (e.g., MACE, skewness, kurtosis, W). The plots are arranged in a grid with one 
    row for each biomarker and one column for each metric. The plots can be saved in SVG and PNG formats.

    Parameters
    ----------
    metrics_path : str
        The file path to the pickle file containing the aggregated metrics data (e.g., the output from
        `aggregate_metrics_across_runs` function).
    which_biomarkers : list of str
        A list of biomarker names for which to generate the plots. These biomarkers should match the keys
        in the aggregated metrics data.
    biomarkers_new_name : list of str, optional
        A list of new names to assign to the biomarkers in the plot. The length of this list must match the
        number of biomarkers in `which_biomarkers`. If not provided, the original names are used.
    colors : list of str, optional
        A list of colors to use for the KDE plots. Each color corresponds to a different biomarker in the
        `which_biomarkers` list. If not provided, default pastel colors are used.
    save_path : str, optional
        The directory path where the generated plots will be saved. If not provided, the plots will not be saved.
    x_limits : dict, optional
        A dictionary specifying the x-axis limits for each metric. The dictionary should have metric names as keys 
        (e.g., 'MACE', 'skewness', 'kurtosis', 'W') and each value should be a tuple with the min and max 
        x-axis values (e.g., `{"MACE": (0, 10)}`). If not provided, the x-axis limits are automatically adjusted 
        based on the data.

    Returns
    -------
    None
        The function does not return any value. It directly generates the plots and saves them to the specified 
        `save_path` if provided.

    Notes
    -----
    - The function uses seaborn's `kdeplot` to generate KDE plots for each biomarker and metric combination.
    - The y-axis is hidden for all subplots, and the x-axis ticks are removed for all but the last row of subplots.
    - The plot titles display the metric name (e.g., 'MACE', 'skewness') at the top of each column, and the 
      biomarker names are shown on the left of each row.
    - The function supports KDE plotting for different metrics and can clip the x-axis limits based on the provided 
      `x_limits` dictionary.
    - If `save_path` is provided, the plots are saved in both SVG and PNG formats with high resolution (600 dpi).
    
    Example
    -------
    plot_metrics(
        metrics_path='/path/to/metrics.pkl',
        which_biomarkers=['biomarker_1', 'biomarker_2'],
        biomarkers_new_name=['New_Biomarker_1', 'New_Biomarker_2'],
        colors=['red', 'blue'],
        save_path='/path/to/save/plots',
        x_limits={'MACE': (0, 10), 'skewness': (-2, 2)}
    )
    """
    with open(metrics_path, "rb") as file:
        metrics_dic = pickle.load(file)
    
    # TODO: remove this one
    ordered_metrics = ['MACE', "W", "skewness", "kurtosis"]
    metrics_dic = {k: metrics_dic[k] for k in ordered_metrics}

    number_of_metrics = len(metrics_dic.keys()) 
    number_of_models = len(which_biomarkers)  

    fig, axes = plt.subplots(number_of_models, number_of_metrics, figsize=(11, 10))
    for ax in np.ravel(axes):
        ax.grid(True, axis='x', linestyle='--', color='gray', alpha=0.5)

    sns.set_theme(style="ticks", palette="pastel")
    
    for col_idx, (metric, data) in enumerate(metrics_dic.items()):

        # to select a subset of biomarkers when necessary
        df_temp = pd.DataFrame(data).loc[:, which_biomarkers]
        # If user wants to assign new names to the biomarkers
        if biomarkers_new_name:
            df_temp.columns = biomarkers_new_name  
        
        for row_idx, model in enumerate(df_temp.columns):

            ax = axes[row_idx, col_idx] 
            values = df_temp[model] 

            if colors:
                sns.kdeplot(values, ax=ax, fill=True, color=colors[row_idx], alpha=0.6)
            else:
                sns.kdeplot(values, ax=ax, fill=True, alpha=0.6)

            sns.rugplot(values, ax=ax, color="black", height=0.1)
            
            if x_limits:
                if metric == "MACE":
                    ax.set_xlim(x_limits.get("MACE")[0], x_limits.get("MACE")[1])
                if metric == "skewness":
                    ax.set_xlim(x_limits.get("skewness")[0], x_limits.get("skewness")[1])
                if metric == "kurtosis":
                    ax.set_xlim(x_limits.get("kurtosis")[0], x_limits.get("kurtosis")[1])
                if metric == "W":
                    ax.set_xlim(x_limits.get("W")[0], x_limits.get("W")[1])

            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.set_title(f"{metric.capitalize()}", fontsize=20) if row_idx == 0 else ax.set_title("")
            ax.set_ylabel(biomarkers_new_name[row_idx].capitalize(), fontsize=20) if col_idx == 0 else ax.set_ylabel("") 
            ax.set_xlabel("")
            ax.tick_params(axis="both", labelsize=16)
            
            if row_idx != number_of_models-1: ax.set_xticklabels([]) 

        plt.tight_layout(h_pad=0.01)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "metric.svg"), dpi=600, format="svg")
            plt.savefig(os.path.join(save_path, "metric.png"), dpi=600, format="png")


# ***
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

# ***
def plot_extreme_deviation(
    base_path: str,
    len_runs: int,
    save_path: str,
    mode: str,
    healthy_prefix: str,
    patient_prefix: str,
    legend: list,
    method: str,
    site_id: list = None,
    new_col_name: list = None,
    y_upper_lim :float = 15,
    y_lower_lim : float = 0
):
    """
    Computes and plots the percentage of subjects with extreme Z-scores 
    (positive > 2 and negative < -2) across multiple runs, comparing 
    patient and healthy groups. Two separate plots are generated: one for 
    positive deviations and one for negative deviations.

    Parameters
    ----------
    base_path : str
        Path template pointing to the directory containing run subfolders. 
        Must contain the placeholder "Run_0" to be replaced with run indices.
    len_runs : int
        Number of runs (iterations) to process.
    save_path : str
        Directory where the output plots will be saved.
    mode : str
        Descriptor used in the plot filenames to differentiate modes.
    healthy_prefix : str
        Filename prefix for healthy participant Z-score data (e.g., 'healthy').
    patient_prefix : str
        Filename prefix for patient Z-score data (e.g., 'patient').
    legend : list of str
        Legend labels to display in the plots for the healthy and patient groups.
    method : str
        Subfolder name inside each run directory where Z-score files are located.
    site_id : list of str, optional
        List of site IDs to filter participants by. If None, all participants are used.
    new_col_name : list of str, optional
        Optional list of new column names to rename the Z-score DataFrames.
    y_lower_lim : float, optional
        Lower limit for the y-axis in the plot. Default is 0.

    Returns
    -------
    df_c_pos : pandas.DataFrame
        DataFrame of positive extreme deviation proportions for the healthy group across runs.
    df_p_pos : pandas.DataFrame
        DataFrame of positive extreme deviation proportions for the patient group across runs.
    df_c_neg : pandas.DataFrame
        DataFrame of negative extreme deviation proportions for the healthy group across runs.
    df_p_neg : pandas.DataFrame
        DataFrame of negative extreme deviation proportions for the patient group across runs.
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

    def plot_bar(df_h, df_p, suffix, ylabel, legend_labels):
        means_h = df_h.mean()
        means_p = df_p.mean()
        ci_h = df_h.std() / np.sqrt(len(df_h)) * 1.96
        ci_p = df_p.std() / np.sqrt(len(df_p)) * 1.96

        x = np.arange(len(means_h))
        width = 0.3

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, means_h, yerr=ci_h, label=legend_labels[0],
               color="tomato", width=width, capsize=4, alpha=0.8)
        ax.bar(x + width / 2, means_p, yerr=ci_p, label=legend_labels[1],
               color="darkslategray", width=width, capsize=4, alpha=0.8)

        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(means_h.index, fontsize=12, rotation=45, ha="right")
        if suffix == "positive":
            ax.legend(fontsize=12)
        ax.set_ylim(y_lower_lim, y_upper_lim)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_position(("outward", 5))
        ax.spines["left"].set_position(("outward", 5))
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"extreme_{mode}_{suffix}_dev.svg"), dpi=600)
        plt.savefig(os.path.join(save_path, f"extreme_{mode}_{suffix}_dev.png"), dpi=600)
        plt.show()

    # Plot and save
    plot_bar(df_c_pos, df_p_pos, "positive", "Percentage", legend)
    plot_bar(df_c_neg, df_p_neg, "negative", "Percentage", legend)

    return df_c_pos, df_p_pos, df_c_neg, df_p_neg

