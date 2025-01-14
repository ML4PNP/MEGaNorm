import os
import matplotlib
import pickle
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go


def KDE_plot(data, experiments, metric, xlim = 'auto', fontsize=24):
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
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)
    
    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    
    # Define and use a simple function to label the plot in axes coordinates
    def label_KDE(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=fontsize)
        ax.plot([x.median(), x.median()], [0,10], c='w')
        if xlim == 'auto':
            ax.set_xlim([x_min - 0.05, x_max])  
        else:
            ax.set_xlim([xlim[0],xlim[1]])
        plt.xticks(fontsize=fontsize)    
    plt.yticks(fontsize=fontsize) 
    g.map(label_KDE, "x")
    
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)
    
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.set_xlabels(metric.upper(), fontsize=fontsize)
    g.set_ylabels("")
    g.despine(bottom=True, left=True)

    
def plot_nm_range(processing_dir, data_dir, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], 
                   age_range=[15, 90], ind=0, parallel=True, save_plot=True, outputsuffix='estimate'):
    
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
    testrespfile_path = os.path.join(data_dir, 'y_test.pkl')
    testcovfile_path = os.path.join(data_dir, 'x_test.pkl')
    tsbefile = os.path.join(data_dir, 'b_test.pkl')
        
    if parallel:
        nm = pickle.load(open(os.path.join(processing_dir, 'batch_' + str(ind+1),
                                       'Models/NM_0_0_'+ outputsuffix + '.pkl'), 'rb'))
        meta_data = pickle.load(open(os.path.join(processing_dir, 'batch_' + str(ind+1), 
                                                  'Models/meta_data.md'), 'rb'))
        in_scaler = meta_data['scaler_cov'][0]
        out_scaler = meta_data['scaler_resp'][0]
    else:
        nm = pickle.load(open(os.path.join(processing_dir,
                                       'Models/NM_0_' + str(ind) + '_'+ outputsuffix + '.pkl'), 'rb'))
        meta_data = pickle.load(open(os.path.join(processing_dir, 
                                                  'Models/meta_data.md'), 'rb'))
        in_scaler = meta_data['scaler_cov'][ind][0]
        out_scaler = meta_data['scaler_resp'][ind][0]

    synthetic_X = np.linspace(age_range[0], age_range[1], 200)[:,np.newaxis] # Truncated
    
    X_test = pickle.load(open(testcovfile_path, 'rb')).to_numpy(float)
    be_test = pickle.load(open(tsbefile, 'rb')).to_numpy(float)
    Y_test = pickle.load(open(testrespfile_path, 'rb'))
    bio_name = Y_test.columns[ind]
    Y_test = Y_test.to_numpy(float)[:,ind:ind+1]

    fig, ax = plt.subplots(1,1, figsize=(8,6), sharex=True, sharey=True)
    
    colors = ['#00BFFF', '#FF69B4']
    labels = ['Males', 'Females'] # assumes 0 for males and 1 for females
    
    for be1 in list(np.unique(be_test[:,0])):
        model_be = np.repeat(np.array([[be1]]), synthetic_X.shape[0], axis=0)
        q = nm.get_mcmc_quantiles(in_scaler.transform(synthetic_X), model_be, z_scores=z_scores) 
        ts_idx = be_test[:,0]==be1
        ax.scatter(X_test[ts_idx], Y_test[ts_idx], s = 15, alpha = 0.6, 
                    label=labels[int(be1)], color=colors[int(be1)])
        for i, v in enumerate(z_scores):
            if v == 0:
                thickness = 3
                linestyle = "-"
            else:
                linestyle = "--"
                thickness = 1
            y = out_scaler.inverse_transform(q[i,:]).T
            ax.plot(synthetic_X, y, linewidth = thickness, 
                    linestyle = linestyle,  alpha = 0.9, color=colors[int(be1)]) 
            if be1 ==0:
                plt.annotate(str(int(quantiles[i]*100))+'%', xy=(synthetic_X[-1], y[-1]),
                         xytext=(synthetic_X[-1] + 0.6, y[-1]), 
                            ha='left', va='center', fontsize=14)
        ax.grid(True, linewidth=0.5, alpha=0.5, linestyle='--')
        ax.set_ylabel(bio_name.replace('_', ' '), fontsize=10)
        ax.set_xlabel('Age', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    for spine in ax.spines.values():
        spine.set_visible(False)  
        
    ax.legend()
    plt.tight_layout()
    
    
    if save_plot:
        save_path = os.path.join(processing_dir, 'Figures')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, str(ind) + '_' + bio_name + '.png'), dpi=300)
        
        
        
def plot_comparison(path, hbr_configs, biomarker_num, metrics = ['Rho','SMSE','MSLL','MACE'], plot_type='boxplot'):
    
    
    results = {metric:np.zeros([biomarker_num, len(hbr_configs.keys())]) for metric in metrics}

    for m, method in enumerate(hbr_configs.keys()):
        for metric in metrics:
            with open(os.path.join(path, method, metric + '_estimate.pkl'), 'rb') as file:
                temp = pickle.load(file)
            results[metric][:,m] = temp.squeeze()

    methods = hbr_configs.keys()
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  
    axs = axs.flatten()  

    index = np.arange(len(methods))

    for ax, (metric_name, values) in zip(axs, results.items()):
        
        if plot_type == 'boxplot':
            ax.boxplot(values, positions=index, notch=True, showfliers=False)
        elif plot_type == 'violin':
            violin_parts = ax.violinplot(values, index, showmedians=True, showextrema=False)    

            for partname in ['cmedians']:
                vp = violin_parts[partname]
                vp.set_edgecolor("black")
                vp.set_linewidth(1)
                
            # Make the violin body blue with a red border:
            for vp in violin_parts['bodies']:
                vp.set_facecolor("#929591")
                vp.set_edgecolor("#000000")
                vp.set_alpha(1)
                
        ax.set_title(f'{metric_name} Comparison', fontsize=14)
        ax.set_xticks(index)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=12)  
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(path, plot_type + '_metric_comparison.png'), dpi=300)
    
    

def plot_age_dist(data, save_path=None):
    
    ## This function customized for camcan data and needs adaptation for other datasets
    
    data_copy = data.copy()
    data_copy['Gender'] = data_copy['gender'].map({0: 'Males', 1: 'Females'})

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    # KDE plot of age
    sns.kdeplot(data=data_copy, x='age', ax=ax1, fill=True,
                common_norm=False, palette="crest", hue='Gender',
                alpha=.3, linewidth=0)
    ax1.set_xlabel('Age', fontsize=16)
    ax1.set_ylabel('Density', fontsize=16)
    ax1.set_title('Age Distribution', fontsize=18)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.set_xticks(range(0, 101, 10))  

    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Secondary y-axis for histogram
    ax2 = ax1.twinx()
    sns.histplot(data=data_copy, x='age', hue='Gender', bins=10, palette="crest",
                 alpha=0.8, ax=ax2, element='step', linewidth=1.5)
    ax2.set_ylabel('Count', fontsize=16)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.1)  # Adjust y-axis to ensure visibility
    

    plt.tight_layout()
        
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'age_dist.png'), dpi=600)

    plt.show()
    

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
        
        bar_plot = df_means.plot(kind='bar', yerr=df_stds, capsize=4, stacked=True, ax=ax, alpha=0.7, 
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
        ax.set_xlabel('Age Ranges', fontsize=16)
        if legend:
            ax.legend(loc='upper right', bbox_to_anchor=(1.1,1))  
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
    
    axes[1].set_xlabel('Age Ranges', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'Chrono-NeuroOscilloChart.svg'), dpi=600)
    else:
        plt.show()



def plot_age_dist2(df, site_names, save_path):
    
    bins = list(range(5, 90, 5))
    ages = []

    for counter in range(len(site_names)):
        ages.append(df[df["site"]==counter]["age"].to_numpy()*100)
    
    plt.figure(figsize=(14, 8))
    plt.hist(ages, bins=bins, color=['#006685' ,'#591154' ,'#E84653' ,'black' ,'#E6B213'], 
             edgecolor="black", 
             alpha=0.6, 
             histtype="barstacked", 
             rwidth=0.9,)
    
    # Remove the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Offset the bottom and left spines
    plt.gca().spines['bottom'].set_position(('outward', 15))  # Set offset in points
    plt.gca().spines['left'].set_position(('outward', 15))

    # Optionally, trim the axis ticks to fit the visible range
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.grid(axis="y", color = 'black', linestyle = '--')
    plt.xlabel("Age (years)", fontsize=25)
    plt.legend(site_names, prop={'size': 20}, loc='upper right')
    plt.tick_params(axis="both", labelsize=17)
    plt.xticks(bins)
    plt.ylabel("Count",  fontsize=25)
    plt.savefig(os.path.join(save_path, "age_dis.svg"), format="svg", dpi=600, bbox_inches="tight")
    plt.close()

    

def plot_nm_range_site(processing_dir, data_dir, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], 
                        save_plot=True, outputsuffix='estimate', experiment_id=0,
                        batch_curve={0:["Male", "Female"]}, batch_marker={1:['CAMCAN', 'BTNRH']}):
    
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
    testrespfile_path = os.path.join(data_dir, 'y_test.pkl')
    testcovfile_path = os.path.join(data_dir, 'x_test.pkl')
    tsbefile = os.path.join(data_dir, 'b_test.pkl')
    quantiles_path = os.path.join(processing_dir, 'Quantiles_' + outputsuffix + '.pkl')
        
    X_test = pickle.load(open(testcovfile_path, 'rb')).to_numpy(float)
    be_test = pickle.load(open(tsbefile, 'rb')).to_numpy(float)
    Y_test = pickle.load(open(testrespfile_path, 'rb'))
    temp = pickle.load(open(quantiles_path, 'rb'))
    q = temp['quantiles']
    synthetic_X = temp['synthetic_X']
    quantiles_be = temp['batch_effects']
    
    X_test = X_test * 100
    
    colors = ['#00BFFF', '#FF69B4']

    curve_indx = list(batch_curve.keys())[0]
    hue_indx = list(batch_marker.keys())[0]
    
    for ind in range(q.shape[2]):
        
        bio_name = Y_test.columns[ind]
        y_test = Y_test.to_numpy(float)[:,ind:ind+1]

        fig, ax = plt.subplots(1,1, figsize=(8,6), sharex=True, sharey=True)
        
        for be1 in list(np.unique(be_test[:,curve_indx])):
                            
            # Male
            ts_idx = np.logical_and(be_test[:,curve_indx]==be1, be_test[:,hue_indx]==0)
            ax.scatter(X_test[ts_idx], y_test[ts_idx], s = 35, alpha = 0.6, 
                        label=[*batch_curve.values()][0][int(be1)]+ " " + [*batch_marker.values()][0][0], color=colors[int(be1)], marker="o")
            # Female
            ts_idx = np.logical_and(be_test[:,curve_indx]==be1, be_test[:,hue_indx]==1)
            ax.scatter(X_test[ts_idx], y_test[ts_idx], s = 35, alpha = 0.6, 
                        label=[*batch_curve.values()][0][int(be1)]+ " " + [*batch_marker.values()][0][1], color=colors[int(be1)], marker="^")

            q_idx = np.logical_and(quantiles_be[:,curve_indx]==be1, quantiles_be[:,hue_indx]==0) # only for males
            for i, v in enumerate(z_scores):
                if v == 0:
                    thickness = 3
                    linestyle = "-"
                else:
                    linestyle = "--"
                    thickness = 1
                
                y = q[q_idx,i:i+1,ind]
                
                ax.plot(synthetic_X[q_idx], y, linewidth = thickness, 
                        linestyle = linestyle,  alpha = 0.9, color=colors[int(be1)]) 
                if be1 ==0:
                    plt.annotate(str(int(quantiles[i]*100))+'%', xy=(synthetic_X[-1], y[-1]),
                            xytext=(synthetic_X[-1] + 0.6, y[-1]), 
                                ha='left', va='center', fontsize=14)
            ax.grid(True, linewidth=0.5, alpha=0.5, linestyle='--')
            ax.set_ylabel(bio_name.replace('_', ' '), fontsize=10)
            ax.set_xlabel('Age', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
        
        for spine in ax.spines.values():
            spine.set_visible(False)  
            
        ax.legend()
        plt.tight_layout()
        
        
        if save_plot:
            save_path = os.path.join(processing_dir, f'Figures_experiment{experiment_id}')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, str(ind) + '_' + bio_name + '.png'), dpi=300)
            

def plot_growthchart(age_vector, centiles_matrix, cut=0, idp='', save_path=None):
    
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
        'male': ['#0a2f66', '#0d3a99', '#1350d0', '#4b71db', '#8195e6'],
        'female': ['#b30059', '#cc0066', '#e60073', '#ff3399', '#ff66b2']
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

    genders = ['male', 'female']

    for j, gender in enumerate(genders):
        for i in range(5):
            linestyle = '-' if i in [1,2,3] else '--'
            linewidth = 5 if i == 2 else (3 if i in [1, 3] else 2)
            axes[j].plot(transformed_age, centiles_matrix[:, i, j], label=f'{[5, 25, 50, 75, 95][i]}th Percentile', 
                         linestyle=linestyle, color=colors[gender][i], linewidth=linewidth, alpha=1 if i == 2 else 0.8)
        
        axes[j].fill_between(transformed_age, centiles_matrix[:, 0, j], centiles_matrix[:, 4, j], 
                             color=colors[gender][2], alpha=0.1)
        axes[j].fill_between(transformed_age, centiles_matrix[:, 1, j], centiles_matrix[:, 3, j], 
                             color=colors[gender][2], alpha=0.1)
        
        transformed_ticks = [age_transform(age) for age in np.concatenate((np.arange(min_age, cut+1, 2, dtype=int), 
                                                                           np.arange(np.ceil((cut+1)/10)*10, max_age+1, 10, dtype=int)))]
        axes[j].set_xticks(transformed_ticks)
        axes[j].set_xticklabels(np.concatenate((np.arange(min_age, cut+1, 2, dtype=int), 
                                                np.arange(np.ceil((cut+1)/10)*10, max_age+1, 10, dtype=int))), fontsize=22)
        axes[j].tick_params(axis='y', labelsize=22)
        axes[j].grid(True, which='both', linestyle='--', linewidth=2, alpha=0.85)
        axes[j].spines['top'].set_visible(False)
        axes[j].spines['right'].set_visible(False)
        axes[j].set_xlabel('Age (years)', fontsize=28)
        
        #axes[j].legend(loc='upper left', fontsize=20)

        for i, label in enumerate(['5th', '25th', '50th', '75th', '95th']):
            axes[j].annotate(label, xy=(transformed_age[-1], centiles_matrix[-1, i, j]),
                             xytext=(8, 0), textcoords='offset points', fontsize=18, color=colors[gender][i], fontweight='bold')

        #axes[j].axvline(x=age_transform(cut), color='k', linestyle='--', linewidth=2, alpha=0.5)

    axes[0].set_ylabel(idp, fontsize=28)
    axes[0].set_title("Males", fontsize=28)
    axes[1].set_title("Females", fontsize=28)

    plt.tight_layout(pad=2)
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, idp.replace(" ", "_") + '_growthchart.png'), dpi=600)


def plot_growthcharts(path, idp_indices, idp_names, site=1, point_num=100):
    """Plotting growth charts for multiple idps.

    Args:
        path (string): Path to processing directory in which the quantiles are saved.
        idp_indices (list): A list of IDP indices.
        idp_names (list): A list of IDP names corresponding to IDP indices.
        site (int, optional): The site id to plot. Defaults to 0.
        point_num (int, optional): Number of points used in creating the synthetic X. Defaults to 100.
    """

    temp = pickle.load(open(os.path.join(path, 'Quantiles_estimate.pkl'),'rb'))

    q = temp['quantiles']
    x = temp['synthetic_X']
    b = temp['batch_effects']

    for i, idp in enumerate(idp_indices):

        data = np.concatenate([q[b[:,0]== 0,:,idp:idp+1], 
                            q[b[:,0]== 1,:,idp:idp+1]], axis=2)
        data = data.reshape(5, 100, 5, 2) 
        data = data.mean(axis=0)

        plot_growthchart(x[0:point_num].squeeze(), data, cut=0, idp=idp_names[i], save_path=path)
        


def plot_quantile_gauge(current_value, q1, q3, percentile_5, percentile_95, percentile_50, 
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

    if current_value < percentile_5:
        value_color = "rgba(115, 90, 63, 1)"  # Purple 
    elif current_value < q1:
        value_color = "rgba(255, 215, 0, 1)"  # Gold 
    elif current_value <= q3:
        value_color = "rgba(34, 139, 34, 1)"  # Green 
    elif current_value <= percentile_95:
        value_color = "rgba(255, 99, 71, 1)"  # Tomato red
    else:
        value_color = "rgba(128, 0, 128, 1)"  # Purple

    if show_legend:
        number_font_size = 75
        delta_font_size = 30
    else:
        number_font_size = 150
        delta_font_size = 50
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_value,
        number={'font': {'size': number_font_size, 'family': 'Arial', 'color': value_color}},  
        delta={'reference': percentile_50, 'position': "top", 'font': {'size': delta_font_size}},
        gauge={
            'axis': {
                'range': [min_value, max_value],
                'tickfont': {'size': 30, 'family': 'Arial', 'color': 'black'},
                'showticklabels': True,
                'tickwidth': 2,
                'tickcolor': "lightgrey",
                'tickvals': [round(min_value + i * (max_value - min_value) / 10, 2) for i in range(11)],  
            },
            'bar': {'color': "rgb(255, 69, 58)"},  
            'steps': [
                {'range': [min_value, percentile_5], 'color': "rgba(115, 90, 63, 1)"},  # Purple 
                {'range': [percentile_5, q1], 'color': "rgba(255, 215, 0, 0.6)"},  # Warm gold 
                {'range': [q1, q3], 'color': "rgba(34, 139, 34, 0.7)"},  # Forest green 
                {'range': [q3, percentile_95], 'color': "rgba(255, 99, 71, 0.6)"},  # Soft tomato red
                {'range': [percentile_95, max_value], 'color': "rgba(128, 0, 128, 0.9)"},  # dark Purple
            ],
            'threshold': {
                'line': {'color': "black", 'width': 6},  # Black line for the 0.5th percentile marker
                'thickness': 0.75,
                'value': percentile_50, 
            },
        },
        title={
            'text': bio_name,
            'font': {'size': 50, 'family': 'Arial', 'color': 'black'}
        }
    ))

    if show_legend:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color="rgba(115, 90, 63, 1)"),
                                 name="0-5th Percentile (Extremely Low)"))

        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color="rgba(255, 215, 0, 0.6)"),
                                 name="5th-25th Percentile (Below Normal)"))

        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color="rgba(34, 139, 34, 0.7)"),
                                 name="25th-75th Percentile (Normal)"))

        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color="rgba(255, 99, 71, 0.6)"),
                                 name="75th-95th Percentile (Above Normal)"))

        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color="rgba(128, 0, 128, 0.9)"),
                                 name="95th-100th Percentile (Extremely High)"))
    
    # Update layout for better aesthetics
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=50, b=100 if show_legend else 30, l=30, r=30),  # Adjust bottom margin for legend
        showlegend=show_legend,
        width=1100,
        height=700,
        legend=dict(
            orientation="h",      # Horizontal orientation for legend
            yanchor="top",        # Align legend to top
            y=-0.2,               # Place below the chart
            xanchor="center",     # Center legend horizontally
            x=0.5,                # Centered under the chart
            font=dict(size=14)    # Set font size for readability
        ),
        xaxis=dict(visible=False),  # Hide x-axis
        yaxis=dict(visible=False)   # Hide y-axis   
    )

    fig.write_image(os.path.join(save_path, f"{bio_name}.png"))



def plot_feature_scatter(df, feature_names, save_fig_path):
    """
    pltots the scatter plot of the specified features
    """
    import pandas as pd
    import math

    fig, ax = plt.subplots(math.ceil(len(feature_names)/2), 2, figsize=(15,10))
    ax = ax.flatten()

    for coutner, name in enumerate(feature_names):
        sns.scatterplot(data=df, x="age", y=name, hue="site", size=10, ax=ax[coutner], palette=["orange", "teal", "black"], legend=False)
        ax[coutner].set_title(name)
        
        ax[coutner].set_xlabel("Age")
    # ax[coutner].legend(["BTH", "CAMCAN", "NIMH"])
    plt.savefig(save_fig_path, dpi=600)






def plot_nm_range_site2(processing_dir, data_dir, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], 
                        save_plot=True, outputsuffix='estimate', experiment_id=0,
                        batch_curve={"sex":["Male", "Female"]}, batch_marker={"site":['BTH', 'Cam-Can', "NIMH", "OMEGA", "HCP"]},
                        new_names = ['Theta', 'Alpha','Beta', 'Gamma']):
    
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
    matplotlib.rcParams['pdf.fonttype']=42

    z_scores = st.norm.ppf(quantiles)
    # paths
    testrespfile_path = os.path.join(data_dir, 'y_test.pkl')
    testcovfile_path = os.path.join(data_dir, 'x_test.pkl')
    tsbefile = os.path.join(data_dir, 'b_test.pkl')
    quantiles_path = os.path.join(processing_dir, 'Quantiles_' + outputsuffix + '.pkl')
    # reading the paths
    X_test = pickle.load(open(testcovfile_path, 'rb')).to_numpy(float)
    be_test = pickle.load(open(tsbefile, 'rb'))
    Y_test = pickle.load(open(testrespfile_path, 'rb'))
    temp = pickle.load(open(quantiles_path, 'rb'))

    q = temp['quantiles']
    synthetic_X = temp['synthetic_X']
    quantiles_be = temp['batch_effects']
    # converting age values to original space
    X_test = X_test * 100

    colors =  ['#006685' ,'#591154' ,'#E84653' ,'black' ,'#E6B213']
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
        fig, ax = plt.subplots(1,1, figsize=(8,6), sharex=True, sharey=True)
        
        for unique_hue in np.unique(be_test[:,hue_indx]).tolist():

        
            # Extract all male and female subjects and use different 
            # markers for them
            for unique_marker in np.unique(be_test[:,curve_indx]).tolist():

                ts_idx = np.logical_and(be_test[:,hue_indx]==unique_hue, 
                                        be_test[:,curve_indx]==unique_marker)
                
                ax.scatter(X_test[ts_idx], 
                           y_test[ts_idx], 
                            s = 35, 
                            alpha = 0.6, 
                            label=([*batch_curve.values()][0][int(unique_marker)] 
                                        + " " 
                                        + [*batch_marker.values()][0][int(unique_hue)]), 
                            color=colors[int(unique_hue)], 
                            marker=markers[int(unique_marker)])
        
        for unique_marker in np.unique(be_test[:,curve_indx]).tolist():
            q_idx = np.where(quantiles_be[:,curve_indx]==unique_marker)[0] # only for males
            
            for i, v in enumerate(z_scores):

                if v == 0:
                    thickness = 3
                    linestyle = "-"
                else:
                    linestyle = "--"
                    thickness = 1

                x = np.asarray(synthetic_X[q_idx]).flatten()
                x = np.mean(x.reshape(-1, 100), axis=0)

                y = q[q_idx,i:i+1,ind]
                y = np.asarray(y).flatten()
                y = np.mean(y.reshape(-1, 100), axis=0)

                ax.plot(x.tolist(), y.tolist(), linewidth = thickness, 
                        linestyle = linestyle,  alpha = 1, color=curves_colors[int(unique_marker)]) 

            ax.grid(True, linewidth=0.5, alpha=0.5, linestyle='--')
            ax.set_ylabel(f"{new_names[ind]} (proportion)", fontsize=10)
            ax.set_xlabel('Age (years)', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)

            for spine in ax.spines.values():
                spine.set_visible(False)  
                
            ax.legend()
            plt.tight_layout()
        
        if save_plot:
            save_path = os.path.join(processing_dir, f'Figures_experiment{experiment_id}')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, str(ind) + '_' + bio_name + '.svg'), dpi=300, format="svg")



def box_plot_auc(df, save_path):

    # Melt the DataFrame to long format for Seaborn
    data_long = pd.melt(df)


    plt.figure(figsize=(6, 5))
    # colors = ['#E6B213', 'sandybrown', '#E84653', 'lightseagreen']
    sns.boxplot(x='variable', y='value', data=data_long, color="lightgray")#, palette=colors)

    sns.stripplot(x='variable', y='value', data=data_long, color='black', marker='o', size=6, alpha=0.7, jitter=True)

    means = df.mean(axis=0)
    for i, mean in enumerate(means):
        plt.text(i, mean, '', color='black', ha='center', va='center', fontsize=2)


    # Customize the plot
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title('AUCs Across 10 Runs', fontsize=16)
    plt.ylabel('AUC', fontsize=16)
    plt.xlabel("")
    plt.grid()


    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)

    plt.tight_layout()
    # Show the plot
    plt.savefig(os.path.join(save_path, "AUCs.svg"), dpi=600, format="svg")



def z_scores_scatter_plot(X, Y, bands_name=["theta", "beta"], thr=0.68, save_path=None):



    plt.figure(figsize=(8, 8))

    plt.ylim((-4, 4))
    plt.xlim((-4, 4))

    # Define the fixed order of labels and corresponding colors
    order = [
        (f'High {bands_name[1]} - Low {bands_name[0]}', 'red'),
        (f'High {bands_name[0]} - Low {bands_name[1]}', 'purple'),
        (f'High {bands_name[1]} - Normal {bands_name[0]}', 'blue'),
        (f'Normal {bands_name[0]} - Low {bands_name[1]}', 'orange'),
        (f'Normal {bands_name[1]} - High {bands_name[0]}', 'green'),
        (f'Normal {bands_name[1]} - Low {bands_name[0]}', 'teal'),
        (f'Low {bands_name[1]} - Low {bands_name[0]}', 'pink'),
        (f'High {bands_name[1]} - High {bands_name[0]}', 'mediumvioletred'),
        (f'Normal range', 'black')
    ]

    # Initialize lists for colors and labels
    colors = []
    labels = []

    # Assign colors and labels based on conditions
    for x, y in zip(X, Y):
        if y > thr and x < -thr:
            colors.append('red')
            labels.append('High beta - Low theta')
        elif thr > thr and y < -thr:
            colors.append('purple')
            labels.append('High theta - Low beta')
        elif y > thr and -thr < x < thr:
            colors.append("blue")
            labels.append('High beta - Normal theta')
        elif -thr < x < thr and y < -thr:
            colors.append("orange")
            labels.append('Normal theta - Low beta')
        elif -thr < y < thr and x > thr:
            colors.append("olive")
            labels.append('Normal beta - High theta')
        elif -thr < y < thr and x < -thr:
            colors.append("teal")
            labels.append('Normal beta - Low theta')
        elif y < -thr and x < -thr:
            colors.append("pink")
            labels.append('Low beta - Low theta')
        elif  y > thr and x > thr:
            colors.append("mediumvioletred")
            labels.append('High beta - High theta')
        else:
            colors.append('black')
            labels.append('Normal range')

    # Create the legend handles in the correct order
    handles = []
    for label, color in order:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label))

    # Plot the scatter plot
    plt.scatter(X, Y, color=colors)

    # Add the gray region and lines
    plt.fill_betweenx(y=[-thr, thr], x1=-thr, x2=thr, color='gray', alpha=0.5, label=f"|z| < {thr}")
    plt.hlines(y=[-thr, thr], xmin=-thr, xmax=thr, colors='black', linestyles='--', linewidth=1.5)
    plt.vlines(x=[-thr, thr], ymin=-thr, ymax=thr, colors='black', linestyles='--', linewidth=1.5)

    # Set axis ticks
    ticks = [-3, -thr, 0, thr, 3]
    plt.xticks(ticks)
    plt.yticks(ticks)

    # Labeling
    plt.xlabel('Theta z-scores', fontsize=16)
    plt.ylabel('Beta z-scores', fontsize=16)

    # Style the plot
    plt.grid(alpha=0.5)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)

    # Add the legend with the correct order
    plt.legend(handles=handles, fontsize=13)

    # Finalize and save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "z_scores_scatter.png"), dpi=600, format="svg")




def plot_metrics(metrics_path, which_features,
                  feature_new_name=[], save_path = None):

    # Use valid hexadecimal colors
    colors = ["#9E6240", "#819595", "#5F0F40", "#0F4C5C"]
    
    with open(metrics_path, "rb") as file:
        metrics_dic = pickle.load(file)
    
    for metric in metrics_dic.keys():
        df_temp = pd.DataFrame(metrics_dic.get(metric)).loc[:, which_features]
        df_temp.columns = feature_new_name
        
        # Reshape the data for boxplot
        df_temp = df_temp.melt(var_name='Variable', value_name='Value')

        sns.set_theme(style="ticks", palette="pastel")
        
        # Use palette instead of color
        sns.boxplot(x='Variable', y='Value', data=df_temp, palette=colors)
        sns.despine(offset=0, trim=True)
        plt.xlabel("Frequency Bands")
        plt.ylabel(metric.title())

        plt.savefig(os.path.join(save_path, "z_scores_scatter.png"), dpi=600, format="svg")
        plt.close()