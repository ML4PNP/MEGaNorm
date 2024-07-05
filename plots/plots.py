import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import pandas as pd



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
            ax.violinplot(values, index, showmedians=True, showextrema=False)    
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(index)
        ax.set_xticklabels(methods, rotation=45, ha="right")  
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
    

def plot_neurooscillochart(data, save_path=None):
    # Age ranges
    ages = [f"{i*10}-{(i+1)*10}" for i in range(1, 9)]
    
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    def plot_gender_data(ax, gender_data, title, legend=True):
        
        means = {k: [item[0] * 100 for item in v] for k, v in gender_data.items()}  
        stds = {k: [item[1] * 100 * 1.96 for item in v] for k, v in gender_data.items()}  
        
        df_means = pd.DataFrame(means, index=ages)
        df_stds = pd.DataFrame(stds, index=ages)
        
        bar_plot = df_means.plot(kind='bar', yerr=df_stds, capsize=4, stacked=True, ax=ax, alpha=0.7, colormap='Set2')
        for p in bar_plot.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            bar_plot.text(x + width / 2, 
                          y + height / 2, 
                          f'{height:.0f}%', 
                          ha='center', 
                          va='center')
        ax.set_title(title)
        ax.set_xlabel('Age Ranges', fontsize=16)
        if legend:
            ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))  
            ax.get_legend().remove()
            
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        ax.grid(False, axis='x')  
        ax.tick_params(axis='x', labelsize=14)
        ax.set_yticklabels([])  
    
    plot_gender_data(axes[0], data['Male'], "Males' Chrono-NeuroOscilloChart")
    
    plot_gender_data(axes[1], data['Female'], "Females' Chrono-NeuroOscilloChart", legend=False)
    
    axes[1].set_xlabel('Age Ranges', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'Chrono-NeuroOscilloChart.png'), dpi=600)
    else:
        plt.show()

