import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def plot_nm_range(processing_dir, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], 
                  ind=0, parallel=True, save_plot=True, outputsuffix='estimate'):
    
    """Function to plot notmative ranges.

    Args:
        processing_dir (str): Path to normative modeling processing directory.
        quantiles (list, optional): Plotted centiles. Defaults to [0.05, 0.25, 0.5, 0.75, 0.95].
        ind (int, optional): Index of target biomarker to plot. Defaults to 0.
        parallel (bool, optional): Is parallel NM used to estimate the model?. Defaults to True.
        save_plot (bool, optional): Save the plot?. Defaults to True.
        outputsuffix (str, optional): outputsuffix in normative modeling. Defaults to 'estimate'.
    """
    
    z_scores = st.norm.ppf(quantiles)
    respfile = os.path.join(processing_dir, 'y_train.pkl')
    covfile = os.path.join(processing_dir, 'x_train.pkl')
    testrespfile_path = os.path.join(processing_dir, 'y_test.pkl')
    testcovfile_path = os.path.join(processing_dir, 'x_test.pkl')
    trbefile = os.path.join(processing_dir, 'b_train.pkl')
    tsbefile = os.path.join(processing_dir, 'b_test.pkl')
        
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

    synthetic_X = np.linspace(0.05, 0.95, 200)[:,np.newaxis] # Truncated
    
    X_train = pickle.load(open(covfile, 'rb')).to_numpy(float)
    X_test = pickle.load(open(testcovfile_path, 'rb')).to_numpy(float)
    be_test = pickle.load(open(tsbefile, 'rb')).to_numpy(float)
    be_train = pickle.load(open(trbefile, 'rb')).to_numpy(float)
    Y_train = pickle.load(open(respfile, 'rb'))
    bio_name = Y_train.columns[ind]
    Y_train = Y_train.to_numpy(float)[:,ind:ind+1]
    Y_test = pickle.load(open(testrespfile_path, 'rb')).to_numpy(float)[:,ind:ind+1]
    
    fig, ax = plt.subplots(1,2, figsize=(12,6), sharex=True, sharey=True)
        
    for be1 in list(np.unique(be_test[:,0])):
        model_be = np.repeat(np.array([[be1]]), synthetic_X.shape[0], axis=0)
        q = nm.get_mcmc_quantiles(synthetic_X, model_be, z_scores=z_scores) 
        tr_idx = be_train[:,0]==be1
        ts_idx = be_test[:,0]==be1
        ax[int(be1)].scatter(X_train[tr_idx], Y_train[tr_idx], s = 10, alpha = 0.5, 
                    label='Training')
        ax[int(be1)].scatter(X_test[ts_idx], Y_test[ts_idx], s = 10, alpha = 0.5, 
                    label='Test')
        for i, v in enumerate(z_scores):
            if v == 0:
                thickness = 3
                linestyle = "-"
            else:
                linestyle = "--"
                thickness = 1

            ax[int(be1)].plot(in_scaler.inverse_transform(synthetic_X), out_scaler.inverse_transform(q[i,:]).T, 
                        linewidth = thickness, linestyle = linestyle,  alpha = 0.9, color='k') 
        ax[int(be1)].grid(True, linewidth=0.5, alpha=0.5, linestyle='--')
        ax[int(be1)].set_title('Sex:' + str(int(be1)))
        ax[int(be1)].set_ylabel(bio_name, fontsize=10)
        ax[int(be1)].set_xlabel('Age', fontsize=16)
        
    ax[0].legend()
    plt.tight_layout()
    
    
    if save_plot:
        save_path = os.path.join(processing_dir, 'Figures')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, str(ind) + '_' + bio_name + '.png'), dpi=300)
    