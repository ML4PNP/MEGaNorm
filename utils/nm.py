import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from plots.plots import KDE_plot



def hbr_data_split(data, save_path, covariates=['age'], batch_effects=None, train_split=0.5, 
                  validation_split=None, drop_nans=False, random_seed=42):
    
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
        data = data.dropna(axis=1)

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
        b_train = train_set.loc[:, batch_effects] if batch_effects is not None else pd.DataFrame(np.zeros([x_train.shape[0],1], dtype=int), 
                                                                                    index=x_train.index, columns=['site'])
        y_train = train_set.drop(columns=covariates+batch_effects) if batch_effects is not None else train_set.drop(columns=covariates) 
        x_train_all.append(x_train), y_train_all.append(y_train), b_train_all.append(b_train)

        if validation_split is not None:
            validation_num = np.round(validation_split * data_site.shape[0]).astype(int)
            validation_set = data_site.iloc[rand_idx[train_num:train_num + validation_num]]
            x_val = validation_set.loc[:, covariates]
            b_val = validation_set.loc[:, batch_effects] if batch_effects is not None else pd.DataFrame(np.zeros([x_val.shape[0],1], dtype=int), 
                                                                                            index=x_val.index, columns=['site'])
            y_val = validation_set.drop(columns=covariates+batch_effects) if batch_effects is not None else validation_set.drop(columns=covariates) 
            x_val_all.append(x_val), y_val_all.append(y_val), b_val_all.append(b_val)

            test_set = data_site.iloc[rand_idx[train_num + validation_num:]]
        else:
            validation_set = None
            test_set = data_site.iloc[rand_idx[train_num:]]

        x_test = test_set.loc[:, covariates]
        b_test = test_set.loc[:, batch_effects] if batch_effects is not None else pd.DataFrame(np.zeros([x_test.shape[0],1], dtype=int), 
                                                                                    index=x_test.index, columns=['site'])
        y_test = test_set.drop(columns=covariates+batch_effects) if batch_effects is not None else test_set.drop(columns=covariates) 
        x_test_all.append(x_test), y_test_all.append(y_test), b_test_all.append(b_test)


    # train
    pd.concat(x_train_all, axis=0).to_pickle(os.path.join(save_path, 'x_train.pkl'))
    pd.concat(y_train_all, axis=0).to_pickle(os.path.join(save_path, 'y_train.pkl'))
    pd.concat(b_train_all, axis=0).to_pickle(os.path.join(save_path, 'b_train.pkl'))
    # validation
    if validation_split is not None:
        pd.concat(x_val_all, axis=0).to_pickle(os.path.join(save_path, 'x_val.pkl'))
        pd.concat(y_val_all, axis=0).to_pickle(os.path.join(save_path, 'y_val.pkl'))
        pd.concat(b_val_all, axis=0).to_pickle(os.path.join(save_path, 'b_val.pkl'))
    # test
    pd.concat(x_test_all, axis=0).to_pickle(os.path.join(save_path, 'x_test.pkl'))
    pd.concat(y_test_all, axis=0).to_pickle(os.path.join(save_path, 'y_test.pkl'))
    pd.concat(b_test_all, axis=0).to_pickle(os.path.join(save_path, 'b_test.pkl'))
    
    
    return y_test.shape[1]



def mace(nm, x_test, y_test, be_test, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], plot=False):
    
    z_scores = st.norm.ppf(quantiles)
    batch_ids = np.unique(be_test)    
    batch_mace = np.zeros([len(batch_ids),])
    empirical_quantiles = []
    
    for b, batch_id in enumerate(batch_ids):
        model_be = np.repeat(np.array([[batch_id]]), x_test[be_test==batch_id,:].shape[0])
        mcmc_quantiles = nm.get_mcmc_quantiles(x_test[be_test==batch_id,:], model_be, z_scores=z_scores).T
        empirical_quantiles.append((mcmc_quantiles >= y_test[be_test==batch_id,0:1]).mean(axis=0))
        batch_mace[b] = np.abs((np.array(quantiles) - empirical_quantiles[b])).mean()
        
    if plot:
        plt.figure(figsize=(10, 6))
        sns.set_context("notebook", font_scale=2)
        sns.lineplot(x = quantiles, y = quantiles, color = "magenta", linestyle='--', linewidth=3, label = "ideal")
        for b, batch_id in enumerate(batch_ids): 
            sns.lineplot(x = quantiles, y = empirical_quantiles[b], color = "black", linestyle = "dashdot", 
                         linewidth=3, label = f"observed {b}")
            sns.scatterplot(x = quantiles, y = empirical_quantiles[b], marker="o", s = 150, alpha=0.5)
        plt.legend()
        plt.xlabel("True Quantile")
        plt.ylabel("Empirical Quantile")
        _ = plt.title("Reliability diagram")
    
    return batch_mace.mean()



def evaluate_mace(model_path, X_path, y_path, be_path, save_path=None, model_id=0,
                       quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], plot=False, outputsuffix='ms'):
    
    nm = pickle.load(open(os.path.join(model_path, 'NM_0_' + str(model_id) + '_' + outputsuffix + '.pkl'), 'rb'))
    x_test = pickle.load(open(X_path, 'rb')).to_numpy()
    be_test = pickle.load(open(be_path, 'rb')).to_numpy().squeeze()
    y_test = pickle.load(open(y_path, 'rb')).to_numpy()[:,model_id:model_id+1]

    meta_data = pickle.load(open(os.path.join(model_path, 'meta_data.md'), 'rb'))
    
    cov_scaler =  meta_data['scaler_cov']
    res_scaler =  meta_data['scaler_resp']
    
    if len(cov_scaler)>0:
        x_test = cov_scaler[model_id][0].transform(x_test)
    if len(res_scaler)>0:    
        y_test = res_scaler[model_id][0].transform(y_test)

    z_scores = st.norm.ppf(quantiles)
    batch_num = be_test.shape[1]
    
    batch_mace = []
    empirical_quantiles = []
    
    b = 0
    for i in range(batch_num):
        batch_ids = list(np.unique(be_test[:,i]))   
        if len(batch_ids)>1:
            for batch_id in batch_ids:
                model_be = be_test[be_test[:,i]==batch_id,:]            
                mcmc_quantiles = nm.get_mcmc_quantiles(x_test[be_test[:,i]==batch_id,:], model_be, z_scores=z_scores).T
                empirical_quantiles.append((mcmc_quantiles >= y_test[be_test[:,i]==batch_id,:]).mean(axis=0))
                batch_mace.append(np.abs(np.array(quantiles) - empirical_quantiles[b]).mean())  
                b += 1              
    
    batch_mace = np.array(batch_mace)
        
    if plot:
        plt.figure(figsize=(10, 6))
        sns.set_context("notebook", font_scale=2)
        sns.lineplot(x = quantiles, y = quantiles, color = "magenta", linestyle='--', linewidth=3, label = "ideal")
        b = 0
        for i in range(batch_num):
            batch_ids = list(np.unique(be_test[:,i]))
            for batch_id in batch_ids:
                sns.lineplot(x = quantiles, y = empirical_quantiles[b], color = "black", linestyle = "dashdot", 
                         linewidth=3, label = f"observed {b}")
                sns.scatterplot(x = quantiles, y = empirical_quantiles[b], marker="o", s = 150, alpha=0.5)
                b += 1
        plt.legend()
        plt.xlabel("True Quantile")
        plt.ylabel("Empirical Quantile")
        _ = plt.title("Reliability diagram")
        plt.savefig(os.path.join(save_path, 'MACE_' + str(model_id) + '.png'), dpi=300)
    
    return batch_mace.mean()



def model_quantile_evaluation(configs, save_path, valcovfile_path, 
                              valrespfile_path, valbefile, bio_num, plot=True, outputsuffix='ms'):
    
    mace = np.zeros([len(configs.keys()), bio_num])
    best_models = []

    for c, config in enumerate(configs.keys()):
        for ind in range(bio_num):
            mace[c,ind] = evaluate_mace(os.path.join(save_path, config, 'Models'), valcovfile_path, 
                                                    valrespfile_path, valbefile, model_id=ind,
                                                    quantiles=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
                                                    outputsuffix=outputsuffix)
            print(f'Config:{config}, id:{ind}')
        
        with open(os.path.join(save_path, config, 'MACE_' + outputsuffix + '.pkl'), 'wb') as file:
             pickle.dump(mace[c,:].T, file)
       

    for ind in range(bio_num):
        best_models.append(list(configs.keys())[np.argmin(mace[:,ind])])
    
    bio_ids = dict()
    for model in np.unique(best_models):
        bio_ids[model] = np.where(np.array(best_models)==model)[0]
    
    with open(os.path.join(save_path, 'model_selection_results.pkl'), 'wb') as file:
        pickle.dump({'best_models':best_models, 'bio_ids':bio_ids, 'mace':mace}, file)


    if plot:
        KDE_plot(mace, list(configs.keys()), 'MACE')
        plt.savefig(os.path.join(save_path, 'model_comparison_mace.png'), dpi=600)
    
    return mace, best_models, bio_ids


def calculate_oscilochart(model_path, gender_ids, frequency_band_model_ids, age_range = [5,90], 
                         outputsuffix='_estimate'):
    
    quantile = [0.5]
    samples = (age_range[1] - age_range[0]) * 2

    mcmc_quantiles = {gender:dict.fromkeys(frequency_band_model_ids.keys()) for gender in gender_ids.keys()}
    oscilogram = {gender:dict.fromkeys(frequency_band_model_ids.keys()) for gender in gender_ids.keys()}

    for fb in frequency_band_model_ids.keys():
        model_id = frequency_band_model_ids[fb]
        nm = pickle.load(open(os.path.join(model_path, 'NM_0_' + str(model_id) + outputsuffix + '.pkl'), 'rb'))

        meta_data = pickle.load(open(os.path.join(model_path, 'meta_data.md'), 'rb'))

        cov_scaler =  meta_data['scaler_cov'][model_id][0]
        res_scaler =  meta_data['scaler_resp'][model_id][0]

        synthetic_X = np.linspace(age_range[0], age_range[1], samples)[:,np.newaxis] # Truncated
        
        z_scores = st.norm.ppf(quantile)

        for gender in gender_ids.keys():
            batch_id = gender_ids[gender]
            model_be = np.repeat(np.array([[batch_id]]), synthetic_X.shape[0], axis=0)   
            q = res_scaler.inverse_transform(nm.get_mcmc_quantiles(cov_scaler.transform(synthetic_X), model_be, z_scores=z_scores))   
            mcmc_quantiles[gender][fb] = q.T
            oscilogram[gender][fb] = []
            for i in range(int((age_range[1] - age_range[0])/10)):
                m = np.mean(mcmc_quantiles[gender][fb][i*20:(i+1)*20])
                s = np.std(mcmc_quantiles[gender][fb][i*20:(i+1)*20])
                oscilogram[gender][fb].append([m,s])
    
    return oscilogram