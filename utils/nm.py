import os
import numpy as np
import pandas as pd

def hbr_data_split(data, save_path, covariates=['age', 'gender'], site_ids=None, train_split=0.5, 
                  validation_split=None, drop_nans=False, random_seed=42):
    
    """Utility function for splitting data into training, validation and test sets
    before HBR normative modeling. The sets are save as picke file in the specified 
    save path.

    Args:
        data (DataFrame): Panda dataframe of data created for example using "load_camcan_data"
        function.
        save_path (str): Path to save the results.
        covariates (list, optional): List of covariates. Defaults to ['age', 'gender'].
        site_ids (str, optional): Used for multisite data and decides the column label for site ids in data.
        Defaults to None.
        train_split (float, optional): train set split ratio. Defaults to 0.5.
        validation_split (_type_, optional): If not None will be used for validation set 
        split ratio. Defaults to None.
        drop_nans (boolean, optional): If True, drops columns with missing values. Defaults to False.
        random_seed (int, optional): Random seed in the splitting data. Defaults to 42.
    """

    os.makedirs(save_path, exist_ok=True)

    np.random.seed(random_seed)
    
    if drop_nans:
        data = data.dropna(axis=1)

    rand_idx = np.random.permutation(data.shape[0])

    train_num = np.round(train_split * data.shape[0]).astype(int)

    train_set = data.iloc[rand_idx[0:train_num]]
    x_train = train_set.loc[:, covariates]
    b_train = train_set.loc[:, site_ids] if site_ids is not None else pd.DataFrame(np.zeros([x_train.shape[0],1], dtype=int), 
                                                                                   index=x_train.index, columns=['site'])
    y_train = train_set.drop(columns=covariates+[site_ids]) if site_ids is not None else train_set.drop(columns=covariates) 

    if validation_split is not None:
        validation_num = np.round(validation_split * data.shape[0]).astype(int)
        validation_set = data.iloc[rand_idx[train_num:train_num + validation_num]]
        x_val = validation_set.loc[:, covariates]
        b_val = validation_set.loc[:, site_ids] if site_ids is not None else pd.DataFrame(np.zeros([x_val.shape[0],1], dtype=int), 
                                                                                          index=x_val.index, columns=['site'])
        y_val = validation_set.drop(columns=covariates+[site_ids]) if site_ids is not None else validation_set.drop(columns=covariates) 

        x_val.to_pickle(os.path.join(save_path, 'x_val.pkl'))
        y_val.to_pickle(os.path.join(save_path, 'y_val.pkl'))
        b_val.to_pickle(os.path.join(save_path, 'b_val.pkl'))
        test_set = data.iloc[rand_idx[train_num + validation_num:]]
    else:
        validation_set = None
        test_set = data.iloc[rand_idx[train_num:]]

    x_test = test_set.loc[:, covariates]
    b_test = test_set.loc[:, site_ids] if site_ids is not None else pd.DataFrame(np.zeros([x_test.shape[0],1], dtype=int), 
                                                                                 index=x_test.index, columns=['site'])
    y_test = test_set.drop(columns=covariates+[site_ids]) if site_ids is not None else test_set.drop(columns=covariates) 

    x_train.to_pickle(os.path.join(save_path, 'x_train.pkl'))
    y_train.to_pickle(os.path.join(save_path, 'y_train.pkl'))
    b_train.to_pickle(os.path.join(save_path, 'b_train.pkl'))
    x_test.to_pickle(os.path.join(save_path, 'x_test.pkl'))
    y_test.to_pickle(os.path.join(save_path, 'y_test.pkl'))
    b_test.to_pickle(os.path.join(save_path, 'b_test.pkl'))
    
    return