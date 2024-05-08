import os
import numpy as np
import pandas as pd

def nm_data_split(data, save_path, covariates=['age', 'gender'], train_split=0.5, 
                  validation_split=None, random_seed=42):
    
    """Utility function for splitting data into training, validation and test sets
    before normative modeling. The sets are save as picke file in the specified 
    save path.

    Args:
        data (DataFrame): Panda dataframe of data created for example using "load_camcan_data"
        function.
        save_path (str): Path to save the results.
        covariates (list, optional): List of covariates. Defaults to ['age', 'gender'].
        train_split (float, optional): train set split ratio. Defaults to 0.5.
        validation_split (_type_, optional): If not None will be used for validation set 
        split ratio. Defaults to None.
        random_seed (int, optional): Random seed in the splitting data. Defaults to 42.
    """

    os.makedirs(save_path, exist_ok=True)

    np.random.seed(random_seed)

    rand_idx = np.random.permutation(data.shape[0])

    train_num = np.round(train_split * data.shape[0]).astype(int)

    train_set = data.iloc[rand_idx[0:train_num]]
    x_train = train_set.loc[:, covariates]
    y_train = train_set.drop(columns=covariates)

    if validation_split is not None:
        validation_num = np.round(validation_split * data.shape[0]).astype(int)
        validation_set = data.iloc[rand_idx[train_num:train_num + validation_num]]
        x_val = validation_set.loc[:, covariates]
        y_val = validation_set.drop(columns=covariates)
        x_val.to_pickle(os.path.join(save_path, 'x_val.pkl'))
        y_val.to_pickle(os.path.join(save_path, 'y_val.pkl'))
        test_set = data.iloc[rand_idx[train_num + validation_num:]]
    else:
        validation_set = None
        test_set = data.iloc[rand_idx[train_num:]]

    x_test = test_set.loc[:, covariates]
    y_test = test_set.drop(columns=covariates)

    x_train.to_pickle(os.path.join(save_path, 'x_train.pkl'))
    y_train.to_pickle(os.path.join(save_path, 'y_train.pkl'))
    x_test.to_pickle(os.path.join(save_path, 'x_test.pkl'))
    y_test.to_pickle(os.path.join(save_path, 'y_test.pkl'))
    
    return