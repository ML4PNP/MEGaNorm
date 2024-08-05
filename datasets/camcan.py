import numpy as np
import pandas as pd
    

def load_covariates(path):
    
    """Load age and gender for CamCAN dataset.

    Args:
        path (str): path to covariates.

    Returns:
        DataFrame: Pandas dataframe containing age and gender for camcan dataset.
    """
    
    df = pd.read_csv(path, sep='\t', index_col=0)
    df = df[['age', 'gender_code']]
    df = df.rename(columns={'gender_code':'gender'})
    df.gender = df.gender - 1 # 0 for males and 1 for females
    df.index.name = None
    df['site'] = np.zeros([df.shape[0],1], dtype=int)
    
    return df   


def load_camcan_data(feature_path, covariates_path):
    
    """Load camcan dataset

    Args:
        feature_path (str): Path to the the feature csv file.
        covariates_path (str): path to the covariates tsv file.

    Returns:
        DataFrame: Pandas dataframe with camcan covariates and features.
    """
    
    camcan_covariates = load_covariates(covariates_path)
    camcan_features = pd.read_csv(feature_path, index_col=0)
    camcan_data = camcan_covariates.join(camcan_features, how='inner')
    
    return camcan_data



    
    
