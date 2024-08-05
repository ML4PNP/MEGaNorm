import pandas as pd


def load_BTNRH_data(feature_path, covariates_path):
    """
    load BTNRH dataset
    """
    BTNRH_covariates = pd.read_excel("/project/meganorm/Data/BTNRH/Rempe_Ott_PNAS_2022_Data.xlsx", index_col=0)
    BTNRH_features = pd.read_csv(feature_path, index_col=0)

    BTNRH_covariates = BTNRH_covariates.rename(columns={"Sex" : "gender", "Age" : "age"})
    BTNRH_covariates.gender = BTNRH_covariates.gender.replace({"M": 0, "F" : 1})
    
    
    BTNRH_data = BTNRH_covariates.join(BTNRH_features, how='inner')

    #Assigning 1 as the site for the BTNRH dataset    
    BTNRH_data["site"] = 1
    
    return BTNRH_data
