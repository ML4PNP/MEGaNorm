import pandas as pd
import json



def normativePrepare(featurePath:str, 
                    metaDataPath:str, 
                    covSavePath:str, 
                    featSavePath:str) -> None:
    """
    This function reads the feature and target
    matrices, combines them, and then saves them
    as two separate .txt files for future modeling purposes.
    """

    # in future version, you must have specified features
    # name before getting to this step

    featMat = pd.read_csv(featurePath)
    metaData = pd.read_csv(metaDataPath, sep="\t")

    featMat = featMat.merge(metaData, on="participant_id", how="left")
    # print(featMat.columns)

    covMat = featMat[["gender_code", "age"]]
    

    # getting average values over channels (nice method, hah?!)
    # remember, including non-int will results in an errror
    featMat = featMat.iloc[:,:-5]
    featMat = featMat.set_index(["participant_id"])
    # featMat = featMat.T.groupby(
    #     lambda x: x.split(" ")[0]).mean(numeric_only=True).T
    
    # print(featMat.info(verbose=True, show_counts=True))
    covMat.to_csv(covSavePath,
                    sep=" ",
                    header=False,
                    index=False)
    
    featMat.to_csv(featSavePath,
                    sep=" ",
                    header=False,
                    index=False)
    


    return 


if __name__=="__main__":

    featurePath = "data/features/summarizedResVar.csv"
    metaDataPath = "data/participants.tsv"

    covSavePath = "data/normativeInput/covariateNormSample.txt"
    featSavePath = "data/normativeInput/responseVarNorm.txt"

    normativePrepare(featurePath, metaDataPath, covSavePath, featSavePath)


