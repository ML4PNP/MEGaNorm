from fooof.plts.annotate import plot_annotated_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import fooof as f
import json








class normative:
    """
    A class of various functions, related to
    normative modeling
    """

    @staticmethod
    def prepareData(featurePath:str, metaDataPath:str) -> None:
        """
        This function reads the feature and target
        matrices, combines them, and then saves them
        as two separate .txt files for future modeling purposes.
        """

        # in future version, you must have specified features
        # name before getting to this step
        with open ("data/features/featuresNames.json", "r") as file:
            featNames = json.load(file)
        featMat = pd.read_csv(featurePath, header=None, names=featNames)
        metaData = pd.read_csv(metaDataPath, sep="\t")

        featMat = featMat.merge(metaData, on="participant_id", how="left")
        featMat.dropna(axis=0, inplace=True)

        covMat = featMat[["gender_code", "age"]]
        

        # getting average values over channels (nice method, hah?!)
        # remember, including non-int will results in an errror
        featMat = featMat.iloc[:,:-5]
        featMat = featMat.set_index(["participant_id"])
        featMat = featMat.T.groupby(
            lambda x: x.split(" ")[0]).mean(numeric_only=True).T
        
        # print("heeeere: ", featMat.head())

        covPath = "data/normativeInput/covariateNormSample.txt"
        featPath = "data/normativeInput/responseVarNorm.txt"

        covMat.to_csv(covPath,
                      sep=" ",
                      header=False,
                      index=False)
        
        featMat.to_csv(featPath,
                       sep=" ",
                       header=False,
                       index=False)
        


        return covPath, featPath
    

    @staticmethod
    def covForward(min, max, step):
        """
        This function create a covariate matrix for forward
        modeling
        
        parameters
        -------------
        min: int
        the lowest age that your model can represent

        max: int
        the highest age that your model can represent
        
        step:int
        bins

        return
        ------------
        savePath: str
        forward covariate path
        """

        bins = int(((max - min)/ step)+1)

        ageCov = np.arange(min, max, step).tolist()
        sex0 = (np.ones(bins).astype("int32")+1).tolist()
        sex1 = np.ones(bins).astype("int32").tolist()

        covariateForward = {"sex": sex0 + sex1,
                            "age": ageCov + ageCov}
        
        savePath = "data/normativeInput/'covariate_forwardmodel.txt'"
        pd.DataFrame(covariateForward).to_csv(savePath,
                                            sep = ' ',
                                            header = False,
                                            index = False)

        return savePath
    




def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))








    




    








