import pandas as pd
import matplotlib.pyplot as plt
import json
import config
import numpy as np
import seaborn as sns
from config import featuresCategories


def visualize(metaDataPath, featuresName, featuresPath, savePath):

    features = pd.read_csv(featuresPath, header=None, names=featuresName)
    metadata = pd.read_csv(metaDataPath, sep="\t")
    features = features.merge(metadata, on="participant_id", how="left")
    # features.dropna(axis=0, inplace=True)

    covMat = features[["gender_text", "age"]]

    averagedFeatures = pd.DataFrame({})

    for featuresCategory in featuresCategories:

        if featuresCategory in ["Canonical Relative Power", 
                                "Canonical Absolute Power",
                                "Individualized Relative Power ",
                                "Individualized Absolute Power"]:
            
            for typee in ["adjusted", "original psd"]:
                array1 = features.iloc[:, features.columns.str.contains(typee)]

                for band in list(config.freqBands.keys())[:-1]:
                    array2 = array1.iloc[:, array1.columns.str.contains(band)]
                    averagedFeatures[f"{typee}_{featuresCategory}_{band}"] = array2.iloc[:, array2.columns.str.contains(featuresCategory)].mean(axis=1)
        



        elif featuresCategory in ["frequency of dominant peak", "power of dominant peak", "width of dominant peak"]:
            for band in config.freqBands.keys():
                array = features.iloc[:, features.columns.str.contains(band)]
                averagedFeatures[f"{featuresCategory}_{band}"] = array.iloc[:, array.columns.str.contains(featuresCategory)].mean(axis=1)
        



        else:
            averagedFeatures[featuresCategory] = features.iloc[:, features.columns.str.contains(featuresCategory)].mean(axis=1)
        
        
    averagedFeatures = pd.concat([averagedFeatures, covMat], axis=1)


    for typee in ["adjusted", "original psd"]:
        df = averagedFeatures.iloc[:, averagedFeatures.columns.str.contains(typee)]
        df = pd.concat([df, covMat], axis=1)
        
        counter=0
        fig, ax = plt.subplots(4,4, figsize=(30,30))
        for i in range(4):
            for j in range(4):
                
                sns.scatterplot(data=df, x="age", y=(df.columns)[counter], hue="gender_text", ax=ax[i,j])
                ax[i,j].set_title((df.columns)[counter])
                ax[i,j].spines["right"].set_visible(False)
                ax[i,j].spines["top"].set_visible(False)
                ax[i,j].set(xlabel=None)
                ax[i,j].set(ylabel=None)

                counter+=1

        plt.savefig(f"pictures/featuresDis/{typee}.png", dpi=300)



# ====================================================================================
    aperFeat = ["offset", # 1
                "exponent"] # 1
    
    # for typee in aperFeat:
    df = averagedFeatures.loc[:, ["offset", "exponent"]]
    df = pd.concat([df, covMat], axis=1)
    
    counter=0
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    for i in range(1):
        for j in range(2):
            
            sns.scatterplot(data=df, x="age", y=(df.columns)[counter], hue="gender_text", ax=ax[j])
            ax[j].set_title((df.columns)[counter])
            ax[j].spines["right"].set_visible(False)
            ax[j].spines["top"].set_visible(False)
            ax[j].set(xlabel=None)
            ax[j].set(ylabel=None)

            counter+=1

    plt.savefig("pictures/featuresDis/aperiodic.png", dpi=300)
# ====================================================================================

    peakFeat = ["frequency of dominant peak", # 5,
                "power of dominant peak", # 5,
                "width of dominant peak"] # 5,
    
    
    
    fig, ax = plt.subplots(3,5, figsize=(30,20))
    for i, typee in enumerate(peakFeat):
        counter=0
        df = averagedFeatures.iloc[:, averagedFeatures.columns.str.contains(typee)]
        df = pd.concat([df, covMat], axis=1)
        

        for j in range(5):
            
            sns.scatterplot(data=df, x="age", y=(df.columns)[counter], hue="gender_text", ax=ax[i,j])
            ax[i,j].set_title((df.columns)[counter])
            ax[i,j].spines["right"].set_visible(False)
            ax[i,j].spines["top"].set_visible(False)
            ax[i,j].set(xlabel=None)
            ax[i,j].set(ylabel=None)

            counter+=1

    plt.savefig("pictures/featuresDis/peak features.png", dpi=300)

    

        



if __name__=="__main__":

    metaDataPath = "data/participants.tsv"
    savePath = "pictures/featureDis.png"
    featuresPath = "data/features/featureMatrix.csv"

    with open("data/features/featuresNames.json", "r") as file:
        featuresName = json.load(file)

    visualize(metaDataPath, featuresName, featuresPath, savePath)