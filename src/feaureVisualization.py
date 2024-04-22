import pandas as pd
import matplotlib.pyplot as plt
import json
import config
import numpy as np
import seaborn as sns
from config import featuresCategories




metadata = pd.read_csv("data/participants.tsv", sep="\t")

with open("data/features/featuresNames.json", "r") as file:
    featuresName = json.load(file)

features = pd.read_csv("data/features/featureMatrix.csv", header=None, names=featuresName)

# print(features.shape)

features = features.merge(metadata, on="participant_id", how="left")
# features.dropna(axis=0, inplace=True)

covMat = features[["gender_text", "age"]]

averagedFeatures = pd.DataFrame({})

for featuresCategory in featuresCategories:

    if featuresCategory in ["integrated power", "Individualized power"]:
        for band in list(config.freqBands.keys())[:-1]:
            array = features.iloc[:, features.columns.str.contains(band)]
            averagedFeatures[f"{featuresCategory}_{band}"] = array.iloc[:, array.columns.str.contains(featuresCategory)].mean(axis=1)
    
    elif featuresCategory in ["frequency of dominant peak", "power of dominant peak", "width of dominant peak"]:
        for band in config.freqBands.keys():
            array = features.iloc[:, features.columns.str.contains(band)]
            averagedFeatures[f"{featuresCategory}_{band}"] = array.iloc[:, array.columns.str.contains(featuresCategory)].mean(axis=1)
    
    else:
        averagedFeatures[featuresCategory] = features.iloc[:, features.columns.str.contains(featuresCategory)].mean(axis=1)
    
    
averagedFeatures = pd.concat([averagedFeatures, covMat], axis=1)


counter=0
fig, ax = plt.subplots(5,5, figsize=(30,30))
for i in range(5):
    for j in range(5):
        
        
        sns.scatterplot(data=averagedFeatures, x="age", y=(averagedFeatures.columns)[counter], hue="gender_text", ax=ax[i,j])
        ax[i,j].set_title((averagedFeatures.columns)[counter])
        ax[i,j].spines["right"].set_visible(False)
        ax[i,j].spines["top"].set_visible(False)
        ax[i,j].set(xlabel=None)
        ax[i,j].set(ylabel=None)


        counter+=1

plt.savefig("pictures/featureDis.png", dpi=400)