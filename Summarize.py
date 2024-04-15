import pandas as pd
import numpy as np
import json

from config.config import channels_spatial_layouts, featuresCategories, freqBands





def featureSummarize(featureMatrix, savePath):

    dfs_to_concat = []

    # no band-related features
    for featureCategory in ["offset", "exponent"]:

            # loop over channels_spatial_layouts
            for brainReigion, channelsList in channels_spatial_layouts.items():
                                
                    selectedColumns = []
                    for channelName in channelsList:

                        col = np.where(np.logical_and(
                            featureMatrix.columns.str.contains(featureCategory),
                            featureMatrix.columns.str.contains(channelName)))
                        selectedColumns.append(col[0].item())

                    newName = f"{featureCategory} - {brainReigion}"
                    dfs_to_concat.append(featureMatrix.iloc[:,selectedColumns].mean(axis=1))

    for featureCategory in featuresCategories:
        if featureCategory in ["offset", "exponent"]: continue

        for band in freqBands.keys():

            # loop over channels_spatial_layouts
            for brainReigion, channelsList in channels_spatial_layouts.items():
                
                    selectedColumns = []
                    for channelName in channelsList:

                        col = np.where(np.logical_and(np.logical_and(
                            featureMatrix.columns.str.contains(featureCategory),
                            featureMatrix.columns.str.contains(band)),
                            featureMatrix.columns.str.contains(channelName)))

                        # print(len(col[0]))
                        if len(col[0]) ==1: 
                            selectedColumns.append(col[0].item())
                        
                    if band == "Broadband" and featureCategory not in featuresCategories[-4:-1]:  continue
                    newName = f"{featureCategory} - {band} - {brainReigion}"
                    dfs_to_concat.append(featureMatrix.iloc[:,selectedColumns].mean(axis=1))

    df = pd.concat(dfs_to_concat, axis=1)
    df.to_csv(savePath)


if __name__=="__main__":
     

    with open("data/features/featuresNames.josn", "r") as file:
        featureNames = json.load(file)
    featureMatrix = pd.read_csv("data/features/featureMatrix.csv", names=featureNames)

    savePath = "data/features/summarizedResVar.csv"

    featureSummarize(featureMatrix, savePath)
