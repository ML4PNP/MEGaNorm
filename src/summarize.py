import pandas as pd
import numpy as np
import json
import sys
import os

# Add utils folder to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'utils')
sys.path.append(config_path)

from IO import make_config




def summarize(path):

    df = pd.read_csv(path)
    categories = []
    
    for name in df.columns.to_list()[2:]:
        if "offset" in name or "exponent" in name:
            categories.append(name.split("_")[:-1][0])
        if "r_squared" in name: 
            continue
        else:
            categories.append("_".join(name.split("_")[:-1]))

    categories = np.unique(categories)

    
    dfs = []
    for uniqueName in categories:
        dfs.append(df.loc[:, df.columns.str.startswith(uniqueName)].mean(axis=1))

    averaged = pd.concat(dfs)
    
    print(dfs)





summarize("featureData.csv")





  











# def featureSummarize(featureMatrix, savePath):

#     dfs_to_concat = []
#     newNames = []

#     # no band-related features
#     for featureCategory in ["offset", "exponent"]:

#             # loop over channels_spatial_layouts
#             for brainReigion, channelsList in channels_spatial_layouts.items():
                                
#                     selectedColumns = []
#                     for channelName in channelsList:

#                         col = np.where(np.logical_and(
#                             featureMatrix.columns.str.contains(featureCategory),
#                             featureMatrix.columns.str.contains(channelName)))
#                         selectedColumns.append(col[0].item())

#                     newNames.append(f"{featureCategory} - {brainReigion}")
#                     dfs_to_concat.append(featureMatrix.iloc[:,selectedColumns].mean(axis=1))

#     for featureCategory in featuresCategories:
#         if featureCategory in ["offset", "exponent"]: continue

#         for band in freqBands.keys():

#             # loop over channels_spatial_layouts
#             for brainReigion, channelsList in channels_spatial_layouts.items():
                
#                     selectedColumns = []
#                     for channelName in channelsList:

#                         col = np.where(np.logical_and(np.logical_and(
#                             featureMatrix.columns.str.contains(featureCategory),
#                             featureMatrix.columns.str.contains(band)),
#                             featureMatrix.columns.str.contains(channelName)))

#                         # print(len(col[0]))
#                         if len(col[0]) ==1: 
#                             selectedColumns.append(col[0].item())
                        
#                     if band == "Broadband" and featureCategory not in featuresCategories[-4:-1]:  continue
#                     newNames.append(f"{featureCategory} - {band} - {brainReigion}")
#                     dfs_to_concat.append(featureMatrix.iloc[:,selectedColumns].mean(axis=1))

    
#     df = pd.concat(dfs_to_concat, axis=1)
#     df.columns = newNames
#     df["participant_id"] = featureMatrix.loc[:,"participant_id"]

#     df.to_csv(savePath)


# if __name__=="__main__":
     

#     with open("data/features/featuresNames.josn", "r") as file:
#         featureNames = json.load(file)
#         featureNames.insert(0, "participant_id")
#     featureMatrix = pd.read_csv("data/features/featureMatrix.csv", names=featureNames)

#     savePath = "data/features/summarizedResVar.csv"

#     featureSummarize(featureMatrix, savePath)
