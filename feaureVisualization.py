import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns




metadata = pd.read_csv("data/participants.tsv", sep="\t")

with open("data/features/featuresNames.json", "r") as file:
    featuresName = json.load(file)
features = pd.read_csv("data/features/featureMatrix.csv", header=None, names=featuresName)

print(metadata.columns)

features = features.merge(metadata, on="participant_id", how="left")
features.dropna(axis=0, inplace=True)

covMat = features[["gender_text", "age"]]


# getting average values over channels (nice method, hah?!)
# remember, including non-int will results in an errror
features = features.iloc[:,:-5]
features = features.set_index(["participant_id"])
features = features.T.groupby(
    lambda x: x.split(" ")[0]).mean(numeric_only=True).T

features = pd.concat([features, covMat])


fig, ax = plt.subplots(4,2, figsize=(15, 15))
sns.scatterplot(data=features, x="age", y="alpha", ax=ax[0, 0])

plt.savefig("FeatureDis.png")
plt.close()

