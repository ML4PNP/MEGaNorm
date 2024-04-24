import fooof as f
import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt

dir = "data/fooofResults/fooofModels.pkl"

with open(dir, "rb") as fooofFile:
    counter = 0
    while True:
        try: pickle.load(fooofFile) ; counter += 1
        except: break


r_squared = []
with open(dir, "rb") as fooofFile:
    for j in tqdm.tqdm(range(counter)):
        subjectId, (fmGroup, psds, freqs) = next(iter(pickle.load(fooofFile).items()))

        r_squared.extend(fmGroup.get_params(name="r_squared"))


plt.figure(figsize=(10,6))
plt.hist(r_squared, range=(0.8, 1), bins=60, color="black")
plt.title("R Squared distribution")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.savefig("pictures/fooof/R_squared.png")