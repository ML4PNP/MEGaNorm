import h5py
import tqdm
import numpy as np

from dataManagementUtils import readFooofres, subjectList, saveFeatures
from config.config import freq_bands
from processUtils import features


def featureEx(path, subjectsId):
    """
    This function extract features from periodic data
    and save them along with aperiodic paramereters
    """

    # loading data
    (_, # aperiodic
    periodic, 
    freqs, 
    aperiodicParams, 
    _,  # periodicParams
    ) = readFooofres(path, subjectsId)

    # mean power of periodic signals
    (delta, theta, alpha, beta,
    gamma) = features.meanPower(np.asarray(periodic), freqs, freq_bands)

    aperiodicParams = np.hstack(aperiodicParams)
    featureSet = np.hstack([delta, theta, alpha, beta, gamma, aperiodicParams]).T.tolist()
    featureSet.insert(0, subjectsId)

    return featureSet





if __name__ == "__main__":

    path = "data/fooofResults/fooofResults.h5"
    subjectsIds = subjectList(path)

    savePath = "/home/zamanzad/trial1/data/features/featureMatrix.csv"

    for subjectId in tqdm.tqdm(subjectsIds[:]):

        featureSet = featureEx(path, subjectId)
        saveFeatures(savePath, featureSet)
    
    

    

