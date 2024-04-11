import h5py
import pickle
from csv import writer

def storeFooofModels(path, subjId, fooofModels) -> None:
    """
    This function stores the periodic and aperiodic 
    results in a h5py file

    parameters
    ------------
    path: str
    where to save

    subjid: str
    subject ID

    fooofModels: object

    returns
    -------------
    None

    """

    with open(path, "ab") as file:
        pickle.dump({subjId: fooofModels}, file)
    # with h5py.File(path, "a") as hdf:
        
    #     participant = hdf.create_group(subjId)
    #     participant.create_dataset("fooof Models", data=fooofModels)




def logFunc(path, content, mode="a") -> None:
    """
    this function logs results in a 
    .txt file 
    """

    with open(path, mode) as logFile:
        logFile.write(content)




def readFooofres(path:str, subject):
    """
    this function returns the following data:
    1. aperiodic signal
    2. periodic signal
    3. aperiodic parameters
    4. periodic parameters
    """

    with h5py.File(path, 'r') as file:
        aperiodic = list(file.get(f"{subject}/aperiodic"))
        periodic = list(file.get(f"{subject}/periodic"))
        freqs = list(file.get(f"{subject}/freqs"))
        aperiodicParams = list(file.get(f"{subject}/aperiodic Params"))
        periodicParams = list(file.get(f"{subject}/periodic Params"))
      
    
    return aperiodic, periodic, freqs, aperiodicParams, periodicParams



def subjectList(path):
    """
    This function returns a list of all subject in
    .hd fies
    """
    with h5py.File(path, "r") as file:
        subjects = list(file.keys())

    return subjects


def saveFeatures(path, arr):
    
    with open(path, "a") as file:
        writerObj = writer(file)
        writerObj.writerow(arr)
        file.close()



