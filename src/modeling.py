
import pcntoolkit as pcn

from processUtils import normative



def model(covPath, featPath, alg, cv):

    covForward = normative.covForward(20, 80+1, 5)

    pcn.normative.estimate(
                 covfile=covPath,
                 respfile=featPath,
                 testcov=covForward,
                 alg=alg,
                 cvfolds = cv,
                 saveoutput=True,
                 suffix="_forward",
                 inscaler="standardize")
    


if __name__ == "__main__":
    
    alg = "gpr"
    cv = 2
    featurePath = "data/features/featureMatrix.csv"
    metaDataPath = "data/participants.tsv"
    covPath, featPath = normative.prepareData(featurePath, metaDataPath)
    
    model(covPath, featPath, alg, cv)