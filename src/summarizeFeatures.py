import pandas as pd
import numpy as np
import argparse
import json
import sys
import os

# Add utils folder to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'utils')
sys.path.append(config_path)

from IO import make_config




def summarizeFeatures(df, sensorsInf, whichSensor):

    """
    average features across the whole brain. 
    """
    
    sensorsAverageds = []

    for sensorType, sensorsID in sensorsInf.items():
        
        if sensorType != whichSensor: continue

        dfSenssor = df.loc[:,df.columns.str.endswith(sensorsID)]
        categories = set()
    
        for name in dfSenssor.columns:
            
            if "offset" in name or "exponent" in name:
                categories.add(name.split("_")[:-1][0] + f"{sensorType}")
            elif "r_squared" not in name and "participant_ID" not in name:
                categories.add("_".join(name.split("_")[:-1]) + f"_{sensorType}")

        dfs = [dfSenssor.loc[:, dfSenssor.columns.str.startswith(uniqueName[:-4])].mean(axis=1) for uniqueName in categories]
        dfNames = list(categories)

        averaged = pd.concat(dfs, axis=1)
        averaged.columns = dfNames
        
        sensorsAverageds.append(averaged)

    sensorsAverageds = pd.concat(sensorsAverageds, axis=1)


    return sensorsAverageds





    
    



if __name__=="__main__":

    parser = argparse.ArgumentParser()

    # positional arguments
    parser.add_argument("dir",
                        help="address to a .csv dataframe")
    parser.add_argument("saveDir",
                        help="where to save results")
	# optional arguments
    parser.add_argument("--configs", type=str, default=None,
        help="Address of configs json file")

    args = parser.parse_args()

    # args.dir = "featureData.csv"
    # args.saveDir = "dataTest"

	# Loading configs
    if args.configs is not None:
        with open(args.configs, 'r') as f:
            configs = json.load(f)
    else:
        configs = make_config()
                     

    df = pd.read_csv(args.dir)
    averaged = summarizeFeatures(df, configs["sensorsID"])
    averaged.to_csv(f"{args.saveDir}/averagedOverBrain.csv")


