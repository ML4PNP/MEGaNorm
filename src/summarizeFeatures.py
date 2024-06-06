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




def summarizeFeatures(df, layoutPath):

    """
    average features across the whole brain. 
    """
    
    with open(layoutPath, "r") as file:
        layout = json.load(file)

    summarized = []

    for layoutName, layoutRow in layout.items():

        sensorType = layoutName.split("_")[1]

        parcell = df.loc[:, [col for col in df.columns if col.split("_")[-1] in layoutRow]]
        categories = set()

        for name in parcell.columns:
            if "offset" in name or "exponent" in name:
                categories.add(name.split("_")[:-1][0] + f"{sensorType}")
            elif "r_squared" not in name and "participant_ID" not in name:
                categories.add("_".join(name.split("_")[:-1]) + f"_{sensorType}")
        
        dfs = [parcell.loc[:, parcell.columns.str.startswith(uniqueName[:-4])].mean(axis=1) for uniqueName in categories]

        averaged = pd.concat(dfs, axis=1); averaged.columns = list(categories)
        

        summarized.append(averaged)
    summarized = pd.concat(summarized, axis=1)

    return summarized





    
    



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



