import json
import os
from meganorm.src import mainParallel
from meganorm.utils.parallel import auto_parallel_feature_extraction

def sbatch_feature_extraction_runner(job_name, 
                                      env,
                                      features_dir,
                                      feature_extraction_runner_path,
                                      save_sbatch_file_name="runner_driver.sbatch",
                                      time="48:00:00", 
                                      mem="16GB", 
                                      partition_name="normal",
                                    ):
    
    sbatch_text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=1
#SBATCH --partition={partition_name}

# Activate your environment
source activate {env}

python {feature_extraction_runner_path}
"""

    save_path = os.path.join(features_dir, save_sbatch_file_name)
    with open(save_path, "w") as f:
        f.write(sbatch_text)

    print("Created run_driver.sbatch")


def save_feature_extraction_runner_params(mainParallel_path,
                                           features_dir,
                                           subjects,
                                           job_configs,
                                           config_file,
                                           slurm_username,
                                           freesurfer_home,
                                           freesurfer_license,
                                           auto_rerun=True,
                                           auto_collect=True,
                                           max_try=5):
    
    params = {
        "mainParallel_path": mainParallel_path,
        "features_dir": features_dir,
        "subjects": subjects,
        "job_configs": job_configs,
        "config_file": config_file,
        "username": slurm_username,
        "freesurfer_home": freesurfer_home,
        "freesurfer_license": freesurfer_license,
        "auto_rerun": auto_rerun,
        "auto_collect": auto_collect,
        "max_try": max_try
    }

    save_path = os.path.join(features_dir, "runner_params.json")
    with open(save_path, "w") as f:
        json.dump(params, f, indent=4)

    print("Saved!")


if __name__ == "__main__":

    with open("Features/runner_params.json") as f:
        params = json.load(f)

    if params.get("mainParallel_path", None) is None:
        params["mainParallel_path"] = os.path.abspath(mainParallel.__file__)

    # Run
    auto_parallel_feature_extraction(**params)



    