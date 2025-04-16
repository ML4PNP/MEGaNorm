import mne_bids
import glob
import shutil
import os
import mne
from pathlib import Path
import scipy
import sys
import numpy as np 
# Add utils folder to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'utils')
sys.path.append(config_path)
# from utils.EEGlab import read_raw_eeglab
import pandas as pd


def make_demo_file_bids( file_dir:str, save_dir:str, id_col:int, age_col:int, *argv) -> None:

    """
    This function read demographic data and convert them to a common
    template so can be used later in processing.

    Parameters:
        file_dir (str): Path to the input demographic file (e.g., CSV).
        save_dir (str): Directory where the BIDS-formatted file should be saved.
        id_col (int): Column index for the participant ID.
        age_col (int): Column index for the age.
        argv (dict):  

    Returns:
        None
    """
    new_df = pd.DataFrame({})

    if "xlsx" in file_dir[-4:]:
        df = pd.read_excel(file_dir, index_col=None)
    if "csv" in file_dir[-4:]:
        df = pd.read_csv(file_dir, index_col=None)
    if "tsv" in file_dir[-4:]:
        df = pd.read_csv(file_dir, sep='\t', index_col=None)

    new_df["participant_id"] = df.iloc[:, id_col]
    new_df["age"] = df.iloc[:, age_col]
    
    for arg in argv:

        if not arg.get("col_id"):
            new_df[arg.get("col_name")] = arg.get("single_value")
            continue

        col_id = arg.get("col_id")
        col_name = arg.get("col_name")
        mapping = arg.get("mapping", None)

        if col_id is None and col_name is None:
            raise ValueError("Both'col_id' and 'col_name' can not be None.")

        new_df[col_name] = df.iloc[:, col_id]

        if mapping:
            new_df[col_name] = new_df[col_name].map(mapping)
        

        if col_name == "diagnosis":
            new_df[col_name] = new_df[col_name].replace(np.nan, "nan")

        new_df = new_df.drop_duplicates(subset=["participant_id"], keep="first") 

    new_df.to_csv(save_dir, sep='\t', index=False)




if __name__ == "__main__":

    # Preparing demographic data according to mne_bids format
    # BTH
    file_dir = "/project/meganorm/Data/BTNRH/Rempe_Ott_PNAS_2022_Data.xlsx"
    save_dir = "/project/meganorm/Data/BTNRH/BIDS/participants_bids.tsv"
    make_demo_file_bids(file_dir, 
                        save_dir, 
                        0, 
                        1, 
                        {"col_name": "sex", "col_id": 2, "mapping": {"M": "Male", "F": "Female"}, "single_value":None},
                        {"col_name": "eyes", "col_id": None, "mapping": None, "single_value":"eyes_closed"},
                        {"col_name": "diagnosis", "col_id": None, "mapping": None, "single_value":"control"})

    # CAMCAN
    file_dir = "/project/meganorm/Data/camcan/CamCAN/cc700/participants.tsv"
    save_dir = "/project/meganorm/Data/camcan/BIDS/participants_bids.tsv"
    make_demo_file_bids(file_dir, 
                        save_dir, 
                        0, 
                        1, 
                        {"col_name": "sex", "col_id": 3, "mapping": {"MALE": "Male", "FEMALE": "Female"}, "single_value":None},
                        {"col_name": "eyes", "col_id": None, "mapping": None, "single_value":"eyes_closed"},
                        {"col_name": "diagnosis", "col_id": None, "mapping": None, "single_value":"control"})

    
    
    # # NIMH 
    file_dir = "/project/meganorm/Data/NIMH/participants_original.tsv"
    save_dir = "/project/meganorm/Data/NIMH/participants_bids.tsv"
    make_demo_file_bids(file_dir, 
                        save_dir, 
                        0, 
                        1, 
                        {"col_name": "sex", "col_id": 2, "mapping": {"male": "Male", "female": "Female"}, "single_value":None},
                        {"col_name": "eyes", "col_id": None, "mapping": None, "single_value":"eyes_closed"},
                        {"col_name": "diagnosis", "col_id": None, "mapping": None, "single_value":"control"})

    
    # # OMEGA 
    file_dir = "/project/meganorm/Data/Omega/participants.tsv"
    save_dir = "/project/meganorm/Data/Omega/participants_bids.tsv"
    make_demo_file_bids(file_dir, 
                        save_dir, 
                        0, 
                        3, 
                        {"col_name": "sex", "col_id": 1, "mapping": {"M": "Male", "F": "Female"}, "single_value":None},
                        {"col_name": "eyes", "col_id": None, "mapping": None, "single_value":"eyes_open"},
                        {"col_name": "diagnosis", "col_id": 4, "mapping": {"Control": "control", 
                                                                           "Parkinson": "parkinson", 
                                                                           "Chronic Pain": "chronic pain", 
                                                                           "Control	1" : "control",
                                                                           "ADHD": "adhd"}, "single_value":None})

    
    # # # HCP 
    file_dir = "/project/meganorm/Data/HCP/info/RESTRICTED_smkia_11_25_2024_7_16_58_merged.csv"
    save_dir = "/project/meganorm/Data/HCP/participants_bids.tsv"
    make_demo_file_bids(file_dir, 
                        save_dir, 
                        0, 
                        1, 
                        {"col_name": "sex", "col_id": 203, "mapping": {"M": "Male", "F": "Female"}, "single_value":None},
                        {"col_name": "eyes", "col_id": None, "mapping": None, "single_value":"eyes_open"},
                        {"col_name": "diagnosis", "col_id": None, "mapping": None, "single_value":"control"})

    
    # CMI
    file_dir = "/project/meganorm/Data/EEG_CMI/EEG_BIDS/covariates.tsv"
    save_dir = "/project/meganorm/Data/EEG_CMI/EEG_BIDS/participants_bids.tsv"
    make_demo_file_bids(file_dir, 
                        save_dir, 
                        0, 
                        1, 
                        {"col_name": "sex", "col_id": 2, "mapping":{"0": "Male", "1": "Female"}, "single_value":None}, 
                        {"col_name": "eyes", "col_id": None, "mapping": None, "single_value":"eyes_closed"},
                        {"col_name": "diagnosis", "col_id": 5, "mapping": {
                            "ADHD-Combined Type": "adhd combined type",
                            "Generalized Anxiety Disorder": "generalized anxiety disorder",
                            "ADHD-Inattentive Type": "adhd inattentive type",
                            "Specific Learning Disorder with Impairment in Reading": "specific learning disorder with impairment in reading",
                            "Disruptive Mood Dysregulation Disorder": "disruptive mood dysregulation disorder",
                            "Oppositional Defiant Disorder": "oppositional defiant disorder",
                            "Major Depressive Disorder": "major depressive disorder",
                            "Tourettes Disorder": "tourettes disorder",
                            "Other Specified Anxiety Disorder": "other specified anxiety disorder",
                            "Other Specified Attention-Deficit/Hyperactivity Disorder": "other specified attention deficit hyperactivity disorder",
                            "No Diagnosis Given": "control",
                            "Autism Spectrum Disorder": "autism spectrum disorder",
                            "Language Disorder": "language disorder",
                            "Specific Learning Disorder with Impairment in Mathematics": "specific learning disorder with impairment in mathematics",
                            "No Diagnosis Given: Incomplete Eval": "no diagnosis given incomplete eval",
                            "Separation Anxiety": "separation anxiety",
                            "Social (Pragmatic) Communication Disorder": "social pragmatic communication disorder",
                            "Provisional Tic Disorder": "provisional tic disorder",
                            "Social Anxiety (Social Phobia)": "social anxiety social phobia",
                            "Specific Phobia": "specific phobia",
                            "Borderline Intellectual Functioning": "borderline intellectual functioning",
                            "ADHD-Hyperactive/Impulsive Type": "adhd hyperactive impulsive type",
                            "Intellectual Disability-Moderate": "intellectual disability moderate",
                            "Intellectual Disability-Mild": "intellectual disability mild",
                            "Adjustment Disorders": "adjustment disorders",
                            "Bipolar I Disorder": "bipolar i disorder",
                            "Obsessive-Compulsive Disorder": "obsessive compulsive disorder",
                            "Conduct Disorder-Childhood-onset type": "conduct disorder childhood onset type",
                            "Selective Mutism": "selective mutism",
                            "Other Specified Depressive Disorder": "other specified depressive disorder",
                            "Unspecified Attention-Deficit/Hyperactivity Disorder": "unspecified attention deficit hyperactivity disorder",
                            "Other Specified Disruptive, Impulse-Control, and Conduct Disorder": "other specified disruptive impulse control and conduct disorder",
                            "Persistent Depressive Disorder (Dysthymia)": "persistent depressive disorder dysthymia",
                            "Other Specified Trauma- and Stressor-Related Disorder": "other specified trauma and stressor related disorder",
                            "Other Specified Tic Disorder": "other specified tic disorder",
                            "Posttraumatic Stress Disorder": "posttraumatic stress disorder",
                            "Excoriation (Skin-Picking) Disorder": "excoriation skin picking disorder",
                            "Substance/Medication-Induced Bipolar and Related Disorder": "substance medication induced bipolar and related disorder",
                            "Specific Learning Disorder with Impairment in Written Expression": "specific learning disorder with impairment in written expression",
                            "Enuresis": "enuresis",
                            "Major Neurocognitive Disorder Due to Epilepsy": "major neurocognitive disorder due to epilepsy",
                            "Speech Sound Disorder": "speech sound disorder",
                            "Encopresis": "encopresis",
                            "Bipolar II Disorder": "bipolar ii disorder",
                            "Intermittent Explosive Disorder": "intermittent explosive disorder",
                            "Persistent (Chronic) Motor or Vocal Tic Disorder": "persistent chronic motor or vocal tic disorder",
                            "Other Specified Neurodevelopmental Disorder": "other specified neurodevelopmental disorder",
                            "Unspecified Anxiety Disorder": "unspecified anxiety disorder",
                            "Other Specified Feeding or Eating Disorder": "other specified feeding or eating disorder",
                            "Cannabis Use Disorder": "cannabis use disorder",
                            "Bulimia Nervosa": "bulimia nervosa",
                            "Avoidant/Restrictive Food Intake Disorder": "avoidant restrictive food intake disorder",
                            " ": "unspecified",
                            "Reactive Attachment Disorder": "reactive attachment disorder",
                            "Unspecified Neurodevelopmental Disorder": "unspecified neurodevelopmental disorder",
                            "Agoraphobia": "agoraphobia",
                            "Depressive Disorder Due to Another Medical Condition": "depressive disorder due to another medical condition",
                            "Delirium due to another medical condition": "delirium due to another medical condition",
                            "Specific Learning Disorder with Impairment in Reading ": "specific learning disorder with impairment in reading",
                            "Cyclothymic Disorder": "cyclothymic disorder",
                            "Schizophrenia": "schizophrenia",
                            "Delirium due to multiple etiologies": "delirium due to multiple etiologies",
                            "Gender Dysphoria in Adolescents and Adults": "gender dysphoria in adolescents and adults",
                            "Other Specified Obsessive-Compulsive and Related Disorder": "other specified obsessive compulsive and related disorder",
                            "Developmental Coordination Disorder": "developmental coordination disorder",
                            "Acute Stress Disorder": "acute stress disorder"}, "single_value":None})
    



    # MOUS 
    file_dir = "/project/meganorm/Data/MOUS/participants.tsv"
    save_dir = "/project/meganorm/Data/MOUS/participants_bids.tsv"
    make_demo_file_bids(file_dir, 
                        save_dir, 
                        0, 
                        2, 
                        {"col_name": "sex", "col_id": 1, "mapping": {"M": "Male", "F": "Female"}, "single_value":None},
                        {"col_name": "eyes", "col_id": None, "mapping": None, "single_value":"eyes_open"},
                        {"col_name": "diagnosis", "col_id": None, "mapping": None, "single_value":"control"})
    
    #TDBrain
    file_dir = "/project/meganorm/Data/EEG_TDBrain/EEG/TDBRAIN_participants_V2.tsv"
    save_dir = "/project/meganorm/Data/EEG_TDBrain/EEG/participants_bids.tsv"
    make_demo_file_bids(file_dir, 
                        save_dir, 
                        0, 
                        10, 
                        {"col_name": "sex", "col_id": 11, "mapping": {"1.0": "Male", "0.0": "Female"}, "single_value":None},
                        {"col_name": "eyes", "col_id": None, "mapping": None, "single_value":"eyes_closed"},
                        {"col_name": "diagnosis", "col_id": 2, "mapping": {
                            "REPLICATION": "replication", "BURNOUT": "burnout",  "SMC": "smc", 
                            "HEALTHY": "control", "Dyslexia": "dyslexia", "CHRONIC PAIN": "chronic pain", 
                            "MDD": "mdd", "nan": "nan", "ADHD": "adhd", 
                            "ADHD/ASPERGER": "adhd/asperger", "PDD NOS/DYSLEXIA": "pdd nos/dyslexia", 
                            "PDD NOS": "pdd nos", "WHIPLASH": "whiplash", "ANXIETY": "anxiety",
                            "ADHD/DYSLEXIA": "adhd/dyslexia", "ASD": "asd", "TINNITUS": "tinnitus",
                            "OCD": "ocd", "Tinnitus": "tinnitus", "PDD NOS ": "pdd nos", "PANIC": "panic",
                            "MDD/ANXIETY": "mdd/anxiety", "MIGRAINE": "migraine", "PDD NOS/ANXIETY": "pdd nos/anxiety",
                            "PARKINSON": "parkinson",  "BIPOLAR": "bipolar",  "MDD/bipolar": "mdd/bipolar",
                            "DYSPRAXIA": "dyspraxia", "TINNITUS/MDD": "tinnitus/mdd", "ADHD/ASD/ANXIETY": "adhd/asd/anxiety",
                            "MDD/ADHD": "mdd/adhd",  "ADHD/PDD NOS": "adhd/pdd nos", "MDD/BIPOLAR": "mdd/bipolar",
                            "ASPERGER": "asperger", "ADHD/EPILEPSY": "adhd/epilepsy", "MDD/PAIN": "mdd/pain",
                            "PDD NOS/GTS": "pdd nos/gts",  "PDD NOS/ADHD": "pdd nos/adhd", "PDD NOS/ASD": "pdd nos/asd",
                            "TBI": "tbi", "ADHD/ANXIETY": "adhd/anxiety",  "ADHD/DYSLEXIA/DYSCALCULIA": "adhd/dyslexia/dyscalculia",
                            "ADHD/MDD": "adhd/mdd", "MDD/PANIC": "mdd/panic", "DEPERSONALIZATION": "depersonalization",
                            "MDD/TRAUMA": "mdd/trauma", "PTSD/ADHD": "ptsd/adhd",  "OCD/DPS": "ocd/dps","MDD/OCD": "mdd/ocd",
                            "MDD/TUMOR": "mdd/tumor", "ADHD/GTS": "adhd/gts", "OCD/MDD": "ocd/mdd", "CONVERSION DX": "conversion dx",
                            "ASD/ASPERGER": "asd/asperger", "MDD/ADHD/LYME": "mdd/adhd/lyme", "ADHD/OCD": "adhd/ocd",
                            "MSA-C": "msa-c", "OCD/ASD": "ocd/asd", "STROKE/PAIN": "stroke/pain",
                            "STROKE ": "stroke", "MDD/OCD/ADHD": "mdd/ocd/adhd",  "EPILEPSY/OCD": "epilepsy/ocd",
                            "ADHD ": "adhd", "INSOMNIA": "insomnia", "MDD/ADHD/ANOREXIA": "mdd/adhd/anorexia",
                            "MDD/ANXIETY/TINNITUS": "mdd/anxiety/tinnitus"
                        }})

