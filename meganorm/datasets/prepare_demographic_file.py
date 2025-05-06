import pandas as pd


def make_demo_file_bids(
    file_dir: str,
    save_dir: str,
    id_col: int,
    age_col: int,
    *columns
) -> None:
    """
    Convert formats of demographic data into a single format so it can be used
    in later stages.

    Parameters
    ----------
    file_dir : str
        Path to the input demographic file (supports CSV, TSV, or XLSX).
    save_dir : str
        Path where the BIDS-formatted demographic file will be saved (as TSV).
    id_col : int
        Column index containing the participant ID.
    age_col : int
        Column index containing participant age.
    *extra_columns : dict
        Additional column definitions. While age and participants id were defined
        using positional arguments, extra coulmn modification (e.g., sex and eyes 
        condition) can be revised and converted to a single format across dataset
        using this function. Each dict can contain:
            - 'col_name': str, required name for the output column. This does not
                necessarly match the column name before being passed to this function.
            - 'col_id': int, index of the column that the revision should be applied to.
            - 'single_value': value to assign to all rows if no col_id and mapping are given.
                This can be helpful when all subjects in a dataset have the same properties
                e.g., eyes open condition.
            - 'mapping': dict, if single value is not defined, value mapping can be passed
                to map the initial values to the target values.

    Returns
    -------
    None
    """
    for col in columns:    
        if col.get(single_value) == col.get(single_value) and col.get(mapping) == col.get(mapping):
            raise ValueError("'single_value' and 'mapping' can not be both defined.")
        if col.get(single_value) == col.get(single_value) and col.get(mapping) == col.get(mapping):
            raise ValueError("'single_value' and 'mapping' can not be both defined.")

    # Load input file based on extension
    if file_dir.endswith(".xlsx"):
        df = pd.read_excel(file_dir)
    elif file_dir.endswith(".csv"):
        df = pd.read_csv(file_dir)
    elif file_dir.endswith(".tsv"):
        df = pd.read_csv(file_dir, sep="\t")
    else:
        raise ValueError(f"Unsupported file type for: {file_dir}")

    # Initialize new dataframe with required fields
    new_df = pd.DataFrame({
        "participant_id": df.iloc[:, id_col],
        "age": df.iloc[:, age_col]
    })

    for col in columns:
        col_name = col.get("col_name")
        col_id = col.get("col_id")
        mapping = col.get("mapping")
        single_value = col.get("single_value")

        if col_name is None:
            raise ValueError("Each column dictionary must contain a 'col_name'.")

        if col_id is not None:
            new_df[col_name] = df.iloc[:, col_id]
            if mapping:
                new_df[col_name] = new_df[col_name].map(mapping)
        elif single_value is not None:
            new_df[col_name] = single_value
        else:
            raise ValueError(f"Column '{col_name}' must have either 'col_id' or 'single_value'.")

        # Special case handling
        if col_name == "diagnosis":
            new_df[col_name] = new_df[col_name].fillna("nan")

    # Remove duplicate participants
    new_df = new_df.drop_duplicates(subset="participant_id", keep="first")

    # Save as BIDS-compatible TSV
    new_df.to_csv(save_dir, sep="\t", index=False)


