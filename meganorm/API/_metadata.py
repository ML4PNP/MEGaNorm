"""Explicit participant identities and one-to-one demographic joins."""

import pandas as pd


def normalize_ids(frame, participant_id):
    """Copy a frame, validating IDs before normalizing its named string index."""
    frame = frame.copy(deep=True)
    if not frame.columns.is_unique:
        raise ValueError("Table contains duplicate column names.")
    if participant_id in frame.columns:
        ids = frame[participant_id]
        if frame.index.name == participant_id and list(frame.index.astype(str)) != list(
            ids.astype(str)
        ):
            raise ValueError("Participant ID column and index disagree.")
        frame = frame.drop(columns=participant_id)
    elif frame.index.name == participant_id:
        ids = pd.Series(frame.index, index=frame.index)
    else:
        raise ValueError(
            f"Missing participant identifier column/index: {participant_id}"
        )
    if ids.isna().any():
        raise ValueError("Participant IDs contain missing values.")
    strings = ids.astype(str)
    if (strings.str.strip() == "").any():
        raise ValueError("Participant IDs contain missing/empty values.")
    if (strings != strings.str.strip()).any():
        raise ValueError("Participant IDs contain surrounding whitespace.")
    if strings.duplicated().any():
        raise ValueError(
            "Participant IDs contain duplicate values after string conversion."
        )
    frame.index = pd.Index(strings.to_numpy(), name=participant_id)
    return frame


def load_demographics(dataset):
    if dataset.demographics is None:
        return None
    path = dataset.demographics
    if not path.is_file():
        raise FileNotFoundError(f"Demographics missing for {dataset.name}: {path}")
    dtype = {dataset.participant_id: str}
    if path.suffix.lower() in {".tsv", ".txt", ".csv"}:
        frame = pd.read_csv(
            path, sep="," if path.suffix.lower() == ".csv" else "\t", dtype=dtype
        )
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        frame = pd.read_excel(path, dtype=dtype)
    else:
        raise ValueError(f"Unsupported demographic file type: {path}")
    return normalize_ids(frame, dataset.participant_id)


def attach_metadata(features, datasets, manifest):
    features = features.copy(deep=True)
    features.index = features.index.astype(str)
    if "subject" in features:
        if list(features.subject.astype(str)) != list(features.index):
            raise ValueError("Collector subject column disagrees with feature index.")
        features = features.drop(columns="subject")
    counts = {"matched_demographics": 0, "unused_demographics": {}}
    tables = []
    for dataset in datasets:
        discovered = manifest.loc[
            manifest.dataset == dataset.name, "participant_id"
        ].tolist()
        demo = load_demographics(dataset)
        if demo is not None:
            missing = set(discovered) - set(demo.index)
            if missing:
                raise ValueError(
                    f"Missing demographics for {dataset.name}: {sorted(missing)}"
                )
            counts["unused_demographics"][dataset.name] = len(
                set(demo.index) - set(discovered)
            )
            ids = [p for p in discovered if p in features.index]
            counts["matched_demographics"] += len(ids)
            demo = demo.loc[ids].copy()
        else:
            ids = [p for p in discovered if p in features.index]
            demo = pd.DataFrame(index=ids)
            counts["unused_demographics"][dataset.name] = 0
        if "dataset" in demo and not demo.dataset.eq(dataset.name).all():
            raise ValueError(
                f"Demographic dataset labels conflict with {dataset.name}."
            )
        demo["dataset"] = dataset.name
        if "site" not in demo:
            demo["site"] = dataset.name
        if overlap := set(demo.columns) & set(features.columns):
            raise ValueError(f"Demographic/feature column overlap: {sorted(overlap)}")
        tables.append(demo)
    demo = pd.concat(tables).reindex(features.index)
    result = demo.join(features, how="left", validate="one_to_one")
    result.index.name = "participant_id"
    return result, counts
