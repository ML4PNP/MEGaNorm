"""Deterministic FIF cohort exercising the real local processing workflow."""

import numpy as np
import pandas as pd
import mne
import pytest


@pytest.fixture
def api_features(api_cohort, tmp_path):
    """Extract a real feature table using the public API for model workflows."""
    from meganorm.API import Pipeline

    dataset, config = api_cohort
    return Pipeline(config=config, output_dir=tmp_path / "features").run(dataset)


@pytest.fixture
def api_cohort(tmp_path):
    from meganorm.API import Config, Dataset

    root = tmp_path / "cohort"
    rng = np.random.default_rng(42)
    sfreq = 512
    time = np.arange(60 * sfreq) / sfreq
    ids = [f"sub-{i:03}" for i in range(24)]
    info = mne.create_info([f"MEG{i:03}" for i in range(1, 7)], sfreq, ch_types="mag")
    for i, subject in enumerate(ids):
        samples = 1e-12 * (
            (1 + i / 48) * np.sin(2 * np.pi * 10 * time)[None, :]
            + rng.normal(0, 1, (6, len(time)))
        )
        folder = root / subject
        folder.mkdir(parents=True)
        raw = mne.io.RawArray(samples, info, verbose=False)
        raw.info["line_freq"] = 50
        raw.save(folder / f"{subject}_task-rest_meg.fif", overwrite=True, verbose=False)
    pd.DataFrame(
        {
            "participant_id": ids,
            "age": np.linspace(20, 70, 24),
            "sex": ["F", "M"] * 12,
            "site": ["A"] * 24,
        }
    ).to_csv(root / "participants.tsv", sep="\t", index=False)
    config = Config(
        apply_source_localization=False,
        apply_ica=False,
        apply_oversampled_temporal_projection=False,
        apply_Head_movement_correction=False,
        apply_environmental_noise_correction=False,
        drop_noisy_flat_channel=False,
        bad_segment_removal_method=None,
        which_layout=None,
        save_psds=False,
        psd_parametrization_method="specparam",
        aperiodic_mode="fixed",
        min_r_squared=0.0,
    )
    config.feature_categories = {
        key: key == "OriginalPSD_Canonical_Relative_Power"
        for key in config.feature_categories
    }
    return (
        Dataset(
            name="synthetic",
            root=root,
            demographics="participants.tsv",
            task="rest",
            extension=".fif",
            device="MEGIN",
        ),
        config,
    )
