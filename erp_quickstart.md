# EEG ERP analysis

Use the local API to process EEG recordings and collect amplitude and latency
IDPs in one participant-by-feature table. For a runnable example without a data
download, open the [ERP tutorial](notebooks/erp_idp_extraction_tutorial.ipynb).

## 1. Describe your recordings and events

Install MEGaNorm and its dependencies in your Python environment. Put recordings
in participant folders under a dataset root, for example
`/data/oddball/sub-001/sub-001_oddball_eeg.vhdr`. Use unique participant folder
names and one intended recording per participant. The `task` text must appear
in the filenames, and `extension` must match the recording format.

```python
from meganorm.API import (
    Config, Dataset, Pipeline, ERPSettings, ERPPreprocessing, ERPEventSpec,
)

# Adapt paths and annotation descriptions to your study.
dataset = Dataset(
    name="oddball", root="/data/oddball", task="oddball", extension=".vhdr",
    erp_events=ERPEventSpec(
        mode="annotations",
        annotation_map={"standard": "Stimulus/S  5", "target": "Stimulus/S  6"},
    ),
)
print(dataset.discover()[["participant_id", "recording_paths"]])
```

The map connects logical condition names on the left to exact recording
annotation descriptions on the right. Inspect a representative recording's
`raw.annotations.description` to find the correct descriptions.

If events are stored in a stimulus channel, use this instead:

```python
erp_events = ERPEventSpec(
    mode="stim", stim_channel="STI 014",
    event_id={"standard": 5, "target": 6},
)
# Supply erp_events=erp_events when constructing Dataset.
```

Use the actual stimulus channel and codes. External BIDS `events.tsv` files are
not read automatically.

## 2. Define your ERP analysis

Choose components, sensors, windows and preprocessing for your protocol. The
following values are illustrative starting settings, not a validated protocol
for every study. Time windows and baseline values are in seconds; rejection
thresholds are in volts.

```python
config = Config(
    analysis_type="erp", which_sensor="eeg",
    erp=ERPSettings(
        preprocessing=ERPPreprocessing(
            tmin=-0.2, tmax=0.8, baseline=(-0.2, 0.0),
            reference="average", l_freq=0.1, h_freq=35.0,
            ica_mode="auto", ica_l_freq=1.0,
            ica_corr_threshold=0.8, ica_label_threshold=0.9,
            min_trials=10, reject={"eeg": 150e-6},
        ),
        components={
            "P300": {
                "conditions": ["target"], "channels": ["Pz"],
                "window": [0.25, 0.5], "polarity": "pos",
                "metrics": ["peak_amplitude", "peak_latency", "mean_amplitude"],
            },
        },
    ),
)
```

Add components to the dictionary as needed. Each sensor is measured separately;
all components share epoch and baseline settings. Component conditions must be
present in the dataset event specification.

Ensure channel types, bad-channel markings and electrode positions are correct.
The worker uses embedded metadata or matching `*_channels.tsv` and a unique
local `*_montage.csv`; it does not guess positions. Channel tables require
`name` and `type`, with optional `status=bad`. Montage tables require
`Channel,X,Y,Z` in head-frame metres.

`ica_mode="auto"` uses usable EOG/ECG signals, otherwise ICLabel. ICLabel requires
average reference and valid sensor positions; its training used a 1–100 Hz fit
filter, so a different filter produces a QC warning. Explicit `physiological`,
`iclabel`, and `none` modes are also available. `none` disables ICA only.
Only `config.erp.preprocessing` controls ERP cleaning; resting-state settings
do not apply. Review the chosen settings on representative recordings.

## 3. Run and inspect

```python
result = Pipeline(
    config=config, output_dir="results", on_error="continue",
).run(dataset)

table = result.to_dataframe()
features = table.loc[:, result.feature_names]  # IDPs only
print(features)
print(result.processing)
print(result.summary)
print("Saved results:", result.output_dir)

import pandas as pd
qc = pd.read_csv(result.paths["qc"])
print(qc)
```

The pipeline processes participants and combines their results automatically.
For several datasets, use `.run([dataset_a, dataset_b, ...])`. Their event codes
may differ, but logical conditions must match the shared component definitions
and participant IDs must be unique across datasets. Optional demographics can
be supplied through `Dataset(demographics=..., participant_id=...)`.

EEG amplitudes are in µV and latencies in ms. A column such as
`ERP__target__P300__peak_amplitude_uV__Pz` identifies the condition, component,
measurement and sensor. Amplitudes retain their sign. Peaks are window extrema;
tied peaks use the earliest sample.

Missing conditions/sensors and insufficient trials produce `NaN` with a QC
reason. Complete and partial participant rows enter the table; failures are
recorded in `result.processing`. If no participant has usable IDPs, the run
raises an error. `on_error="raise"` stops at the first failed participant.
Inspect trial counts and `boundary_peak` flags before interpreting measurements.

Outputs are saved automatically under `results/erp/<run-id>/`, with a fresh
folder for each invocation. Each participant has feature/QC tables, a
`preprocessing.json` report and a processing log. Earlier runs are retained.
Keep the output directory outside your input dataset folder. QC is separate
from `result.feature_names`.

## Optional AutoReject epoch cleaning

Threshold rejection remains the default. To learn channel thresholds and repair
or reject epochs with AutoReject, update the ERP definition before running:

```python
config.erp = ERPSettings(
    preprocessing=config.erp.preprocessing,
    components=config.erp.components,
    rejection_mode="autoreject",
    autoreject={"cv": 5},
)
```

AutoReject replaces `preprocessing.reject`; that fixed amplitude cutoff is not
applied first. BAD annotations, recording edges and any `preprocessing.flat`
threshold still exclude epochs. It fits each participant's conditions together
and preserves the existing epoch times and baseline.

At least four usable EEG channels with valid distinct sensor positions and
`max(5, cv)` retained epochs are required. These are computational minimums,
not evidence that a small dataset is adequate for reliable cleaning. Unmet
requirements cause a participant failure; there is no silent fallback.
The two-channel synthetic tutorial therefore keeps threshold rejection.

Inspect `n_autoreject_rejected` and `n_interpolated_trials` in QC. The participant's
`preprocessing.json` also records learned settings, channel thresholds and
per-trial interpolation decisions. Interpolation counts refer to retained trials.
All-rejected recordings have unavailable IDPs and are reported as failed.
Validate cleaning and ERP measurements on representative real recordings.

## Optional standalone workflow

For in-memory epochs and evoked waveforms, or extraction from already-cleaned
evoked responses, see the final sections of the
[ERP tutorial](notebooks/erp_idp_extraction_tutorial.ipynb). The cohort API does
not currently save raw or epoch objects; keep its raw/epoch saving flags disabled.

The local raw-data workflow supports EEG with threshold rejection or AutoReject.
Automatic continuous bad-channel detection, sensor-group
averages, condition contrasts, source reconstruction and ERP SLURM execution are
not yet supported. Multiple recordings are not automatically combined into
sessions. Prepared-evoked extraction can also support MEG; this is separate from
raw EEG cohort processing.

## Optional single-participant command line

Save the same configuration and event specification:

```python
from pathlib import Path
config.save("erp_config.json")
Path("erp_events.json").write_text(dataset.erp_events.model_dump_json(indent=2))
```

Then run:

```bash
python -m meganorm.src.mainParallel \
  /data/oddball/sub-001/sub-001_oddball_eeg.vhdr results sub-001 erp_config.json \
  --erp_events erp_events.json
```

This uses the same participant worker as the local cohort API. For a cohort,
prefer `Pipeline.run()` so that discovery, collection and processing reports
are handled together.
