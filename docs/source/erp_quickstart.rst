EEG ERP analysis
================

Use the local API to process EEG recordings and collect amplitude and
latency IDPs in one participant-by-feature table. For a runnable example
without a data download, open the `ERP
tutorial <https://github.com/smkia/MEGaNorm/blob/dev_erp/notebooks/erp_idp_extraction_tutorial.ipynb>`__.

1. Describe your recordings and events
--------------------------------------

Install MEGaNorm and its dependencies in a Python 3.12 environment. Put
recordings in participant folders under a dataset root, for example
``/data/oddball/sub-001/sub-001_oddball_eeg.vhdr``. Use unique
participant folder names and one intended recording per participant. The
``task`` text must appear in the filenames, and ``extension`` must match
the recording format.

.. code:: python

   from meganorm.API import (
       Config, Dataset, Pipeline, ERPEventSpec,
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

The map connects logical condition names on the left to exact recording
annotation descriptions on the right. Inspect a representative
recording’s ``raw.annotations.description`` to find the correct
descriptions.

If events are stored in a stimulus channel, use this instead:

.. code:: python

   erp_events = ERPEventSpec(
       mode="stim", stim_channel="STI 014",
       event_id={"standard": 5, "target": 6},
   )
   # Supply erp_events=erp_events when constructing Dataset.

Use the actual stimulus channel and codes. External BIDS ``events.tsv``
files are not read automatically.

2. Define your ERP analysis
---------------------------

Choose components, sensors, windows and preprocessing for your protocol.
The following values are illustrative starting settings, not a validated
protocol for every study. Time windows and baseline values are in
seconds; rejection thresholds are in volts.

.. code:: python

   config = Config(
       analysis_type="erp", which_sensor="eeg",
       cutoffFreqLow=0.1, cutoffFreqHigh=30, resampling_rate=250,
       rereference_method="average",
       ica_mode="auto", ica_l_freq=1, ica_h_freq=100, ica_resample_sfreq=250,
       auto_ica_corr_thr=0.8, ica_label_threshold=0.9,
       bad_segment_removal_method="fixed_thr", eeg_var_threshold=150e-6,
       erp={
           "tmin": -0.2, "tmax": 0.8, "baseline": (-0.2, 0.0),
           "min_trials": 10,
           "components": {
               "P300": {
                   "conditions": ["target"], "channels": ["Pz"],
                   "window": [0.25, 0.5], "polarity": "pos",
                   "metrics": ["peak_amplitude", "peak_latency", "mean_amplitude"],
               },
           },
       },
   )

Add components to the dictionary as needed. Each sensor is measured
separately; all components share epoch and baseline settings. Component
conditions must be present in the dataset event specification.

Ensure channel types, bad-channel markings and electrode positions are
correct. The worker uses embedded metadata or matching
``*_channels.tsv`` and supported position sidecars. Channel tables
require ``name`` and ``type``, with optional ``status=bad``. Montage
tables require ``Channel,X,Y,Z`` in head-frame metres.

``ica_mode="auto"`` uses usable EOG/ECG signals, otherwise ICLabel.
ICLabel requires average reference and valid sensor positions; its
training used a 1–100 Hz fit filter, so a different filter produces a QC
warning. Explicit ``physiological``, ``iclabel``, and ``none`` modes are
also available. ``none`` disables ICA only. Shared preprocessing options
live directly on ``Config``; ``erp`` contains only epoch timing,
baseline, minimum trial count, component definitions and ERP output flags.
Resting-state segmentation and spectral options do not control ERP analysis. Review
the chosen settings on representative recordings.

To fit ICA independently of the ERP analysis filter, set these fields in
``Config`` (the recording must have a sample rate above 200 Hz):

.. code:: python

   Config(
       analysis_type="erp", which_sensor="eeg",
       cutoffFreqLow=0.1, cutoffFreqHigh=30, resampling_rate=250,
       ica_l_freq=1, ica_h_freq=100, ica_resample_sfreq=250,
   )

This fits ICA on a 1–100 Hz copy and applies the cleaning to the 0.1–30
Hz ERP data. Event timing follows the analysis sample rate. By default,
``ica_h_freq=None`` inherits ``cutoffFreqHigh``, while
``ica_resample_sfreq=None`` keeps the original sample rate for fitting.

Electrode positions
~~~~~~~~~~~~~~~~~~~

Valid embedded positions take priority, followed by adjacent BIDS
electrode files and then montage CSV files. If positions are absent,
select a template once in ``Dataset``; no preparation script or
generated sidecars are needed:

.. code:: python

   # Add this argument to Dataset(...).
   erp_montage={
       "template": "auto",  # 1005 template; does not infer the actual cap geometry.
       "exclude_channels": ["O9", "O10"],  # Only if these should be excluded.
   }

You can select an MNE built-in template name instead of ``auto``. Omit
this argument to use supplied positions only. Templates are
approximations, not participant measurements. Ear labels
``TP9 LEFT EAR`` and ``TP10 RIGHT EAR`` map to TP9 and TP10 while
preserving the original recording names. Unmatched channels produce an
error; explicit exclusions cannot include requested IDP channels. The
input files remain untouched.

Template use is announced and recorded in ``processing.log``,
``montage.json``, ``preprocessing.json``, and ``provenance.json``. If an
existing CSV came from a template, add ``"supplied_kind": "template"``;
use ``"measured"`` only for known measured positions. Otherwise, the
origin of supplied positions is recorded as unknown.

Adjacent BIDS ``*_electrodes.tsv`` and matching ``*_coordsystem.json``
support **CapTrak head coordinates** in ``m``, ``cm``, or ``mm``. Other
coordinate systems require an explicit transformation before loading;
full BIDS inheritance is not implemented. CSV files require
``Channel,X,Y,Z`` in head-frame metres. A recording-specific CSV takes
priority over a single unambiguous local CSV. Incomplete supplied
positions are never mixed with template coordinates.

3. Run and inspect
------------------

.. code:: python

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

The pipeline processes participants and combines their results
automatically. For several datasets, use
``.run([dataset_a, dataset_b, ...])``. Their event codes may differ, but
logical conditions must match the shared component definitions and
participant IDs must be unique across datasets. Optional demographics
can be supplied through
``Dataset(demographics=..., participant_id=...)``.

EEG amplitudes are in µV and latencies in ms. A column such as
``ERP__target__P300__peak_amplitude_uV__Pz`` identifies the condition,
component, measurement and sensor. Amplitudes retain their sign. Peaks
are window extrema; tied peaks use the earliest sample.

Missing conditions/sensors and insufficient trials produce ``NaN`` with
a QC reason. Complete and partial participant rows enter the table;
failures are recorded in ``result.processing``. If no participant has
usable IDPs, the run raises an error. ``on_error="raise"`` stops at the
first failed participant. Inspect trial counts and ``boundary_peak``
flags before interpreting measurements.

Outputs are saved automatically under ``results/erp/<run-id>/``, with a
fresh folder for each invocation. Each participant has feature/QC
tables, a ``preprocessing.json`` report and a processing log. By default,
it also has reusable condition averages in ``evoked-ave.fif``, their
``erp_result.json`` descriptor, and a self-contained ``report.html``. Earlier
runs are retained. Keep the output directory outside your input dataset
folder. QC is separate from ``result.feature_names``.

4. Open reports and plot saved responses
----------------------------------------

Open the cohort index at ``result.paths["report"]`` in a browser. It links
participant reports with annotated ERP curves, measurement tables, trial
counts and preprocessing QC. Reports show missing measurements and their
reasons as well as available IDPs.

.. code:: python

   print("Cohort report:", result.paths["report"])

   import matplotlib.pyplot as plt
   participant = result.load_erp("sub-001")
   fig, axes = participant.plot(channels=("Cz", "Pz"), show=False)
   plt.show()

   # Equivalent convenience method on the cohort result:
   fig, axes = result.plot_erp("sub-001", channels=("Cz", "Pz"), show=False)
   plt.show()

Use the exact participant folder name, including prefixes and leading
zeros. An integer such as ``1`` is not converted to ``"sub-001"``. Request
channels that exist in your recordings.

To reopen a participant later without running preprocessing again:

.. code:: python

   from meganorm.API import load_erp_result
   participant = load_erp_result(
       result.output_dir / "participants" / "sub-001",
   )
   print(participant.features)
   print(participant.qc)

The FIF contains condition averages, not single trials. Reloaded results
have ``epochs=None``; features and QC come from the stored tables and are
not recalculated. FIF waveform values can differ slightly due to floating
point storage precision. The descriptor preserves condition mappings,
missing conditions, component definitions and participant identity.

ERP output defaults are ``save_evokeds=True``, ``save_report=True`` and
``save_figures=False`` inside ``config.erp``. Set these when constructing
``Config(erp={...})``. Enable ``save_figures`` for separate PNG/SVG plots
under each participant's ``plots/`` folder. Disabling ``save_evokeds``
prevents later waveform reloading. These options do not change resting-state
outputs. Core feature/QC outputs are retained if report or figure generation
fails; inspect ``report_outputs.json`` for artifact status. Failure to save
requested evoked FIF data is a participant processing failure.

Optional AutoReject epoch cleaning
----------------------------------

Threshold rejection remains the default. To learn channel thresholds and
repair or reject epochs with AutoReject, update the ERP definition
before running:

.. code:: python

   options = config.model_dump()
   options.update(bad_segment_removal_method="autoreject", autoreject_cv=5)
   config = Config(**options)

AutoReject replaces ``eeg_var_threshold``; that fixed amplitude cutoff
is not applied first. BAD annotations, recording edges and any
``eeg_flat_threshold`` threshold still exclude epochs. It fits each
participant’s conditions together and preserves the existing epoch times
and baseline.

At least four usable EEG channels with valid distinct sensor positions
and ``max(5, cv)`` retained epochs are required. These are computational
minimums, not evidence that a small dataset is adequate for reliable
cleaning. Unmet requirements cause a participant failure; there is no
silent fallback. The two-channel synthetic tutorial therefore keeps
threshold rejection.

Inspect ``n_autoreject_rejected`` and ``n_interpolated_trials`` in QC.
The participant’s ``preprocessing.json`` also records learned settings,
channel thresholds and per-trial interpolation decisions. Interpolation
counts refer to retained trials. All-rejected recordings have
unavailable IDPs and are reported as failed. Validate cleaning and ERP
measurements on representative real recordings.

Optional standalone workflow
----------------------------

For a selected participant's in-memory epochs, run the same discovery and
metadata preparation used by the cohort:

.. code:: python

   participant = dataset.extract_erp("sub-001", config=config)
   fig, axes = participant.plot(channels=("Cz", "Pz"), show=False)
   plt.show()
   participant.save("participant-export", report=True, figures=False)

Extraction itself writes no output files. ``save`` writes reusable condition
averages and feature/QC tables, plus the requested report and figures;
``overwrite=False`` protects existing outputs by default. Dataset extraction
uses the configured session selection, events, montage and line frequency.
For a recording path supplied directly, pass the dataset metadata explicitly:

.. code:: python

   from meganorm.API import extract_recording
   participant = extract_recording(
       "/data/oddball/sub-001/sub-001_oddball_eeg.vhdr", "sub-001",
       config=config, event_spec=dataset.erp_events,
       montage_settings=dataset.erp_montage, line_freq=dataset.line_freq,
   )

For extraction from already-cleaned evoked responses, see the final sections
of the `ERP
tutorial <https://github.com/smkia/MEGaNorm/blob/dev_erp/notebooks/erp_idp_extraction_tutorial.ipynb>`__.
The cohort API does not currently save raw or epoch objects; keep its
raw/epoch saving flags disabled.

The local raw-data workflow supports EEG with threshold rejection or
AutoReject. Automatic continuous bad-channel detection, sensor-group
averages, condition contrasts, source reconstruction and ERP SLURM
execution are not yet supported. Multiple recordings are not
automatically combined into sessions. Prepared-evoked extraction can
also support MEG; this is separate from raw EEG cohort processing.

Optional single-participant command line
----------------------------------------

Save the same configuration and event specification:

.. code:: python

   from pathlib import Path
   config.save("erp_config.json")
   Path("erp_events.json").write_text(dataset.erp_events.model_dump_json(indent=2))

Then run:

.. code:: bash

   python -m meganorm.src.mainParallel \
     /data/oddball/sub-001/sub-001_oddball_eeg.vhdr results sub-001 erp_config.json \
     --erp_events erp_events.json

This uses the same participant worker as the local cohort API. For a
cohort, prefer ``Pipeline.run()`` so that discovery, collection and
processing reports are handled together.
