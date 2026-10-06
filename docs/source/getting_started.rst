Getting Started
===============

The local scientific API follows four steps:
**describe a dataset → configure processing → extract features → fit a normative model**.
It processes participants sequentially and preserves the existing MEGaNorm
processing and PCNtoolkit modeling functions.

Install MEGaNorm with ``pip install meganorm``. These examples require a cohort
of usable recordings and a participant table containing numeric age and the
chosen batch-effect labels. A directory layout might be::

   dataset/
       participants.tsv
       sub-001/sub-001_task-rest_meg.fif
       sub-002/sub-002_task-rest_meg.fif

The participant table might begin::

   participant_id    age    sex
   sub-001           24     F
   sub-002           31     M

These two rows illustrate the format; they are not a sufficient training cohort.
Participant identifiers must match the directory names and be unique across
input datasets. The identifier column need not be the first table column.

1. Describe the dataset
-----------------------

.. code-block:: python

   from pathlib import Path
   import logging
   from meganorm.API import Config, Dataset, Pipeline, NormativeModel
   from pcntoolkit import HBR

   logging.basicConfig(level=logging.INFO)
   output_dir = Path("./meganorm_example")
   dataset = Dataset(
       name="example_dataset",
       root="/path/to/dataset",
       demographics="participants.tsv",
       participant_id="participant_id",
       task="rest",
       extension=".fif",
       device="MEGIN",
       line_freq=50,
   )

Relative demographic and auxiliary paths resolve against the dataset root.
``line_freq`` is a fallback when recording metadata are unavailable. To inspect
recording candidates without processing, call ``dataset.discover()``.

2. Configure the analysis
-------------------------

.. code-block:: python

   config = Config(
       which_sensor="meg",
       which_layout="all",
       apply_source_localization=False,
   )

This is the existing :class:`~meganorm.utils.IO.Config`. The pipeline saves it
automatically and snapshots it when constructed. Source localization is off,
but the defaults still include substantial preprocessing, including ICA. GEDAI is disabled by default. Choose settings appropriate for your recordings.

If several recordings match a participant, ``config.which_meg_session`` selects
one from the sorted candidate list. The API does not concatenate all sessions.

3. Extract and inspect features
-------------------------------

Extract features through the public API and inspect the participant table:

.. code-block:: python

   features = Pipeline(config=config, output_dir=output_dir).run(dataset)
   print(features.data.head())
   print("Available features:", features.feature_names[:10])
   # For research, replace this demonstration choice with your selected response.
   response = features.feature_names[0]
   print("Modeling:", response)

``features.data`` contains one row per successful participant, with demographic
variables, dataset/site labels, and extracted features. ``features.feature_names``
contains only extracted feature names. ``features.processing`` records each
attempted participant's outcome; ``features.paths`` points to saved artifacts.
The first feature is selected above only to demonstrate execution. Select the
response according to your scientific question for actual research.

The pipeline creates configuration, feature, and report directories. If a
``site`` column is absent, the dataset name supplies it. Missing values in an
existing site column remain missing. Supplied demographics must match all
discovered participants; extra demographic rows are reported and allowed.
For extraction without demographics, omit the ``demographics`` argument.

By default, processing stops at the first failure and preserves a partial
report. Use ``Pipeline(..., on_error="continue")`` to process the remaining
participants and collect only successful outputs. All-failed runs raise an
error. Managed outputs are never appended or reused. When rerunning in the same
output directory, the pipeline validates inputs first, warns, and removes its
previous ``Features/`` tree (including temporary CSVs, logs, plots, and saved
preprocessing), ``config.json``, ``manifest.csv``, ``processing.csv``,
``run_summary.json``, and ``features_with_demographics.csv`` before processing.
Save a copy elsewhere if you need to retain those results. Unrelated files and
separate model-output directories are preserved. If input validation fails,
previous outputs are preserved; if processing fails, old aggregate tables are
not left behind as apparent current results. The output directory must be
outside every input dataset root, and managed output paths must not contain
input data.

Processing several datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a list of ``Dataset`` objects to one ``Pipeline.run`` call to extract and
combine their participants. For example, keep ``dataset`` from step 1 and add
a second cohort with its own recording and demographic paths:

.. code-block:: python

   second_dataset = Dataset(
       name="second_dataset",
       root="/path/to/second_dataset",
       demographics="participants.tsv",
       participant_id="participant_id",
       task="rest",
       extension=".fif",
       device="MEGIN",
       line_freq=60,
   )
   combined_features = Pipeline(
       config=config,
       output_dir=Path("./meganorm_combined"),
       on_error="continue",
   ).run([dataset, second_dataset])
   print(combined_features.data.groupby("dataset").size())
   print(combined_features.processing)

Use the line frequency appropriate for each acquisition; 60 Hz here is only
an example. Relative paths resolve against each dataset's own root. Each
cohort can specify its own device, task, extension, demographic identifier
column, line frequency, and auxiliary ``options``. Participants from all
cohorts are processed sequentially, with one shared ``Config`` and one combined
``FeatureDataset``. Use ``combined_features`` in place of ``features`` in step 4
to model this combined cohort, with a separate model output directory if needed.

Dataset names and participant IDs must be unique across the entire run. For
example, if the first cohort uses ``sub-001`` and ``sub-002``, the second could
use ``sub-101`` and ``sub-102``, with matching demographic identifiers. Two
cohorts both containing ``sub-001`` are rejected before output cleanup or
processing; the API does not automatically prefix IDs. All participants must
produce the same extracted feature columns for collection. Choose compatible
recordings and layouts; this API does not harmonize mismatched feature schemas.

The combined table retains each participant's ``dataset`` label and supplied
``site`` label. When a cohort has no ``site`` column, its dataset name supplies
one; missing values in an existing ``site`` column remain missing. Metadata
validation covers all discovered participants before processing, including
when ``on_error="continue"`` is used. That option only permits participant
processing failures and excludes failed participants from the combined table.

If cohorts require different processing configurations or have overlapping
participant IDs, run them separately with distinct output directories:

.. code-block:: python

   cohort_a_features = Pipeline(
       config=config,
       output_dir=Path("./results/cohort_a"),
   ).run(dataset)
   cohort_b_features = Pipeline(
       config=config,
       output_dir=Path("./results/cohort_b"),
   ).run(second_dataset)

You can supply a different ``Config`` to each separate pipeline. These calls
return independent results; they do not automatically merge tables or resolve
participant-ID collisions. Calling ``run`` twice with the same output directory
replaces the first run's managed outputs after a warning; it does not append
the second cohort. To combine cohorts in one extraction run, pass their list
in a single call as shown above.

Phase 1 local extraction is sequential and has no participant-level ``n_jobs``
option. The existing SLURM workflow remains available for parallel extraction.

4. Fit and inspect a normative model
------------------------------------

.. code-block:: python

   estimator = HBR(
       name="quickstart_hbr",
       cores=1,
       chains=2,
       draws=500,
       tune=500,
       progressbar=True,
   )

.. code-block:: python

   results = NormativeModel(
       estimator=estimator,
       covariates=["age"],
       batch_effects=["sex", "site"],
       output_dir=Path(output_dir) / "quickstart",
       name="MEGaNorm_quickstart",
       save_plots=False,
   ).fit(features, responses=[response], train_fraction=0.8, random_state=42)
   print(results.z_scores.head())
   print(results.predictions.head())
   print(results.metrics)

``results.predictions`` contains predictive means on the original response
scale. ``results.z_scores`` contains PCNtoolkit standardized individual
deviations. Both tables use held-out participant IDs and selected response
names. ``results.metrics`` contains available held-out evaluation statistics;
``results.model`` is the live fitted PCNtoolkit model. Actual train/test and
excluded participants are recorded in ``results.participants``.

Reduced HBR sampling keeps the demonstration practical; it does not establish
convergence or scientific adequacy. The example disables saved model plots to
keep execution simple. Enable ``save_plots=True`` for native PCNtoolkit plots,
and inspect model diagnostics using the advanced tools.

Covariates and responses must be numeric; explicitly encode categorical
covariates if needed. Default ``missing="raise"`` identifies invalid selected
model values. ``missing="drop"`` records complete-case exclusions without
imputation. Unselected feature missingness does not discard participants.

Use ``train_fraction=None`` to fit the entire eligible cohort without held-out
results. To supply an explicit test cohort, pass
``fit(training_data, test_data=test_data, responses=[response], train_fraction=None)``.
Training and test IDs must not overlap. New test batch labels are rejected by
this Phase 1 path; use PCNtoolkit's advanced transfer workflow for new sites.
A random participant split is a demonstration, not site-level generalization.

The model outputs remain under ``quickstart/Normative_models/``. Predictions,
deviations, metrics, participant membership, and the analysis summary are also
exported as ordinary tables/reports. If ``evaluate=False``, metrics are absent;
on PCNtoolkit 1.3.0 the API disables native result saving in that mode because
its writer requires statistics, while still exporting predictions/deviations.

Executable verification and advanced workflows
----------------------------------------------

From a development checkout with test dependencies installed, run:

.. code-block:: bash

   python -m pytest tests/integration/test_api_workflow.py -q
   python -m pytest tests/integration/test_api_hbr.py -q -m slow

The first test runs the real four-step workflow on a reproducible 24-participant
synthetic FIF cohort with a lightweight BLR estimator. The second executes HBR
with the sampling settings shown above. Both exercise the public API directly;
the synthetic fixture explicitly disables
processing options that its generated recordings cannot support and uses
specparam with a fixed aperiodic model and ``min_r_squared=0`` to exercise
feature extraction on these artificial signals. These checks
verify orchestration and output alignment, not the complete default
preprocessing stack or model adequacy for a clinical application.

Existing dataset dictionaries can be adapted through
``Dataset.from_dict(name, settings)``. Optional acquisition-specific inputs
are available through ``Dataset(options={...})``; see :doc:`meganorm.API`.

Existing low-level functions, CLI, and SLURM workflows remain available. See the
`full feature extraction tutorial
<https://github.com/ML4PNP/MEGaNorm/blob/main/notebooks/feature_extraction_tutorial.ipynb>`_
and the `PCNtoolkit tutorials
<https://pcntoolkit.readthedocs.io/en/stable/pages/tutorials.html>`_
for advanced processing, source localization, basis functions, likelihoods,
priors, and diagnostics.
