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
but the defaults still include substantial preprocessing, including ICA and
GEDAI. Choose settings appropriate for your recordings.

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
error. Managed outputs are never appended or reused: choose a fresh directory
for another run, outside every input dataset root.

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

For multiple datasets use ``pipeline.run([dataset_a, dataset_b])`` with globally
unique participant IDs. Existing dataset dictionaries can be adapted through
``Dataset.from_dict(name, settings)``. Optional acquisition-specific inputs
are available through ``Dataset(options={...})``; see :doc:`meganorm.API`.

Existing low-level functions, CLI, and SLURM workflows remain available. See the
`full feature extraction tutorial
<https://github.com/ML4PNP/MEGaNorm/blob/main/notebooks/feature_extraction_tutorial.ipynb>`_
and the `PCNtoolkit tutorials
<https://pcntoolkit.readthedocs.io/en/stable/pages/tutorials.html>`_
for advanced processing, source localization, basis functions, likelihoods,
priors, and diagnostics.
