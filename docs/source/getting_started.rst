Getting Started
===============

This guide shows how to extract features from a cohort of MEG recordings and
fit a normative model. Each participant has a folder containing their
recording, and a participant table supplies their age and other information.
Use a cohort large enough for model training and evaluation.

Install MEGaNorm:

.. code-block:: bash

   pip install meganorm

Your dataset can contain many participants, for example::

   dataset/
       participants.tsv
       sub-001/sub-001_task-rest_meg.fif
       sub-002/sub-002_task-rest_meg.fif
       sub-003/sub-003_task-rest_meg.fif
       ...
       sub-100/sub-100_task-rest_meg.fif

The participant table uses the same identifiers as the participant folders::

   participant_id    age    sex
   sub-001           24     F
   sub-002           31     M
   sub-003           45     F
   ...               ...   ...
   sub-100           67     M

The dots illustrate additional participants; they should not appear as rows
in the actual table. Store ``participants.tsv`` as a tab-separated file.

1. Describe your dataset
----------------------------

.. code-block:: python

   from pathlib import Path
   from meganorm.API import Config, Dataset, Pipeline, NormativeModel
   from pcntoolkit import HBR

   output_dir = Path("./meganorm_results")
   dataset = Dataset(
       name="my_cohort",
       root="/path/to/dataset",
       demographics="participants.tsv",
       participant_id="participant_id",
       task="rest",
       extension=".fif",
       device="MEGIN",
       line_freq=50,
   )

Replace ``root`` with your dataset directory. Set the task, file extension,
device, and line frequency to match your recordings. The example uses MEGIN
recordings with a 50 Hz line frequency. The participant table path is relative
to the dataset directory. Choose an output directory outside your dataset.

2. Choose processing settings
---------------------------------

.. code-block:: python

   config = Config(
       which_sensor="meg",
       which_layout="all",
       apply_source_localization=False,
   )

This example extracts sensor-level features averaged across channels. The
default settings include preprocessing such as ICA; GEDAI is disabled.
Choose settings appropriate for your recordings. If a participant has several
matching recordings, ``which_meg_session`` selects one; the default is the first.

3. Extract features for the cohort
--------------------------------------

.. code-block:: python

   features = Pipeline(config=config, output_dir=output_dir).run(dataset)
   print(features.data.head())
   print("Available features:", features.feature_names)
   print("Processing outcomes:")
   print(features.processing)

``features.data`` contains the extracted features and participant information.
Processing continues if an individual participant fails, and only successful
participants enter the feature table. Inspect ``features.processing`` for
failed participants and their log locations. If no participants succeed, the
pipeline raises an error. To stop at the first processing failure, use
``Pipeline(config=config, output_dir=output_dir, on_error="raise")``.

Rerunning extraction in the same output directory warns and replaces the
previous extraction results, including saved processing files. Use a different
output directory if you want to retain an earlier run.

4. Fit a normative model
----------------------------

Choose the feature you want to model. The first available feature is used below
only as an example.

.. code-block:: python

   response = features.feature_names[0]
   print("Modeling:", response)

   estimator = HBR(
       name="my_cohort_hbr",
       cores=1,
       chains=2,
       draws=500,
       tune=500,
       progressbar=True,
   )

   results = NormativeModel(
       estimator=estimator,
       covariates=["age"],
       batch_effects=["sex", "site"],
       output_dir=output_dir / "normative_model",
       name="my_cohort_model",
       save_plots=False,
   ).fit(features, responses=[response], train_fraction=0.8, random_state=42)

   print(results.predictions.head())
   print(results.z_scores.head())
   print(results.metrics)

The example uses 80% of eligible participants for training and the remainder
for evaluation. Age must be numeric, and the selected model columns must have
no missing values. If your participant table has no ``site`` column, the dataset
name supplies it. Each sex/site label in the evaluation group must also occur
in the training group. Choose a supported split for your cohort.

``results.predictions`` contains predicted feature values for the evaluation
participants. ``results.z_scores`` describes their individual deviations from
the model, and ``results.metrics`` reports evaluation performance. Results are
saved under ``meganorm_results/normative_model/``.

The sampling settings are a starting example; check model diagnostics and
adjust them for your analysis. Fitting a new model into the same output directory
warns and replaces previous model results. Use a new directory to retain them.

Optional: combine several datasets
--------------------------------------

Describe each cohort separately and pass their list to the pipeline:

.. code-block:: python

   second_dataset = Dataset(
       name="second_cohort",
       root="/path/to/second_dataset",
       demographics="participants.tsv",
       participant_id="participant_id",
       task="rest",
       extension=".fif",
       device="MEGIN",
       line_freq=50,
   )

   combined_features = Pipeline(
       config=config,
       output_dir=Path("./combined_results"),
   ).run([dataset, second_dataset])

   print(combined_features.data.groupby("dataset").size())

Dataset names and participant identifiers must be unique across all input
cohorts. For example, the second cohort could use ``sub-101`` through
``sub-200``. Use compatible recordings and the same processing settings so
that the extracted features match. Use ``combined_features`` in place of
``features`` when fitting a model for the combined cohort.

If cohorts require different settings or have overlapping identifiers, process
each separately with its own ``Config`` and output directory.

Further examples
--------------------

The ``notebooks/getting_started.ipynb`` notebook follows this guide. For more
processing options, see :doc:`meganorm.API` and the
`feature extraction tutorial
<https://github.com/ML4PNP/MEGaNorm/blob/main/notebooks/feature_extraction_tutorial.ipynb>`_.
For model choices and diagnostics, see the
`PCNtoolkit tutorials
<https://pcntoolkit.readthedocs.io/en/stable/pages/tutorials.html>`_.
