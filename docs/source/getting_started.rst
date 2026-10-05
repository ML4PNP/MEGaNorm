Getting Started
===============

This quick-start example demonstrates the complete MEGaNorm workflow on a
single computer:

**MEG dataset → feature extraction → normative modeling**

The example intentionally uses a minimal configuration and processes subjects
sequentially. For large datasets, advanced preprocessing, source localization,
and SLURM-based parallel processing, see the
`full feature extraction tutorial
<https://github.com/ML4PNP/MEGaNorm/blob/main/notebooks/feature_extraction_tutorial.ipynb>`_.


Prerequisites
-------------

Install MEGaNorm:

.. code-block:: bash

   pip install meganorm

This example assumes that the dataset contains one directory per subject and
that the MEG recordings can be identified using a common task name and file
extension.

For example:

.. code-block:: text

   dataset/
   ├── participants.tsv
   ├── sub-01/
   │   └── sub-01_task-rest_meg.fif
   ├── sub-02/
   │   └── sub-02_task-rest_meg.fif
   └── sub-03/
       └── sub-03_task-rest_meg.fif

The demographic table should use the participant identifier as its first
column and contain the variables required for the normative model, for example:

.. code-block:: text

   participant_id    age    sex
   sub-01            24     F
   sub-02            31     M
   sub-03            42     F

For simplicity, source localization is disabled in this example, so FreeSurfer
data are not required.


1. Define the dataset
---------------------

MEGaNorm uses a dataset dictionary to describe where recordings and
demographic information are stored.

.. code-block:: python

   datasets = {
       "example_dataset": {
           "base_dir": "/path/to/dataset",
           "task": "rest",
           "ending": ".fif",
           "device_type": "MEGIN",
           "line_freq": 50,
           "demographic_path": "/path/to/dataset/participants.tsv",
       }
   }

The most important entries are:

``base_dir``
   Root directory containing the subject folders.

``task``
   Text used to identify the desired recording, such as ``"rest"``.

``ending``
   File extension used to identify recordings, such as ``".fif"``.

``device_type``
   MEG acquisition system.

``line_freq``
   Power-line frequency in Hz.

``demographic_path``
   Path to the participant-level demographic table.

Additional options, including empty-room recordings, event files, FreeSurfer
derivatives, head-position files, and acquisition-system-specific inputs, are
described in the full feature extraction tutorial.


2. Discover the subjects
------------------------

MEGaNorm can automatically discover matching recordings for all subjects in
the dataset:

.. code-block:: python

   from meganorm.utils.IO import merge_datasets_with_glob

   subjects = merge_datasets_with_glob(datasets)

   print(f"Found {len(subjects)} subjects")

   for subject in subjects:
       print(subject)

The resulting ``subjects`` dictionary contains the recording path and
dataset-specific information required to process each participant.


3. Create a processing configuration
------------------------------------

Preprocessing and feature-extraction options are controlled through
:class:`~meganorm.utils.IO.Config`.

For a first run, most settings can remain at their defaults:

.. code-block:: python

   from pathlib import Path
   from meganorm.utils.IO import Config

   project_dir = Path("./meganorm_example")
   project_dir.mkdir(parents=True, exist_ok=True)

   config = Config(
       which_sensor="meg",
       which_layout="all",
       apply_source_localization=False,
   )

   config_path = project_dir / "config.json"
   config.save(str(config_path), overwrite=True)

The complete set of preprocessing, spectral-analysis, and feature-extraction
parameters is documented in the
:class:`~meganorm.utils.IO.Config` API reference and in the full tutorial.


4. Process the dataset locally
------------------------------

Create directories for the extracted features:

.. code-block:: python

   features_dir = project_dir / "Features"
   temp_dir = features_dir / "temp"

   temp_dir.mkdir(parents=True, exist_ok=True)

MEGaNorm's processing function operates on one subject at a time. On a local
computer, all subjects can therefore be processed sequentially:

.. code-block:: python

   from meganorm.src.mainParallel import main

   for subject, info in subjects.items():
       print(f"Processing {subject}")

       main(
           [
               info["rest_record"],
               str(temp_dir),
               subject,
               str(config_path),
               "--line_freq",
               info["line_freq"],
               "--device_type",
               info["device"],
           ]
       )

Each subject is preprocessed and its electrophysiological features are
extracted according to the configuration.

The per-subject results are saved in:

.. code-block:: text

   meganorm_example/
   └── Features/
       └── temp/
           ├── sub-01.csv
           ├── sub-02.csv
           └── sub-03.csv


5. Combine the extracted features
---------------------------------

After all subjects have been processed, combine the individual feature tables
into a single file:

.. code-block:: python

   from meganorm.utils.parallel import collect_results

   collect_results(
       target_dir=str(features_dir),
       subjects=subjects,
       temp_path=str(temp_dir),
       file_name="all_features",
       clean=False,
   )

This produces:

.. code-block:: text

   meganorm_example/
   └── Features/
       ├── all_features.csv
       └── temp/
           ├── sub-01.csv
           ├── sub-02.csv
           └── sub-03.csv

At this stage, the electrophysiological features can be analyzed directly or
combined with demographic information for normative modeling.


6. Combine features and demographics
------------------------------------

MEGaNorm provides a helper for combining the extracted functional
imaging-derived phenotypes (f-IDPs) with participant-level demographic data:

.. code-block:: python

   from meganorm.utils.IO import merge_fidp_demo

   data = merge_fidp_demo(
       demographic_paths=[
           datasets["example_dataset"]["demographic_path"]
       ],
       features_dir=str(features_dir),
       dataset_names=["example_dataset"],
   )

   data.index.name = "participant_id"
   data = data.reset_index()

   print(data.head())

If the demographic table does not already contain a ``site`` column,
MEGaNorm assigns the dataset name as the site identifier.

The resulting dataframe contains the demographic variables and extracted
MEGaNorm features in a form suitable for normative modeling.


7. Prepare data for normative modeling
--------------------------------------

For this quick example, we model one extracted MEG feature as a function of
age.

MEGaNorm feature names contain ``__``, so we can identify the available
features and select one for demonstration:

.. code-block:: python

   feature_columns = [
       column for column in data.columns
       if "__" in column
   ]

   print(f"Available features: {len(feature_columns)}")
   print(feature_columns[:10])

   response_var = feature_columns[0]

   print(f"Modeling: {response_var}")

MEGaNorm's :func:`~meganorm.src.normative_modeling.prepare_nm_data` helper
converts the dataframe into PCNtoolkit ``NormData`` objects and creates a
train/test split:

.. code-block:: python

   from meganorm.src.normative_modeling import prepare_nm_data

   train, test = prepare_nm_data(
       df=data,
       response_vars=[response_var],
       covariate_list=["age"],
       batch_effect_list=["sex", "site"],
       subject_id_col_name="participant_id",
       train_split_size=0.8,
       random_state=42,
   )

Here:

* ``age`` is the model covariate;
* the selected MEGaNorm feature is the response variable;
* ``sex`` and ``site`` are used as batch-effect dimensions;
* 80% of participants are used for model fitting and 20% for testing.

These choices are intended only to demonstrate the workflow. Covariates,
batch effects, response variables, preprocessing, and model structure should
be selected according to the scientific question and study design.


8. Fit a normative model
------------------------

A simple hierarchical Bayesian regression model can now be fitted using
PCNtoolkit.

For a quick demonstration, use a Normal-likelihood HBR model with reduced
sampling settings:

.. code-block:: python

   from pcntoolkit import HBR
   from meganorm.src.normative_modeling import nm_model_train

   hbr = HBR(
       name="quickstart_hbr",
       cores=1,
       chains=2,
       draws=500,
       tune=500,
       progressbar=True,
   )

   nm_model_train(
       train=train,
       test=test,
       project_dir=str(project_dir),
       experiment_name="quickstart",
       template_regression_model=hbr,
       model_name="MEGaNorm_quickstart",
       if_parallel=False,
   )

The fitted model, predictions, evaluation results, and generated plots are
stored under:

.. code-block:: text

   meganorm_example/
   └── Normative_models/

The test data are used to evaluate how each participant deviates from the
normative trajectory learned from the training data.

.. note::

   The reduced sampling settings above are intended only to keep the quick
   start practical. For scientific analyses, sampling settings, model
   specification, convergence diagnostics, likelihood choice, and model
   evaluation should be considered carefully.


9. Next steps
-------------

This quick start demonstrates the basic MEGaNorm workflow on a local computer:

.. code-block:: text

   MEG recordings
         ↓
   preprocessing
         ↓
   spectral feature extraction
         ↓
   cohort-level f-IDP table
         ↓
   demographics + f-IDPs
         ↓
   PCNtoolkit NormData
         ↓
   normative model
         ↓
   individual deviations

For larger or more advanced studies, see the
`full feature extraction tutorial
<https://github.com/ML4PNP/MEGaNorm/blob/main/notebooks/feature_extraction_tutorial.ipynb>`_,
which covers:

* processing multiple datasets;
* detailed preprocessing and feature-extraction configuration;
* SLURM-based parallel processing;
* event-related recordings;
* empty-room recordings and environmental-noise correction;
* source localization and FreeSurfer integration;
* acquisition-system-specific options;
* demographic data preparation.

For more advanced normative modeling, including custom priors, nonlinear
basis functions, alternative likelihoods, model diagnostics, and SHASH
models, see the
`PCNtoolkit tutorials
<https://pcntoolkit.readthedocs.io/en/stable/pages/tutorials.html>`_.

For details about individual MEGaNorm classes and functions, see the
:doc:`API reference <modules>`.
