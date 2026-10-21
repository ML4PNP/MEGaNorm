Changes in v0.2.2
=================

Local cohort workflow
----------------------

The Python API connects ``Dataset``, ``Config``, ``Pipeline``, ``FeatureDataset``,
``NormativeModel``, and ``NormativeResults``. It supports feature extraction for
one or more cohorts, explicit model fitting, and prediction and deviation tables
aligned with participant identifiers. See :doc:`getting_started`.

Extraction continues after a participant fails and records the outcome and log
path in ``features.processing``. Use ``on_error="raise"`` to stop at the first
failure. Only successful participants from the current run enter the results.

Reruns warn before replacing managed extraction or model outputs. Input checks
run before cleanup, and unrelated files are retained. Use separate output
directories to keep earlier analyses. Response names must be safe directory
names, because PCNtoolkit uses them when saving models.
Native train and test files use distinct names so that their participant
identities remain separate in the saved results.

The existing SLURM workflow remains available in the cluster notebook. The local
``Pipeline`` processes participants sequentially and does not submit SLURM jobs.

Processing and configuration
-----------------------------

The :doc:`configuration` guide covers JSON files, common settings, feature
families, and source-localization inputs. The configuration reference and
notebooks use the current field names and defaults.

Several defaults differ from earlier releases: the low-pass cutoff is 40 Hz,
the target sampling rate is 300 Hz, and epochs are 5 seconds long with 2 seconds
of overlap. Oversampled temporal projection and reference-channel environmental
ICA are off by default. The default spectral method is specparam with a fixed
aperiodic slope. Intermediate processing files are saved only when requested.

The spectral configuration uses ``psd_parametrization_*`` fields shared by
specparam and IRASA. Older ``fooof_*`` names and ``parametrization_method`` need
to be updated. GEDAI and its configuration fields are absent from the current
pipeline. Save a full configuration and check its settings when updating an
existing analysis.

Numerically failed IRASA aperiodic fits are retried per channel with power
scaling, while preserving original spectral units and the fit-quality threshold.
EEG fixed-threshold rejection uses a flatness limit below its rejection limit.
Invalid processing ranges and incomplete feature-family dictionaries now fail
when the configuration is created.

Additional fixes preserve annotation timing in cropped recordings, select
participant-specific annotation files by exact identifier, and create requested
intermediate-output folders. EEG ICA handles missing physiological channels
independently and records its ICLabel algorithm choice.

The legacy ``prepare_nm_data`` helper supports released PCNtoolkit schemas.
When splitting data, it estimates imputation donors and outlier bounds from
training rows and applies them to held-out rows. Age-centile limits use original
age units. Source-localization fixes handle a supplied transform without
digitization and keep scaled anatomy available beside its morph target.

Cluster jobs are tracked by their submitted IDs, including accounting delays
and terminal failures. Rerunning a driver no longer passes stale subject data
as an unsupported argument. MRI QC excludes failed and missing participants,
and collected participant IDs retain leading zeros. FreeSurfer scripts preserve
paths containing spaces and report reconstruction failures to the scheduler.

Release metadata and checks
---------------------------

Package version information comes from ``meganorm/_version.py``. Repository
links and DOI information come from ``pyproject.toml``; the metadata helper
keeps README and software citation fields consistent. Tag builds check that
the release version and date match before publishing.

Tests cover the local API, real feature-extraction and model workflows, and
regressions in processing and cluster utilities. Python 3.12 is the supported
runtime for this release. Source localization still requires the relevant
FreeSurfer installation, license, anatomy, and acquisition-specific inputs.

Continuous integration builds documentation with warnings treated as errors.
Manual test-workflow runs include both slow HBR and IRASA integration checks.
Source distributions include the documentation, full test suite, and release
instructions.
