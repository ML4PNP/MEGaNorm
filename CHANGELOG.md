# Changelog

This file provides a summary of notable changes to **MEGaNorm** across released versions.

Detailed release notes for each version are available on the [GitHub Releases](https://github.com/ML4PNP/MEGaNorm/releases) page.

## [0.2.2] - 2026-10-07

- Added the local scientific API under `meganorm.API`: cohort discovery,
  multi-dataset feature extraction, explicit normative-model fitting, and
  participant-aligned prediction, deviation, and evaluation tables.
- Fix EEG fixed-threshold flatness limits, ICA method/seed handling, independent
  physiological-artifact passes, removal counts, and artifact annotation
  timing for cropped recordings.
- Retry failed IRASA aperiodic fits per channel with scaling while preserving
  original spectral units and the fit-quality threshold. Check the bandwidth
  needed for IRASA resampling before processing.
- Validate processing ranges, Welch windows, complete feature-family settings,
  and three-layer EEG source models when loading configurations.
- Make `prepare_nm_data` work with released PCNtoolkit schemas. Fit imputation
  donors, feature eligibility, and outlier bounds on training rows before
  applying them to held-out rows. Correct age centile units and limits.
- Normalize outlier group columns and the z-score method name for PCNtoolkit's
  extended interface while retaining the helper's existing calling convention.
- Handle MEG source localization without digitization when a transform is
  provided; retain scaled anatomy beside its morph target when both are needed.
- Monitor exact submitted SLURM job IDs, wait for accounting records, classify
  terminal failures, and keep the driver reusable after a run. Exclude all
  failed/missing MRI QC participants and retain leading zeros in collected IDs.
- Correct legacy train/validation splitting, diagnosis exclusion, MACE batch
  handling, site-zero centile statistics, and site-specific growth charts.
- Use consistent image names and tags for Docker build/pull/run/push. Retain
  Jupyter login tokens and expose the Makefile's notebook port on localhost.
- Add a configuration guide, configuration migration notes, release instructions,
  and API/real-workflow regression tests. Update tutorials and remove obsolete
  development notes and commented-out experiments.
- Current defaults use a 40 Hz low-pass filter, 300 Hz resampling, five-second
  epochs, and specparam with a fixed aperiodic slope. Oversampled temporal
  projection, reference-channel environmental ICA, and intermediate saves are
  disabled by default. GEDAI and its configuration fields are no longer present.
- Replace `parametrization_method` and old `fooof_*` settings with the current
  `psd_parametrization_*` fields. See the configuration guide for the full mapping.
- IRASA with the default 3–40 Hz fitting range needs a wider input passband;
  use `cutoffFreqHigh=80` with `resampling_rate=300`. Saving a complete
  configuration records defaults explicitly for reproducibility.

## [0.2.1] - 2026-09-16

- Added PSD visualization functionality.
- Improved session selection and example workflows.
- Improved support and examples for high-performance computing environments.
- Expanded and updated tutorials and documentation.
- Improved the documentation infrastructure, including automatic version retrieval.
- Updated the package build and publishing workflow.

See the [v0.2.1 release][0.2.1] for detailed release notes.

## [0.2.0] - 2026-07-10

- Added source-space MEG feature extraction and normative modeling.
- Expanded preprocessing with trial extraction/rejection, improved movement handling, cHPI processing, and environmental-noise removal.
- Added support for Artemis123 MEG data, configurable MEG device types, individual MRI scaling, and watershed BEM reconstruction.
- Improved parallel processing, including event-file/event-ID support and more robust job handling.
- Improved peak-related measurements, preprocessing argument/type handling, and fixed several processing and configuration issues.
- Updated installation requirements by removing MNE as a direct package dependency.

See the [v0.2.0 release][0.2.0] for detailed release notes.

## [0.1.1] - 2025-06-03

- Fixed configuration-key inconsistencies affecting processing workflows.
- Improved argument handling for both serial and parallel execution.
- Fixed minor processing bugs and documentation issues.
- Added an automated GitHub Actions workflow for publishing releases to PyPI.
- Improved documentation build requirements and project metadata.

See the [v0.1.1 release][0.1.1] for detailed release notes.

## [0.1.0] - 2025-05-16

Initial public release of MEGaNorm, providing workflows for large-scale EEG/MEG processing, extraction of functional imaging-derived phenotypes (f-IDPs), and normative modeling.

See the [v0.1.0 release][0.1.0] for detailed release notes.

[Unreleased]: https://github.com/ML4PNP/MEGaNorm/compare/v0.2.1...dev
[0.2.1]: https://github.com/ML4PNP/MEGaNorm/releases/tag/v0.2.1
[0.2.0]: https://github.com/ML4PNP/MEGaNorm/releases/tag/v0.2.0
[0.1.1]: https://github.com/ML4PNP/MEGaNorm/releases/tag/v0.1.1
[0.1.0]: https://github.com/ML4PNP/MEGaNorm/releases/tag/v0.1.0
