# Changelog

This file provides a summary of notable changes to **MEGaNorm** across released versions.

Detailed release notes for each version are available on the [GitHub Releases](https://github.com/ML4PNP/MEGaNorm/releases) page.

## [Unreleased]

Changes currently under development on the `dev` branch.

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
