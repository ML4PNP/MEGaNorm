Local scientific API
====================

The canonical import path is ``meganorm.API`` (uppercase ``API``). The existing
``Config`` is re-exported. Compatibility aliases
are also available at the package root. Local extraction is sequential and
continues after participant failures by default; use ``on_error="raise"`` to
stop at the first processing failure. See :doc:`getting_started` for
multi-dataset examples, result interpretation, and rerun cleanup behavior.
See :doc:`configuration` for processing settings and JSON configuration files.

The local API and the SLURM runner share the processing configuration. Use the
local API for sequential cohort processing; use the cluster notebook for the
existing SLURM workflow. ``Pipeline`` does not submit cluster jobs.

.. automodule:: meganorm.API.datasets
   :members: Dataset

.. automodule:: meganorm.API.pipeline
   :members: Pipeline

.. automodule:: meganorm.API.modeling
   :members: NormativeModel

.. automodule:: meganorm.API.results
   :members: FeatureDataset, NormativeResults

Advanced dataset options
------------------------

``Dataset(options=...)`` accepts the existing discovery inputs:
``empty_room_task``, ``empty_room_path``, ``empty_room_ending``, ``surfaces_dir``,
``event_file_path``, ``event_file_task``, ``event_file_ending``,
``event_of_interest``, ``trans_path``, ``pos_path``, ``pos_file_ending``,
``annotation_path``, ``annotaion_task_name`` (legacy spelling),
``annotation_ending``, and ``layout_path``. Other keys are rejected.

These remain acquisition-specific compatibility settings; the first local
example needs none of them. Optional paths resolve against the dataset root.

Configured auxiliary inputs must include the complete option group and match
every participant; missing matches fail before processing. Omit unused groups.
Literal glob characters (``*``, ``?``, ``[``, ``]``) in paths are unsupported and
rejected to prevent incorrect participant associations.

Annotation files are looked up under ``annotation_path/<participant_id>/``
using an exact folder-name match, so ``sub-1`` cannot inherit ``sub-10`` files.

Response column names must be portable directory names, unique ignoring case.
Avoid path separators, control characters, Windows reserved names, and trailing
dots or spaces. ``normative_model.json`` is reserved for saved model metadata.
Scientific labels with spaces, brackets, and units remain supported.
