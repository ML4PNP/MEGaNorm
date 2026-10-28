Local scientific API
====================

The canonical import path is ``meganorm.API`` (uppercase ``API``). The existing
``Config`` is re-exported. Compatibility aliases
are also available at the package root. Local extraction is sequential and
continues after participant failures by default; use ``on_error="raise"`` to
stop at the first processing failure. See :doc:`getting_started` for
multi-dataset examples, result interpretation, and rerun cleanup behavior.
See :doc:`configuration` for processing settings and JSON configuration files,
and :doc:`erp_quickstart` for EEG ERP extraction.

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

ERP results and plotting
------------------------

``ERPResult``, ``load_erp_result`` and ``extract_recording`` are available from
``meganorm.API``. A cohort ``FeatureDataset`` provides ``load_erp`` and
``plot_erp`` for saved participant responses:

.. code:: python

   import matplotlib.pyplot as plt
   from meganorm.API import ERPResult, extract_recording, load_erp_result

   participant = result.load_erp("sub-001")
   fig, axes = participant.plot(channels=("Cz", "Pz"), show=False)
   plt.show()
   # Or: fig, axes = result.plot_erp("sub-001", channels=("Cz", "Pz"), show=True)

``Dataset.extract_erp("sub-001", config=config)`` returns an ``ERPResult``
with in-memory epochs using dataset discovery, session selection and metadata.
It writes no files. Participant IDs must be exact strings from discovery;
numeric IDs are not converted to folder names.

``ERPResult.save(directory, report=True, figures=False, overwrite=False)``
exports condition averages, stored features/QC and component metadata, with
optional HTML reports and figures. ``load_erp_result(directory)`` reopens
these outputs without recalculating measurements. Reloaded ``epochs`` is
``None`` because FIF saves averages, not trials. See :doc:`erp_quickstart`
for output flags, report locations and standalone examples.

.. autoclass:: meganorm.src.erp.ERPResult
   :members: plot, save

.. autofunction:: meganorm.API.load_erp_result

.. autofunction:: meganorm.API.extract_recording

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
