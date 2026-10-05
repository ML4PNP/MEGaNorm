Local scientific API
====================

The canonical import path is ``meganorm.API`` (uppercase ``API``). The existing
``Config`` is re-exported without changed defaults. Compatibility aliases are
also available at the package root.

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
