Processing configuration
========================

``Config`` holds the settings for preprocessing, segmentation, source
localization, and feature extraction. Use the same configuration for recordings
that should be processed in the same way. Dataset paths and participant tables
belong in ``Dataset``; model settings belong in ``NormativeModel``.

Start with a few settings
-------------------------

You only need to supply the settings you want to change. All other fields use
the defaults for your installed MEGaNorm version.

.. code-block:: python

   from meganorm.API import Config

   config = Config(
       which_sensor="meg",
       which_layout="all",
       apply_source_localization=False,
       psd_parametrization_method="specparam",
       aperiodic_mode="fixed",
   )

This example extracts sensor-level MEG features averaged across channels.
Use ``which_layout=None`` to retain separate channel features, or ``"lobe"``
to use the sensor groups defined in a layout. For EEG, use
``which_sensor="eeg"`` and a layout that matches your electrode names.

Inspect all settings, including those you did not supply:

.. code-block:: python

   print(config.model_dump_json(indent=4))

Save and load a JSON file
-------------------------

Save the full configuration to record the settings used in an analysis:

.. code-block:: python

   config.save("config.json")
   loaded_config = Config.load("config.json")

The destination folder must already exist. Saving over an existing file raises
``FileExistsError``; use ``config.save("config.json", overwrite=True)`` when
you intend to replace it.

You can also write a short JSON file containing only your chosen settings:

.. code-block:: json

   {
       "which_sensor": "meg",
       "which_layout": "all",
       "apply_source_localization": false,
       "psd_parametrization_method": "specparam",
       "aperiodic_mode": "fixed",
       "segments_length": 5,
       "segments_overlap": 2
   }

Load it with ``Config.load("config.json")``. The file is a single JSON object,
without a surrounding ``config`` key. Use JSON ``true``, ``false``, and ``null``
in place of Python ``True``, ``False``, and ``None``. JSON does not allow
comments or trailing commas. Pairs such as frequency ranges are JSON arrays,
for example ``"muscle_activity_filter_freq": [110, 140]``.

Unknown setting names and invalid values raise a validation error when the
configuration is loaded. Read the field name in the error and compare it with
the :class:`~meganorm.utils.IO.Config` reference.

``same_environmental_noise_removal`` and ``specparam_res_save_path`` are
retained for compatibility and have no effect in the current pipeline.
Choose the individual environmental-noise settings and use ``save_psds``
to save spectral outputs.

Use the saved settings
----------------------

Pass the loaded configuration to the local pipeline:

.. code-block:: python

   from meganorm.API import Pipeline

   features = Pipeline(
       config=Config.load("config.json"),
       output_dir="./meganorm_results",
   ).run(dataset)

Here, ``dataset`` is the ``Dataset`` described in :doc:`getting_started`.
``Pipeline.run`` also accepts a list of datasets. Describe cohorts with
different processing needs separately and use a configuration suited to each.

For the SLURM workflow, pass a ``Config`` object as ``config_file`` to
``sbatch_feature_extraction_runner``. The runner saves the JSON file for the
participant jobs. Scheduler settings such as ``conda_env``, ``modules``,
``partition``, and ``memory`` belong in ``job_configs``. See the
`cluster feature-extraction notebook
<https://github.com/ML4PNP/MEGaNorm/blob/main/notebooks/feature_extraction_tutorial.ipynb>`_
for a complete example.

Settings to check for your recordings
-------------------------------------

The table shows commonly used settings and their v0.2.2 defaults. Defaults are
a starting point; check them against your recordings and analysis goals.

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - Setting
     - Default
     - Meaning
   * - ``which_sensor``
     - ``"meg"``
     - Sensor selection: ``"meg"``, ``"mag"``, ``"grad"``, ``"eeg"``, or ``"opm"``.
   * - ``which_layout``
     - ``"all"``
     - Average across channels, group by lobe, or retain channels with ``None``.
   * - ``which_meg_session``
     - ``0``
     - Select the first matching recording; increase the index for another session.
   * - ``cutoffFreqLow``, ``cutoffFreqHigh``
     - ``1.0``, ``40``
     - Bandpass limits in Hz when ``digital_filter=True``.
   * - ``resampling_rate``
     - ``300``
     - Target sampling rate in Hz. Keep the analysis band below its Nyquist frequency.
   * - ``apply_ica``
     - ``True``
     - Remove physiological artifacts using ICA.
   * - ``apply_oversampled_temporal_projection``
     - ``False``
     - Enable oversampled temporal projection when appropriate for the recording.
   * - ``bad_segment_removal_method``
     - ``"autoreject"``
     - Use AutoReject, ``"fixed_thr"`` amplitude thresholds, or ``None``.
   * - ``segments_tmin``, ``segments_tmax``
     - ``20``, ``-20``
     - Crop 20 seconds from each end of a resting-state recording before segmentation.
   * - ``segments_length``, ``segments_overlap``
     - ``5``, ``2``
     - Epoch length and overlap in seconds. Overlap must be shorter than the epoch.
   * - ``psd_method``
     - ``"welch"``
     - Welch or multitaper PSD estimation for the specparam workflow.
   * - ``psd_n_fft``, ``psd_n_per_seg``, ``psd_n_overlap``
     - ``2``, ``2``, ``1``
     - Welch FFT/window durations and overlap in seconds, converted to samples internally.
   * - ``psd_parametrization_method``
     - ``"specparam"``
     - Choose specparam or ``"irasa"`` to separate periodic and aperiodic components.
   * - ``psd_parametrization_freq_range_low``, ``psd_parametrization_freq_range_high``
     - ``3``, ``40``
     - Spectral fitting range in Hz.
   * - ``aperiodic_mode``
     - ``"fixed"``
     - Choose a fixed aperiodic slope or a ``"knee"`` model.
   * - ``min_r_squared``
     - ``0.9``
     - Minimum fit quality for spectral features. Inspect the fits before changing it.
   * - ``apply_source_localization``
     - ``False``
     - Enable source features when the required anatomy and head-model inputs are available.
   * - ``save_preprocessed_data``, ``save_segmented_data``, ``save_psds``
     - ``False``
     - Save intermediate processing files; large cohorts can need substantial disk space.

For ``"fixed_thr"`` rejection, the fields ending in ``_var_threshold`` are
peak-to-peak amplitude limits, despite their historical names. MEG thresholds
use T or T/m and EEG thresholds use V. Flatness thresholds must be lower than
the corresponding rejection thresholds. The default EEG limits are
``eeg_flat_threshold=1e-6`` and ``eeg_var_threshold=40e-6``.

For EEG recordings without a matching ECG or EOG channel, the automatic
pipeline uses ICLabel with extended Infomax to identify that artifact type.
This step uses Infomax even when ``ica_method`` selects another algorithm for
the other ICA steps; the choice is recorded in the participant log.

IRASA uses its own spectral estimates; the Welch settings above apply to
specparam. IRASA resamples the signal, so its fitting range also needs enough
bandwidth in the original recording. For its default 3–40 Hz fitting range
and ``irasa_hset=(1.05, 2.0, 0.05)``, retain a 1.5–80 Hz passband:

.. code-block:: python

   irasa_config = Config(
       psd_parametrization_method="irasa",
       cutoffFreqLow=1.0,
       cutoffFreqHigh=80,
       resampling_rate=300,
       aperiodic_mode="fixed",
   )

Changing only the method while keeping a 40 Hz low-pass filter cannot support
that IRASA range and raises a configuration error. Previously filtered inputs
must also retain the needed bandwidth. Review spectra and fit quality for both
methods, especially near filter edges.

Choose feature families
-----------------------

``feature_categories`` enables or disables families of extracted features.
To retain the other defaults, start with the existing dictionary:

.. code-block:: python

   settings = Config().feature_categories.copy()
   settings["Peak_Center"] = True
   settings["Peak_Power"] = True
   config = Config(feature_categories=settings)

Providing a new dictionary replaces the whole field; its entries are not
merged with the defaults. This also applies to ``freq_bands`` and
``individualized_band_ranges``.

Band ratios use named bands:

.. code-block:: python

   config = Config(
       power_band_ratios_list=[
           {"numerator": "Theta", "denominator": "Beta"},
           {"numerator": "Alpha", "denominator": "Beta"},
       ]
   )

A ratio is computed only when both bands are present in ``freq_bands``. The
default bands are Theta (3–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), and Gamma
(30–40 Hz). If you add a band, include it in the fitting range and set its
individualized range when requesting individualized features.

Source localization
-------------------

Set ``apply_source_localization=True`` only after supplying the anatomy and
FreeSurfer inputs for your workflow. Keep ``source_space_spacing`` and
``source_space_spacing_number`` consistent, such as ``"ico4"`` and ``4``.
For ``source_space_spacing="all"``, use ``source_space_spacing_number=None``.

For EEG source localization, supply three conductivity values in
``SL_conductivity``. A template analysis also needs
``apply_mri_template=True`` and ``freesurfer_template_path``. Dataset-specific
paths such as ``surfaces_dir`` and a precomputed transform belong in
``Dataset(options=...)``; see :doc:`meganorm.API`.

Update an older configuration
-----------------------------

Recheck saved settings when moving from an older MEGaNorm release. Several
defaults changed, and old spectral setting names are no longer accepted.
Saving a full configuration makes these choices explicit across releases.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Older name
     - Current name
   * - ``parametrization_method``
     - ``psd_parametrization_method``; use ``"specparam"`` for the former FOOOF workflow.
   * - ``fooof_freq_range_low``, ``fooof_freq_range_high``
     - ``psd_parametrization_freq_range_low``, ``psd_parametrization_freq_range_high``
   * - ``fooof_peak_width_limits``
     - ``psd_parametrization_peak_width_limits``
   * - ``fooof_min_peak_height``
     - ``psd_parametrization_min_peak_height``
   * - ``fooof_peak_threshold``
     - ``psd_parametrization_peak_threshold``
   * - ``make_new_watershed_bem``
     - ``force_new_watershed_bem``

GEDAI is absent from the current processing pipeline. Remove ``apply_gedai``,
``gedai_*``, and ``sensai_method`` entries from older configuration files.
For the full field reference, see :class:`~meganorm.utils.IO.Config`.

ERP analysis settings
---------------------

``Config.analysis_type`` defaults to ``"rest"`` for existing configurations.
For ERP analysis set ``analysis_type="erp"``, ``which_sensor="eeg"`` and
provide ``erp=ERPSettings(...)``. Source localization is not supported in this
mode. ``ERPSettings`` contains ``preprocessing``, named ``components`` and
``rejection_mode`` (``"threshold"`` by default, or ``"autoreject"``). Component windows must lie inside the epoch;
condition, channel and metric selections must be nonempty and unique.

Each ``Dataset`` can hold an ``ERPEventSpec`` as ``erp_events``. Annotation mode
maps logical condition names to exact annotation descriptions through
``annotation_map``. Stimulus mode uses ``event_id`` and ``stim_channel``.
``ERPSettings.validate_events(spec)`` checks that a dataset declares every
condition requested by the components. Both settings groups support JSON
serialization. Dataset discovery preserves the event specification in the
manifest without resolving or processing events.

These models are exported from ``meganorm.API``. Local ``Pipeline.run`` and the
single-participant CLI support ERP analysis. Cluster submission remains guarded.
Each local ERP invocation writes a new ``output_dir/erp/<run-id>/`` directory;
previous ERP runs and resting-state outputs are retained. ``FeatureDataset``
contains complete and partial participant rows, and ``paths["qc"]`` points to
the separate cohort QC table. Partial means some requested IDPs are unavailable;
participants with no usable IDPs are reported as failed. ``summary["succeeded"]``
counts complete participants, ``partial`` counts partial participants and
``collected`` is their sum. Existing ``on_error`` behavior applies to failures.

Use embedded EEG metadata or supported local channel/montage sidecars.
Ambiguous sidecars fail preflight. Sidecar channel types and bad-channel flags
are explicit overrides; an embedded montage is preserved. Montage CSV coordinates
must be in metres in the head frame. No montage is guessed.
Legacy dataset options and cohort raw/epoch saving are not supported for ERP;
use the standalone API when you need access to epochs and evoked responses.
Only nested ERP preprocessing settings control ERP cleaning; resting-state
filtering, segmentation and spectral settings do not apply.

For standalone use, ``ERPSettings.to_runtime()`` returns an ``ERPConfig`` and
component mapping. Pass ``event_spec=dataset.erp_events`` to the standalone ERP
API; explicit events and annotation maps remain supported. See the repository's
``erp_quickstart.md`` for local cohort and CLI examples.

ERP ICA strategy
~~~~~~~~~~~~~~~~

Set ``erp.preprocessing.ica_mode`` to one of:

* ``"auto"`` (default): use usable EOG/ECG channels; otherwise use ICLabel.
  Missing ICLabel prerequisites fail the participant; cleaning is never silently
  skipped.
* ``"physiological"``: require at least one usable EOG/ECG channel. As in the
  existing correlation helper, select the strongest correlated component per
  auxiliary channel when its absolute Pearson correlation reaches the threshold.
* ``"iclabel"``: use ICLabel even if auxiliary channels are available.
* ``"none"``: explicitly disable ICA; referencing, filtering and epoch rejection
  still run.

``ica_corr_threshold`` and ``ica_label_threshold`` independently control
correlation and winning-label probability (both in ``(0, 1]``). An unset value
inherits the legacy ``ica_threshold`` (default ``0.8``). Thus existing
configurations retain their thresholds. For example::

    ERPPreprocessing(
        ica_mode="auto",
        ica_corr_threshold=0.8,
        ica_label_threshold=0.9,
    )

Legacy ``apply_ica=False`` disables ``auto``; combining it with an explicit
``physiological`` or ``iclabel`` request is an error. ``ica_mode="none"`` also
works with the default ``apply_ica=True``.

Auxiliary channels marked bad, flat, nonfinite, or with insufficient samples
outside BAD annotations are excluded from strategy selection. Correlations use
those same retained samples. ICA is fitted on a separately filtered EEG copy,
with the same channels and reference as the data being cleaned. The fitted EEG
rank must be at least two; explicit component counts cannot exceed that rank.
The default component count uses that rank. Variance fractions remain supported.

ICLabel requires ``reference="average"``, ``ica_l_freq >= 1`` Hz and finite,
nonzero sensor locations for every usable EEG channel. Its training used a
1–100 Hz filter. Other fit filter settings, including the current 1–35 Hz
default, generate a QC warning because classification performance may differ.
No filter is silently changed to meet the training conditions.

Each participant's ``preprocessing.json`` records requested/effective modes,
selection reason, resolved thresholds, auxiliary quality, fit rank/filter,
component count, exclusions and selected correlation scores or classifier
labels/probabilities. It also records supplied bad-channel markings and that
automatic continuous bad-channel detection was not performed. A successful ICA
fit with no components meeting the threshold is recorded with an empty exclusion
list; this does not establish that the recording is artifact-free.

ERP epoch rejection
~~~~~~~~~~~~~~~~~~~

``erp.rejection_mode="threshold"`` remains the default and uses the existing
MNE peak-to-peak/flat rejection. Set ``"autoreject"`` to learn EEG channel
thresholds and cross-validate interpolation/rejection choices on each
participant's pooled conditions. The existing event-locked epochs, baseline,
metadata and original trial indices are preserved. Only retained trials are
averaged for IDPs.

AutoReject replaces ``erp.preprocessing.reject``; it does not first discard
trials using that fixed amplitude cutoff. Recording-edge loss, BAD annotation
rejection and any ``erp.preprocessing.flat`` threshold still apply before
fitting. Fewer than ``max(5, cv)`` retained epochs, fewer than four usable EEG
channels, nonfinite EEG samples or missing/duplicate sensor locations cause an
error. Explicit candidates that interpolate all usable channels or have no
valid consensus/interpolation combination also fail. These checks are minimum
computational requirements; real-data validation remains necessary. There is no
fallback to another method when fitting fails.

Optional ``erp.autoreject`` settings (also exported as ``ERPAutoRejectSettings``):

* ``cv``: cross-validation folds, integer >= 2; default 5.
* ``n_interpolate``: unique nonnegative integer candidates. Each must be below
  the number of usable EEG channels. By default, use 1, 2 and 4, omitting values
  at or above that channel count. Zero allows rejection without interpolation.
* ``consensus``: unique fractions in (0, 1]; default 0.2, 0.4, 0.6, 0.8, 1.0.

``erp.preprocessing.random_state`` supplies the seed. Fitting uses one worker.
Only unmarked EEG channels participate in fitting/interpolation; pre-existing
bad channels remain marked. AutoReject options supplied with threshold mode
are rejected as conflicting settings. The standalone ``ERPConfig`` accepts the
same ``rejection_mode`` and ``autoreject`` dictionary.

``preprocessing.json`` contains ``epoch_rejection`` with resolved candidates,
selected values, learned channel thresholds and per-trial decisions. Each trial
entry identifies its original event index, sample, code, retained status,
detected bad channels and channels interpolated in retained data. Trial entries
cover epochs reaching AutoReject after annotation/edge/flat exclusion;
``drop_log`` retains the complete rejection history. The report's
``n_input_epochs`` counts epochs entering the rejection helper, after
annotation/edge exclusion; ``n_prefit_flat_rejected`` counts subsequent flatness
losses. Interpolation counts describe retained trials only.

The IDP QC table adds ``n_autoreject_rejected`` and ``n_interpolated_trials`` per
condition. Edge, annotation, amplitude/flat, AutoReject and other loss categories
sum to ``n_rejected``; interpolated trials remain in ``n_trials``. If AutoReject
drops every epoch, IDPs are NaN with ``no_retained_trials`` and the participant
worker publishes a failed outcome with QC, as with threshold all-rejected data.
