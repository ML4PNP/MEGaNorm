Portable normative models
=========================

MEGaNorm can export a fitted normative model for local reuse without bundling
participant records or prediction outputs. The export is a single ZIP archive;
loading it restores a native PCNToolkit model. Remote publication is not yet
available through this interface.

Supported models
----------------

The initial adapter supports PCNToolkit 1.4.0 models with one or more responses, one
continuous covariate, one site batch effect, and standard input/output scalers:

* ``blr-default-site``: default, unwarped BLR with linear basis functions.
* ``hbr-shashb-linear-site``: SHASHb with a linear mean, a random site intercept,
  and the audited constant sigma, epsilon, and delta priors. Arbitrary SHASHb
  configurations are not interchangeable with this profile.

The HBR profile uses Normal(0, 1) for the mean slope; default random-intercept
priors; Normal(0, 1) with softplus(0, 1) for sigma; Normal(0, 0.5) for epsilon;
and Normal(1, 0.5) with softplus(0, 1, 0.6) for delta. It uses MCMC with the
``pymc`` sampler and ``auto`` initialization. Sampling counts may vary.

Unsupported configurations are rejected explicitly. Exporting already calibrated
or extended children is not enabled yet. These are sharing-adapter restrictions,
not limitations of PCNToolkit's modeling capabilities.

Multiple IDPs in one bundle
---------------------------

Export the fitted PCNToolkit ``NormativeModel`` containing all your IDPs. List
all response names in ``metadata["responses"]``, each with its unit and a unique
safe ``storage_key``. The names must match the fitted models exactly; this
interface exports the whole collection. Loading returns the same native model
container, with response order taken from the metadata.

For example, a model fitted on three responses can declare:

.. code-block:: python

   metadata["responses"] = [
       {"name": "alpha_power", "unit": "V^2", "storage_key": "alpha"},
       {"name": "beta_power", "unit": "V^2", "storage_key": "beta"},
       {"name": "theta_power", "unit": "V^2", "storage_key": "theta"},
   ]

Each IDP retains its own native regression model and output scaler. The input
scalers and site configuration are shared. All IDPs must satisfy the same
supported adapter profile. Predictions still use PCNToolkit's ``model.predict``;
no separate model container or prediction implementation is introduced.

For ``p`` HBR responses, the archive holds one shared ``normative_model.json``,
``p`` per-response ``regression_model.json`` files (including output scalers),
and ``p`` posterior-only ``idata.nc`` files. Thus there are ``2p + 1`` model
payload files, plus the manifest, README, and license. BLR has no posterior
files and uses ``p + 1`` model payload files. All are packaged in one
``model.zip``; the individual files do not need separate repository entries.

Export and load
---------------

Use the native fitted model, available as ``results.model`` after fitting through
MEGaNorm's modeling API. Supply publication metadata as a dictionary, leaving
out ``files``: the exporter builds the inventory and hashes itself.

.. code-block:: python

   from meganorm.sharing import export_model, validate_bundle, load_backend

   bundle = export_model(
       results.model,
       metadata=metadata,
       output_dir="shared_model_v1",
       metadata_reviewed=True,
   )
   report = validate_bundle(bundle.path, expected_sha256=bundle.bundle_sha256)
   model = load_backend(bundle.path, expected_sha256=bundle.bundle_sha256)

   # local_data is a PCNToolkit NormData object with matching feature definitions.
   model.predict(local_data)
   predictions = local_data["Yhat"]
   z_scores = local_data["Z"]

The destination must not already exist. ``bundle.path`` points to
``shared_model_v1/model.zip``. Keep ``bundle.bundle_sha256`` separately if you
want to verify the exact archive later. Hashes prove consistency with the
expected bytes, not publisher identity or scientific validity.

Loading disables ordinary prediction output writes. Predictions and z-scores
remain in your local data object. The native backend's ``transfer`` and ``extend``
methods retain PCNToolkit's own output behavior; this interface does not yet
provide a managed adaptation workflow.

Predict with local measurements
-------------------------------

Use ``SharedNormativeModel`` for validated prediction from a DataFrame or an
existing ``FeatureDataset``. For the single-response example below:

.. code-block:: python

   from meganorm.API import SharedNormativeModel

   model = SharedNormativeModel.load(
       "shared_model_v1/model.zip",
       expected_sha256=bundle.bundle_sha256,
   )
   results = model.predict(
       local_data,
       units={"age": "years", "roi": "V"},
   )
   predictions = results.predictions
   z_scores = results.z_scores

``local_data`` must contain a ``participant_id`` column or named index, the
model's covariates, site labels, and measurements for every bundled response.
Use ``participant_id="your_id_column"`` for another identifier name. Input rows
are preserved, IDs are normalized to strings, and result columns follow the
manifest's response order. Column order in the input does not matter; unrelated
columns are ignored. The input table is not modified.

Declare the units your values actually use, for every response and continuous
covariate. They must match ``model.manifest`` exactly. Convert values before
calling ``predict`` when necessary; the interface does not infer or convert
units. Feature definitions and preprocessing must also match the shared model.
A matching unit declaration alone cannot verify how data were processed.

The method returns both predictions and z-scores, so it requires all measured
responses. Missing/nonfinite values, duplicate IDs/columns and unknown sites
are rejected; no participants are dropped or measurements imputed. New-site
calibration is not exposed by this interface yet. In the audited PCNToolkit
1.4.0 build, HBR prediction on a single observation fails because native array
squeezing removes its observation dimension. This needs a PCNToolkit fix;
HBR cohort prediction is supported. BLR single-observation prediction works.

By default, results remain in memory and temporary working files are cleaned
up. To save participant-level outputs locally, pass a new directory:

.. code-block:: python

   results = model.predict(
       local_data,
       units={"age": "years", "roi": "V"},
       output_dir="local_predictions",
   )

This writes only ``predictions.csv`` and ``z_scores.csv``. Existing output
directories are rejected. These CSVs are local analysis results and are never
added to the shared model archive. Prediction does not refit the model or its
scalers, change the bundle, or contact a hosting service. The public
``capabilities`` property currently reports only ``("predict",)``.

Metadata example
----------------

The following example describes one synthetic HBR model using the
``hbr-shashb-linear-site`` profile above. Replace the values
with the model's actual feature definitions, training provenance, preprocessing
fingerprint, and reviewed publication information. Category codes must match
the fitted model's encoding.

.. code-block:: python

   metadata = {
       "schema_version": 1,
       "model_id": "publisher/reference-model",
       "version": "1.0.0",
       "adapter": {
           "name": "pcntoolkit", "format_version": 1,
           "pcntoolkit_version": "1.4.0", "profile": "hbr-shashb-linear-site",
       },
       "software": {"meganorm_version": "0.2.2", "python_version": "3.12.14"},
       "responses": [{"name": "roi", "unit": "V", "storage_key": "roi"}],
       "covariates": [{"name": "age", "kind": "continuous", "unit": "years"}],
       "batch_effects": [{
           "name": "site",
           "categories": [{"label": "A", "code": 0}, {"label": "B", "code": 1}],
       }],
       "preprocessing": {
           "modality": "MEG", "paradigm": "rest", "space": "sensor",
           "pipeline_version": "0.2.2",
           "config_sha256": "a" * 64,  # Replace with the real configuration hash.
           "description": "Describe the approved feature extraction settings.",
       },
       "metadata": {
           "title": "Reference model", "description": "Describe the model.",
           "authors": ["Author name"], "license": "CC-BY-4.0",
           "datasets": ["Dataset name"], "sites": ["A", "B"],
       },
       "lineage": {"operation": "fit", "parent": None},
       "capabilities": ["predict", "calibrate", "extend"],
   }

The software versions above are example provenance, not recommended versions.
``capture_runtime_versions()`` can capture the environment at training time.
Do not replace an existing model's training provenance with the environment
used later to export it. Native PCNToolkit objects do not reliably retain all
historical environment information, so this metadata remains the publisher's
responsibility. Compatibility is checked separately against the adapter's
audited producer/runtime pairs.

Privacy and file contents
-------------------------

``metadata_reviewed=True`` affirms that the public descriptions, feature names,
and site/category labels contain no information identifying individual
participants. Automated validation cannot establish this for arbitrary text.
Dataset and site names are permitted provenance.

Export reconstructs only approved model state. It excludes participant identifiers,
measurements, observation-indexed variables, prediction outputs, diagnostic groups,
and source paths. It retains fitted scaler statistics and necessary aggregate
site counts and covariate ranges. Unknown serialized fields and unexpected
posterior variables, coordinates, or attributes are rejected. Known native local
analysis state is excluded rather than copied.

The archive contains ``manifest.json``, a generated ``README.md``, a ``LICENSE``
declaration, shared and per-response model JSON, and (for HBR) a posterior-only
NetCDF3 file per response. The license file records the publisher's declaration; it is not an automatically generated
copy of the license's full legal text. No pickles or arbitrary attachments are
accepted. Portable archives are not native PCNToolkit save directories: use
``load_backend`` to reconstruct the native representation.

Limits and failures
-------------------

``BundleLimits`` accepts positive limits for archive bytes, extracted bytes,
JSON bytes, member count, and total posterior array bytes across all IDPs.
Defaults are 20 GiB, 80 GiB, 32 MiB, 100,000 members, and 512 MiB respectively. The initial format
also requires a canonical, uncompressed ZIP32 archive; ZIP64, multipart,
compressed, linked, and manually repacked containers are not accepted. The
container's ZIP32 limits apply even when a larger byte budget is configured.

Validation checks actual bytes as well as declared sizes and hashes. It rejects
extra members and ignored trailing data. It does not promise a formal privacy
guarantee against inference from learned parameters or a fixed total process
memory ceiling; array budgets bound stored array sizes, while reconstruction
requires additional working memory.

``validate_bundle`` returns a sanitized ``ValidationReport``. Export and load
raise sharing exceptions on failure. Error messages do not include rejected
values or source paths.
