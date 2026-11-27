# MEGaNorm Model Accessibility and Sharing — Staged Development Plan

**Date:** 13 November 2026

**Status (27 November 2026):** Stages 0–4 complete for the audited HBR/SHASHb and BLR profiles. Stage 5 local implementation and review are complete; its live-service gate is pending. Stages 6–9 remain unimplemented. No live model publication performed.

**Source baseline:** User-supplied `MEGaNorm(20261113-144803).zip`, described as the dev branch after merging ERP developments.

**Goal:** Let researchers publish, fetch, apply, calibrate, and extend normative models without publishing participant-level data or derivatives.

**Architecture:** A portable, validated model bundle is the core artifact. A PCNToolkit adapter supplies modeling operations; Hugging Face and Zenodo adapters handle distribution and archival publication through one resumable workflow.

**Tech stack:** Existing MEGaNorm Python environment, PCNToolkit, its supported data serialization, SHA-256, optional Hugging Face client and Zenodo HTTP client, pytest, and Sphinx.

**Spec:** Sections 1–7 below are the self-contained design baseline implemented by stages 0–9. Proposed interfaces and defaults remain subject to the evidence gathered in stage 0.

**Execution guidance:** Implement and review one stage at a time. Use the executing-plans workflow when resuming this plan; use subagent-driven-development only if explicitly selected. Each stage has its own tests and completion evidence. Do not mark subsequent stages complete on the strength of mocks alone. Suggested repository location: `docs/superpowers/plans/2026-11-13-model-sharing.md`.

## 1. Agreed requirements

| ID | Requirement |
| --- | --- |
| R1 | Publish fitted models, never participant identifiers or individual data records. Fitted scaler statistics and necessary aggregate model summaries are permitted under the clarified privacy policy. |
| R2 | Never publish participant-level derivatives, including predictions, z-scores, residuals, or per-observation likelihoods. Generate outputs locally after fetching. |
| R3 | Dataset and site names may be included as provenance. Necessary feature, covariate, batch, and preprocessing definitions must accompany the model. |
| R4 | Support calibration to new sites and model extension using PCNToolkit functionality, without reimplementing its statistical methods. |
| R5 | Provide automated model push and fetch using Hugging Face plus Zenodo. |
| R6 | Any user may publish models under an account or organization they are authorized to write to. |
| R7 | Hosting services enforce authorization. Other users must not be able to modify a publisher's models without permission. |
| R8 | MEGaNorm publication creates versioned releases and refuses to replace an existing release with different content. |
| R9 | Adapted models preserve parent provenance and do not alter the fetched parent. Publishing an adaptation is a separate explicit action. |
| R10 | Keep the workflow simple for researchers; advanced implementation detail belongs in developer documentation. |
| R11 | Develop, test, and review the feature stage by stage. |

The term “derived data” in the discussion is interpreted as **derived models**, consistent with the explicit prohibition on sharing measurements and participant-level derivatives.

## 2. Scope and architecture decisions

### Chosen approach and alternatives

| Approach | Strength | Limitation | Decision |
| --- | --- | --- | --- |
| Hugging Face only | Model discovery, versioned downloads, Python integration | Does not supply the chosen independent archival copy | Not selected |
| Zenodo only | Archival records and version-specific citation | Requires more custom discovery and distribution work | Not selected |
| Hugging Face + Zenodo, portable bundle | Convenient access plus an archived, citable copy | Needs reconciliation across two services | Selected |

Hugging Face is the primary distribution service; Zenodo archives the identical model bundle for each complete release. Local export, loading, prediction, and adaptation must work without either service. Network clients are optional dependencies, imported only when used.

First-release scope includes local export/import, schema validation, prediction and z-score generation, PCNToolkit transfer and extension for validated model configurations, resumable two-service publication, exact-version fetch, offline reuse, and user documentation.

Not in the initial scope: hosted inference, raw data upload, central approval of every community model, federated training, cross-modality conversion, arbitrary model classes, a custom web portal, or a formal differential-privacy guarantee. A CLI is a later convenience; the Python API is the initial interface.

### Evidence from the attached package

- `meganorm/API/modeling.py` exposes fitting and writes analysis metadata and participant-level outputs. It rejects unseen test batch labels.
- `meganorm/src/normative_modeling.py` delegates fitting and persistence to PCNToolkit's `NormativeModel`.
- Its prior-predictive helper reads JSON model configuration and NetCDF inference data. This is evidence of existing serialization use, not proof that all saved files are safe to share.
- `meganorm/API/results.py` contains `NormativeResults`; its training/test objects and output paths must not become publication payloads.
- `requirements.txt` specifies `pcntoolkit>=1.3.0`. This broad dependency declaration is not a promise that every future version has a compatible bundle format.
- The relevant public wrapper has no dedicated fetch/load/predict interface yet.

These were the initial source-inspection observations. Stage 0 subsequently ran the baseline suite and synthetic round-trip/adaptation experiments; see `../../development/sharing/stage0-audit.md` and its evidence files for the tested scope.

## 3. Privacy boundary and bundle contract

### Export policy

Build a new bundle in a clean staging directory using a schema-defined allowlist. Never recursively upload an analysis directory, copy a whole model directory and remove selected files, or trust file extensions as evidence of safety.

| Content | Policy |
| --- | --- |
| Fitted coefficients, posterior parameter draws, approved model-level state | Include only through an audited adapter |
| Necessary scaler parameters, transforms, basis definitions, category encodings | Include as model state, including fitted extrema; inspect for stored participant records |
| Feature names, units, regions/channels, covariate definitions, batch labels | Include validated definitions; no participant-level axes or mappings |
| Dataset/site names, authors, methods, citations, license | Include intentionally supplied provenance |
| Participant IDs, dates of birth, measurement rows, source paths | Exclude |
| Training/test/calibration arrays; observed or constant data containing rows | Exclude |
| Predictions, z-scores, residuals, individual log likelihoods, predictive draws by participant | Exclude |
| Logs, reports, raw/processed EEG or MEG, participant tables | Exclude |
| Arbitrary attachments, executable code, pickles, object arrays | Reject in the initial portable format |
| Model-generated synthetic samples used during extension | Keep local; do not automatically publish |
| Necessary aggregate synthesis state (site counts and covariate ranges) | Retain for native extension, provided metadata does not identify individual participants |
| Aggregate performance tables and arbitrary cohort tables | Exclude from automatic export initially; narrative methodology and paper citations suffice |

Inspect NetCDF groups, variables, dimensions, coordinates, and attributes as well as JSON keys and values. A posterior file can contain observation-indexed data even when its filename looks harmless. Allowlisted parameter draws must not contain participant-indexed latent variables. Names, attributes, category labels, and free-text metadata also need checks; a participant ID used as a batch label is not safe merely because batch labels are normally allowed.

Automated checks cannot establish that arbitrary free text contains no personal information. Require intentional publisher-supplied metadata and an explicit reviewed-metadata declaration before push, with a dry-run inventory available. No interactive prompt is required on every retry. Never include rejected sensitive values in validation logs.

If prediction or adaptation requires prohibited state, block export for that model/version and report the incompatibility. Do not weaken the privacy rule to make loading work. Necessary aggregate scaler statistics (including minima/maxima), fitted parameters, and model-level synthesis summaries are permitted learned model state. The governing boundary is information traceable to individual participants; do not treat every data-derived scalar as prohibited. This is not a formal protection against membership or reconstruction inference.

### Proposed schema version 1

| Bundle member | Purpose |
| --- | --- |
| `manifest.json` | Schema version, model identity/version, adapter identity, software compatibility, response/covariate/batch definitions, model capabilities, preprocessing contract, lineage, file hashes and sizes |
| `model/` | Only adapter-produced, audited state in supported data-only formats |
| `README.md` | Model card produced from validated publication metadata |
| `LICENSE` | Explicit publisher-selected model license |

The manifest lists every other file's SHA-256 and byte size. It does not hash itself. Hash the canonical complete archive externally to identify the bundle, avoiding recursive hash dependencies. Use deterministic member ordering, normalized archive timestamps, and canonical JSON where practical. Identical staged bytes are uploaded to both services.

The bundle does not contain a Hugging Face commit hash that would depend on its own upload. A separate release descriptor records bundle digest, repository, exact commit, Zenodo record/version DOI, and release status. Dataset/site provenance is optional; model input definitions and operational batch mappings are mandatory where required.

Keep schema version, adapter format version, model release version, and software version separate. Only tested schema/adapter combinations load. Unknown combinations fail with actionable messages; they do not trigger a best-effort deserialization.

## 4. Application, calibration, and extension

Use a dedicated loaded-model object, separate from the existing fit wrapper. It validates inputs before constructing PCNToolkit data objects.

- Match responses and covariates by declared names; reorder explicitly and reject duplicates or omissions. Never infer meaning from column position alone.
- Validate units, transforms, category encodings, batch definitions, and declared feature-extraction settings. Missing compatibility metadata must remain visibly unverified, never silently “compatible.”
- Require observed response values for z-scores. Covariate-only predictions are offered only if validated in the selected PCNToolkit adapter.
- Reject nonfinite selected inputs by default with a local error summary. Preserve participant ordering and identifiers in local outputs only.
- Reject unknown sites in direct prediction and direct users to calibration. Never map an unknown site to an existing one automatically.
- Warn about declared applicability-range violations without silently changing inputs; scientific applicability beyond that range is not guaranteed.
- Call PCNToolkit transfer for new-site calibration and expose extension separately. Stage 0 must verify exact semantics, supported estimators, parameters, retained old-site capabilities, and serialization requirements for the chosen version.
- Use distinct calibration and evaluation datasets. Reject overlapping identifiers where available; document that changed identifiers cannot be automatically recognized as the same person.
- Adapt into a new local directory outside the cache. Preserve the parent files and parent identifier/digest. Do not assume extension preserves old-site behavior: test and document the backend's actual behavior.
- Record adaptation method and approved settings, parent release, updated batch support, and software versions. Calibration data and participant-level diagnostics remain local.

Do not promise support for every HBR likelihood, BLR configuration, or custom estimator in the first release. Stage 0 selects a tested matrix, starting with the MEGaNorm HBR/SHASH use case and evaluating BLR separately. The milestone is not complete until at least one real supported model passes export → reload → transfer → extension → re-export.

## 5. Publication, ownership, and reproducibility

### Authorization and provenance

Publishers supply their own service credentials through supported credential stores or environment variables. No tokens in model metadata, logs, exceptions, URLs, notebooks, or source-controlled configuration. Never forward a credential to another host or redirected download endpoint.

Hugging Face and Zenodo enforce write access server-side. Preflight checks improve error messages but do not replace those controls. ML4PNP repositories are ordinary protected organizational destinations; users publish elsewhere unless granted access. A model card claiming “official” is not authoritative: provenance is based on the actual source namespace and a documented curated list, if one is maintained.

Use a new model version for changed payloads. Existing version + same digest is an idempotent success; existing version + different digest is a conflict. Do not expose `overwrite=True` for released payloads. Platform owners may have capabilities outside MEGaNorm to change refs or metadata; the client cannot promise universal immutability. Reproducibility relies on exact revision references, archived copies, and verified digests.

### Proposed automated push transaction

The two services do not form an atomic transaction. Implement a resumable state machine with a secret-free local journal and remote reconciliation:

1. Validate metadata, ownership intent, licenses, model state, and destination/version conflicts; build the clean bundle. Dry-run stops here without network mutation.
2. Create or recover a Zenodo draft and reserve the version DOI; save the returned record identity immediately.
3. Upload the bundle to a unique version path on Hugging Face using optimistic concurrency where supported. Record the returned exact commit; never force-update an existing release tag.
4. Upload the identical archive to the Zenodo draft. Record the Hugging Face revision and bundle digest in approved publication metadata, then publish the draft and verify the remote publication state.
5. Publish a completion descriptor on Hugging Face referencing the payload commit, bundle digest, and version DOI. This is a metadata commit, not a change to the payload. Record both commits explicitly.
6. Return a complete release receipt only after both services and the descriptor agree.

Do not mark an uploaded payload as a completed release before archival publication succeeds. A public staging upload can still be visible on its host; all bytes must already be publication-safe before the first upload. No automatic deletion or rollback of a published archive.

The journal records destination, version, digest, draft/record IDs, commits, phase, and error category. A timeout after a successful request must trigger status reconciliation before retrying a mutation. If a lost journal leaves ambiguous remote records, stop with a recoverable conflict rather than create duplicates. Two simultaneous pushes for one destination/version must not both claim success for different payloads.

### Fetch behavior

Accept a fully qualified Hugging Face model reference plus explicit version, or a Zenodo version DOI. Resolve once to an exact release and return a local lock receipt with identifiers and digest. Do not silently track a moving `main` branch for reproducible analyses. Convenience “latest” selection may be added later; if provided, always report and lock the resolved version.

Fetch verifies the descriptor, archive digest, member hashes, schema, adapter compatibility, safe extraction, and approved state before loading. Hashes show integrity, not scientific validity or safety by themselves. Reject path traversal, links, unexpected files, decompression bombs, oversized declarations, and executable payloads. Define configurable download/extraction budgets with documented defaults during stage 1.

Fallback to Zenodo only for the exact recorded bundle and only for availability failures. Never silently bypass access denials, digest mismatches, or validation failures. Preserve access restrictions across both services; the first release supports public model publication only and refuses restricted drafts rather than accidentally making them public elsewhere.

Cache by bundle digest/exact revision, use atomic writes and locks, and never modify cached parent files during prediction or adaptation. Offline fetch succeeds only for a fully validated cached artifact; an incomplete cache produces a clear error.

## 6. Proposed interfaces and file ownership

These are proposed MEGaNorm APIs, not claims about existing functions. Stage 0 finalizes PCNToolkit calls; subsequent stages maintain these public contracts unless a documented review changes them.

```python
from meganorm.API import ModelHub, SharedNormativeModel

# All metadata is deliberately supplied and reviewed by the publisher.
bundle = ModelHub.export(fitted_model, metadata=metadata, output_dir="model_bundle")
report = ModelHub.validate(bundle.path)
release = ModelHub.push(bundle.path, repo_id="publisher/model-name",
                        version="1.0.0", zenodo_metadata=archive_metadata,
                        metadata_reviewed=True)

fetched = ModelHub.fetch("publisher/model-name", version="1.0.0")
model = SharedNormativeModel.load(fetched.path)
results = model.predict(local_data, output_dir="local_predictions")
adapted = model.calibrate(calibration_data, output_dir="local_adapted_model")
extended = model.extend(extension_data, output_dir="local_extended_model")
# Adapted/extended models must go through export and validation before push.
```

Additional contracts: `push(..., dry_run=True) -> PublicationPreview`; `push(..., resume=True) -> PublishedRelease`; `fetch(..., offline=True) -> FetchedModel`. Credentials are resolved by backend configuration, not included in these objects. Local `export` and `load` make no network requests.

| Type | Required fields/behavior |
| --- | --- |
| `ModelBundle` | `path`, `model_id`, `version`, `schema_version`, `bundle_sha256` |
| `ValidationReport` | `valid`, structured issue codes, sanitized descriptions; no participant values |
| `PublicationPreview` | destination/version, file inventory, digest, metadata readiness; no remote mutation |
| `PublishedRelease` | repo ID, payload commit, descriptor commit, Zenodo record/version DOI, digest, completed state |
| `FetchedModel` | verified local path, exact source references, digest, local lock receipt |
| `SharedNormativeModel` | predict, calibrate, extend; explicit validated capabilities |
| `PredictionResults` | local predictions/z-scores and ordered participant index; deliberately not exportable as a model |

`export` accepts an audited fitted backend model or the backend state of a loaded/adapted model. It does not serialize a `NormativeResults` container. Provide a documented convenience path from existing fit results to their underlying model without copying data attributes.

| Proposed file | Responsibility |
| --- | --- |
| `meganorm/sharing/schema.py` | Bundle and release schemas, typed result objects, validated metadata |
| `meganorm/sharing/errors.py` | Stable privacy, compatibility, integrity, authorization, version-conflict, and incomplete-publication errors |
| `meganorm/sharing/bundle.py` | Clean export, validation, deterministic packing and safe unpacking |
| `meganorm/sharing/pcn_adapter.py` | Version-specific model state extraction/restoration and capability checks |
| `meganorm/sharing/huggingface.py` | Hugging Face transport and permission/conflict handling |
| `meganorm/sharing/zenodo.py` | Zenodo draft/version/upload/publish/read operations |
| `meganorm/sharing/publication.py` | Resumable transaction and release descriptors |
| `meganorm/sharing/cache.py` | Fetch orchestration, exact-version cache, locks, offline/fallback behavior |
| `meganorm/API/sharing.py` | Thin public `ModelHub` facade |
| `meganorm/API/pretrained.py` | Loaded-model prediction, calibration, and extension |
| `meganorm/API/__init__.py` | Lazy public exports consistent with existing import conventions |
| `tests/sharing/` | Unit, privacy, transport, publication, cache, and backend integration tests |
| `tests/API/test_pretrained.py` | Public workflow and participant-alignment tests |
| `docs/source/model_sharing.rst` | Researcher-facing usage, privacy boundary, accounts, citations, troubleshooting |
| `docs/source/model_sharing_development.rst` | Format, adapter matrix, publication state machine, maintenance |
| `docs/notebooks/model_sharing_tutorial.ipynb` | Synthetic end-to-end example, cleared outputs and no credentials |

Modify `setup.py` for optional dependencies following its existing packaging conventions; inspect `MANIFEST.in`, Sphinx index, CI configuration, and existing metadata automation before editing. Do not manually duplicate release metadata already managed elsewhere.

## 7. Stage rules and review focus

For each stage: write the meaningful tests first, observe their initial failure, implement the smallest complete deliverable, run focused checks, review the diff, and commit only the stage's work. Use an isolated feature branch/worktree when implementation starts. Keep existing user changes intact.

A completion record must state commit, exact dependency versions, commands run, outcomes, remaining limitations, and next-stage readiness. Do not count mocked transport success as evidence of live service compatibility or serialization mocks as evidence of privacy.

Five review priorities, with owning stages:

1. Hidden observation coordinates, participant-level posterior state, or identifiers in metadata → stages 0–2.
2. Correct shape but wrong feature units, ordering, category encodings, or site → stages 3–4.
3. Remote success followed by local timeout, lost journal, or concurrent publisher → stages 5–7.
4. Tampered archive, unsafe paths, interrupted cache, or unauthorized fallback → stages 2 and 8.
5. New-site calibration modifies parent state or leaks calibration rows on re-export → stage 4.

## 8. Stage-by-stage implementation

### Stage 0 — Audit PCNToolkit and freeze the supported contract

**Audit status (13 November 2026):** Completed for the bounded HBR/SHASHb and BLR configurations recorded in `../../development/sharing/stage0-audit.md`. The scripts are diagnostic tools, not production sharing APIs. User-approved clarification: scaler statistics and necessary aggregate model state are allowed; participant-traceable information remains excluded.

**Purpose:** Resolve feasibility and privacy before building a publisher.

**Depends on:** Nothing.

**Files:** Create `docs/source/model_sharing_development.rst` and local synthetic audit fixtures; later commit reusable fixtures under `tests/sharing/conftest.py`.

- [x] Record archive checksum, implementation baseline commit, installed dependency versions, and current test status.
- [x] Inspect the actual installed PCNToolkit source for save/load, prediction, transfer, extension, scalers, category mappings, posterior groups, and incidental output writing. Online documentation alone is insufficient.
- [x] Fit a small synthetic HBR/SHASH model with two known sites and prepare a third-site calibration/evaluation split using `random_state=2026` wherever supported.
- [x] Inventory every saved file and internal group; insert distinctive synthetic identifiers and measurements to trace unintended retention. Record a recursive field/type inventory. Exhaustive validated export allowlists are implemented in Stages 1–2; the diagnostic reconstruction is not such an allowlist.
- [x] Prove local reload, prediction, transfer, and extension using clean state. Compare to the native unsanitized model in the same environment.
- [x] Evaluate BLR independently. Publish an explicit matrix of estimator/likelihood/backend-version capabilities, including unsupported cases.
- [x] Record stochastic comparison tolerances and test fixtures. Propose exact tested dependency bounds, archive limits, metadata schema, and the license-selection policy.

**Tests/evidence:** A reproducible audit script and report with state inventory, before/after comparisons, and backend method signatures. Synthetic data is used only locally. Do not infer privacy from a text search alone.

**Exit gate:** At least one supported HBR/SHASH configuration can predict, transfer, and extend without prohibited state. If not, report the backend blocker and revise the design before stage 1. Record any blocked BLR support rather than silently claiming it.

### Stage 1 — Define schemas, types, and metadata validation

**Purpose:** Establish a precise format independent of hosting.

**Depends on:** Stage 0.

**Files:** `meganorm/sharing/{__init__,schema,errors}.py`; `tests/sharing/test_schema.py`.

**Interfaces:** Define all types in section 6 and `validate_manifest(document: dict) -> ValidationReport`.

- [x] Write tests for required fields, duplicate features, invalid versions, unknown schema/adapter, reserved path names, prohibited metadata fields, and parent references.
- [x] Run `pytest tests/sharing/test_schema.py -q`; confirm failures reflect missing behavior.
- [x] Implement strict schema validation, canonical field ordering rules, digest definitions, and error codes. Keep publication metadata separate from local analysis metadata.
- [x] Run the focused tests; verify every rejected case has a sanitized actionable error.
- [x] Review and commit the schema and fixtures together.

**Exit gate:** Valid fixtures round-trip; invalid or ambiguous manifests are rejected. No network access is needed.

**Completion evidence (15 November 2026):** See `../../development/sharing/stage1-schema.md` and the progress ledger. The initial profiles deliberately cover only the Stage 0 shapes. Metadata validation does not certify payload privacy; actual export and inspection remain Stage 2.

**Version-policy follow-up (16 November):** Software provenance is independent of compatibility. Stage 2 must call `validate_compatibility` with the actual runtime version before export/loading. Expand `ADAPTER_COMPATIBILITY` only after validating each producer/consumer pair. See the Stage 1 schema notes.

### Stage 2 — Build clean export and local loading

**Purpose:** Produce a reusable bundle with no participant-level content.

**Depends on:** Stages 0–1.

**Files:** `bundle.py`, `pcn_adapter.py`; `tests/sharing/test_bundle.py`, `test_privacy.py`, `test_pcn_roundtrip.py`.

**Interfaces:** `export_model(model, *, metadata, output_dir) -> ModelBundle`; `validate_bundle(path) -> ValidationReport`; `load_backend(path) -> fitted backend model`.

- [x] Write tests that reject observation-indexed state, unknown groups/attributes, unsafe object serialization, links, path traversal, unexpected files, and extraction-budget violations.
- [x] Add fixtures containing identifiers in coordinates, paths, attributes, batch labels, and free text. Assert rejection or exclusion according to the contract and no sensitive values in errors.
- [x] Run the focused suite and confirm intended failures.
- [x] Implement reconstruction of only allowlisted model state into a clean directory, deterministic archive creation, and validated restoration. Unknown state fails closed.
- [x] Compare native and restored predictions using stage-0 tolerances. Assert input model files remain unchanged and repeated packing of the same staged state produces the same digest.
- [x] Run `pytest tests/sharing/test_bundle.py tests/sharing/test_privacy.py tests/sharing/test_pcn_roundtrip.py -q`, review, and commit.

**Exit gate:** Real supported-model round trip passes; prohibited content is absent from all bundle members, not merely the top-level directory. No upload code is required yet.

### Stage 3 — Add local pretrained prediction API

**Purpose:** Make a shared model useful before adding remote storage.

**Depends on:** Stage 2.

**Files:** `meganorm/API/pretrained.py`, API exports; `tests/API/test_pretrained.py`.

**Interfaces:** `SharedNormativeModel.load(path)` and `.predict(data, *, output_dir=None) -> PredictionResults`.

- [x] Write tests for reordered columns, missing/duplicate responses, unknown sites/categories, incompatible units, nonfinite values, participant ordering, and measurements required for z-scores.
- [x] Confirm failure, then implement input validation and PCNToolkit invocation with local output paths.
- [x] Verify predictions and z-scores against direct PCNToolkit calls; verify calls neither refit nor mutate scalers or the bundle.
- [x] Run `pytest tests/API/test_pretrained.py tests/sharing/test_pcn_roundtrip.py -q` and existing modeling API tests, review, and commit.

**Exit gate:** A new process loads the bundle and reproduces outputs from local inputs without network access or access to training data.

### Stage 4 — Add new-site calibration and model extension

**Purpose:** Complete scientific reuse through native PCNToolkit operations.

**Depends on:** Stage 3.

**Files:** `pretrained.py`, `pcn_adapter.py`, lineage schema; `tests/sharing/test_adaptation.py`.

**Interfaces:** `.calibrate(data, *, output_dir, **validated_backend_options) -> SharedNormativeModel`; `.extend(...) -> SharedNormativeModel`.

- [x] Write tests for unknown-site direct prediction rejection, supported transfer, separate extension, unsupported backend options, overlap detection, category collisions, and parent-file immutability.
- [x] Confirm failures and implement the validated backend mapping from stage 0. Do not alias transfer and extension as if they were interchangeable.
- [x] Compare adapted/extended predictions with direct PCNToolkit results on held-out synthetic data. Assess declared old-site behavior and record capabilities accurately.
- [x] Export and reload both outputs; assert calibration rows, generated synthetic samples, predictions, and z-scores are excluded. Assert parent provenance remains intact.
- [x] Run `pytest tests/sharing/test_adaptation.py tests/sharing/test_privacy.py tests/API/test_pretrained.py -q`, review, and commit.

**Exit gate:** Fetch-equivalent local load → calibrate/extend → predict → re-export works for the supported matrix, with no parent mutation or calibration-data exposure.

**Completion evidence (27 November 2026):** Stage 4 and its single-site HBR cache follow-up are included in `dev_sharing` commit `f4d12978dc9549446b41ddc144c57e5b0bd31f18`. All 152 tracked files match the tested implementation: 1201 passed, 8 expected skips; the user also confirmed all tests pass in their development environment. Native adaptation parity, parent immutability, lineage, privacy checks, reload, and re-export are covered. Earlier Stage 2–3 checkboxes have been reconciled with the implemented and tested baseline.

### Stage 5 — Implement Hugging Face transport

**Status (27 November 2026):** Local transport implementation, independent review and regression-tested fixes are complete. Full suite: 1253 passed, 9 skipped; focused transport/security suite: 52 passed, 1 skipped. The live-service exit gate remains pending.

**Purpose:** Add authenticated, version-aware storage without yet claiming two-service publication.

**Depends on:** Stage 2; stage 4 remains required before the full feature release.

**Files:** `huggingface.py`, private `_hf_upload.py` worker, optional dependency configuration and developer documentation; `tests/sharing/test_huggingface.py`, `test_huggingface_security.py` and opt-in `test_huggingface_live.py`.

**Interfaces:** Backend methods `preflight`, `upload_bundle`, `read_descriptor`, `download_bundle`, and `write_descriptor`; keep these internal to the sharing package.

- [x] Write mocked tests for permitted and forbidden namespaces, same-version same-digest retry, conflicting payload, optimistic concurrency, token redaction, timeouts, and revision pinning.
- [x] Confirm failures, implement using the supported Hugging Face client, and run `pytest tests/sharing/test_huggingface.py -q`.
- [ ] Run an opt-in test against a dedicated test repository using a synthetic bundle and scoped credentials. Verify read-only credentials cannot write. Ordinary CI must not publish.
- [x] Review and commit; document test-only remote artifacts and any authorized cleanup.

**Exit gate:** Exact bytes upload/download successfully; unauthorized writes fail at the service; a different payload cannot replace the same version through MEGaNorm.

**Local verification evidence (27 November 2026):** The full suite passed with 1253 tests and 9 expected skips in 351.79 seconds. Five important independent-review findings were fixed using failing regressions: ignored required files, streaming byte limits, transfer credential/log handling, canonical repository identity, and private output-path errors (including write failures). The focused suite passed 52 tests with the live test skipped. Sphinx with warnings treated as errors, Black, whitespace, release metadata and installed-wheel checks passed; local HBR loading works with Hugging Face imports blocked. SDK 1.33.0 is pinned and uploads use an isolated HTTP LFS worker. Repeated descriptor validation is a deferred performance improvement. No test repository was supplied, so no live model publication or remote cleanup was performed; the opt-in live command and retained synthetic artifacts are documented in `docs/source/model_sharing_development.rst`.

### Stage 6 — Implement Zenodo archival transport

**Purpose:** Manage citable versioned archives programmatically.

**Depends on:** Stage 2.

**Files:** `zenodo.py`; `tests/sharing/test_zenodo.py`.

**Interfaces:** `prepare_draft`, `upload_bundle`, `publish`, `get_record`, and `download_bundle`, returning structured provider IDs rather than reconstructed URLs.

- [ ] Write mocked tests for required metadata/license, DOI reservation, first/new versions, permission errors, sandbox isolation, upload verification, and ambiguous publish timeouts.
- [ ] Confirm failures; implement the current documented API and follow provider-returned record links.
- [ ] Run `pytest tests/sharing/test_zenodo.py -q`.
- [ ] Run opt-in Zenodo sandbox tests for publication and a second version; reconcile timed-out requests without duplicate publication. Production publication is not a test default.
- [ ] Review and commit.

**Exit gate:** Two versions of a synthetic bundle can be archived and retrieved by exact record/version identity; metadata cannot accidentally turn a restricted workflow public.

### Stage 7 — Implement automatic push and recovery

**Purpose:** Expose one reliable push operation across both services.

**Depends on:** Stages 4–6.

**Files:** `publication.py`, `meganorm/API/sharing.py`, release schema; `tests/sharing/test_publication.py`.

**Interfaces:** `ModelHub.export`, `.validate`, and `.push` as specified in section 6.

- [ ] Write fault-injection tests at every transition in section 5: before/after remote success, before journal persistence, during archival publication, and before descriptor completion.
- [ ] Test repeated calls, competing processes, lost journals, changed bundle bytes between validation and upload, inconsistent remote descriptors, missing credentials, and unreviewed metadata.
- [ ] Confirm failures; implement a durable journal, locks, remote reconciliation, immutable staged payload, and conflict detection. Never retry a mutation blindly after an ambiguous response.
- [ ] Assert dry-run performs no remote mutation; assert retries produce one logical release and never replace different content.
- [ ] Run `pytest tests/sharing/test_publication.py tests/sharing/test_huggingface.py tests/sharing/test_zenodo.py -q` and the opt-in cross-service synthetic workflow.
- [ ] Review and commit with an explicit failure/recovery matrix.

**Exit gate:** One push produces matching payload digests on both hosts and a complete receipt. Every injected interruption either resumes correctly or returns a clear incomplete/conflict state.

### Stage 8 — Implement fetch, cache, offline use, and archival recovery

**Purpose:** Make published models easy to reuse reproducibly.

**Depends on:** Stage 7.

**Files:** `cache.py`, API facade; `tests/sharing/test_fetch.py`, `test_cache.py`.

**Interfaces:** `ModelHub.fetch(reference, *, version=None, cache_dir=None, offline=False) -> FetchedModel`; require an exact version unless the reference is already version-specific.

- [ ] Write tests for explicit revision resolution, corrupted/missing members, path attacks, incomplete descriptors, interrupted download, concurrent fetch, offline hit/miss, and cache immutability.
- [ ] Test exact-digest Zenodo fallback on availability failure. Assert authorization denial, integrity failure, and an unrelated archive never trigger silent fallback.
- [ ] Confirm failures; implement atomic cache population and lock receipts; validate before any backend reconstruction.
- [ ] Run `pytest tests/sharing/test_fetch.py tests/sharing/test_cache.py tests/sharing/test_bundle.py -q`.
- [ ] In a clean environment, push a synthetic model, fetch, predict, calibrate, and extend. Disconnect network access and repeat cached loading/prediction.
- [ ] Review and commit.

**Exit gate:** Exact-version online and offline workflows pass; cached predictions and adaptations preserve parent bytes and do not send user inputs to either host.

### Stage 9 — Documentation, integration audit, and release readiness

**Purpose:** Deliver a maintainable, researcher-facing feature.

**Depends on:** Stages 0–8.

**Files:** User/developer documentation and notebook in section 6; Sphinx index/API docs; README, CHANGELOG, packaging, and CI as needed.

- [ ] Write concise examples for export/push, exact-version fetch/predict, calibration, extension, and publishing an adaptation to the user's own namespace.
- [ ] Explain account setup, least-privilege credentials, model licenses, citation of the specific release, privacy exclusions, compatibility errors, and incomplete-publication recovery.
- [ ] Keep development-stage language out of getting-started documentation. Provide synthetic examples supporting cohorts of arbitrary size, with no real participant data or secrets.
- [ ] Run `pytest -q` using the validated development environment; review new warnings for relevance rather than blanket suppression.
- [ ] Run `python -m sphinx -W --keep-going -b html docs/source docs/build/html`.
- [ ] Build and install a wheel in a clean environment; test core local functionality without sharing extras and remote functionality with extras.
- [ ] Execute the tutorial with isolated synthetic fixtures, clear outputs before committing, and inspect notebook metadata for paths/secrets.
- [ ] Review the complete diff and requirement matrix; repeat the final privacy inspection on the actual release candidate bundle. Record all checks and known limitations.
- [ ] Prepare release notes and a merge-ready change set. Official scientific model publication is a distinct action using approved metadata and credentials.

**Exit gate:** All mandatory checks pass; documentation matches implemented capabilities; no claims of unsupported model versions or formal privacy guarantees; the user can reproduce the complete supported workflow.

## 9. Requirement-to-test traceability

| Requirement | Primary stages | Acceptance evidence |
| --- | --- | --- |
| R1–R2: no participant data/derivatives | 0, 2, 4, 9 | Deep state audit, adversarial fixtures, adapted-model re-export, release bundle inspection |
| R3: useful provenance and definitions | 1–3 | Strict manifest fixtures and input compatibility tests |
| R4: calibration and extension | 0, 4 | Direct PCNToolkit parity and held-out synthetic evaluation |
| R5: automated push/fetch | 5–8 | Cross-service workflow, exact-version fetch, offline and recovery tests |
| R6–R7: user publishing and authorization | 5–7 | Separate-owner/read-only tests; provider-denied writes |
| R8: version preservation | 5–8 | Same-digest retry; conflicting-digest rejection; locked fetch |
| R9: lineage and parent preservation | 4, 7–8 | Parent byte hashes unchanged and lineage survives re-export |
| R10: researcher usability | 3, 9 | Fresh-environment tutorial and concise public API |
| R11: staged development | All | Per-stage completion records and reviewed exit gates |

## 10. Open technical decisions and stop conditions

These are bounded investigations, not missing user requirements:

| Decision | Owner stage | Required resolution |
| --- | --- | --- |
| Safe PCNToolkit model state and supported versions | 0 | Field-level audit and real round-trip/adaptation evidence |
| HBR/SHASH and BLR support matrix | 0 | Explicit supported/unsupported combinations; no blanket support claim |
| Predictive comparison tolerances and stochastic seeding | 0 | Fix tolerances from backend behavior and independent repeatability checks |
| Archive size/extraction limits | 0–1 | Document defaults based on representative posterior sizes; support explicit bounded overrides |
| Zenodo API details and cross-provider linking | 6–7 | Verify current endpoints in sandbox; keep digest identity independent of mutable metadata |
| Licenses for original and adapted models | 1, 9 | Require an explicit license; preserve parent license obligations; no automatic relicensing |

Stop progression if clean state cannot support a required backend operation, a test reveals data leakage, a conflicting release is detected, or provider state is ambiguous. Record the evidence and resolve the cause within that stage. Do not relax privacy, invent remote identifiers, silently downgrade to another model version, or mark partial publication complete.

## 11. Stage completion record template

```text
Stage:
Status: not started / in progress / blocked / ready for review / complete
Baseline and completion commits:
Dependency versions:
Implemented behavior:
Tests and commands:
Results:
Privacy and integrity evidence:
Remote test environment (if applicable):
Known limitations / blockers:
Exit gate met:
Next stage:
```

## 12. References and maintenance

Official sources consulted on 13 November 2026. Recheck the installed backend and current provider APIs at implementation time; documentation version labels are not sufficient compatibility evidence.

- [PCNToolkit NormativeModel API](https://pcntoolkit.readthedocs.io/en/stable/autoapi/pcntoolkit/normative_model/index.html) — documented prediction, transfer, and extension entry points; exact adapter behavior must be verified against the installed version.
- [Hugging Face model uploads](https://huggingface.co/docs/hub/models-uploading) — custom model files and repositories.
- [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/main/guides/download) — cached downloads and exact revisions.
- [Hugging Face organization access controls](https://huggingface.co/docs/hub/organizations-security) — roles controlling write access.
- [Hugging Face model cards](https://huggingface.co/docs/huggingface_hub/en/guides/model-cards) — human-readable model documentation.
- [Zenodo developer API](https://developers.zenodo.org/) — draft, upload, publish, and version operations.
- [Zenodo version management](https://help.zenodo.org/docs/deposit/manage-versions/) — versioned archival records.

Update this plan as decisions are resolved, retaining a dated decision log. A future implementation session should review the completed Stage 0 evidence and current repository state, then start Stage 1. Repeat the audit only when backend changes or expanded capabilities warrant it.

### Decision log

| Date | Decision | Status |
| --- | --- | --- |
| 2026-11-13 | Hugging Face + Zenodo; automated push/fetch | Agreed |
| 2026-11-13 | No participant identifiers, measurements, predictions, or z-scores in shared models | Agreed |
| 2026-11-13 | New-site calibration and extension reuse PCNToolkit | Agreed |
| 2026-11-13 | Any user may publish; modification requires provider authorization | Agreed |
| 2026-11-13 | Develop and test stage by stage | Agreed |
| 2026-11-13 | Portable allowlisted bundle, separate loaded-model API, resumable release descriptor, public-only initial publication | Proposed implementation baseline for review |

| 2026-11-13 | Pin Stage 0 experiments to released PCNToolkit 1.4.0; create local `dev_sharing` from merged dev | Implemented |
| 2026-11-13 | Privacy boundary is participant-traceable information; fitted scaler statistics are acceptable | User clarification |
| 2026-11-13 | Reuse necessary aggregate synthesis summaries; recommend upstream portable export/output controls | Stage 0 recommendation |
