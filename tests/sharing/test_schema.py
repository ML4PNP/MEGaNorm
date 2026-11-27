"""Manifest contracts: metadata only, without model loading or networking."""

import copy
import json
from pathlib import Path

import pytest

HASH = "a" * 64
COMMIT = "b" * 40


@pytest.fixture
def manifest():
    return {
        "schema_version": 1,
        "model_id": "publisher/example",
        "version": "1.0.0",
        "adapter": {
            "name": "pcntoolkit",
            "format_version": 1,
            "pcntoolkit_version": "1.4.0",
            "profile": "blr-default-site",
        },
        "software": {"meganorm_version": "0.2.2", "python_version": "3.12.14"},
        "responses": [{"name": "roi", "unit": "V", "storage_key": "roi"}],
        "covariates": [{"name": "age", "kind": "continuous", "unit": "years"}],
        "batch_effects": [
            {
                "name": "site",
                "categories": [{"label": "A", "code": 0}, {"label": "B", "code": 1}],
            }
        ],
        "preprocessing": {
            "modality": "EEG",
            "paradigm": "rest",
            "space": "sensor",
            "pipeline_version": "0.2.2",
            "config_sha256": HASH,
            "description": "Approved feature extraction settings.",
        },
        "metadata": {
            "title": "Synthetic reference model",
            "description": "Test fixture.",
            "authors": ["Example Researcher"],
            "license": "CC-BY-4.0",
            "datasets": ["Synthetic"],
            "sites": ["A", "B"],
            "citations": [],
        },
        "lineage": {"operation": "fit", "parent": None},
        "capabilities": ["predict", "calibrate", "extend"],
        "files": [
            {"path": p, "sha256": HASH, "size_bytes": 128}
            for p in [
                "README.md",
                "LICENSE",
                "model/normative_model.json",
                "model/roi/regression_model.json",
            ]
        ],
    }


def validate(document):
    from meganorm.sharing.schema import validate_manifest

    return validate_manifest(document)


def test_valid_manifest_roundtrips_canonically(manifest):
    from meganorm.sharing.schema import canonical_manifest, parse_manifest

    assert validate(manifest).valid
    encoded = canonical_manifest(manifest)
    assert parse_manifest(json.loads(encoded)).model_dump(mode="json") == json.loads(
        encoded
    )
    assert canonical_manifest(json.loads(encoded)) == encoded
    assert manifest["files"][0]["path"] == "README.md"  # no input mutation


@pytest.mark.parametrize(
    "problem", ["calibration_overlap", "extension_missing_parent", "blr_freedom"]
)
def test_adaptation_metadata_matches_profile_and_site_support(manifest, problem):
    manifest["lineage"] = {
        "operation": "calibrate",
        "parent": {
            "model_id": "publisher/parent",
            "version": "1.0.0",
            "bundle_sha256": HASH,
        },
        "settings": {
            "parent_categories": [{"label": "A", "code": 0}, {"label": "B", "code": 1}],
        },
    }
    manifest["capabilities"] = ["predict"]
    manifest["batch_effects"][0]["categories"] = [{"label": "C", "code": 0}]
    assert validate(manifest).valid
    if problem == "calibration_overlap":
        manifest["lineage"]["settings"]["parent_categories"][0]["label"] = "c"
    elif problem == "extension_missing_parent":
        manifest["lineage"]["operation"] = "extend"
    else:
        manifest["lineage"]["settings"]["freedom"] = 0.5
    assert not validate(manifest).valid


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "adapter",
        "software",
        "responses",
        "preprocessing",
        "metadata",
        "lineage",
        "files",
    ],
)
def test_missing_required_fields(manifest, field):
    del manifest[field]
    report = validate(manifest)
    assert not report.valid
    assert "missing_field" in [i.code for i in report.issues]


@pytest.mark.parametrize("value", ["1.0", "v1.0.0", "01.0.0", "1.0.0-dev", "", 1, True])
def test_noncanonical_release_versions_rejected(manifest, value):
    manifest["version"] = value
    assert not validate(manifest).valid


@pytest.mark.parametrize(
    "field,value", [("schema_version", 2), ("schema_version", True)]
)
def test_unsupported_schema_or_wrong_type(manifest, field, value):
    manifest[field] = value
    assert not validate(manifest).valid


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "other"),
        ("format_version", 2),
        ("pcntoolkit_version", "invalid"),
        ("profile", "arbitrary-hbr"),
    ],
)
def test_unknown_adapter_rejected(manifest, field, value):
    manifest["adapter"][field] = value
    assert not validate(manifest).valid


def test_profile_is_bounded_to_stage0_shapes(manifest):
    manifest["covariates"].append(
        {"name": "height", "kind": "continuous", "unit": "cm"}
    )
    assert "unsupported_configuration" in [i.code for i in validate(manifest).issues]


@pytest.mark.parametrize("name", ["roi", "ROI", "ＲＯＩ"])
def test_duplicate_response_names_rejected(manifest, name):
    manifest["responses"].append({"name": name, "unit": "V", "storage_key": "another"})
    assert "duplicate_name" in [i.code for i in validate(manifest).issues]


def test_cross_role_name_collision_rejected(manifest):
    manifest["covariates"][0]["name"] = "ROI"
    assert "duplicate_name" in [i.code for i in validate(manifest).issues]


@pytest.mark.parametrize(
    "categories",
    [
        [],
        [{"label": "A", "code": 0}, {"label": "a", "code": 1}],
        [{"label": "A", "code": 1}],
        [{"label": "A", "code": 0}, {"label": "B", "code": 0}],
    ],
)
def test_category_labels_and_codes_unambiguous(manifest, categories):
    manifest["batch_effects"][0]["categories"] = categories
    assert not validate(manifest).valid


@pytest.mark.parametrize(
    "path",
    [
        "../private.json",
        "/private.json",
        "model/../x.json",
        "model\\x.json",
        "model//x.json",
        "model/.hidden.json",
        "model/CON.json",
        "model/foo./x.json",
        "model/a:stream.json",
        "model/%2e%2e/x.json",
        "manifest.json",
        "model/private.csv",
    ],
)
def test_unsafe_or_unapproved_files_rejected(manifest, path):
    manifest["files"].append({"path": path, "sha256": HASH, "size_bytes": 10})
    assert not validate(manifest).valid


def test_file_inventory_is_exact(manifest):
    manifest["files"].pop()
    assert "invalid_inventory" in [i.code for i in validate(manifest).issues]


def test_case_colliding_file_paths_rejected(manifest):
    manifest["files"].append({"path": "readme.md", "sha256": HASH, "size_bytes": 1})
    assert "duplicate_path" in [i.code for i in validate(manifest).issues]


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "invalid"),
        ("sha256", "A" * 64),
        ("size_bytes", True),
        ("size_bytes", -1),
        ("size_bytes", 0),
        ("size_bytes", 1.5),
        ("size_bytes", float("inf")),
    ],
)
def test_file_hash_and_size_strict(manifest, field, value):
    manifest["files"][0][field] = value
    assert not validate(manifest).valid


def test_declared_json_size_limit(manifest):
    manifest["files"][-1]["size_bytes"] = 32 * 1024**2 + 1
    assert "artifact_limit" in [i.code for i in validate(manifest).issues]


@pytest.mark.parametrize(
    "location,key",
    [
        ("root", "participants"),
        ("metadata", "predictions"),
        ("preprocessing", "recording_path"),
        ("response", "measurements"),
        ("metadata", "PRIVATE_subject_123"),
    ],
)
def test_unknown_fields_rejected_without_echoing_inputs(
    manifest, location, key, caplog
):
    from meganorm.sharing.schema import parse_manifest
    from meganorm.sharing.errors import ManifestValidationError

    target = (
        manifest
        if location == "root"
        else manifest["responses"][0] if location == "response" else manifest[location]
    )
    target[key] = {"PRIVATE_subject_123": "token-secret-canary"}
    report = validate(manifest)
    assert not report.valid
    assert "unknown_field" in [i.code for i in report.issues]
    with pytest.raises(ManifestValidationError) as exc:
        parse_manifest(manifest)
    exposed = repr(report) + str(exc.value) + caplog.text
    assert "PRIVATE_subject_123" not in exposed
    assert "token-secret-canary" not in exposed


def test_invalid_value_never_echoed(manifest):
    manifest["version"] = "PRIVATE_subject_123"
    report = validate(manifest)
    assert "PRIVATE_subject_123" not in repr(report)


def test_hbr_requires_posterior_file(manifest):
    manifest["adapter"]["profile"] = "hbr-shashb-linear-site"
    assert not validate(manifest).valid
    manifest["files"].append(
        {"path": "model/roi/idata.nc", "sha256": HASH, "size_bytes": 128}
    )
    assert validate(manifest).valid


def test_lineage_requires_exact_parent_and_limits_capabilities(manifest):
    manifest["lineage"]["operation"] = "calibrate"
    assert not validate(manifest).valid
    manifest["lineage"]["parent"] = {
        "model_id": "publisher/parent",
        "version": "1.0.0",
        "bundle_sha256": HASH,
    }
    assert not validate(manifest).valid  # further adaptation not audited yet
    manifest["capabilities"] = ["predict"]
    assert validate(manifest).valid


def test_self_parent_rejected(manifest):
    manifest["lineage"] = {
        "operation": "extend",
        "parent": {
            "model_id": manifest["model_id"],
            "version": manifest["version"],
            "bundle_sha256": HASH,
        },
    }
    manifest["capabilities"] = ["predict"]
    assert "invalid_lineage" in [i.code for i in validate(manifest).issues]


def test_previous_version_of_same_model_is_valid_parent(manifest):
    manifest["version"] = "1.0.1"
    manifest["lineage"] = {
        "operation": "extend",
        "parent": {
            "model_id": manifest["model_id"],
            "version": "1.0.0",
            "bundle_sha256": HASH,
        },
    }
    manifest["capabilities"] = ["predict"]
    assert validate(manifest).valid


def test_canonical_order_independent_of_dictionary_and_file_order(manifest):
    from meganorm.sharing.schema import canonical_manifest

    alternate = dict(reversed(list(copy.deepcopy(manifest).items())))
    alternate["files"].reverse()
    assert canonical_manifest(manifest) == canonical_manifest(alternate)


def test_local_results_are_not_valid_manifests(manifest):
    from meganorm.sharing.schema import PredictionResults

    result = PredictionResults(
        predictions=None, z_scores=None, participant_index=("private",)
    )
    report = validate(result)
    assert not report.valid
    assert "private" not in repr(report)


def test_local_result_contracts_and_release_references():
    from meganorm.sharing.schema import (
        ModelBundle,
        PublishedRelease,
        FetchedModel,
        PublicationPreview,
        HuggingFaceReference,
        ZenodoReference,
    )

    hf = HuggingFaceReference(repo_id="publisher/example", commit=COMMIT)
    zenodo = ZenodoReference(record_id="12345", version_doi="10.5281/zenodo.12345")
    bundle = ModelBundle(
        path=Path("/local/bundle"),
        model_id="publisher/example",
        version="1.0.0",
        schema_version=1,
        bundle_sha256=HASH,
    )
    release = PublishedRelease(
        model_id=bundle.model_id,
        version=bundle.version,
        huggingface=hf,
        descriptor_commit="c" * 40,
        zenodo=zenodo,
        bundle_sha256=HASH,
        status="complete",
    )
    fetched = FetchedModel(
        path=bundle.path,
        source=hf,
        bundle_sha256=HASH,
        lock_receipt=Path("/local/lock.json"),
    )
    preview = PublicationPreview(
        repo_id=bundle.model_id,
        version=bundle.version,
        files=[],
        bundle_sha256=HASH,
        metadata_ready=False,
    )
    assert release.status == "complete" and fetched.path == bundle.path
    assert not preview.metadata_ready


@pytest.mark.parametrize("commit", ["main", "v1.0.0", "abcdef0"])
def test_mutable_or_short_revision_rejected(commit):
    from meganorm.sharing.schema import HuggingFaceReference

    with pytest.raises(ValueError):
        HuggingFaceReference(repo_id="publisher/model", commit=commit)


def test_mismatched_archive_identity_rejected():
    from meganorm.sharing.schema import ZenodoReference

    with pytest.raises(ValueError):
        ZenodoReference(record_id="12", version_doi="10.5281/zenodo.13")


def test_numeric_unknown_key_is_not_mistaken_for_an_array_index(manifest):
    manifest["metadata"][987654321] = "private"
    report = validate(manifest)
    assert not report.valid
    assert "987654321" not in repr(report)


@pytest.mark.parametrize("field", ["name", "unit"])
def test_control_characters_not_allowed_in_labels(manifest, field):
    manifest["responses"][0][field] = "first\nsecond"
    assert not validate(manifest).valid


def test_boolean_not_a_schema_version_in_local_bundle():
    from meganorm.sharing.schema import ModelBundle

    with pytest.raises(ValueError):
        ModelBundle(
            path=Path("/local"),
            model_id="owner/model",
            version="1.0.0",
            schema_version=True,
            bundle_sha256=HASH,
        )


def test_capability_order_is_not_semantic(manifest):
    from meganorm.sharing.schema import canonical_manifest

    other = copy.deepcopy(manifest)
    other["capabilities"].reverse()
    assert canonical_manifest(manifest) == canonical_manifest(other)


def test_batch_code_mapping_order_is_semantic(manifest):
    from meganorm.sharing.schema import canonical_manifest

    other = copy.deepcopy(manifest)
    other["batch_effects"][0]["categories"][0]["label"] = "B"
    other["batch_effects"][0]["categories"][1]["label"] = "A"
    assert canonical_manifest(manifest) != canonical_manifest(other)


def test_sandbox_archive_identity_is_explicit():
    from meganorm.sharing.schema import ZenodoReference

    source = ZenodoReference(
        environment="sandbox", record_id="12345", version_doi="10.5072/zenodo.12345"
    )
    assert source.environment == "sandbox"
    with pytest.raises(ValueError):
        ZenodoReference(record_id="12345", version_doi="10.5072/zenodo.12345")


def test_sanitized_exception_has_no_raw_validation_context(manifest):
    from meganorm.sharing.schema import parse_manifest
    from meganorm.sharing.errors import ManifestValidationError

    manifest["PRIVATE_IDENTIFIER"] = "PRIVATE_VALUE"
    with pytest.raises(ManifestValidationError) as exc:
        parse_manifest(manifest)
    assert exc.value.__context__ is None
    assert exc.value.__cause__ is None


def test_modified_nested_model_is_revalidated(manifest):
    from meganorm.sharing.schema import parse_manifest

    metadata = parse_manifest(manifest).metadata
    metadata.authors.clear()
    manifest["metadata"] = metadata
    assert not validate(manifest).valid


def test_modified_deeply_nested_model_is_revalidated(manifest):
    from meganorm.sharing.schema import parse_manifest

    batches = parse_manifest(manifest).batch_effects
    batches[0].categories[0].label = "PRIVATE\nIDENTIFIER"
    manifest["batch_effects"] = batches
    report = validate(manifest)
    assert not report.valid
    assert "PRIVATE" not in repr(report)


@pytest.mark.parametrize("field", ["adapter", "batch_effects"])
def test_modified_nested_contract_cannot_be_canonicalized(manifest, field):
    from meganorm.sharing.schema import parse_manifest, canonical_manifest
    from meganorm.sharing.errors import ManifestValidationError

    parsed = parse_manifest(manifest)
    if field == "adapter":
        parsed.adapter.pcntoolkit_version = "UNSUPPORTED"
    else:
        parsed.batch_effects[0].categories[0].code = 99
    manifest[field] = getattr(parsed, field)
    assert not validate(manifest).valid
    with pytest.raises(ManifestValidationError):
        canonical_manifest(manifest)


@pytest.mark.parametrize("value", ["3.11.9", "3.13.1", "3.14.0rc1"])
def test_python_provenance_not_a_runtime_gate(manifest, value):
    manifest["software"]["python_version"] = value
    assert validate(manifest).valid


@pytest.mark.parametrize(
    "field,value",
    [("meganorm_version", "0.3.0.dev1"), ("meganorm_version", "0.3.0rc1+local")],
)
def test_software_provenance_accepts_development_versions(manifest, field, value):
    manifest["software"][field] = value
    assert validate(manifest).valid


def test_backend_provenance_separate_from_compatibility(manifest):
    from meganorm.sharing import validate_compatibility, canonical_manifest

    manifest["adapter"]["pcntoolkit_version"] = "1.5.0"
    assert validate(manifest).valid
    assert b"1.5.0" in canonical_manifest(manifest)
    assert not validate_compatibility(
        manifest, runtime_pcntoolkit_version="1.4.0"
    ).valid


@pytest.mark.parametrize(
    "producer,runtime,expected",
    [
        ("1.4.0", "1.4.0", True),
        ("1.4.1", "1.4.0", False),
        ("1.4.0", "1.4.1", False),
        ("1.4.0+local", "1.4.0", False),
        ("1.4.0", "PRIVATE_ID", False),
    ],
)
@pytest.mark.parametrize("profile", ["blr-default-site", "hbr-shashb-linear-site"])
def test_compatibility_checks_producer_and_consumer(
    manifest, producer, runtime, expected, profile
):
    from meganorm.sharing import validate_compatibility

    manifest["adapter"]["profile"] = profile
    if profile == "hbr-shashb-linear-site":
        manifest["files"].append(
            {"path": "model/roi/idata.nc", "sha256": HASH, "size_bytes": 128}
        )
    manifest["adapter"]["pcntoolkit_version"] = producer
    report = validate_compatibility(manifest, runtime_pcntoolkit_version=runtime)
    assert report.valid is expected
    assert "PRIVATE_ID" not in repr(report)


def test_runtime_provenance_tracks_environment(monkeypatch):
    from meganorm.sharing import versions

    monkeypatch.setattr(versions.platform, "python_version", lambda: "3.13.2")
    monkeypatch.setattr(versions.metadata, "version", lambda name: "1.5.0rc1")
    actual = versions.capture_runtime_versions()
    from meganorm._version import __version__

    assert actual.python_version == "3.13.2"
    assert actual.pcntoolkit_version == "1.5.0rc1"
    assert actual.meganorm_version == __version__


def test_missing_backend_has_no_fabricated_provenance(monkeypatch):
    from meganorm.sharing import versions

    def missing(name):
        raise versions.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(versions.metadata, "version", missing)
    with pytest.raises(versions.metadata.PackageNotFoundError):
        versions.capture_runtime_versions()


@pytest.mark.parametrize("field", ["python_version", "meganorm_version"])
@pytest.mark.parametrize("value", ["not-a-version", "", " 3.12.0", 312])
def test_invalid_software_provenance_rejected(manifest, field, value):
    manifest["software"][field] = value
    assert not validate(manifest).valid


def test_compatibility_does_not_skip_schema_validation(manifest):
    from meganorm.sharing import validate_compatibility

    manifest["PRIVATE_PARTICIPANT"] = "PRIVATE_VALUE"
    report = validate_compatibility(manifest, runtime_pcntoolkit_version="1.4.0")
    assert not report.valid
    assert "PRIVATE" not in repr(report)
