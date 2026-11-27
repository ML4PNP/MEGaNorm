"""Transport privacy, immutable versions, pinned reads and mutation recovery."""

import copy
import hashlib
import importlib
import io
from urllib.parse import urlsplit
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("huggingface_hub")
import httpx
from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError

from .conftest import data, hbr
from .test_bundle import rewrite_bundle
from meganorm.sharing.errors import (
    AuthorizationError,
    IncompletePublicationError,
    IntegrityError,
    PrivacyError,
    VersionConflictError,
)

REPO = "publisher/example"
VERSION = "1.0.0"
TOKEN = "hf_PRIVATE_TOKEN_DO_NOT_PRINT"
PREFIX = "releases/1.0.0/"


def api_error(status):
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://huggingface.co/api/test")
    )
    return HfHubHTTPError(f"provider response contains {TOKEN}", response=response)


class HubService:
    """Only remote SDK operations are replaced; bundles and commits are real bytes."""

    def __init__(self, tmp_path):
        self.repo_id = REPO
        self.head = "a" * 40
        self.snapshots = {self.head: {}}
        self.private = self.gated = False
        self.allow_write = True
        self.calls = []
        self.attempts = 0
        self.before_commit = None
        self.after_commit = None
        self.on_info = None
        self.fail_info = None
        self.oversized = False
        self.bad_download = False

    def auth_check(self, repo_id, *, repo_type, write):
        assert repo_id == self.repo_id and repo_type == "model"
        self.calls.append(("auth", write))
        if write and not self.allow_write:
            raise api_error(403)

    def repo_info(self, repo_id, *, revision, repo_type, timeout):
        assert repo_id == self.repo_id and repo_type == "model"
        self.calls.append(("info", revision))
        if self.fail_info:
            raise self.fail_info
        if self.on_info:
            callback, self.on_info = self.on_info, None
            callback()
        commit = self.head if revision == "main" else revision
        if commit not in self.snapshots:
            raise api_error(404)
        return SimpleNamespace(
            id=self.repo_id, sha=commit, private=self.private, gated=self.gated
        )

    def get_paths_info(self, repo_id, paths, *, revision, repo_type):
        assert repo_id == self.repo_id and repo_type == "model" and len(revision) == 40
        self.calls.append(("paths", revision))
        return [
            SimpleNamespace(path=p, size=(10**14 if self.oversized else len(content)))
            for p, content in self.snapshots[revision].items()
            if p in paths
        ]

    def download(self, url):
        repo, rest = urlsplit(url).path[1:].split("/resolve/")
        revision, filename = rest.split("/", 1)
        assert repo == self.repo_id and len(revision) == 40
        self.calls.append(("download", revision, filename))
        content = self.snapshots[revision][filename]
        result = io.BytesIO(b"corrupt" if self.bad_download else content)
        result.headers = {}
        return result

    def create_commit(
        self,
        repo_id,
        operations,
        *,
        commit_message,
        repo_type,
        revision,
        parent_commit,
        create_pr,
        num_threads,
    ):
        assert repo_id == self.repo_id and repo_type == "model" and revision == "main"
        assert create_pr is False and num_threads == 1
        self.attempts += 1
        if not self.allow_write:
            raise api_error(403)
        if self.before_commit:
            callback, self.before_commit = self.before_commit, None
            callback(operations)
        if parent_commit != self.head:
            raise api_error(409)
        contents = copy.deepcopy(self.snapshots[self.head])
        for operation in operations:
            assert isinstance(operation, CommitOperationAdd)
            with operation.as_file() as stream:
                contents[operation.path_in_repo] = stream.read()
        self.advance(contents)
        if self.after_commit:
            callback, self.after_commit = self.after_commit, None
            callback()
        return SimpleNamespace(oid=self.head)

    def advance(self, contents):
        self.head = hashlib.sha1(str(len(self.snapshots)).encode()).hexdigest()
        self.snapshots[self.head] = contents


@pytest.fixture(scope="module")
def native_model(tmp_path_factory):
    from pcntoolkit import NormativeModel

    model = NormativeModel(
        hbr(),
        save_dir=str(tmp_path_factory.mktemp("hf-synthetic")),
        savemodel=False,
        saveresults=False,
        saveplots=False,
        evaluate_model=False,
    )
    model.fit(data())
    return model


@pytest.fixture
def bundle(native_model, manifest, tmp_path):
    from meganorm.sharing import export_model

    manifest.pop("files")
    manifest["adapter"]["profile"] = "hbr-shashb-linear-site"
    return export_model(
        native_model,
        metadata=manifest,
        output_dir=tmp_path / "source",
        metadata_reviewed=True,
    )


@pytest.fixture
def service(monkeypatch, tmp_path):
    import huggingface_hub

    hub = HubService(tmp_path)

    def client(*, endpoint, token, library_name):
        assert endpoint == "https://huggingface.co" and library_name == "meganorm"
        assert token == TOKEN
        return hub

    monkeypatch.setattr(huggingface_hub, "HfApi", client)
    module = importlib.import_module("meganorm.sharing.huggingface")
    monkeypatch.setattr(module, "_open_public", hub.download)

    def commit(self, *args, **kwargs):
        return self._call(hub.create_commit, *args, **kwargs)

    monkeypatch.setattr(module.HuggingFaceTransport, "_create_commit", commit)
    return hub


def transport(**kwargs):
    module = importlib.import_module("meganorm.sharing.huggingface")
    return module.HuggingFaceTransport(token=TOKEN, **kwargs)


def upload(client, bundle):
    return client.upload_bundle(
        bundle.path, repo_id=REPO, version=VERSION, metadata_reviewed=True
    )


def test_upload_retry_and_pinned_download(bundle, service, tmp_path):
    client = transport()
    receipt = upload(client, bundle)
    original_commit = receipt.commit
    assert receipt.bundle_sha256 == bundle.bundle_sha256
    contents = service.snapshots[original_commit]
    assert set(contents) == {PREFIX + "model.zip", PREFIX + "payload.json"}
    assert contents[PREFIX + "model.zip"] == bundle.path.read_bytes()
    assert all(b"PRIVATE_" not in content for content in contents.values())
    assert upload(client, bundle) == receipt and service.attempts == 1
    service.advance({})  # a moving main must not affect a previously issued receipt
    result = client.download_bundle(receipt, output_dir=tmp_path / "download")
    assert result.path.read_bytes() == bundle.path.read_bytes()
    assert result.bundle_sha256 == bundle.bundle_sha256
    assert all(
        call[1] == original_commit for call in service.calls if call[0] == "download"
    )


@pytest.mark.parametrize("mode", ["denied", "private", "gated", "missing"])
def test_preflight_blocks_unauthorized_or_restricted_repositories(
    bundle, service, mode
):
    if mode == "denied":
        service.allow_write = False
    if mode == "private":
        service.private = True
    if mode == "gated":
        service.gated = "auto"
    if mode == "missing":
        service.fail_info = api_error(404)
    with pytest.raises(AuthorizationError) as error:
        upload(transport(), bundle)
    assert service.attempts == 0
    assert TOKEN not in str(error.value) and error.value.__context__ is None


@pytest.mark.parametrize(
    "repo,version,reviewed",
    [
        ("publisher/../private", VERSION, True),
        ("https://elsewhere/model", VERSION, True),
        ("other/model", VERSION, True),
        (REPO, "../1.0.0", True),
        (REPO, "2.0.0", True),
        (REPO, VERSION, False),
    ],
)
def test_invalid_identity_or_unreviewed_metadata_never_reaches_service(
    bundle, service, repo, version, reviewed
):
    with pytest.raises((IntegrityError, PrivacyError, ValueError)):
        transport().upload_bundle(
            bundle.path, repo_id=repo, version=version, metadata_reviewed=reviewed
        )
    assert service.calls == [] and service.attempts == 0


def test_validation_precedes_upload_and_source_changes_do_not_change_snapshot(
    bundle, service
):
    original = bundle.path.read_bytes()
    service.on_info = lambda: bundle.path.write_bytes(b"PRIVATE_replacement")
    receipt = upload(transport(), bundle)
    assert service.snapshots[receipt.commit][PREFIX + "model.zip"] == original


def test_rehashed_participant_state_is_rejected_before_network(
    bundle, service, tmp_path
):
    unsafe = tmp_path / "unsafe.zip"

    def mutate(contents):
        key = "model/roi/regression_model.json"
        state = json.loads(contents[key])
        state["model"]["PRIVATE_rows"] = [1, 2]
        contents[key] = json.dumps(state).encode()

    rewrite_bundle(bundle.path, unsafe, mutate)
    with pytest.raises((PrivacyError, IntegrityError)):
        transport().upload_bundle(
            unsafe, repo_id=REPO, version=VERSION, metadata_reviewed=True
        )
    assert service.calls == [] and service.attempts == 0


def test_conflicting_version_never_overwrites(bundle, service, tmp_path):
    client = transport()
    upload(client, bundle)
    changed = tmp_path / "changed.zip"

    def mutate(contents):
        state = json.loads(contents["model/roi/regression_model.json"])
        state["model"]["draws"] += 1
        contents["model/roi/regression_model.json"] = json.dumps(state).encode()

    rewrite_bundle(bundle.path, changed, mutate)
    before = copy.deepcopy(service.snapshots)
    with pytest.raises(VersionConflictError):
        client.upload_bundle(
            changed, repo_id=REPO, version=VERSION, metadata_reviewed=True
        )
    assert service.snapshots == before and service.attempts == 1


@pytest.mark.parametrize(
    "fault", ["timeout_after", "race_same", "race_different", "timeout_before"]
)
def test_mutation_reconciles_without_blind_retries(bundle, service, fault):
    client = transport()

    def timeout(*args):
        raise httpx.ReadTimeout(TOKEN)

    if fault == "timeout_after":
        service.after_commit = timeout
    elif fault == "timeout_before":
        service.before_commit = timeout
    else:

        def race(operations):
            contents = {}
            for operation in operations:
                with operation.as_file() as stream:
                    contents[operation.path_in_repo] = stream.read()
            if fault == "race_different":
                record = json.loads(contents[PREFIX + "payload.json"])
                record["bundle_sha256"] = "f" * 64
                contents[PREFIX + "payload.json"] = json.dumps(record).encode()
            service.advance(contents)

        service.before_commit = race
    if fault in {"timeout_after", "race_same"}:
        receipt = upload(client, bundle)
        assert (
            receipt.commit == service.head
            and receipt.bundle_sha256 == bundle.bundle_sha256
        )
    else:
        expected = (
            VersionConflictError
            if fault == "race_different"
            else IncompletePublicationError
        )
        with pytest.raises(expected) as error:
            upload(client, bundle)
        assert TOKEN not in str(error.value) and error.value.__context__ is None
    assert service.attempts == 1


@pytest.mark.parametrize(
    "fault", ["partial", "oversized", "corrupt", "wrong_identity", "extra_field"]
)
def test_untrusted_remote_payload_is_rejected(bundle, service, tmp_path, fault):
    client = transport()
    receipt = upload(client, bundle)
    state = service.snapshots[receipt.commit]
    if fault == "partial":
        del state[PREFIX + "payload.json"]
    elif fault == "oversized":
        service.oversized = True
    elif fault == "corrupt":
        service.bad_download = True
    else:
        record = json.loads(state[PREFIX + "payload.json"])
        record["repo_id" if fault == "wrong_identity" else "PRIVATE_rows"] = (
            "other/model"
        )
        state[PREFIX + "payload.json"] = json.dumps(record).encode()
    service.calls.clear()
    with pytest.raises(IntegrityError):
        client.download_bundle(receipt, output_dir=tmp_path / "bad-download")
    assert not (tmp_path / "bad-download").exists()
    if fault == "oversized":
        assert not any(call[0] == "download" for call in service.calls)


def descriptor(receipt):
    return dict(
        schema_version=1,
        repo_id=REPO,
        version=VERSION,
        payload_commit=receipt.commit,
        bundle_sha256=receipt.bundle_sha256,
        zenodo=dict(
            provider="zenodo",
            environment="sandbox",
            record_id="123",
            version_doi="10.5072/zenodo.123",
        ),
    )


def test_descriptor_is_separate_immutable_and_pinned(bundle, service):
    client = transport()
    assert client.read_descriptor(REPO, VERSION) is None
    payload = upload(client, bundle)
    assert client.read_descriptor(REPO, VERSION) is None
    document = descriptor(payload)
    stored = client.write_descriptor(document, metadata_reviewed=True)
    assert stored.commit != payload.commit
    assert stored.descriptor.payload_commit == payload.commit
    assert (
        service.snapshots[stored.commit][PREFIX + "model.zip"]
        == bundle.path.read_bytes()
    )
    assert client.write_descriptor(document, metadata_reviewed=True) == stored
    assert service.attempts == 2
    changed = copy.deepcopy(document)
    changed["zenodo"].update(record_id="456", version_doi="10.5072/zenodo.456")
    with pytest.raises(VersionConflictError):
        client.write_descriptor(changed, metadata_reviewed=True)
    service.advance({})
    assert client.read_descriptor(REPO, VERSION, revision=stored.commit) == stored
    assert client.read_descriptor(REPO, VERSION, revision=payload.commit) is None


@pytest.mark.parametrize(
    "fault",
    ["bad_digest", "unknown_commit", "extra", "unreviewed", "payload_missing_now"],
)
def test_invalid_descriptors_never_commit(bundle, service, fault):
    client = transport()
    payload = upload(client, bundle)
    document = descriptor(payload)
    if fault == "bad_digest":
        document["bundle_sha256"] = "f" * 64
    elif fault == "unknown_commit":
        document["payload_commit"] = "f" * 40
    elif fault == "extra":
        document["PRIVATE_rows"] = [1]
    elif fault == "payload_missing_now":
        service.advance({})
    with pytest.raises(
        (IntegrityError, PrivacyError, AuthorizationError, VersionConflictError)
    ):
        client.write_descriptor(document, metadata_reviewed=fault != "unreviewed")
    assert service.attempts == 1


def test_descriptor_timeout_reconciles(bundle, service):
    client = transport()
    payload = upload(client, bundle)

    def timeout():
        raise httpx.ReadTimeout(TOKEN)

    service.after_commit = timeout
    stored = client.write_descriptor(descriptor(payload), metadata_reviewed=True)
    assert stored.commit == service.head and service.attempts == 2


def test_missing_sdk_does_not_break_local_imports():
    code = """
import sys
sys.modules['huggingface_hub'] = None
from meganorm.sharing import validate_bundle
from meganorm.API import SharedNormativeModel
from meganorm.sharing.huggingface import HuggingFaceTransport
try:
    HuggingFaceTransport()
except ImportError as e:
    assert 'meganorm[sharing]' in str(e)
else:
    raise AssertionError('Missing optional client was not reported')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_permission_revoked_at_commit_is_not_retried(bundle, service):
    def revoked(operations):
        raise api_error(403)

    service.before_commit = revoked
    with pytest.raises(AuthorizationError) as error:
        upload(transport(), bundle)
    assert service.attempts == 1 and len(service.snapshots) == 1
    assert TOKEN not in str(error.value) and error.value.__context__ is None


def test_unknown_commit_outcome_is_resolved_on_next_call(bundle, service):
    def lose_response_and_connectivity():
        service.fail_info = httpx.ConnectError(TOKEN)
        raise httpx.ReadTimeout(TOKEN)

    service.after_commit = lose_response_and_connectivity
    client = transport()
    with pytest.raises(IncompletePublicationError) as error:
        upload(client, bundle)
    assert TOKEN not in str(error.value) and error.value.__context__ is None
    assert service.attempts == 1
    service.fail_info = None
    recovered = upload(client, bundle)
    assert recovered.commit == service.head and service.attempts == 1


def test_unrelated_concurrent_commit_requires_explicit_retry(bundle, service):
    service.before_commit = lambda operations: service.advance(
        {"unrelated.txt": b"test"}
    )
    client = transport()
    with pytest.raises(VersionConflictError):
        upload(client, bundle)
    assert service.attempts == 1
    receipt = upload(client, bundle)
    assert receipt.bundle_sha256 == bundle.bundle_sha256
    assert service.snapshots[receipt.commit]["unrelated.txt"] == b"test"


@pytest.mark.parametrize("kind", ["directory", "file", "link"])
def test_download_never_overwrites_existing_outputs(bundle, service, tmp_path, kind):
    client = transport()
    receipt = upload(client, bundle)
    dest = tmp_path / "protected"
    if kind == "directory":
        dest.mkdir()
    elif kind == "file":
        dest.write_bytes(b"keep")
    else:
        dest.symlink_to(tmp_path / "elsewhere")
    service.calls.clear()
    with pytest.raises(FileExistsError):
        client.download_bundle(receipt, output_dir=dest)
    assert service.calls == []
    if kind == "file":
        assert dest.read_bytes() == b"keep"
    if kind == "link":
        assert dest.is_symlink()


def test_descriptor_concurrent_conflict_is_not_overwritten(bundle, service):
    client = transport()
    payload = upload(client, bundle)
    document = descriptor(payload)

    def competing_descriptor(operations):
        contents = copy.deepcopy(service.snapshots[service.head])
        other = copy.deepcopy(document)
        other["zenodo"].update(record_id="456", version_doi="10.5072/zenodo.456")
        contents[PREFIX + "release.json"] = json.dumps(other).encode()
        service.advance(contents)

    service.before_commit = competing_descriptor
    with pytest.raises(VersionConflictError):
        client.write_descriptor(document, metadata_reviewed=True)
    assert service.attempts == 2
    stored = client.read_descriptor(REPO, VERSION)
    assert stored.descriptor.zenodo.record_id == "456"


@pytest.mark.parametrize(
    "namespace,allowed", [("independent-lab", True), ("protected-lab", False)]
)
def test_namespace_access_is_decided_by_the_provider(
    native_model, manifest, service, tmp_path, namespace, allowed
):
    from meganorm.sharing import export_model

    manifest.pop("files")
    manifest["adapter"]["profile"] = "hbr-shashb-linear-site"
    manifest["model_id"] = service.repo_id = f"{namespace}/example"
    service.allow_write = allowed
    bundle = export_model(
        native_model,
        metadata=manifest,
        output_dir=tmp_path / "other",
        metadata_reviewed=True,
    )
    client = transport()
    if allowed:
        receipt = client.upload_bundle(
            bundle.path,
            repo_id=service.repo_id,
            version=VERSION,
            metadata_reviewed=True,
        )
        assert (
            receipt.repo_id == service.repo_id
            and receipt.bundle_sha256 == bundle.bundle_sha256
        )
    else:
        with pytest.raises(AuthorizationError):
            client.upload_bundle(
                bundle.path,
                repo_id=service.repo_id,
                version=VERSION,
                metadata_reviewed=True,
            )
        assert service.attempts == 0


def test_download_write_error_does_not_retain_private_path(
    bundle, service, tmp_path, monkeypatch
):
    client = transport()
    receipt = upload(client, bundle)
    output = tmp_path / "PRIVATE_DIRECTORY" / "new"
    original = Path.mkdir

    def blocked(path, *args, **kwargs):
        if path == output.parent:
            raise PermissionError(
                "PRIVATE_DIRECTORY must not appear in exception context"
            )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", blocked)
    with pytest.raises(IntegrityError) as error:
        client.download_bundle(receipt, output_dir=output)
    assert "PRIVATE_DIRECTORY" not in str(error.value)
    assert error.value.__context__ is None
    assert not output.exists()
