"""Internal Hugging Face transport for validated model bundles.

Only existing public, ungated model repositories are supported. Payload commits
and archival-linkage descriptors are separate. This module never certifies
Zenodo publication or returns a completed two-provider release.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import base64
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request
from tempfile import TemporaryDirectory
from typing import Annotated

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from .bundle import (
    DEFAULT_LIMITS,
    BundleLimits,
    copy_stream,
    read_json,
    sanitized,
    staged,
)
from .errors import (
    AuthorizationError,
    IncompletePublicationError,
    IntegrityError,
    PrivacyError,
    SharingError,
    TransportError,
    VersionConflictError,
)
from .schema import (
    GitCommit,
    ModelBundle,
    ModelId,
    ReleaseVersion,
    SHA256,
    ZenodoReference,
    _StrictModel,
)

ENDPOINT = "https://huggingface.co"
DESCRIPTOR_LIMIT = 64 * 1024


class _UploadPolicyError(IntegrityError):
    """The worker refused a known incomplete commit before submission."""


class _PublicRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _public_url(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise IntegrityError("Unsafe public download redirect.")
    return url


def _open_public(url):
    # A fresh opener has no cookies or SDK token store. urllib does not log
    # signed redirect URLs; only HTTPS targets without URL credentials are used.
    return build_opener(_PublicRedirect()).open(
        Request(_public_url(url), headers={"Accept-Encoding": "identity"}), timeout=60
    )


class PayloadRecord(_StrictModel):
    """Identity record committed atomically with the version's model archive."""

    schema_version: Annotated[int, Field(ge=1, le=1)] = 1
    repo_id: ModelId
    version: ReleaseVersion
    bundle_sha256: SHA256
    size_bytes: Annotated[int, Field(gt=0)]


class PayloadReceipt(PayloadRecord):
    """An exact commit containing verified payload bytes; not a publication receipt."""

    commit: GitCommit


class HubDescriptor(_StrictModel):
    """Coordinator-supplied archival linkage, without a self-referential commit.

    Structural validity does not establish that the archive exists or is public.
    The future publication coordinator must verify those provider-side facts.
    """

    schema_version: Annotated[int, Field(ge=1, le=1)] = 1
    repo_id: ModelId
    version: ReleaseVersion
    payload_commit: GitCommit
    bundle_sha256: SHA256
    zenodo: ZenodoReference


@dataclass(frozen=True)
class DescriptorReceipt:
    """A validated linkage document and the exact commit from which it was read."""

    descriptor: HubDescriptor
    commit: str


def _parse(kind, value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    try:
        return TypeAdapter(kind).validate_python(value, strict=True)
    except (ValidationError, TypeError, ValueError):
        pass
    raise IntegrityError("Invalid transport identity, revision, or descriptor.")


def _json(value):
    return json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _paths(version):
    prefix = f"releases/{_parse(ReleaseVersion, version)}/"
    return prefix + "model.zip", prefix + "payload.json", prefix + "release.json"


def _reviewed(value):
    if value is not True:
        raise PrivacyError("Review public model metadata before uploading.")


class HuggingFaceTransport:
    """Internal versioned transport using the optional supported Hub SDK.

    Credentials use the SDK's token argument or its usual local credential store.
    The endpoint is fixed; downloads from public repositories are anonymous.
    There is no repository creation, visibility change, overwrite, or deletion API.
    """

    def __init__(self, *, token=None, limits=DEFAULT_LIMITS):
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "Install the optional client with pip install 'meganorm[sharing]'."
            ) from None
        if not isinstance(limits, BundleLimits):
            raise ValueError("Use BundleLimits for transport budgets.")
        self.limits = limits
        self._token = token
        self._api = HfApi(endpoint=ENDPOINT, token=token, library_name="meganorm")

    @staticmethod
    def _repo(repo_id):
        from huggingface_hub.utils import validate_repo_id

        repo_id = _parse(ModelId, repo_id)
        try:
            validate_repo_id(repo_id)
        except ValueError:
            pass
        else:
            return repo_id
        raise IntegrityError("Invalid Hugging Face model repository identity.")

    @staticmethod
    def _call(method, *args, **kwargs):
        """Never retain provider messages, URLs, response objects or exception context."""
        try:
            return method(*args, **kwargs)
        except SharingError:
            raise
        except Exception as error:
            status = getattr(
                getattr(error, "response", None),
                "status_code",
                getattr(error, "code", None),
            )
            if status in (401, 403, 404):
                failure = AuthorizationError(
                    "Repository access or permission was denied; check local credentials and destination."
                )
            elif status in (409, 412):
                failure = VersionConflictError(
                    "Repository changed concurrently; reconcile the version before retrying."
                )
            else:
                failure = TransportError(
                    "Hugging Face request failed; check connectivity and retry the operation."
                )
        raise failure

    def _head(self, repo_id, revision="main"):
        info = self._call(
            self._api.repo_info,
            repo_id,
            revision=revision,
            repo_type="model",
            timeout=30,
        )
        if (
            getattr(info, "id", None) != repo_id
            or getattr(info, "private", None) is not False
            or getattr(info, "gated", None) is not False
        ):
            raise AuthorizationError(
                "Only the exact requested public, ungated model repository is supported."
            )
        commit = _parse(GitCommit, getattr(info, "sha", None))
        if revision != "main" and revision != commit:
            raise IntegrityError(
                "The provider returned a different revision than requested."
            )
        return commit

    def preflight(self, repo_id, *, write=False):
        """Check repository visibility and, when requested, server-side write access.

        This read-only operation returns the current main commit. Authorization
        is enforced again by the provider when committing; it is not inferred
        from the namespace name or a broad token role.
        """
        repo_id = self._repo(repo_id)
        if type(write) is not bool:
            raise ValueError("write must be a boolean.")
        head = self._head(repo_id)
        if write:
            self._call(self._api.auth_check, repo_id, repo_type="model", write=True)
        return head

    def _inventory(self, repo_id, version, commit):
        paths = _paths(version)
        entries = self._call(
            self._api.get_paths_info,
            repo_id,
            list(paths),
            revision=commit,
            repo_type="model",
        )
        result = {}
        for entry in entries:
            name, size = getattr(entry, "path", None), getattr(entry, "size", None)
            limit = (
                self.limits.max_archive_bytes
                if name == paths[0]
                else min(DESCRIPTOR_LIMIT, self.limits.max_json_bytes)
            )
            if (
                name not in paths
                or name in result
                or type(size) is not int
                or not 0 < size <= limit
            ):
                raise IntegrityError("Remote inventory or file size is invalid.")
            result[name] = size
        return result

    def _fetch(self, repo_id, name, commit, size, work):
        """Stream anonymously from an exact commit, bounding bytes before writes."""
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(
            repo_id, name, revision=commit, repo_type="model", endpoint=ENDPOINT
        )
        target = work / Path(name).name
        try:
            with (
                sanitized(),
                self._call(_open_public, url) as src,
                target.open("xb") as dst,
            ):
                if src.headers.get("Content-Encoding", "identity") != "identity":
                    raise IntegrityError("Encoded download responses are unsupported.")
                actual, _ = self._call(copy_stream, src, dst, size)
                if actual != size:
                    raise IntegrityError(
                        "Downloaded size does not match the inventory."
                    )
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return target

    @contextmanager
    def _payload(self, repo_id, version, commit, expected=None):
        archive_name, record_name, descriptor_name = _paths(version)
        entries = self._inventory(repo_id, version, commit)
        if (
            archive_name not in entries
            and record_name not in entries
            and descriptor_name not in entries
        ):
            yield None
            return
        if archive_name not in entries or record_name not in entries:
            raise IntegrityError("Remote version contains an incomplete payload.")
        with sanitized(), TemporaryDirectory(prefix="meganorm-hf-read-") as temp:
            work = Path(temp)
            path = self._fetch(repo_id, record_name, commit, entries[record_name], work)
            record = _parse(PayloadRecord, read_json(path, DESCRIPTOR_LIMIT))
            if (
                record.repo_id != repo_id
                or record.version != version
                or record.size_bytes != entries[archive_name]
            ):
                raise IntegrityError("Remote payload identity or size does not match.")
            if expected is not None and record != expected:
                raise VersionConflictError(
                    "This model version already contains different content."
                )
            archive = self._fetch(
                repo_id, archive_name, commit, entries[archive_name], work
            )
            with staged(archive, self.limits, record.bundle_sha256) as (
                root,
                manifest,
                digest,
            ):
                if manifest.model_id != repo_id or manifest.version != version:
                    raise IntegrityError(
                        "Bundle identity does not match the remote version."
                    )
                # staged owns this private, validated snapshot for the context's lifetime.
                yield record, root.parent / "archive.zip"

    def _payload_receipt(self, repo_id, version, commit, expected):
        with self._payload(repo_id, version, commit, expected) as payload:
            if payload is None:
                return None
            record, _ = payload
            return PayloadReceipt(**record.model_dump(), commit=commit)

    def _create_commit(self, repo_id, operations, **options):
        """Run the SDK's global transfer machinery in a private interpreter."""
        files = []
        for operation in operations:
            item = {"path": operation.path_in_repo}
            if isinstance(operation.path_or_fileobj, (str, Path)):
                item["local_path"] = str(operation.path_or_fileobj)
            else:
                with operation.as_file() as stream:
                    item["data"] = base64.b64encode(stream.read()).decode("ascii")
            files.append(item)
        payload = dict(
            repo_id=repo_id,
            token=self._token,
            parent_commit=options["parent_commit"],
            files=files,
        )
        result = self._call(
            subprocess.run,
            [sys.executable, str(Path(__file__).with_name("_hf_upload.py"))],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=600,
        )
        with sanitized():
            answer = json.loads(result.stdout) if result.returncode == 0 else {}
        kind = answer.get("error")
        if kind == "ignored":
            raise _UploadPolicyError(
                "Repository ignore rules exclude a required release file; adjust them before retrying."
            )
        if kind == "authorization":
            raise AuthorizationError("Repository access or permission was denied.")
        if kind == "conflict":
            raise VersionConflictError(
                "Repository changed concurrently; retry explicitly."
            )
        if kind or not answer.get("oid"):
            raise TransportError("Upload request failed; reconcile before retrying.")
        return SimpleNamespace(oid=answer["oid"])

    def _commit(self, repo_id, head, operations, reconcile):
        """One conditional mutation, followed by verification or status reconciliation."""
        failure = None
        try:
            info = self._create_commit(
                repo_id,
                operations,
                commit_message="Store validated MEGaNorm model-sharing artifact",
                repo_type="model",
                revision="main",
                parent_commit=head,
                create_pr=False,
                num_threads=1,
            )
            commit = _parse(GitCommit, getattr(info, "oid", None))
        except _UploadPolicyError:
            raise
        except (TransportError, VersionConflictError, IntegrityError) as error:
            failure = type(error)
        if failure is not None:
            try:
                commit = self.preflight(repo_id)
            except TransportError:
                commit = None
            if commit is None:
                raise IncompletePublicationError(
                    "Commit outcome is unknown; rerun to reconcile before any new write."
                )
        try:
            result = reconcile(commit)
        except TransportError:
            result = None
        if result is not None:
            return result
        if failure is VersionConflictError:
            raise VersionConflictError(
                "Repository changed concurrently; rerun after checking the version."
            )
        raise IncompletePublicationError(
            "Commit could not be verified; rerun to reconcile before any new write."
        )

    def upload_bundle(self, path, *, repo_id, version, metadata_reviewed=False):
        """Upload one fully validated snapshot, or verify an identical existing version.

        The receipt's commit identifies verified bytes at an exact snapshot. An
        idempotent retry can return a newer commit containing those same bytes.
        """
        from huggingface_hub import CommitOperationAdd

        _reviewed(metadata_reviewed)
        repo_id, version = self._repo(repo_id), _parse(ReleaseVersion, version)
        with staged(path, self.limits) as (root, manifest, digest):
            if manifest.model_id != repo_id or manifest.version != version:
                raise IntegrityError(
                    "Destination and version must match the validated bundle."
                )
            archive = root.parent / "archive.zip"
            record = PayloadRecord(
                repo_id=repo_id,
                version=version,
                bundle_sha256=digest,
                size_bytes=archive.stat().st_size,
            )
            head = self.preflight(repo_id, write=True)
            reconcile = lambda commit: self._payload_receipt(
                repo_id, version, commit, record
            )
            existing = reconcile(head)
            if existing is not None:
                return existing
            archive_name, record_name, _ = _paths(version)
            operations = [
                CommitOperationAdd(path_in_repo=archive_name, path_or_fileobj=archive),
                CommitOperationAdd(
                    path_in_repo=record_name, path_or_fileobj=_json(record)
                ),
            ]
            return self._commit(repo_id, head, operations, reconcile)

    @staticmethod
    def _output_path(output_dir):
        try:
            output = Path(output_dir).expanduser()
            occupied = output.is_symlink() or output.exists()
            resolved = output.resolve() if not occupied else None
        except (OSError, ValueError, TypeError, RuntimeError):
            pass
        else:
            if occupied:
                raise FileExistsError(
                    "Download output directory already exists or is a link."
                )
            return resolved
        # Raise outside the handler; suppressed display alone retains __context__.
        raise IntegrityError("Download output directory could not be safely resolved.")

    @staticmethod
    def _save_download(archive, output, limit):
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.mkdir(mode=0o700)
            try:
                with (
                    archive.open("rb") as src,
                    (output / "model.zip").open("xb") as dst,
                ):
                    copy_stream(src, dst, limit)
            except BaseException:
                (output / "model.zip").unlink(missing_ok=True)
                output.rmdir()
                raise
        except (OSError, ValueError, TypeError, RuntimeError):
            pass
        else:
            return
        raise IntegrityError(
            "Downloaded bundle could not be safely written to the output directory."
        )

    def download_bundle(self, receipt, *, output_dir):
        """Download and validate the exact receipt commit into a new local directory."""
        receipt = _parse(PayloadReceipt, receipt)
        repo_id = self._repo(receipt.repo_id)
        output = self._output_path(output_dir)
        self._head(repo_id, receipt.commit)
        expected = _parse(PayloadRecord, receipt.model_dump(exclude={"commit"}))
        with self._payload(
            repo_id, receipt.version, receipt.commit, expected
        ) as payload:
            if payload is None:
                raise IntegrityError("The pinned commit does not contain the payload.")
            _, archive = payload
            self._save_download(archive, output, self.limits.max_archive_bytes)
        return ModelBundle(
            path=output / "model.zip",
            model_id=repo_id,
            version=receipt.version,
            schema_version=1,
            bundle_sha256=receipt.bundle_sha256,
        )

    def _descriptor_at(self, repo_id, version, commit):
        _, _, name = _paths(version)
        entries = self._inventory(repo_id, version, commit)
        if name not in entries:
            return None
        with sanitized(), TemporaryDirectory(prefix="meganorm-hf-descriptor-") as temp:
            path = self._fetch(repo_id, name, commit, entries[name], Path(temp))
            descriptor = _parse(HubDescriptor, read_json(path, DESCRIPTOR_LIMIT))
        if descriptor.repo_id != repo_id or descriptor.version != version:
            raise IntegrityError(
                "Descriptor identity does not match its repository version."
            )
        self._verify_linkage(descriptor, commit)
        return DescriptorReceipt(descriptor, commit)

    def _verify_linkage(self, descriptor, current):
        self._head(descriptor.repo_id, descriptor.payload_commit)
        with self._payload(
            descriptor.repo_id, descriptor.version, descriptor.payload_commit
        ) as payload:
            if payload is None or payload[0].bundle_sha256 != descriptor.bundle_sha256:
                raise IntegrityError("Descriptor does not identify the pinned payload.")
            expected = payload[0]
        with self._payload(
            descriptor.repo_id, descriptor.version, current, expected
        ) as payload:
            if payload is None:
                raise IntegrityError(
                    "Current version no longer contains the referenced payload."
                )

    def read_descriptor(self, repo_id, version, *, revision=None):
        """Resolve main once or read an exact commit; return None when no linkage exists."""
        repo_id, version = self._repo(repo_id), _parse(ReleaseVersion, version)
        revision = "main" if revision is None else _parse(GitCommit, revision)
        commit = self._head(repo_id, revision)
        return self._descriptor_at(repo_id, version, commit)

    def write_descriptor(self, descriptor, *, metadata_reviewed=False):
        """Store immutable coordinator-supplied linkage after checking its payload.

        The caller must establish Zenodo publication independently. This transport
        validates only the linkage structure and the referenced Hugging Face bytes.
        """
        from huggingface_hub import CommitOperationAdd

        _reviewed(metadata_reviewed)
        descriptor = _parse(HubDescriptor, descriptor)
        repo_id = self._repo(descriptor.repo_id)
        head = self.preflight(repo_id, write=True)
        self._verify_linkage(descriptor, head)

        def reconcile(commit):
            existing = self._descriptor_at(repo_id, descriptor.version, commit)
            if existing is not None and existing.descriptor != descriptor:
                raise VersionConflictError(
                    "This version already has different archival linkage."
                )
            return existing

        existing = reconcile(head)
        if existing is not None:
            return existing
        name = _paths(descriptor.version)[2]
        return self._commit(
            repo_id,
            head,
            [CommitOperationAdd(path_in_repo=name, path_or_fileobj=_json(descriptor))],
            reconcile,
        )
