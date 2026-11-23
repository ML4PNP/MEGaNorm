"""Local, deterministic model bundles with validation before reconstruction.

No network client, cache, or executable object serialization is involved. Public
metadata must be deliberately reviewed; this is not a formal inference-privacy
or scientific-validity guarantee.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import struct
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .compatibility import validate_compatibility
from .errors import (
    CompatibilityError,
    IntegrityError,
    IssueCode,
    PrivacyError,
    SharingError,
    ValidationIssue,
    ValidationReport,
    VersionConflictError,
)
from .schema import (
    MAX_ARCHIVE_BYTES,
    MAX_EXTRACTED_BYTES,
    MAX_JSON_BYTES,
    MAX_MEMBERS,
    ModelBundle,
    canonical_manifest,
    parse_manifest,
)
from .versions import capture_runtime_versions


@dataclass(frozen=True)
class BundleLimits:
    """Positive actual-byte/member budgets; posterior arrays default to 512 MiB."""

    max_archive_bytes: int = MAX_ARCHIVE_BYTES
    max_extracted_bytes: int = MAX_EXTRACTED_BYTES
    max_json_bytes: int = MAX_JSON_BYTES
    max_members: int = MAX_MEMBERS
    max_array_bytes: int = 512 * 1024**2

    def __post_init__(self):
        for value in vars(self).values():
            if type(value) is not int or value <= 0:
                raise ValueError("Bundle limits must be positive integers.")


DEFAULT_LIMITS = BundleLimits()


def invalid():
    raise IntegrityError("Bundle inventory, bytes, or resource limits are invalid.")


@contextmanager
def sanitized():
    """Do not expose input-bearing parser/backend exceptions or chained context."""
    try:
        yield
    except SharingError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        RuntimeError,
        RecursionError,
        OverflowError,
        EOFError,
        zipfile.BadZipFile,
    ):
        pass
    else:
        return
    raise IntegrityError(
        "Bundle could not be safely exported, validated, or loaded."
    ) from None


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            invalid()
        result[key] = value
    return result


def read_json(path, limit):
    if path.stat().st_size > limit:
        invalid()
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda _: invalid(),
    )


def copy_stream(src, dst, limit):
    """Copy and hash bounded chunks, including a one-byte overflow probe."""
    digest = hashlib.sha256()
    size = 0
    while chunk := src.read(min(1024**2, limit - size + 1)):
        size += len(chunk)
        if size > limit:
            invalid()
        digest.update(chunk)
        dst.write(chunk)
    return size, digest.hexdigest()


def compatible(document):
    report = validate_compatibility(
        document,
        runtime_pcntoolkit_version=capture_runtime_versions().pcntoolkit_version,
    )
    if not report.valid:
        raise CompatibilityError(
            "Bundle metadata or producer/runtime version pair is unsupported."
        )


def readme(manifest):
    """Render only explicit publication metadata, never source paths or reprs."""
    m = manifest.metadata
    return (
        f'# {m.title}\n\n{m.description}\n\nAuthors: {", ".join(m.authors)}\n'
        f"Model: {manifest.model_id} ({manifest.version})\nLicense: {m.license}\n"
    ).encode("utf-8")


def license_text(manifest):
    """Store the publisher's license declaration; do not invent legal terms."""
    return (manifest.metadata.license + "\n").encode("utf-8")


def zip_preflight(path, limits):
    """Bound ZIP central-directory parsing before ZipFile allocates its entries.

    Initial bundles use a single-disk, comment-free ZIP32 container. Reject ZIP64
    and multipart archives; extending the format requires explicit validation.
    """
    size = path.stat().st_size
    if size < 22:
        invalid()
    with path.open("rb") as f:
        f.seek(-22, 2)
        end = f.read(22)
    magic, disk, start, count_disk, count, central_size, offset, comment = (
        struct.unpack("<4s4H2IH", end)
    )
    if (
        magic != b"PK\x05\x06"
        or disk
        or start
        or comment
        or count_disk != count
        or count > limits.max_members
        or count == 65535
        or central_size > limits.max_members * 302
        or offset + central_size != size - 22
    ):
        invalid()


def write_archive(root, archive, names, limits):
    """Write one canonical ZIP representation, excluding all unlisted bytes."""
    with zipfile.ZipFile(archive, "w", allowZip64=False) as z:
        for name in sorted(names):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            with (root / name).open("rb") as src, z.open(info, "w") as dst:
                copy_stream(src, dst, limits.max_extracted_bytes)


@contextmanager
def staged(path, limits, expected_sha256=None):
    """Validate a private snapshot and yield its safe staged payload."""
    from .pcn_adapter import validate_payload

    with sanitized(), tempfile.TemporaryDirectory(prefix="meganorm-bundle-") as temp:
        root = Path(temp) / "payload"
        root.mkdir()
        archive = Path(temp) / "archive.zip"
        source = Path(path)
        if source.is_symlink():
            invalid()
        fd = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(fd, "rb") as src, archive.open("wb") as dst:
            if not stat.S_ISREG(os.fstat(src.fileno()).st_mode):
                invalid()
            _, digest = copy_stream(src, dst, limits.max_archive_bytes)
        if expected_sha256 is not None and digest != expected_sha256:
            invalid()
        zip_preflight(archive, limits)
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            names = [i.filename for i in infos]
            if (
                len(infos) > limits.max_members
                or len(set(names)) != len(names)
                or names.count("manifest.json") != 1
            ):
                invalid()
            if sum(i.file_size for i in infos) > limits.max_extracted_bytes:
                invalid()
            for i in infos:
                if (
                    i.is_dir()
                    or stat.S_IFMT(i.external_attr >> 16) not in (0, stat.S_IFREG)
                    or i.flag_bits & 1
                    or i.extra
                    or i.comment
                    or i.compress_type != zipfile.ZIP_STORED
                ):
                    invalid()
            info = z.getinfo("manifest.json")
            if info.file_size > limits.max_json_bytes:
                invalid()
            with z.open(info) as src, (root / "manifest.json").open("wb") as dst:
                copy_stream(src, dst, limits.max_json_bytes)
            document = read_json(root / "manifest.json", limits.max_json_bytes)
            manifest = parse_manifest(document)
            compatible(document)
            if (root / "manifest.json").read_bytes() != canonical_manifest(document):
                invalid()
            entries = {f.path: f for f in manifest.files}
            if set(names) != set(entries) | {"manifest.json"}:
                invalid()
            total = (root / "manifest.json").stat().st_size
            for name, entry in entries.items():
                if z.getinfo(name).file_size != entry.size_bytes:
                    invalid()
                budget = min(entry.size_bytes, limits.max_extracted_bytes - total)
                if name.endswith(".json"):
                    budget = min(budget, limits.max_json_bytes)
                dest = (
                    root / name
                )  # Schema and exact inventory already checked this path.
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, dest.open("wb") as dst:
                    n, actual = copy_stream(src, dst, budget)
                total += n
                if n != entry.size_bytes or actual != entry.sha256:
                    invalid()
        if (root / "README.md").read_bytes() != readme(manifest) or (
            root / "LICENSE"
        ).read_bytes() != license_text(manifest):
            invalid()
        canonical = Path(temp) / "canonical.zip"
        write_archive(root, canonical, names, limits)
        with canonical.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                invalid()
        validate_payload(
            root,
            manifest,
            lambda p: read_json(p, limits.max_json_bytes),
            limits.max_array_bytes,
        )
        yield root, manifest, digest


def export_model(
    model, *, metadata, output_dir, metadata_reviewed=False, limits=DEFAULT_LIMITS
) -> ModelBundle:
    """Export an original fitted native model to ``output_dir/model.zip``.

    Supply manifest metadata without ``files``; inventory and hashes are generated.
    Affirm review of public labels/text with ``metadata_reviewed=True``. No source
    model files are copied or changed. Existing destinations are never replaced.
    Aggregate scalers and site summaries remain available for native reuse.
    """
    from .pcn_adapter import export_state

    if metadata_reviewed is not True:
        raise PrivacyError("Review public names and narrative metadata before export.")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise VersionConflictError("Export destination already exists.")
    with sanitized(), tempfile.TemporaryDirectory(prefix="meganorm-export-") as temp:
        document = copy.deepcopy(metadata)
        if type(document) is not dict or "files" in document:
            invalid()
        key = document["responses"][0]["storage_key"]
        names = [
            "README.md",
            "LICENSE",
            "model/normative_model.json",
            f"model/{key}/regression_model.json",
        ]
        if document["adapter"]["profile"] == "hbr-shashb-linear-site":
            names.append(f"model/{key}/idata.nc")
        document["files"] = [dict(path=n, sha256="0" * 64, size_bytes=1) for n in names]
        manifest = parse_manifest(document)
        compatible(document)
        root = Path(temp) / "payload"
        root.mkdir()
        export_state(model, manifest, root, limits.max_array_bytes)
        (root / "README.md").write_bytes(readme(manifest))
        (root / "LICENSE").write_bytes(license_text(manifest))
        document["files"] = []
        for name in names:
            p = root / name
            with p.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            document["files"].append(
                dict(path=name, sha256=digest, size_bytes=p.stat().st_size)
            )
        (root / "manifest.json").write_bytes(canonical_manifest(document))
        archive = Path(temp) / "model.zip"
        write_archive(root, archive, names + ["manifest.json"], limits)
        with staged(archive, limits) as (_, _, digest):
            pass
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            output.mkdir()
        except FileExistsError:
            raise VersionConflictError("Export destination already exists.") from None
        try:
            with archive.open("rb") as src, (output / "model.zip").open("xb") as dst:
                copy_stream(src, dst, limits.max_archive_bytes)
        except BaseException:
            (output / "model.zip").unlink(missing_ok=True)
            output.rmdir()
            raise
        return ModelBundle(
            path=output / "model.zip",
            model_id=manifest.model_id,
            version=manifest.version,
            schema_version=manifest.schema_version,
            bundle_sha256=digest,
        )


def validate_bundle(
    path, *, expected_sha256=None, limits=DEFAULT_LIMITS
) -> ValidationReport:
    """Validate actual archive contents without constructing a fitted backend.

    An external trusted digest is optional. Embedded hashes establish consistency,
    not publisher authenticity. A valid result does not certify scientific quality.
    """
    try:
        with staged(path, limits, expected_sha256):
            pass
    except SharingError as exc:
        code = (
            IssueCode.UNSUPPORTED_ADAPTER
            if isinstance(exc, CompatibilityError)
            else IssueCode.INVALID_INVENTORY
        )
        return ValidationReport(
            (
                ValidationIssue(
                    code,
                    "$",
                    "Bundle failed integrity, privacy, compatibility, or size checks.",
                ),
            )
        )
    return ValidationReport()


def load_backend(path, *, expected_sha256=None, limits=DEFAULT_LIMITS):
    """Validate and restore a native backend with prediction output writes disabled.

    Posterior arrays survive temporary-staging cleanup. The native transfer/extend
    methods still have their own output behavior; the later facade will manage it.
    """
    from .pcn_adapter import restore_backend

    with staged(path, limits, expected_sha256) as (root, manifest, _):
        return restore_backend(root, manifest, limits.max_array_bytes)
