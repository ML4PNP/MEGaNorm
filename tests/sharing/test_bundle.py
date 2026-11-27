"""Exact inventory, archive budgets, and safe error behavior."""

import hashlib
import json
import stat
import zipfile
import pytest


def export(model, metadata, path):
    from meganorm.sharing import export_model

    return export_model(
        model, metadata=metadata, output_dir=path, metadata_reviewed=True
    )


def test_export_is_deterministic(fitted, metadata, tmp_path):
    from meganorm.sharing import validate_bundle

    first = export(fitted[0], metadata, tmp_path / "one")
    second = export(fitted[0], metadata, tmp_path / "two")
    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.bundle_sha256 == hashlib.sha256(first.path.read_bytes()).hexdigest()
    assert validate_bundle(first.path).valid
    with zipfile.ZipFile(first.path) as z:
        assert set(z.namelist()) == {
            "manifest.json",
            *[f["path"] for f in json.loads(z.read("manifest.json"))["files"]],
        }
        assert all(b"PRIVATE_" not in z.read(n) for n in z.namelist())


def test_review_and_overwrite(fitted, metadata, tmp_path):
    from meganorm.sharing import export_model
    from meganorm.sharing.errors import PrivacyError, VersionConflictError

    with pytest.raises(PrivacyError):
        export_model(fitted[0], metadata=metadata, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()
    out = export(fitted[0], metadata, tmp_path / "out")
    before = out.path.read_bytes()
    with pytest.raises(VersionConflictError):
        export(fitted[0], metadata, tmp_path / "out")
    assert out.path.read_bytes() == before


def test_stage3_bundle_remains_loadable(fitted, metadata, tmp_path):
    """Adding adaptation metadata must not invalidate existing canonical archives."""
    from meganorm.sharing import load_backend, validate_bundle

    original = export(fitted[0], metadata, tmp_path / "original")
    legacy = tmp_path / "stage3.zip"
    with zipfile.ZipFile(original.path) as source, zipfile.ZipFile(legacy, "w") as dest:
        for member in source.infolist():
            content = source.read(member)
            if member.filename == "manifest.json":
                document = json.loads(content)
                document["lineage"].pop("settings", None)
                content = json.dumps(
                    document,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode()
            dest.writestr(member, content)
    assert validate_bundle(legacy).valid
    restored = load_backend(legacy)
    assert restored.response_vars == fitted[0].response_vars


@pytest.mark.parametrize(
    "name", ["../PRIVATE_ID", "/tmp/PRIVATE_ID", "model\\PRIVATE_ID", "extra.pkl"]
)
def test_unsafe_members(tmp_path, name):
    from meganorm.sharing import validate_bundle

    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, b"unsafe")
    result = validate_bundle(path)
    assert not result.valid and "PRIVATE_ID" not in repr(result)


def test_link_rejected(tmp_path):
    from meganorm.sharing import validate_bundle

    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        info = zipfile.ZipInfo("manifest.json")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, b"/etc/passwd")
    assert not validate_bundle(path).valid


def test_limits_and_tampering(fitted, metadata, tmp_path):
    from meganorm.sharing import validate_bundle, BundleLimits, load_backend
    from meganorm.sharing.errors import IntegrityError

    result = export(fitted[0], metadata, tmp_path / "out")
    for limits in [
        BundleLimits(max_archive_bytes=1),
        BundleLimits(max_extracted_bytes=1),
        BundleLimits(max_members=1),
        BundleLimits(max_json_bytes=1),
    ]:
        assert not validate_bundle(result.path, limits=limits).valid
    assert not validate_bundle(result.path, expected_sha256="0" * 64).valid
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(result.path) as src, zipfile.ZipFile(bad, "w") as dst:
        for n in src.namelist():
            dst.writestr(n, b"PRIVATE" if n == "LICENSE" else src.read(n))
    assert not validate_bundle(bad).valid
    with pytest.raises(IntegrityError):
        load_backend(bad)


def rewrite_bundle(source, dest, mutate):
    """Rehash a tampered payload so tests exercise validation beyond hashes."""
    from meganorm.sharing import canonical_manifest

    with zipfile.ZipFile(source) as z:
        contents = {n: z.read(n) for n in z.namelist()}
    mutate(contents)
    manifest = json.loads(contents["manifest.json"])
    for entry in manifest["files"]:
        content = contents[entry["path"]]
        entry.update(
            size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
        )
    contents["manifest.json"] = canonical_manifest(manifest)
    with zipfile.ZipFile(dest, "w") as z:
        for name in sorted(contents):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            z.writestr(info, contents[name])


def test_rehashed_unknown_json_rejected(fitted, metadata, tmp_path):
    from meganorm.sharing import validate_bundle

    bundle = export(fitted[0], metadata, tmp_path / "out")
    bad = tmp_path / "bad.zip"

    def mutate(contents):
        name = "model/roi/regression_model.json"
        d = json.loads(contents[name])
        d["model"]["PRIVATE_ID"] = [1, 2, 3]
        contents[name] = json.dumps(d).encode()

    rewrite_bundle(bundle.path, bad, mutate)
    result = validate_bundle(bad)
    assert not result.valid and "PRIVATE" not in repr(result)


def test_archive_preamble_rejected(fitted, metadata, tmp_path):
    from meganorm.sharing import validate_bundle

    bundle = export(fitted[0], metadata, tmp_path / "out")
    bad = tmp_path / "bad.zip"
    # zipfile supports executable/self-extracting prefixes; this format must not.
    with bad.open("wb") as stream:
        stream.write(b"PRIVATE_PARTICIPANT")
    with zipfile.ZipFile(bundle.path) as src, zipfile.ZipFile(bad, "a") as dst:
        for info in src.infolist():
            dst.writestr(info, src.read(info))
    assert not validate_bundle(bad).valid


def test_netcdf_trailing_content_rejected(fitted, metadata, tmp_path):
    from meganorm.sharing import validate_bundle

    if fitted[1] != "hbr":
        pytest.skip("HBR posterior only")
    bundle = export(fitted[0], metadata, tmp_path / "out")
    bad = tmp_path / "bad.zip"

    def mutate(contents):
        contents["model/roi/idata.nc"] += b"PRIVATE_PARTICIPANT"

    rewrite_bundle(bundle.path, bad, mutate)
    assert not validate_bundle(bad).valid


def test_hdf5_not_accepted_as_portable_posterior(fitted, metadata, tmp_path):
    from meganorm.sharing import validate_bundle

    if fitted[1] != "hbr":
        pytest.skip("HBR posterior only")
    bundle = export(fitted[0], metadata, tmp_path / "out")
    bad = tmp_path / "bad.zip"

    def mutate(contents):
        contents["model/roi/idata.nc"] = b"\x89HDF\r\n\x1a\nPRIVATE"

    rewrite_bundle(bundle.path, bad, mutate)
    assert not validate_bundle(bad).valid


def test_duplicate_manifest_keys(fitted, metadata, tmp_path):
    from meganorm.sharing import validate_bundle

    bundle = export(fitted[0], metadata, tmp_path / "out")
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bundle.path) as src, zipfile.ZipFile(bad, "w") as dst:
        for info in src.infolist():
            content = src.read(info)
            if info.filename == "manifest.json":
                content = content.replace(
                    b'"schema_version":1', b'"schema_version":1,"schema_version":1'
                )
            dst.writestr(info, content)
    assert not validate_bundle(bad).valid


def test_nonregular_input_does_not_block(tmp_path):
    import os
    import signal
    from meganorm.sharing import validate_bundle

    if not hasattr(os, "mkfifo") or not hasattr(signal, "SIGALRM"):
        pytest.skip("POSIX FIFO check")
    path = tmp_path / "pipe"
    os.mkfifo(path)

    def interrupted(*args):
        raise AssertionError("Validation blocked on a nonregular file")

    previous = signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, 1)
    try:
        assert not validate_bundle(path).valid
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_truncated_netcdf_reports_failure(fitted, metadata, tmp_path):
    from meganorm.sharing import validate_bundle

    if fitted[1] != "hbr":
        pytest.skip("HBR posterior only")
    bundle = export(fitted[0], metadata, tmp_path / "out")
    bad = tmp_path / "bad.zip"

    def mutate(contents):
        contents["model/roi/idata.nc"] = b"CDF\x02"

    rewrite_bundle(bundle.path, bad, mutate)
    assert not validate_bundle(bad).valid
