"""Explicitly enabled synthetic public-Hub smoke test; never ordinary CI writes."""

import copy
import json
import os
import time

import pytest


@pytest.mark.hf_live
@pytest.mark.skipif(
    os.environ.get("MEGANORM_HF_LIVE") != "1",
    reason="Set MEGANORM_HF_LIVE=1 only for an authorized synthetic test repository.",
)
def test_live_synthetic_versions_and_read_only_denial(manifest, tmp_path):
    pytest.importorskip("huggingface_hub")
    from pcntoolkit import NormativeModel
    from meganorm.sharing import export_model
    from meganorm.sharing.errors import AuthorizationError, VersionConflictError
    from meganorm.sharing.huggingface import HuggingFaceTransport
    from .conftest import data, hbr

    repo_id = os.environ.get("MEGANORM_HF_TEST_REPO")
    if not repo_id or not os.environ.get("MEGANORM_HF_READ_TOKEN"):
        pytest.fail(
            "Live test requires MEGANORM_HF_TEST_REPO and MEGANORM_HF_READ_TOKEN."
        )
    writer = HuggingFaceTransport()  # HF_TOKEN or the SDK's local credential store
    reader = HuggingFaceTransport(token=os.environ["MEGANORM_HF_READ_TOKEN"])
    writer.preflight(repo_id, write=True)
    reader.preflight(repo_id)
    # Check the supplied read credential before making any remote mutation.
    with pytest.raises(AuthorizationError):
        reader.preflight(repo_id, write=True)

    model = NormativeModel(
        hbr(),
        save_dir=str(tmp_path / "synthetic-native"),
        savemodel=False,
        saveresults=False,
        saveplots=False,
        evaluate_model=False,
    )
    model.fit(data())
    metadata = copy.deepcopy(manifest)
    metadata.pop("files")
    metadata["adapter"]["profile"] = "hbr-shashb-linear-site"
    metadata["model_id"] = repo_id
    metadata["metadata"]["title"] = "MEGaNorm synthetic transport test"
    base = time.time_ns()
    receipts = []
    for index in range(2):
        metadata["version"] = f"0.0.{base + index}"
        bundle = export_model(
            model,
            metadata=metadata,
            output_dir=tmp_path / f"export-{index}",
            metadata_reviewed=True,
        )
        options = dict(
            repo_id=repo_id, version=metadata["version"], metadata_reviewed=True
        )
        with pytest.raises(AuthorizationError):
            reader.upload_bundle(bundle.path, **options)
        receipt = writer.upload_bundle(bundle.path, **options)
        again = writer.upload_bundle(bundle.path, **options)
        assert again.bundle_sha256 == receipt.bundle_sha256
        downloaded = reader.download_bundle(
            receipt, output_dir=tmp_path / f"download-{index}"
        )
        assert downloaded.path.read_bytes() == bundle.path.read_bytes()
        assert writer.read_descriptor(repo_id, metadata["version"]) is None
        conflict_metadata = copy.deepcopy(metadata)
        conflict_metadata["metadata"][
            "description"
        ] = "Different synthetic content at the same version."
        conflict = export_model(
            model,
            metadata=conflict_metadata,
            output_dir=tmp_path / f"conflict-{index}",
            metadata_reviewed=True,
        )
        with pytest.raises(VersionConflictError):
            writer.upload_bundle(conflict.path, **options)
        receipts.append(receipt.model_dump(mode="json"))
    # Public identifiers only; no tokens, participant records or invented archive DOI.
    report = json.dumps(receipts, indent=2)
    (tmp_path / "live-receipts.json").write_text(report)
    print(report)
