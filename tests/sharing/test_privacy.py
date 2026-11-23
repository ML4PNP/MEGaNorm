"""Reject unknown serialized state; exclude known local analysis state."""

import copy
import pytest
from .test_bundle import export


@pytest.mark.parametrize("target", ["regression", "scaler", "batch"])
def test_injected_state(fitted, metadata, tmp_path, target):
    from meganorm.sharing.errors import PrivacyError

    model = copy.deepcopy(fitted[0])
    if target == "regression":
        model["roi"].PRIVATE_ID = "PRIVATE_VALUE"
    if target == "scaler":
        model.inscalers["age"].PRIVATE_ID = "PRIVATE_VALUE"
    if target == "batch":
        model["roi"].be_maps = {"site": {"PRIVATE_ID": 0, "B": 1}}
    with pytest.raises(PrivacyError) as exc:
        export(model, metadata, tmp_path / "out")
    assert "PRIVATE" not in str(exc.value)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("attack", ["attribute", "variable", "coordinate", "group"])
def test_posterior_attacks(fitted, metadata, tmp_path, attack):
    import xarray as xr
    import numpy as np
    from meganorm.sharing.errors import PrivacyError

    if fitted[1] != "hbr":
        pytest.skip("HBR posterior only")
    model = copy.deepcopy(fitted[0])
    tree = model["roi"].idata
    if attack == "attribute":
        tree["posterior"].attrs["PRIVATE_ID"] = "PRIVATE_VALUE"
    if attack == "variable":
        tree["posterior"]["PRIVATE_ID"] = xr.DataArray(
            np.ones(24), dims=["observations"]
        )
    if attack == "coordinate":
        ds = tree["posterior"].to_dataset().assign_coords(site=["PRIVATE_ID", "B"])
        tree["posterior"] = xr.DataTree(ds)
    if attack == "group":
        tree["PRIVATE_ID"] = xr.DataTree(xr.Dataset())
    with pytest.raises(PrivacyError):
        export(model, metadata, tmp_path / "out")


def test_object_state_rejected(fitted, metadata, tmp_path):
    from meganorm.sharing.errors import PrivacyError

    model = copy.deepcopy(fitted[0])
    model["roi"].PRIVATE_OBJECT = object()
    with pytest.raises(PrivacyError):
        export(model, metadata, tmp_path / "out")


def test_posterior_array_budget(fitted, metadata, tmp_path):
    from meganorm.sharing import validate_bundle, BundleLimits

    if fitted[1] != "hbr":
        pytest.skip("HBR posterior only")
    result = export(fitted[0], metadata, tmp_path / "out")
    assert not validate_bundle(
        result.path, limits=BundleLimits(max_array_bytes=1)
    ).valid


def test_known_local_state_excluded(fitted, metadata, tmp_path):
    import zipfile

    model = copy.deepcopy(fitted[0])
    # NormativeModel.to_dict intentionally excludes local arbitrary attachments.
    model.local_participants = ["PRIVATE_ID"]
    if fitted[1] == "hbr":
        model["roi"].idata["posterior"].attrs["created_at"] = "PRIVATE_ID"
    result = export(model, metadata, tmp_path / "out")
    with zipfile.ZipFile(result.path) as z:
        assert all(b"PRIVATE_ID" not in z.read(n) for n in z.namelist())


def test_adapted_profile_not_silently_exported(fitted, metadata, tmp_path):
    from meganorm.sharing.errors import CompatibilityError

    metadata["lineage"] = {
        "operation": "calibrate",
        "parent": {
            "model_id": "parent/model",
            "version": "1.0.0",
            "bundle_sha256": "a" * 64,
        },
    }
    metadata["capabilities"] = ["predict"]
    with pytest.raises(CompatibilityError):
        export(fitted[0], metadata, tmp_path / "out")
