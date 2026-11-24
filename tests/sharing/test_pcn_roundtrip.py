"""Real native prediction and z-score parity across a portable round trip."""

import copy
from pathlib import Path
import numpy as np
import pytest
from .conftest import data, hbr
from .test_bundle import export, rewrite_bundle

RESPONSES = ["alpha", "beta", "theta"]


@pytest.fixture(scope="module", params=["hbr", "blr"])
def multi_fitted(request, tmp_path_factory):
    """Three distinct IDPs sharing PCNToolkit's ordinary model container."""
    from pcntoolkit import BLR, NormativeModel

    model = NormativeModel(
        hbr() if request.param == "hbr" else BLR(),
        save_dir=str(tmp_path_factory.mktemp("multi-original")),
        savemodel=False,
        saveresults=False,
        saveplots=False,
        evaluate_model=False,
    )
    model.fit(data(responses=RESPONSES))
    return model, request.param


@pytest.fixture
def multi_metadata(manifest, multi_fitted):
    """Use manifest, training and directory orders that differ deliberately."""
    manifest.pop("files")
    if multi_fitted[1] == "hbr":
        manifest["adapter"]["profile"] = "hbr-shashb-linear-site"
    manifest["responses"] = [
        {"name": name, "unit": "V", "storage_key": key}
        for name, key in zip(["theta", "alpha", "beta"], ["z", "a", "m"])
    ]
    return manifest


def test_native_roundtrip(multi_fitted, multi_metadata, tmp_path):
    from meganorm.sharing import load_backend

    model = multi_fitted[0]
    original = copy.deepcopy(model.to_dict())
    before_files = {
        p.relative_to(model.save_dir): p.read_bytes()
        for p in Path(model.save_dir).rglob("*")
        if p.is_file()
    }
    result = export(model, multi_metadata, tmp_path / "out")
    assert original == model.to_dict()
    assert before_files == {
        p.relative_to(model.save_dir): p.read_bytes()
        for p in Path(model.save_dir).rglob("*")
        if p.is_file()
    }
    loaded = load_backend(result.path, expected_sha256=result.bundle_sha256)
    from pcntoolkit import NormativeModel
    import zipfile

    assert type(loaded) is NormativeModel
    order = [response["name"] for response in multi_metadata["responses"]]
    assert loaded.response_vars == order
    assert list(loaded.regression_models) == list(loaded.outscalers) == order
    for name in order:
        assert loaded.outscalers[name].to_dict() == model.outscalers[name].to_dict()
    repeated = export(model, multi_metadata, tmp_path / "again")
    assert repeated.bundle_sha256 == result.bundle_sha256
    with zipfile.ZipFile(result.path) as archive:
        model_files = [name for name in archive.namelist() if name.startswith("model/")]
        assert len(model_files) == (2 if multi_fitted[1] == "hbr" else 1) * 3 + 1
        assert all(b"PRIVATE_" not in archive.read(name) for name in archive.namelist())
    a = data("test", responses=RESPONSES)
    b = data("test", responses=RESPONSES)
    model.predict(a)
    loaded.predict(b)
    for field in ["Yhat", "Z"]:
        assert np.isfinite(b[field].values).all()
        np.testing.assert_allclose(
            a[field].values, b[field].values, atol=1e-8, rtol=1e-8
        )
    assert loaded.batch_effect_counts == model.batch_effect_counts
    assert not loaded.savemodel and not loaded.saveresults and not loaded.saveplots


def test_fresh_process_load(multi_fitted, multi_metadata, tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    model = multi_fitted[0]
    bundle = export(model, multi_metadata, tmp_path / "out")
    expected = data("test", responses=RESPONSES)
    model.predict(expected)
    np.savez(tmp_path / "expected.npz", Yhat=expected.Yhat.values, Z=expected.Z.values)
    code = """
import numpy as np
import pandas as pd
from pcntoolkit import NormData
from meganorm.sharing import load_backend
rng=np.random.default_rng(2027)
f=pd.DataFrame({'age':np.tile(np.linspace(20,70,12),2),'site':np.repeat(['A','B'],12)})
for i,response in enumerate(['alpha','beta','theta']):
    f[response]=(i+1)*(1+(.1+.02*i)*f.age+rng.normal(0,.3,len(f)))
f['participant']=[f'PRIVATE_test_{i}' for i in range(len(f))]
d=NormData.from_dataframe('test',f,covariates=['age'],response_vars=['alpha','beta','theta'],batch_effects=['site'],subject_ids='participant')
m=load_backend('out/model.zip');m.predict(d)
expected=np.load('expected.npz')
for field in ['Yhat','Z']:np.testing.assert_allclose(d[field].values,expected[field],atol=1e-8,rtol=1e-8)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_multi_response_validation(multi_fitted, multi_metadata, tmp_path, monkeypatch):
    """Check later IDPs too, with one array budget across the whole collection."""
    import json
    from meganorm.sharing import BundleLimits, load_backend, validate_bundle
    from meganorm.sharing.errors import PrivacyError

    model, kind = multi_fitted
    bundle = export(model, multi_metadata, tmp_path / "valid")
    assert validate_bundle(bundle.path).valid
    last = multi_metadata["responses"][-1]
    with monkeypatch.context() as patch:
        patch.setattr(model[last["name"]], "PRIVATE_records", [1], raising=False)
        with pytest.raises(PrivacyError):
            export(model, multi_metadata, tmp_path / "unsafe")
    incomplete = copy.deepcopy(multi_metadata)
    incomplete["responses"].pop()
    with pytest.raises(PrivacyError):
        export(model, incomplete, tmp_path / "incomplete")

    def inject(contents):
        path = f"model/{last['storage_key']}/regression_model.json"
        state = json.loads(contents[path])
        state["model"]["PRIVATE_records"] = [1]
        contents[path] = json.dumps(state).encode()

    forged = tmp_path / "forged.zip"
    rewrite_bundle(bundle.path, forged, inject)
    report = validate_bundle(forged)
    assert not report.valid and "PRIVATE_records" not in repr(report)
    with pytest.raises(PrivacyError):
        load_backend(forged)

    if kind == "hbr":
        # Each individual posterior fits, but their combined size does not.
        sizes = [
            model[name].idata["posterior"].to_dataset().nbytes for name in RESPONSES
        ]
        budget = 2 * max(sizes)
        assert max(sizes) < budget < sum(sizes)
        limits = BundleLimits(max_array_bytes=budget)
        assert not validate_bundle(bundle.path, limits=limits).valid
        with pytest.raises(PrivacyError):
            load_backend(bundle.path, limits=limits)
        from meganorm.sharing import export_model

        with pytest.raises(PrivacyError):
            export_model(
                model,
                metadata=multi_metadata,
                output_dir=tmp_path / "over-budget",
                metadata_reviewed=True,
                limits=limits,
            )
