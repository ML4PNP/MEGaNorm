"""Real native prediction and z-score parity across a portable round trip."""

import copy
from pathlib import Path
import numpy as np
from .conftest import data
from .test_bundle import export


def test_native_roundtrip(fitted, metadata, tmp_path):
    from meganorm.sharing import load_backend

    model = fitted[0]
    original = copy.deepcopy(model.to_dict())
    before_files = {
        p.relative_to(model.save_dir): p.read_bytes()
        for p in Path(model.save_dir).rglob("*")
        if p.is_file()
    }
    result = export(model, metadata, tmp_path / "out")
    assert original == model.to_dict()
    assert before_files == {
        p.relative_to(model.save_dir): p.read_bytes()
        for p in Path(model.save_dir).rglob("*")
        if p.is_file()
    }
    loaded = load_backend(result.path, expected_sha256=result.bundle_sha256)
    a = data("test")
    b = data("test")
    model.predict(a)
    loaded.predict(b)
    for field in ["Yhat", "Z"]:
        assert np.isfinite(b[field].values).all()
        np.testing.assert_allclose(
            a[field].values, b[field].values, atol=1e-8, rtol=1e-8
        )
    assert loaded.batch_effect_counts == model.batch_effect_counts
    assert not loaded.savemodel and not loaded.saveresults and not loaded.saveplots


def test_fresh_process_load(fitted, metadata, tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    model = fitted[0]
    bundle = export(model, metadata, tmp_path / "out")
    expected = data("test")
    model.predict(expected)
    np.savez(tmp_path / "expected.npz", Yhat=expected.Yhat.values, Z=expected.Z.values)
    code = """
import numpy as np
import pandas as pd
from pcntoolkit import NormData
from meganorm.sharing import load_backend
rng=np.random.default_rng(2027)
f=pd.DataFrame({'age':np.tile(np.linspace(20,70,12),2),'site':np.repeat(['A','B'],12)})
f['roi']=1+.1*f.age+rng.normal(0,.3,len(f));f['participant']=[f'PRIVATE_test_{i}' for i in range(len(f))]
d=NormData.from_dataframe('test',f,covariates=['age'],response_vars=['roi'],batch_effects=['site'],subject_ids='participant')
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
