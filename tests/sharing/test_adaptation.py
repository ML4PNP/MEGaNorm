"""Native adaptation parity, immutable parents, and portable child privacy."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from .test_pcn_roundtrip import multi_fitted, multi_metadata, RESPONSES
from .test_bundle import rewrite_bundle

UNITS = {"age": "years", **{name: "V" for name in RESPONSES}}


def frame(split="calibration", sites=("C",)):
    rng = np.random.default_rng(2026 if split == "calibration" else 2027)
    table = pd.DataFrame(
        {
            "age": np.tile(np.linspace(25, 65, 12), len(sites)),
            "site": np.repeat(sites, 12),
        }
    )
    for i, name in enumerate(RESPONSES):
        table[name] = (i + 1) * (2 + 0.1 * table.age + rng.normal(0, 0.3, len(table)))
    table["participant_id"] = [f"PRIVATE_{split}_{i}" for i in range(len(table))]
    return table


def native_data(table):
    from pcntoolkit import NormData

    return NormData.from_dataframe(
        "calibration",
        table.copy(),
        covariates=["age"],
        response_vars=["theta", "alpha", "beta"],
        batch_effects=["site"],
        subject_ids="participant_id",
    )


@pytest.fixture
def parent(multi_fitted, multi_metadata, tmp_path):
    from meganorm.API import SharedNormativeModel
    from meganorm.sharing import export_model

    bundle = export_model(
        multi_fitted[0],
        metadata=multi_metadata,
        output_dir=tmp_path / "parent",
        metadata_reviewed=True,
    )
    return SharedNormativeModel.load(bundle.path), bundle


@pytest.mark.parametrize("operation", ["calibrate", "extend"])
def test_adaptation_matches_native_and_exports_only_child_state(
    parent, multi_fitted, tmp_path, monkeypatch, operation
):
    from meganorm.API import SharedNormativeModel
    from meganorm.sharing import validate_bundle
    import pymc as pm

    shared, bundle = parent
    # Seed only the test's native sampling boundary; production uses PCNToolkit.
    sample, pp = pm.sample, pm.sample_posterior_predictive
    monkeypatch.setattr(
        pm, "sample", lambda *a, **kw: sample(*a, **{**kw, "random_seed": 2026})
    )
    monkeypatch.setattr(
        pm,
        "sample_posterior_predictive",
        lambda *a, **kw: pp(*a, **{**kw, "random_seed": 2026}),
    )
    table = frame()
    original = table.copy(deep=True)
    archive = bundle.path.read_bytes()
    parent_state = copy.deepcopy(shared._backend.to_dict())
    heldout = frame("heldout")
    old_data = frame("old", ("A", "B"))
    old_prediction = shared.predict(old_data, units=UNITS)
    options = {"n_synth_samples": 48} if operation == "extend" else {}
    np.random.seed(2026)
    child = getattr(shared, operation)(
        table,
        units=UNITS,
        output_dir=tmp_path / "child",
        model_id="publisher/adapted",
        version="1.0.0",
        metadata_reviewed=True,
        **options,
    )
    assert isinstance(child, SharedNormativeModel)
    immediate = child.export(output_dir=tmp_path / "immediate", metadata_reviewed=True)
    assert immediate.bundle_sha256 == child.bundle_sha256
    assert set(child.capabilities) == {"predict"}
    assert {p.name for p in (tmp_path / "child").iterdir()} == {"model.zip"}
    lineage = child.manifest.lineage
    assert lineage.operation == operation
    assert lineage.parent.bundle_sha256 == bundle.bundle_sha256
    assert lineage.parent.model_id == shared.manifest.model_id
    assert lineage.parent.version == shared.manifest.version
    assert [c.label for c in lineage.settings.parent_categories] == ["A", "B"]
    if operation == "extend":
        assert lineage.settings.n_synth_samples == 48
    elif multi_fitted[1] == "hbr":
        assert lineage.settings.freedom == 1.0
    labels = [c.label for c in child.manifest.batch_effects[0].categories]
    assert set(labels) == ({"C"} if operation == "calibrate" else {"A", "B", "C"})

    direct_parent = copy.deepcopy(shared._backend)
    direct_parent.save_dir = str(tmp_path / "native-parent")
    np.random.seed(2026)
    native = getattr(
        direct_parent, "transfer" if operation == "calibrate" else "extend"
    )(native_data(table), save_dir=str(tmp_path / "native-child"), **options)
    native.savemodel = native.saveresults = native.saveplots = native.evaluate_model = (
        False
    )
    checks = [heldout] + ([old_data] if operation == "extend" else [])
    for evaluation in checks:
        nd = native_data(evaluation)
        native.predict(nd)
        actual = child.predict(evaluation, units=UNITS)
        for attr, name in [("predictions", "Yhat"), ("z_scores", "Z")]:
            expected = nd[name].sel(response_vars=["theta", "alpha", "beta"]).values
            np.testing.assert_allclose(
                getattr(actual, attr).values, expected, atol=1e-8, rtol=1e-8
            )
    if operation == "calibrate":
        with pytest.raises(ValueError):
            child.predict(old_data, units=UNITS)
    with pytest.raises(ValueError, match="overlap"):
        child.predict(table, units=UNITS)
    with pytest.raises(ValueError):
        child.extend(
            frame("next", ("D",)),
            units=UNITS,
            output_dir=tmp_path / "grandchild",
            model_id="publisher/grandchild",
            version="1.0.0",
            metadata_reviewed=True,
        )

    again = child.export(output_dir=tmp_path / "again", metadata_reviewed=True)
    assert validate_bundle(again.path).valid
    assert again.bundle_sha256 == child.bundle_sha256
    loaded = SharedNormativeModel.load(again.path)
    pd.testing.assert_frame_equal(
        loaded.predict(heldout, units=UNITS).z_scores,
        child.predict(heldout, units=UNITS).z_scores,
    )
    heldout.to_csv(tmp_path / "heldout.csv", index=False)
    np.save(
        tmp_path / "expected.npy", child.predict(heldout, units=UNITS).z_scores.values
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import numpy as np
import pandas as pd
from meganorm.API import SharedNormativeModel
m = SharedNormativeModel.load('again/model.zip')
data = pd.read_csv('heldout.csv')
r = m.predict(data, units={'age':'years','alpha':'V','beta':'V','theta':'V'})
np.testing.assert_allclose(r.z_scores.values, np.load('expected.npy'), atol=1e-8, rtol=1e-8)
""",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    with zipfile.ZipFile(again.path) as z:
        for name in z.namelist():
            assert b"PRIVATE_" not in z.read(name)
        assert len([n for n in z.namelist() if n.startswith("model/")]) == (
            7 if multi_fitted[1] == "hbr" else 4
        )
    assert shared._backend.to_dict() == parent_state
    assert bundle.path.read_bytes() == archive
    pd.testing.assert_frame_equal(table, original)
    pd.testing.assert_frame_equal(
        shared.predict(old_data, units=UNITS).z_scores, old_prediction.z_scores
    )

    # The adapter must still reject extra state even when lineage says adapted.
    def inject(contents):
        key = "model/z/regression_model.json"
        state = json.loads(contents[key])
        state["model"]["PRIVATE_calibration_rows"] = [1, 2]
        contents[key] = json.dumps(state).encode()

    forged = tmp_path / "forged.zip"
    rewrite_bundle(again.path, forged, inject)
    assert not validate_bundle(forged).valid


@pytest.mark.parametrize("multi_fitted", ["hbr"], indirect=True)
def test_calibration_with_single_site_parameter_cache(parent, tmp_path, monkeypatch):
    """A single site's zero-sum effect has no unconstrained parameters."""
    from pcntoolkit import HBR
    from meganorm.API import SharedNormativeModel
    from meganorm.sharing import validate_bundle

    original_transfer = HBR.transfer

    def transfer_with_cache(self, *args, **kwargs):
        child = original_transfer(self, *args, **kwargs)
        # Reproduce development builds' optional cache using native PyMC shapes,
        # including the empty transformed zero-sum vector for a single site.
        child._parameter_dimensions = {
            name: int(value.size)
            for name, value in child.pymc_model.initial_point().items()
        }
        assert (
            child._parameter_dimensions["normalized_site_offset_intercept_mu_zerosum__"]
            == 0
        )
        return child

    monkeypatch.setattr(HBR, "transfer", transfer_with_cache)
    shared, _ = parent
    child = shared.calibrate(
        frame(),
        units=UNITS,
        output_dir=tmp_path / "child",
        model_id="publisher/single-site-child",
        version="1.0.0",
        metadata_reviewed=True,
    )
    bundle = child.export(output_dir=tmp_path / "again", metadata_reviewed=True)
    assert validate_bundle(bundle.path).valid
    assert bundle.bundle_sha256 == child.bundle_sha256
    with zipfile.ZipFile(bundle.path) as archive:
        for response in child.manifest.responses:
            state = json.loads(
                archive.read(f"model/{response.storage_key}/regression_model.json")
            )["model"]
            assert (
                state["_parameter_dimensions"][
                    "normalized_site_offset_intercept_mu_zerosum__"
                ]
                == 0
            )
    restored = SharedNormativeModel.load(bundle.path)
    heldout = frame("heldout")
    pd.testing.assert_frame_equal(
        restored.predict(heldout, units=UNITS).z_scores,
        child.predict(heldout, units=UNITS).z_scores,
    )


@pytest.mark.parametrize("operation", ["calibrate", "extend"])
def test_invalid_adaptation_is_rejected_before_native(
    parent, tmp_path, monkeypatch, operation
):
    shared, bundle = parent
    link = tmp_path / "linked-output"
    link.symlink_to(tmp_path / "link-target", target_is_directory=True)

    def unexpected(*a, **kw):
        pytest.fail("Invalid request reached native adaptation")

    monkeypatch.setattr(type(shared._backend), "transfer", unexpected)
    monkeypatch.setattr(type(shared._backend), "extend", unexpected)
    cases = [
        ({"site": "A"}, {}),
        ({"site": "a"}, {}),
        ({"site": " C "}, {}),
        ({"site": "PRIVATE_calibration_0"}, {}),
        ({"age": np.nan}, {}),
        ({"participant_id": "duplicate"}, {}),
        ({}, {"units": {}}),
        ({}, {"metadata_reviewed": False}),
        (
            {},
            {"model_id": shared.manifest.model_id, "version": shared.manifest.version},
        ),
        ({}, {"output_dir": bundle.path.parent / "nested"}),
        ({}, {"output_dir": link}),
        ({}, {"unexpected_option": 5}),
    ]
    cases += (
        [({}, {"freedom": 0}), ({}, {"freedom": float("nan")})]
        if operation == "calibrate"
        else [({}, {"n_synth_samples": True}), ({}, {"n_synth_samples": 0})]
    )
    for changes, overrides in cases:
        table = frame().assign(**changes)
        options = dict(
            units=UNITS,
            output_dir=tmp_path / "rejected",
            model_id="publisher/child",
            version="1.0.0",
            metadata_reviewed=True,
        )
        options.update(overrides)
        with pytest.raises((ValueError, TypeError)) as error:
            getattr(shared, operation)(table, **options)
        assert "PRIVATE_" not in str(error.value)
        assert not (tmp_path / "rejected").exists()


def test_failed_adaptation_cleans_outputs_and_preserves_parent(
    parent, tmp_path, monkeypatch
):
    shared, bundle = parent
    original = copy.deepcopy(shared._backend.to_dict())
    archive = bundle.path.read_bytes()
    work_paths = []

    def fail(model, data, save_dir, **kw):
        from pathlib import Path

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        work_paths.append(Path(save_dir))
        (Path(save_dir) / "PRIVATE.csv").write_text("PRIVATE_ROWS")
        model.covariates = ["mutated"]
        raise RuntimeError("PRIVATE_FAILURE")

    monkeypatch.setattr(type(shared._backend), "transfer", fail)
    with pytest.raises(RuntimeError) as error:
        shared.calibrate(
            frame(),
            units=UNITS,
            output_dir=tmp_path / "failed",
            model_id="publisher/child",
            version="1.0.0",
            metadata_reviewed=True,
        )
    assert "PRIVATE_" not in str(error.value)
    assert error.value.__suppress_context__
    assert not (tmp_path / "failed").exists()
    assert shared._backend.to_dict() == original
    assert bundle.path.read_bytes() == archive
    assert work_paths and all(not p.exists() for p in work_paths)


def test_native_adaptation_rejects_path_like_response_names(
    parent, tmp_path, monkeypatch
):
    """Valid bundle display names must not become native persistence paths."""
    from meganorm.API import SharedNormativeModel
    from meganorm.sharing import export_model

    shared, _ = parent

    def unexpected(*args, **kwargs):
        pytest.fail("Unsafe response name reached native adaptation")

    monkeypatch.setattr(type(shared._backend), "transfer", unexpected)
    monkeypatch.setattr(type(shared._backend), "extend", unexpected)
    for i, name in enumerate(["../PRIVATE_TARGET", "normative_model.json"]):
        backend = copy.deepcopy(shared._backend)
        backend.response_vars = [
            name if r == "theta" else r for r in backend.response_vars
        ]
        backend.regression_models[name] = backend.regression_models.pop("theta")
        backend.outscalers[name] = backend.outscalers.pop("theta")
        metadata = shared.manifest.model_dump(mode="json", exclude={"files"})
        metadata["responses"][0]["name"] = name
        bundle = export_model(
            backend,
            metadata=metadata,
            output_dir=tmp_path / f"unsafe-{i}",
            metadata_reviewed=True,
        )
        loaded = SharedNormativeModel.load(bundle.path)
        for operation in ("calibrate", "extend"):
            with pytest.raises(ValueError) as error:
                getattr(loaded, operation)(
                    frame().rename(columns={"theta": name}),
                    units={name if k == "theta" else k: v for k, v in UNITS.items()},
                    output_dir=tmp_path / "rejected",
                    model_id="publisher/child",
                    version="1.0.0",
                    metadata_reviewed=True,
                )
            assert "PRIVATE_" not in str(error.value)
            assert error.value.__suppress_context__
            assert not (tmp_path / "rejected").exists()
