"""HBR bookkeeping may be absent, unset, or populated across PCNToolkit builds."""

import copy
import json
import zipfile

import numpy as np
import pytest

from .conftest import data
from .test_bundle import export

DIMENSIONS = {
    "slope_mu": 1,
    "mu_intercept_mu": 1,
    "site_sigma_intercept_mu": 1,
    "normalized_site_offset_intercept_mu_zerosum__": 1,
    "sigma": 1,
    "epsilon": 1,
    "delta": 1,
}


def with_dimensions(fitted, monkeypatch, constructor_has_cache):
    """Reproduce the reported fresh-None/fitted-dict native serialization."""
    from pcntoolkit import HBR

    if constructor_has_cache:
        original_init = HBR.__init__

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._parameter_dimensions = None

        monkeypatch.setattr(HBR, "__init__", init)
    model = copy.deepcopy(fitted[0])
    model.template_regression_model._parameter_dimensions = None
    model["roi"]._parameter_dimensions = DIMENSIONS.copy()
    return model


@pytest.mark.parametrize("fitted", ["hbr"], indirect=True)
@pytest.mark.parametrize("constructor_has_cache", [False, True])
def test_parameter_dimensions_roundtrip(
    fitted, metadata, tmp_path, monkeypatch, constructor_has_cache
):
    from meganorm.sharing import load_backend, validate_bundle

    model = with_dimensions(fitted, monkeypatch, constructor_has_cache)
    original = copy.deepcopy(model.to_dict())
    result = export(model, metadata, tmp_path / "out")
    repeated = export(model, metadata, tmp_path / "repeat")
    assert result.bundle_sha256 == repeated.bundle_sha256
    assert original == model.to_dict()
    assert model["roi"]._parameter_dimensions == DIMENSIONS
    assert validate_bundle(result.path).valid
    with zipfile.ZipFile(result.path) as z:
        state = json.loads(z.read("model/roi/regression_model.json"))
        assert state["model"]["_parameter_dimensions"] == DIMENSIONS
    restored = load_backend(result.path)
    native_data, restored_data = data("test"), data("test")
    model.predict(native_data)
    restored.predict(restored_data)
    for field in ["Yhat", "Z"]:
        np.testing.assert_allclose(
            native_data[field], restored_data[field], rtol=1e-8, atol=1e-8
        )


@pytest.mark.parametrize("fitted", ["hbr"], indirect=True)
@pytest.mark.parametrize("bad", ["PRIVATE_ID", True, 0, -1, 1.5])
def test_invalid_parameter_dimension_rejected(fitted, metadata, tmp_path, bad):
    from meganorm.sharing.errors import PrivacyError

    model = copy.deepcopy(fitted[0])
    model["roi"]._parameter_dimensions = dict(DIMENSIONS, slope_mu=bad)
    with pytest.raises(PrivacyError):
        export(model, metadata, tmp_path / "out")


@pytest.mark.parametrize("fitted", ["hbr"], indirect=True)
def test_unexpected_parameter_name_rejected(fitted, metadata, tmp_path):
    from meganorm.sharing.errors import PrivacyError

    model = copy.deepcopy(fitted[0])
    model["roi"]._parameter_dimensions = dict(DIMENSIONS, PRIVATE_ID=1)
    with pytest.raises(PrivacyError) as exc:
        export(model, metadata, tmp_path / "out")
    assert "PRIVATE_ID" not in str(exc.value)
