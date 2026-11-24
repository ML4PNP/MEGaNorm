"""Model-only reconstruction for the two audited PCNToolkit profiles.

Accept only declared JSON fields and posterior dimensions. Never copy an analysis
folder or deserialize a class selected by unvalidated input. Numeric scaler and
site summaries are intentionally retained; observation-level state is excluded.
"""

from __future__ import annotations

import copy
import json
import hashlib
import tempfile
import math
from pathlib import Path

import numpy as np
import xarray as xr

from .errors import CompatibilityError, PrivacyError

NUMBER = object()
COUNT = object()


def reject():
    """Reject state without reflecting submitted names, values, or paths."""
    raise PrivacyError("Model state does not match the approved sharing profile.")


def primitive(value):
    """Convert native numeric containers into data-only JSON-compatible values."""
    if isinstance(value, np.generic):
        return primitive(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            reject()
        return primitive(value.tolist())
    if type(value) in (list, tuple):
        return [primitive(v) for v in value]
    if type(value) is dict:
        if not all(isinstance(k, str) for k in value):
            reject()
        return {str(k): primitive(v) for k, v in value.items()}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    reject()


def match(value, pattern):
    """Check an explicit bounded tree, including exact keys and array lengths."""
    if pattern is NUMBER:
        if type(value) not in (int, float) or not math.isfinite(value):
            reject()
    elif pattern is COUNT:
        if type(value) is not int or not 0 < value <= 10**9:
            reject()
    elif type(pattern) is dict:
        if type(value) is not dict or value.keys() != pattern.keys():
            reject()
        for key in pattern:
            match(value[key], pattern[key])
    elif type(pattern) is list:
        if type(value) is not list or len(value) != len(pattern):
            reject()
        for actual, expected in zip(value, pattern):
            match(actual, expected)
    elif type(pattern) in (int, float):
        if type(value) not in (int, float) or value != pattern:
            reject()
    elif type(value) is not type(pattern) or value != pattern:
        reject()


def profile_model(profile):
    """Construct trusted templates independently of submitted configuration."""
    from pcntoolkit import BLR, HBR, SHASHbLikelihood, make_prior

    if profile == "blr-default-site":
        return BLR()
    mu = make_prior(
        linear=True,
        intercept=make_prior(random=True),
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 1.0)),
    )
    sigma = make_prior(
        dist_name="Normal",
        dist_params=(0.0, 1.0),
        mapping="softplus",
        mapping_params=(0.0, 1.0),
    )
    epsilon = make_prior(dist_name="Normal", dist_params=(0.0, 0.5))
    delta = make_prior(
        dist_name="Normal",
        dist_params=(1.0, 0.5),
        mapping="softplus",
        mapping_params=(0.0, 1.0, 0.6),
    )
    return HBR(
        likelihood=SHASHbLikelihood(mu, sigma, epsilon, delta),
        nuts_sampler="pymc",
        cores=1,
        progressbar=False,
    )


def model_pattern(manifest, fitted, state=None):
    """Configuration allowlist plus learned parameters and optional HBR counts.

    Some PCNToolkit builds serialize a parameter-dimension cache as None before
    fitting and a dictionary afterwards. Check its known names and integer
    counts explicitly instead of requiring equality with an unfitted template.
    """
    profile = manifest.adapter.profile
    pattern = primitive(profile_model(profile).to_dict())
    pattern.pop("_parameter_dimensions", None)
    name = manifest.responses[0].name if fitted else "template"
    pattern.update(
        name=name, _name=name, is_fitted=fitted, is_from_dict=False, transfered=False
    )
    batch = manifest.batch_effects[0]
    mapping = {batch.name: {c.label: c.code for c in batch.categories}}
    if type(state) is dict and "_parameter_dimensions" in state:
        if profile != "hbr-shashb-linear-site":
            reject()
        pattern["_parameter_dimensions"] = None
        if fitted and state["_parameter_dimensions"] is not None:
            pattern["_parameter_dimensions"] = {
                name: COUNT
                for name in (
                    "slope_mu",
                    "mu_intercept_mu",
                    f"{batch.name}_sigma_intercept_mu",
                    f"normalized_{batch.name}_offset_intercept_mu_zerosum__",
                    "sigma",
                    "epsilon",
                    "delta",
                )
            }
    if profile == "blr-default-site":
        if fitted:
            pattern["basis_function_mean"].update(
                is_fitted=True, min=NUMBER, max=NUMBER
            )
            pattern.update(
                hyp=[NUMBER, NUMBER],
                be_maps=mapping,
                D=2,
                var_D=1,
                n_hyp=2,
                A=[[NUMBER, NUMBER], [NUMBER, NUMBER]],
                m=[NUMBER, NUMBER],
            )
    else:
        for key in ["draws", "tune", "cores", "chains", "vi_iterations", "vi_draws"]:
            pattern[key] = COUNT
        pattern["be_maps"] = mapping if fitted else None
        if fitted:
            mu = pattern["likelihood"]["mu"]
            mu["slope"]["_dims"] = mu["slope"]["dims"] = ["mu_covariates"]
            mu["basis_function"].update(is_fitted=True, min=NUMBER, max=NUMBER)
            mu.update(covariate_dims=["mu_covariates_0"], one_dimensional=True)
            sigma = copy.deepcopy(mu["intercept"]["sigma"])
            sigma["name"] = f"{batch.name}_sigma_intercept_mu"
            mu["intercept"][f"{batch.name}_sigma"] = sigma
    return pattern


def scaler_pattern():
    return dict(
        scaler_type="standardize",
        adjust_outliers=False,
        m=NUMBER,
        s=NUMBER,
        min=NUMBER,
        max=NUMBER,
    )


def normalized_model(state, name):
    """Exclude known local diagnostics and normalize nonstatistical names/flags."""
    result = primitive(state)
    for key in ["N", "nlZ", "dnlZ", "idata_path"]:
        result.pop(key, None)
    result["name"] = result["_name"] = name
    if type(result.get("is_from_dict")) is not bool:
        reject()
    result["is_from_dict"] = False
    return result


def validate_state(top, regression, manifest):
    """Check all stored JSON fields before native deserialization is permitted."""
    if manifest.lineage.operation != "fit":
        raise CompatibilityError("Adapted-model export is not enabled by this adapter.")
    if type(top) is not dict or type(regression) is not dict:
        reject()
    cov = manifest.covariates[0].name
    batch = manifest.batch_effects[0]
    labels = [c.label for c in batch.categories]
    counts = top.get("inverse_batch_effect_counts", {}).get(batch.name, {})
    if type(counts) is not dict or len(counts) != len(labels):
        reject()
    indices = []
    for key, label in counts.items():
        parts = key.split("_")
        if len(parts) != 2 or not all(p.isdecimal() for p in parts):
            reject()
        index, count = map(int, parts)
        if key != f"{index}_{count}" or not 0 < count <= 10**9:
            reject()
        indices.append(index)
    if sorted(indices) != list(range(len(labels))) or sorted(counts.values()) != sorted(
        labels
    ):
        reject()
    ranges = {"min": NUMBER, "max": NUMBER}
    expected = dict(
        name=None,
        save_dir=".",
        savemodel=False,
        saveresults=False,
        saveplots=False,
        evaluate_model=False,
        template_regression_model=model_pattern(
            manifest, False, top.get("template_regression_model")
        ),
        inscalers={cov: scaler_pattern()},
        is_fitted=True,
        inscaler="standardize",
        outscaler="standardize",
        y_transform=None,
        ptk_version=manifest.adapter.pcntoolkit_version,
        covariates=[cov],
        unique_batch_effects={batch.name: labels},
        inverse_batch_effect_counts={batch.name: counts},
        batch_effects_maps={
            batch.name: {f"{i}_{i}": label for i, label in enumerate(labels)}
        },
        batch_effect_covariate_ranges={
            batch.name: {label: {cov: ranges} for label in labels}
        },
        covariate_ranges={cov: ranges},
    )
    match(top, expected)
    match(
        regression,
        {
            "model": model_pattern(manifest, True, regression.get("model")),
            "outscaler": scaler_pattern(),
        },
    )
    for scaler in [top["inscalers"][cov], regression["outscaler"]]:
        if scaler["s"] <= 0 or scaler["min"] > scaler["max"]:
            reject()
    for r in [
        top["covariate_ranges"][cov],
        *[v[cov] for v in top["batch_effect_covariate_ranges"][batch.name].values()],
    ]:
        if r["min"] > r["max"]:
            reject()
    if manifest.adapter.profile == "blr-default-site":
        a = np.asarray(regression["model"]["A"])
        if not np.allclose(a, a.T) or np.linalg.eigvalsh(a).min() <= 0:
            reject()


KNOWN_ATTRS = {
    "created_at",
    "creation_library",
    "creation_library_version",
    "creation_library_language",
    "inference_library",
    "inference_library_version",
    "sample_dims",
    "sampling_time",
    "tuning_steps",
}


def posterior_dataset(ds, manifest, *, source, max_bytes):
    """Rebuild only audited numeric variables and reviewed dimension coordinates."""
    if not set(ds.attrs) <= (KNOWN_ATTRS if source else set()):
        reject()
    batch = manifest.batch_effects[0]
    labels = [c.label for c in batch.categories]
    offset = f"{batch.name}_offset_intercept_mu_dim_0"
    sizes = ds.sizes
    if not sizes.get("chain") or not sizes.get("draw") or ds.nbytes > max_bytes:
        reject()
    coords = {
        "chain": np.arange(sizes["chain"]),
        "draw": np.arange(sizes["draw"]),
        "mu_covariates": ["mu_covariates_0"],
        batch.name: labels,
        offset: np.arange(len(labels)),
    }
    dims = {
        n: ("chain", "draw")
        for n in [
            "mu_intercept_mu",
            f"{batch.name}_sigma_intercept_mu",
            "sigma",
            "epsilon",
            "delta",
        ]
    }
    dims.update(slope_mu=("chain", "draw", "mu_covariates"))
    dims[f"normalized_{batch.name}_offset_intercept_mu"] = ("chain", "draw", batch.name)
    dims[f"{batch.name}_offset_intercept_mu"] = ("chain", "draw", offset)
    if set(ds.data_vars) != set(dims) or set(ds.coords) != set(coords):
        reject()
    for n, values in coords.items():
        if (
            ds[n].dims != (n,)
            or ds[n].attrs
            or not np.array_equal(ds[n].values, values)
        ):
            reject()
    data = {}
    for n in sorted(dims):
        v = ds[n]
        if v.attrs or v.dims != dims[n] or v.dtype.kind not in "iuf":
            reject()
        values = v.values.copy()
        if not np.isfinite(values).all():
            reject()
        data[n] = (dims[n], values)
    return xr.Dataset(data, coords=coords)


def native_posterior(tree, manifest, max_bytes):
    """Exclude known native observation/diagnostic groups, reject unknown groups."""
    allowed = {
        "posterior",
        "observed_data",
        "constant_data",
        "sample_stats",
        "log_likelihood",
        "posterior_predictive",
        "predictions",
    }
    if (
        tree is None
        or not set(tree.children) <= allowed
        or "posterior" not in tree.children
        or tree.attrs
        or tree.data_vars
    ):
        reject()
    if tree["posterior"].children:
        reject()
    return posterior_dataset(
        tree["posterior"].to_dataset(), manifest, source=True, max_bytes=max_bytes
    )


def export_state(model, manifest, root: Path, max_bytes):
    """Write only newly reconstructed state; never save or change the source model."""
    from pcntoolkit import NormativeModel

    if type(model) is not NormativeModel or model.is_fitted is not True:
        raise CompatibilityError("Supply a fitted PCNToolkit NormativeModel.")
    response = manifest.responses[0]
    cls = type(profile_model(manifest.adapter.profile))
    if (
        set(model.regression_models) != {response.name}
        or type(model[response.name]) is not cls
        or type(model.template_regression_model) is not cls
    ):
        reject()
    for scaler in list(model.inscalers.values()) + list(model.outscalers.values()):
        if set(vars(scaler)) != {"adjust_outliers", "m", "s", "min", "max"}:
            reject()
    top = primitive(model.to_dict())
    top.update(
        name=None,
        save_dir=".",
        savemodel=False,
        saveresults=False,
        saveplots=False,
        evaluate_model=False,
    )
    top["template_regression_model"] = normalized_model(
        top["template_regression_model"], "template"
    )
    regression = {
        "model": normalized_model(model[response.name].to_dict(), response.name),
        "outscaler": primitive(model.outscalers[response.name].to_dict()),
    }
    validate_state(top, regression, manifest)
    sub = root / "model" / response.storage_key
    sub.mkdir(parents=True)
    for path, state in [
        (root / "model/normative_model.json", top),
        (sub / "regression_model.json", regression),
    ]:
        path.write_text(
            json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
    if manifest.adapter.profile == "hbr-shashb-linear-site":
        ds = native_posterior(model[response.name].idata, manifest, max_bytes)
        write_posterior(ds, sub / "idata.nc")


def write_posterior(ds, path):
    """Write the deterministic, attribute-free portable NetCDF3 representation."""
    ds.to_netcdf(
        path,
        engine="scipy",
        format="NETCDF3_64BIT",
        encoding={n: {"_FillValue": None} for n in ds.variables},
    )


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_posterior(path, manifest, max_bytes):
    """Read a portable NetCDF3 dataset; never pass untrusted HDF5 to native loading."""
    with path.open("rb") as stream:
        if stream.read(4) not in (b"CDF\x01", b"CDF\x02"):
            reject()
    with xr.open_dataset(path, engine="scipy", decode_times=False) as ds:
        clean = posterior_dataset(ds, manifest, source=False, max_bytes=max_bytes)
    # Reject ignored bytes/metadata too, not just fields exposed by xarray.
    with tempfile.TemporaryDirectory(prefix="meganorm-posterior-") as temp:
        canonical = Path(temp) / "posterior.nc"
        write_posterior(clean, canonical)
        if file_hash(canonical) != file_hash(path):
            reject()
    return clean


def validate_payload(root, manifest, read_json, max_bytes):
    """Validate model contents independently of self-declared file hashes."""
    sub = root / "model" / manifest.responses[0].storage_key
    validate_state(
        read_json(root / "model/normative_model.json"),
        read_json(sub / "regression_model.json"),
        manifest,
    )
    if manifest.adapter.profile == "hbr-shashb-linear-site":
        read_posterior(sub / "idata.nc", manifest, max_bytes)


def restore_backend(root, manifest, max_bytes):
    """Restore validated state and eagerly detach posterior arrays from staging."""
    from pcntoolkit import NormativeModel

    posterior = None
    if manifest.adapter.profile == "hbr-shashb-linear-site":
        path = root / "model" / manifest.responses[0].storage_key / "idata.nc"
        posterior = read_posterior(path, manifest, max_bytes)
        path.unlink()
        xr.DataTree.from_dict({"posterior": posterior}).to_netcdf(path)
    model = NormativeModel.load(str(root))
    model.savemodel = model.saveresults = model.saveplots = model.evaluate_model = False
    if posterior is not None:
        reg = model[manifest.responses[0].name]
        reg.idata.close()
        reg.idata = xr.DataTree.from_dict({"posterior": posterior})
    return model
