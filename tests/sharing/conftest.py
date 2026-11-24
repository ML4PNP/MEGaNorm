"""Synthetic fixtures; short HBR chains exercise persistence, not convergence."""

import numpy as np
import pandas as pd
import pytest
from pcntoolkit import BLR, HBR, NormData, NormativeModel, SHASHbLikelihood, make_prior
from .test_schema import manifest


def data(name="train", responses=("roi",)):
    rng = np.random.default_rng(2026 if name == "train" else 2027)
    frame = pd.DataFrame(
        {"age": np.tile(np.linspace(20, 70, 12), 2), "site": np.repeat(["A", "B"], 12)}
    )
    for i, response in enumerate(responses):
        frame[response] = (i + 1) * (
            1 + (0.1 + 0.02 * i) * frame.age + rng.normal(0, 0.3, len(frame))
        )
    frame["participant"] = [f"PRIVATE_{name}_{i}" for i in range(len(frame))]
    return NormData.from_dataframe(
        name,
        frame,
        covariates=["age"],
        response_vars=list(responses),
        batch_effects=["site"],
        subject_ids="participant",
    )


def hbr():
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
    eps = make_prior(dist_name="Normal", dist_params=(0.0, 0.5))
    delta = make_prior(
        dist_name="Normal",
        dist_params=(1.0, 0.5),
        mapping="softplus",
        mapping_params=(0.0, 1.0, 0.6),
    )
    return HBR(
        likelihood=SHASHbLikelihood(mu, sigma, eps, delta),
        draws=20,
        tune=20,
        chains=2,
        cores=1,
        nuts_sampler="pymc",
        progressbar=False,
    )


@pytest.fixture(scope="session", params=["hbr", "blr"])
def fitted(request, tmp_path_factory):
    model = NormativeModel(
        BLR() if request.param == "blr" else hbr(),
        save_dir=str(tmp_path_factory.mktemp("original")),
        savemodel=False,
        saveresults=False,
        saveplots=False,
        evaluate_model=False,
    )
    model.fit(data())
    return model, request.param


@pytest.fixture
def metadata(manifest, fitted):
    manifest.pop("files")
    if fitted[1] == "hbr":
        manifest["adapter"]["profile"] = "hbr-shashb-linear-site"
    return manifest
