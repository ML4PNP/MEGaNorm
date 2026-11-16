"""Software provenance, independent of adapter compatibility policy."""

import platform
from dataclasses import dataclass
from importlib import metadata
from typing import Annotated

from packaging.version import InvalidVersion, Version
from pydantic import AfterValidator, Field

from .._version import __version__


def _software_version(value: str) -> str:
    # Preserve recorded spelling: this is provenance, not a release identifier.
    if value != value.strip():
        raise ValueError("Invalid software version.")
    try:
        Version(value)
    except InvalidVersion:
        raise ValueError("Invalid software version.") from None
    return value


SoftwareVersion = Annotated[
    str, Field(min_length=1, max_length=128), AfterValidator(_software_version)
]


@dataclass(frozen=True)
class RuntimeVersions:
    meganorm_version: str
    python_version: str
    pcntoolkit_version: str


def capture_runtime_versions() -> RuntimeVersions:
    """Capture this process, not the historical environment of a loaded model.

    Call when recording training/export provenance. Never substitute these values
    for an existing model's training versions. Missing distribution metadata raises
    PackageNotFoundError; no fallback version is invented. No backend import needed.
    """
    return RuntimeVersions(
        meganorm_version=__version__,
        python_version=platform.python_version(),
        pcntoolkit_version=metadata.version("pcntoolkit"),
    )
