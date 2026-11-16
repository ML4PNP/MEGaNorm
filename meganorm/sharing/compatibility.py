"""Audited producer/consumer pairs for each portable adapter format/profile.

Add entries only after serialization and prediction round-trip tests. Installing
an upgrade does not expand this registry. Python provenance is not a loading gate;
package installation requirements remain responsible for supported Python runtimes.
"""

from types import MappingProxyType

from .errors import (
    IssueCode,
    ManifestValidationError,
    ValidationIssue,
    ValidationReport,
)
from .schema import parse_manifest

# Key: (adapter, format, profile). Value: exact (producer, consumer) versions.
# Stage 0 evidence: docs/development/sharing/stage0-audit.md.
ADAPTER_COMPATIBILITY = MappingProxyType(
    {
        ("pcntoolkit", 1, "blr-default-site"): frozenset({("1.4.0", "1.4.0")}),
        ("pcntoolkit", 1, "hbr-shashb-linear-site"): frozenset({("1.4.0", "1.4.0")}),
    }
)


def validate_compatibility(
    document: dict, *, runtime_pcntoolkit_version: str
) -> ValidationReport:
    """Validate metadata and require an audited producer/runtime version pair.

    The caller supplies the actual runtime version (capture_runtime_versions).
    This does not inspect payload bytes or certify a running backend. Future
    export/load entry points must call this before invoking an adapter.
    """
    try:
        manifest = parse_manifest(document)
    except ManifestValidationError as exc:
        return exc.report
    adapter = manifest.adapter
    allowed = ADAPTER_COMPATIBILITY.get(
        (adapter.name, adapter.format_version, adapter.profile), frozenset()
    )
    if (
        type(runtime_pcntoolkit_version) is str
        and (adapter.pcntoolkit_version, runtime_pcntoolkit_version) in allowed
    ):
        return ValidationReport()
    return ValidationReport(
        (
            ValidationIssue(
                IssueCode.UNSUPPORTED_ADAPTER,
                "adapter.pcntoolkit_version",
                "Use an audited producer/runtime version pair from ADAPTER_COMPATIBILITY; "
                "new versions require adapter round-trip tests.",
            ),
        )
    )
