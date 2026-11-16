"""Stable, sanitized errors at the model-sharing boundary.

This file defines how problems are reported. It provides:
- Stable error codes, such as missing fields or unsupported adapters.
- ValidationIssue and ValidationReport, which describe validation results.
- Exception classes for privacy, compatibility, integrity, authorization, 
and publication failures.

"""

from dataclasses import dataclass
from enum import StrEnum


class IssueCode(StrEnum):
    """Stable machine-readable categories for validation failures."""

    INVALID_DOCUMENT = "invalid_document"
    MISSING_FIELD = "missing_field"
    UNKNOWN_FIELD = "unknown_field"
    INVALID_VALUE = "invalid_value"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    UNSUPPORTED_ADAPTER = "unsupported_adapter"
    UNSUPPORTED_CONFIGURATION = "unsupported_configuration"
    DUPLICATE_NAME = "duplicate_name"
    DUPLICATE_PATH = "duplicate_path"
    UNSAFE_PATH = "unsafe_path"
    INVALID_INVENTORY = "invalid_inventory"
    INVALID_LINEAGE = "invalid_lineage"
    INVALID_CAPABILITIES = "invalid_capabilities"
    ARTIFACT_LIMIT = "artifact_limit"


@dataclass(frozen=True)
class ValidationIssue:
    """Describe one failure using a code, field path, and actionable message.

    Producers must sanitize paths and messages before construction. This value
    object does not redact arbitrary text supplied by callers.
    """

    code: IssueCode
    path: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Collect issues from a validation operation; an empty tuple means success.

    Success covers only the checks performed by that operation. This container
    does not inspect model files or attest to their contents.
    """

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether the report contains no issues."""
        return not self.issues


class SharingError(ValueError):
    """Base exception for failures at the model-sharing boundary.

    Callers must provide sanitized messages; this class does not automatically
    redact provider responses, credentials, or participant information.
    """


class ManifestValidationError(SharingError):
    """Expose a validation report with a summary containing only paths and codes."""

    def __init__(self, report: ValidationReport) -> None:
        """Retain the sanitized report and build its compact exception summary."""
        self.report = report
        super().__init__(
            "Manifest rejected: "
            + "; ".join(f"{issue.path}: {issue.code.value}" for issue in report.issues)
        )


class PrivacyError(SharingError):
    """Signal model content that violates the participant-information export policy."""

    pass


class CompatibilityError(SharingError):
    """Signal unsupported schema, adapter, backend, or model configuration."""

    pass


class IntegrityError(SharingError):
    """Signal a mismatch in expected artifact bytes, digests, or inventory."""

    pass


class AuthorizationError(SharingError):
    """Signal insufficient permission for a requested hosting operation."""

    pass


class VersionConflictError(SharingError):
    """Signal an existing release whose content conflicts with the proposed version."""

    pass


class IncompletePublicationError(SharingError):
    """Signal a publication workflow that has not reached verified completion."""

    pass
