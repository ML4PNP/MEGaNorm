"""Version-one metadata contracts for portable MEGaNorm model bundles.

defines the structure and rules for a shared model’s metadata. It specifies
required fields—model identity, software versions, features, preprocessing,
provenance, parent model, and file inventory—and checks their consistency.
It rejects missing or unexpected fields, duplicate names, unsafe paths, and
invalid parent references. It also produces consistent JSON bytes for later hashing.


"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from .versions import SoftwareVersion

from .errors import (
    IssueCode,
    ManifestValidationError,
    ValidationIssue,
    ValidationReport,
)

if TYPE_CHECKING:
    import pandas as pd

SCHEMA_VERSION = 1
MAX_ARCHIVE_BYTES = 20 * 1024**3
MAX_EXTRACTED_BYTES = 80 * 1024**3
MAX_MEMBERS = 100_000
MAX_JSON_BYTES = 32 * 1024**2

ReleaseVersion = Annotated[
    str,
    Field(
        pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$", max_length=64
    ),
]
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
ModelId = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$"),
]


def _label(value: str) -> str:
    """Reject blank, padded, or control-containing display labels."""
    if (
        not value.strip()
        or value != value.strip()
        or any(unicodedata.category(c).startswith("C") for c in value)
    ):
        raise ValueError(
            "Labels must be nonempty single-line names without control characters."
        )
    return value


Label = Annotated[str, Field(min_length=1, max_length=128), AfterValidator(_label)]
Text = Annotated[str, Field(min_length=1, max_length=4096)]
Capability = Literal["predict", "calibrate", "extend"]


class _StrictModel(BaseModel):
    """Base contract: forbid extra fields and coercion; revalidate nested instances."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
        revalidate_instances="always",
    )

    @field_validator("*", mode="after")
    @classmethod
    def _clean_strings(cls, value: object) -> object:
        """Check scalar text without altering it; narrative line breaks are allowed."""
        if isinstance(value, str):
            if not value.strip() or value != value.strip():
                raise ValueError(
                    "Text must be nonempty and have no surrounding whitespace."
                )
            if any(
                unicodedata.category(c).startswith("C") and c not in "\n\t"
                for c in value
            ):
                raise ValueError("Control characters are not permitted.")
        return value


def _key(value: str) -> str:
    """Return the Unicode-normalized, case-insensitive key used for collisions."""
    return unicodedata.normalize("NFKC", value).casefold()


def _unique(values: list[str], code: str = "duplicate_name") -> None:
    """Raise a coded validation error when normalized names collide."""
    keys = [_key(v) for v in values]
    if len(keys) != len(set(keys)):
        raise PydanticCustomError(
            code, "Names must be distinct, including case and Unicode equivalents."
        )


def _path(value: str) -> str:
    # Portable POSIX names; storage keys keep scientific labels out of filenames.
    """Validate a portable relative path lexically; do not access the filesystem."""
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", value) or len(value) > 256:
        raise PydanticCustomError(
            "unsafe_path", "Use a portable relative POSIX file path."
        )
    for segment in value.split("/"):
        stem = segment.split(".")[0].upper()
        if (
            not segment
            or segment.startswith(".")
            or segment.endswith(".")
            or stem in {"CON", "PRN", "AUX", "NUL"}
            or re.fullmatch(r"(COM|LPT)[1-9]", stem)
        ):
            raise PydanticCustomError(
                "unsafe_path", "Reserved or ambiguous path component."
            )
    return value


class AdapterSpec(_StrictModel):
    """Identify the serialization adapter, format, profile, and producer version.

    Software version syntax is checked here. Audited producer/runtime support
    is checked separately by ``validate_compatibility``.
    """

    name: Literal["pcntoolkit"]
    format_version: int
    pcntoolkit_version: SoftwareVersion
    profile: Literal["blr-default-site", "hbr-shashb-linear-site"]

    @field_validator("format_version")
    @classmethod
    def _version(cls, value: int) -> int:
        """Require the currently understood adapter format identifier."""
        if value != 1:
            raise PydanticCustomError(
                "unsupported_adapter", "Unsupported adapter format."
            )
        return value


class SoftwareSpec(_StrictModel):
    """Record software provenance and an optional exact source commit."""

    meganorm_version: SoftwareVersion
    python_version: SoftwareVersion
    source_commit: GitCommit | None = None


class ResponseSpec(_StrictModel):
    """Describe a modeled response, its unit, and portable storage key.

    The scientific display name is independent of the filename component.
    Optional region and description fields must contain publication-safe text.
    """

    name: Label
    unit: Label
    storage_key: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
    description: Text | None = None
    region: Label | None = None

    @field_validator("storage_key")
    @classmethod
    def _storage_path(cls, value: str) -> str:
        """Reject storage keys that would form reserved path components."""
        return _path(value)


class Category(_StrictModel):
    """Map one public category label to its nonnegative integer encoding."""

    label: Label
    code: Annotated[int, Field(ge=0)]


def _categories(categories: list[Category]) -> None:
    """Require unique labels and codes ordered contiguously from zero."""
    if not categories:
        raise ValueError("At least one category is required.")
    _unique([c.label for c in categories])
    if [c.code for c in categories] != list(range(len(categories))):
        raise ValueError(
            "Category codes must be contiguous from zero in declared order."
        )


class CovariateSpec(_StrictModel):
    """Describe a predictor as continuous with units or categorical with codes.

    This type defines metadata; accepted combinations of predictors are further
    restricted by the selected bundle profile.
    """

    name: Label
    kind: Literal["continuous", "categorical"]
    unit: Label | None = None
    categories: list[Category] | None = None

    @model_validator(mode="after")
    def _definition(self) -> CovariateSpec:
        """Require units for continuous predictors and encodings for categorical ones."""
        if self.kind == "continuous":
            if self.unit is None or self.categories is not None:
                raise ValueError(
                    "Continuous covariates require a unit and no categories."
                )
        else:
            if self.unit is not None or self.categories is None:
                raise ValueError(
                    "Categorical covariates require categories and no unit."
                )
            _categories(self.categories)
        return self


class BatchSpec(_StrictModel):
    """Describe a batch effect and the public labels used by its fitted encoding."""

    name: Label
    categories: list[Category]

    @model_validator(mode="after")
    def _definition(self) -> BatchSpec:
        """Validate the declared batch-category encoding."""
        _categories(self.categories)
        return self


class PreprocessingSpec(_StrictModel):
    """Record the feature-extraction contract and configuration fingerprint.

    Source-space features require an atlas. The fingerprint describes approved
    preprocessing configuration, not participant data or an analysis directory.
    """

    modality: Literal["EEG", "MEG"]
    paradigm: Literal["rest", "task"]
    space: Literal["sensor", "source"]
    pipeline_version: SoftwareVersion
    config_sha256: SHA256
    description: Text
    reference: Label | None = None
    atlas: Label | None = None

    @model_validator(mode="after")
    def _atlas(self) -> PreprocessingSpec:
        """Require atlas metadata when features are defined in source space."""
        if self.space == "source" and self.atlas is None:
            raise ValueError("Source-space features require an atlas definition.")
        return self


class PublicationMetadata(_StrictModel):
    """Hold public descriptive metadata, attribution, license, and provenance.

    Dataset and site names are permitted. Structural validation cannot detect
    participant information inside allowed text; publishers must review it.
    """

    title: Annotated[str, Field(min_length=1, max_length=200)]
    description: Text
    authors: Annotated[list[Label], Field(min_length=1, max_length=100)]
    license: Label
    datasets: Annotated[list[Label], Field(max_length=1000)] = Field(
        default_factory=list
    )
    sites: Annotated[list[Label], Field(max_length=1000)] = Field(default_factory=list)
    citations: Annotated[list[Text], Field(max_length=100)] = Field(
        default_factory=list
    )

    @field_validator("authors", "datasets", "sites", "citations")
    @classmethod
    def _entries(cls, values: list[str]) -> list[str]:
        """Reject blank, malformed, or normalized duplicate metadata entries."""
        for value in values:
            cls._clean_strings(value)
        _unique(values)
        return values


class HuggingFaceReference(_StrictModel):
    """Identify a repository at an exact commit without resolving it remotely."""

    provider: Literal["huggingface"] = "huggingface"
    repo_id: ModelId
    commit: GitCommit


class ZenodoReference(_StrictModel):
    """Identify a production or sandbox record and its matching DOI.

    Provider lookup must establish that this is a version record, rather than
    a concept record; this class only checks the declared identity.
    """

    provider: Literal["zenodo"] = "zenodo"
    environment: Literal["production", "sandbox"] = "production"
    record_id: Annotated[str, Field(pattern=r"^[1-9][0-9]*$", max_length=30)]
    version_doi: Annotated[
        str, Field(pattern=r"^10\.(5281|5072)/zenodo\.[1-9][0-9]*$", max_length=64)
    ]

    @model_validator(mode="after")
    def _identity(self) -> ZenodoReference:
        """Require the DOI prefix and record number to match the declared environment."""
        prefix = "10.5281" if self.environment == "production" else "10.5072"
        if self.version_doi != f"{prefix}/zenodo.{self.record_id}":
            raise ValueError("Version DOI and record identity must agree.")
        return self


SourceReference = Annotated[
    HuggingFaceReference | ZenodoReference, Field(discriminator="provider")
]


class ParentReference(_StrictModel):
    """Identify an exact parent release by model, version, digest, and optional source."""

    model_id: ModelId
    version: ReleaseVersion
    bundle_sha256: SHA256
    source: SourceReference | None = None


class LineageSpec(_StrictModel):
    """Distinguish original fitting from calibration or extension of a parent."""

    operation: Literal["fit", "calibrate", "extend"]
    parent: ParentReference | None

    @model_validator(mode="after")
    def _parent(self) -> LineageSpec:
        """Require a parent for adaptation and prohibit one for an original fit."""
        if (self.operation == "fit") != (self.parent is None):
            raise PydanticCustomError(
                "invalid_lineage",
                "Adaptations require a parent; original fits must not declare one.",
            )
        return self


class FileEntry(_StrictModel):
    """Declare a payload member with a portable path, SHA-256, and byte count.

    Values describe expected content; this class neither reads nor hashes files.
    The manifest itself is excluded from the payload inventory.
    """

    path: Annotated[str, Field(min_length=1, max_length=256)]
    sha256: SHA256
    size_bytes: Annotated[int, Field(gt=0, le=MAX_EXTRACTED_BYTES)]

    @field_validator("path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        """Validate the inventory path without following or inspecting filesystem links."""
        return _path(value)

    @model_validator(mode="after")
    def _json_budget(self) -> FileEntry:
        """Enforce the declared JSON-member size limit."""
        if self.path.endswith(".json") and self.size_bytes > MAX_JSON_BYTES:
            raise PydanticCustomError(
                "artifact_limit", "Declared JSON file exceeds the limit."
            )
        return self


class BundleManifest(_StrictModel):
    """Define the complete publication metadata contract for a model bundle.

    Enforce profile shape, unique feature names, lineage, capabilities, and an
    exact file inventory. Passing validation does not verify actual payloads,
    scientific validity, privacy of narrative text, or backend compatibility.
    Use ``parse_manifest`` for sanitized errors on external input.
    """

    schema_version: int
    model_id: ModelId
    version: ReleaseVersion
    adapter: AdapterSpec
    software: SoftwareSpec
    responses: Annotated[list[ResponseSpec], Field(min_length=1)]
    covariates: Annotated[list[CovariateSpec], Field(min_length=1)]
    batch_effects: Annotated[list[BatchSpec], Field(min_length=1)]
    preprocessing: PreprocessingSpec
    metadata: PublicationMetadata
    lineage: LineageSpec
    capabilities: list[Capability]
    files: Annotated[list[FileEntry], Field(min_length=1, max_length=MAX_MEMBERS)]

    @field_validator("schema_version")
    @classmethod
    def _schema(cls, value: int) -> int:
        """Reject bundle schema versions this implementation does not understand."""
        if value != SCHEMA_VERSION:
            raise PydanticCustomError(
                "unsupported_schema", "Unsupported bundle schema."
            )
        return value

    @model_validator(mode="after")
    def _contract(self) -> BundleManifest:
        """Check relationships across fields and declared aggregate size limits."""
        _unique([f.name for f in self.responses + self.covariates + self.batch_effects])
        _unique([f.storage_key for f in self.responses])
        _unique([f.path for f in self.files], "duplicate_path")
        if (
            len(self.covariates) != 1
            or len(self.batch_effects) != 1
            or self.covariates[0].kind != "continuous"
        ):
            raise PydanticCustomError(
                "unsupported_configuration",
                "The current profiles require one or more responses, one continuous covariate and one site batch.",
            )
        parent = self.lineage.parent
        if parent and (parent.model_id, parent.version) == (
            self.model_id,
            self.version,
        ):
            raise PydanticCustomError(
                "invalid_lineage", "A release cannot be its own parent."
            )
        if (
            parent
            and isinstance(parent.source, HuggingFaceReference)
            and parent.source.repo_id != parent.model_id
        ):
            raise PydanticCustomError(
                "invalid_lineage",
                "Parent repository identity must agree with its model identity.",
            )
        expected = (
            {"predict", "calibrate", "extend"}
            if self.lineage.operation == "fit"
            else {"predict"}
        )
        if (
            len(self.capabilities) != len(set(self.capabilities))
            or set(self.capabilities) != expected
        ):
            raise PydanticCustomError(
                "invalid_capabilities",
                "Capabilities must match the audited operation/profile.",
            )
        expected_files = {"README.md", "LICENSE", "model/normative_model.json"}
        for response in self.responses:
            expected_files.add(f"model/{response.storage_key}/regression_model.json")
            if self.adapter.profile == "hbr-shashb-linear-site":
                expected_files.add(f"model/{response.storage_key}/idata.nc")
        if {f.path for f in self.files} != expected_files:
            raise PydanticCustomError(
                "invalid_inventory",
                "File inventory must exactly match the selected adapter profile.",
            )
        if sum(f.size_bytes for f in self.files) > MAX_EXTRACTED_BYTES:
            raise PydanticCustomError(
                "artifact_limit", "Declared extracted size exceeds the limit."
            )
        return self


class ModelBundle(_StrictModel):
    """Describe a locally produced bundle and its declared archive digest.

    Construction does not create or verify files. The local path is not part of
    the published manifest.
    """

    path: Path
    model_id: ModelId
    version: ReleaseVersion
    schema_version: Annotated[int, Field(ge=1, le=1)]
    bundle_sha256: SHA256


class PublicationPreview(_StrictModel):
    """Describe the files and metadata readiness proposed for publication.

    This is a result contract; constructing it does not inspect files or make
    remote changes.
    """

    repo_id: ModelId
    version: ReleaseVersion
    files: list[FileEntry]
    bundle_sha256: SHA256
    metadata_ready: bool


class PublishedRelease(_StrictModel):
    """Describe a completed release with both provider references and digest.

    The coordinator must verify completion before returning this descriptor.
    Constructing it alone does not establish remote publication or ownership.
    """

    model_id: ModelId
    version: ReleaseVersion
    huggingface: HuggingFaceReference
    descriptor_commit: GitCommit
    zenodo: ZenodoReference
    bundle_sha256: SHA256
    status: Literal["complete"]

    @model_validator(mode="after")
    def _publisher(self) -> PublishedRelease:
        """Require the release model identity to match its distribution repository."""
        if self.huggingface.repo_id != self.model_id:
            raise ValueError("Release and repository identity must agree.")
        return self


class FetchedModel(_StrictModel):
    """Describe a local fetch result, exact source, digest, and lock-receipt path."""

    path: Path
    source: SourceReference
    bundle_sha256: SHA256
    lock_receipt: Path


@dataclass
class PredictionResults:
    """Contain local predictions, z-scores, and their participant ordering.

    These analysis outputs are deliberately excluded from publication metadata.
    This dataclass does not enforce DataFrame alignment; the prediction layer
    must validate it.
    """

    predictions: pd.DataFrame | None
    z_scores: pd.DataFrame | None
    participant_index: tuple[object, ...]


class SharedNormativeModel(Protocol):
    """Typing contract only; the executable facade belongs to Stages 3–4."""

    capabilities: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> SharedNormativeModel:
        """Contract for restoring a validated local bundle; no implementation here."""
        ...

    def predict(
        self, data: object, *, output_dir: Path | None = None
    ) -> PredictionResults:
        """Contract for computing local participant outputs from supplied data."""
        ...

    def calibrate(
        self, data: object, *, output_dir: Path, **options: object
    ) -> SharedNormativeModel:
        """Contract for producing a calibrated child without modifying its parent."""
        ...

    def extend(
        self, data: object, *, output_dir: Path, **options: object
    ) -> SharedNormativeModel:
        """Contract for producing an extended child without modifying its parent."""
        ...


_MESSAGES = {
    IssueCode.INVALID_DOCUMENT: "Supply a manifest object containing only declared metadata.",
    IssueCode.MISSING_FIELD: "Supply the required field.",
    IssueCode.UNKNOWN_FIELD: "Remove undeclared fields; analysis records are not publication metadata.",
    IssueCode.INVALID_VALUE: "Use the declared field type, format, and permitted values.",
    IssueCode.UNSUPPORTED_SCHEMA: "Use a supported bundle schema version.",
    IssueCode.UNSUPPORTED_ADAPTER: "Use the audited PCNToolkit version, profile, and adapter format.",
    IssueCode.UNSUPPORTED_CONFIGURATION: "This adapter supports one or more responses, one continuous covariate, and one site batch.",
    IssueCode.DUPLICATE_NAME: "Names must be distinct across roles, case, and Unicode equivalents.",
    IssueCode.DUPLICATE_PATH: "File paths must be distinct across case and Unicode equivalents.",
    IssueCode.UNSAFE_PATH: "Use relative portable paths without reserved components.",
    IssueCode.INVALID_INVENTORY: "Declare exactly the files required by the adapter; omit the manifest itself.",
    IssueCode.INVALID_LINEAGE: "Supply an exact, consistent parent for adaptations; prohibit self-reference.",
    IssueCode.INVALID_CAPABILITIES: "Declare only the operations audited for this profile and lineage.",
    IssueCode.ARTIFACT_LIMIT: "Declared artifact size exceeds a supported limit.",
}
_SAFE_FIELDS = {
    name
    for cls in (
        BundleManifest,
        AdapterSpec,
        SoftwareSpec,
        ResponseSpec,
        CovariateSpec,
        BatchSpec,
        Category,
        PreprocessingSpec,
        PublicationMetadata,
        LineageSpec,
        ParentReference,
        HuggingFaceReference,
        ZenodoReference,
        FileEntry,
    )
    for name in cls.model_fields
}


def _report(error: ValidationError) -> ValidationReport:
    """Convert backend validation errors into fixed messages and sanitized field paths."""
    issues = []
    for item in error.errors(
        include_input=False, include_context=False, include_url=False
    ):
        raw_code = item["type"]
        loc = item["loc"]
        if raw_code == "missing":
            code = IssueCode.MISSING_FIELD
        elif raw_code == "extra_forbidden":
            code = IssueCode.UNKNOWN_FIELD
        elif raw_code in IssueCode._value2member_map_:
            code = IssueCode(raw_code)
        elif loc and loc[0] == "adapter":
            code = IssueCode.UNSUPPORTED_ADAPTER
        elif loc and loc[0] == "schema_version":
            code = IssueCode.UNSUPPORTED_SCHEMA
        else:
            code = IssueCode.INVALID_VALUE
        list_fields = {
            "files",
            "responses",
            "covariates",
            "batch_effects",
            "capabilities",
            "authors",
            "datasets",
            "sites",
            "citations",
            "categories",
        }
        parts = []
        for index, part in enumerate(loc):
            if type(part) is int:
                # A numeric unknown dictionary key may itself be a participant ID.
                safe = (
                    index > 0
                    and loc[index - 1] in list_fields
                    and 0 <= part < MAX_MEMBERS
                )
                parts.append(str(part) if safe else "<unknown>")
            else:
                parts.append(part if part in _SAFE_FIELDS else "<unknown>")
        issues.append(ValidationIssue(code, ".".join(parts) or "$", _MESSAGES[code]))
    return ValidationReport(tuple(issues))


def parse_manifest(document: dict) -> BundleManifest:
    """Validate a plain dictionary and return a typed ``BundleManifest``.

    Raises
    ------
    ManifestValidationError
        If structure or cross-field constraints fail. Its report omits input
        values, unknown field names, and raw validation exception context.

    Notes
    -----
    Does not mutate input, inspect files, or check runtime compatibility.
    Use ``validate_compatibility`` separately before backend operations.
    """
    if type(document) is not dict:
        raise ManifestValidationError(
            ValidationReport(
                (
                    ValidationIssue(
                        IssueCode.INVALID_DOCUMENT,
                        "$",
                        _MESSAGES[IssueCode.INVALID_DOCUMENT],
                    ),
                )
            )
        )
    try:
        return BundleManifest.model_validate(document)
    except ValidationError as exc:
        report = _report(exc)
    # Raise outside the except block so the raw validation error is not chained.
    raise ManifestValidationError(report) from None


def validate_manifest(document: dict) -> ValidationReport:
    """Return a ``ValidationReport`` for a plain manifest dictionary.

    Invalid metadata is reported rather than raised as ManifestValidationError.
    A valid report certifies only the structural checks performed here; it does
    not establish payload integrity, privacy, or runtime compatibility.
    """
    try:
        parse_manifest(document)
    except ManifestValidationError as exc:
        return exc.report
    return ValidationReport()


def canonical_manifest(document: dict) -> bytes:
    """Validate metadata and return deterministic compact UTF-8 JSON bytes.

    Sort object keys, file entries, and capabilities; retain semantic ordering
    of features, categories, and authors. Include explicit defaults and nulls.
    Raise ``ManifestValidationError`` for invalid metadata.

    This is the MEGaNorm canonical format, not an RFC 8785 implementation.
    No archive hash is computed and no files are accessed.
    """
    model = parse_manifest(document)
    data = model.model_dump(mode="json")
    data["files"].sort(key=lambda entry: entry["path"])
    data["capabilities"].sort()
    return json.dumps(
        data, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
