"""Dataset descriptions for the local scientific workflow."""

from pathlib import Path
import glob
from collections.abc import Mapping
import pandas as pd
from meganorm.utils.IO import merge_datasets_with_glob

_PATH_OPTIONS = {
    "empty_room_path",
    "surfaces_dir",
    "event_file_path",
    "trans_path",
    "pos_path",
    "annotation_path",
    "layout_path",
}
_OPTIONS = _PATH_OPTIONS | {
    "empty_room_task",
    "empty_room_ending",
    "event_file_task",
    "event_file_ending",
    "event_of_interest",
    "pos_file_ending",
    "annotaion_task_name",
    "annotation_ending",
}


def _path(value, root=None):
    path = Path(value).expanduser()
    if root is not None and not path.is_absolute():
        path = root / path
    path = path.resolve()
    if any(character in str(path) for character in "*?[]"):
        raise ValueError(
            "Paths containing glob characters '*', '?', '[' or ']' are incompatible with recording discovery."
        )
    return path


class Dataset:
    """Describe one dataset; discovery retains every matching session candidate.

    Relative demographic and auxiliary paths are resolved against root. Device
    and line frequency follow the existing processing engine's inference rules.
    Participant-specific auxiliary directories use the exact recording folder IDs.
    """

    def __init__(
        self,
        *,
        name: str,
        root: str | Path,
        task: str,
        extension: str,
        demographics: str | Path | None = None,
        participant_id: str = "participant_id",
        device: str | None = None,
        line_freq: int | None = 50,
        options: Mapping[str, object] | None = None,
    ):
        for label, value in [
            ("name", name),
            ("task", task),
            ("extension", extension),
            ("participant_id", participant_id),
        ]:
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(
                    f"{label} must be a nonempty string without surrounding whitespace."
                )
        if line_freq is not None and (
            isinstance(line_freq, bool)
            or not isinstance(line_freq, int)
            or line_freq <= 0
        ):
            raise ValueError("line_freq must be a positive integer or None.")
        if device is not None:
            if not isinstance(device, str) or device.upper() not in {
                "MEGIN",
                "CTF",
                "BTI",
                "ARTEMIS123",
            }:
                raise ValueError(f"Unsupported MEG device: {device}")
            device = device.upper()
        options = dict(options or {})
        if unknown := set(options) - _OPTIONS:
            raise ValueError(f"Unknown dataset options: {sorted(unknown)}")
        self.name, self.task, self.extension = name, task, extension
        self.root = _path(root)
        self.demographics = (
            _path(demographics, self.root) if demographics is not None else None
        )
        self.participant_id, self.device, self.line_freq = (
            participant_id,
            device,
            line_freq,
        )
        self.options = {
            key: (
                _path(value, self.root)
                if key in _PATH_OPTIONS and value is not None
                else value
            )
            for key, value in options.items()
        }

    @classmethod
    def from_dict(cls, name: str, settings: Mapping[str, object]) -> "Dataset":
        """Adapt legacy discovery settings without changing the legacy API."""
        settings = dict(settings)
        root = _path(settings.pop("base_dir"))
        demographics = settings.pop("demographic_path", root / "participants_bids.tsv")
        demographic_path = (
            _path(demographics, root) if demographics is not None else None
        )
        participant_id = "participant_id"
        if demographic_path is not None and demographic_path.exists():
            from meganorm.utils.IO import load_demographic_file

            table = load_demographic_file(str(demographic_path), index_col=None)
            participant_id = table.columns[0]
            if not isinstance(participant_id, str) or participant_id.startswith(
                "Unnamed:"
            ):
                raise ValueError(
                    "Name the demographic identifier column before using from_dict."
                )
        return cls(
            name=name,
            root=root,
            task=settings.pop("task"),
            extension=settings.pop("ending"),
            demographics=demographic_path,
            participant_id=participant_id,
            device=settings.pop("device_type", None),
            line_freq=settings.pop("line_freq", 50),
            options=settings,
        )

    def _settings(self):
        return dict(
            base_dir=str(self.root),
            task=self.task,
            ending=self.extension,
            device_type=self.device,
            line_freq=self.line_freq,
            demographic_path=str(self.demographics) if self.demographics else None,
            **{
                k: str(v) if isinstance(v, Path) else v for k, v in self.options.items()
            },
        )

    def discover(self) -> pd.DataFrame:
        """Return a manifest; no recordings are loaded or processing files written."""
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        # Validate literal separators before the legacy helper encodes candidates.
        for folder in self.root.iterdir():
            if folder.is_dir():
                _path(folder)
                for recording in glob.glob(
                    str(folder / "**" / f"*{self.task}*{self.extension}"),
                    recursive=True,
                ):
                    _path(recording)
        subjects = merge_datasets_with_glob({self.name: self._settings()})
        if not subjects:
            raise ValueError(
                f"No recordings found in {self.root} for task={self.task!r}, extension={self.extension!r}."
            )
        rows = []
        for participant, info in subjects.items():
            if participant != participant.strip():
                raise ValueError(
                    f"Participant ID has surrounding whitespace: {participant!r}"
                )
            rows.append(
                dict(
                    info,
                    dataset=self.name,
                    participant_id=participant,
                    recording_paths=tuple(
                        _path(p) for p in info["rest_record"].split("*") if p
                    ),
                    line_freq=self.line_freq,
                )
            )
        return pd.DataFrame(rows)
