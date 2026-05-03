"""Label schema for continuous gesture recordings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from airdesk.recording.jsonl import iter_recording
from airdesk.state.types import EventLogEntry, TrackingFrame, utc_timestamp

LABEL_SCHEMA_VERSION = 1

PHASE_LABELS = frozenset(
    {
        "background",
        "preparation",
        "armed",
        "stroke_left",
        "stroke_right",
        "hold",
        "release",
        "cooldown",
        "aborted",
    }
)
EVENT_LABEL_TYPES = frozenset(
    {
        "gesture",
        "command",
        "false_activation",
        "missed_gesture",
        "abort",
        "note",
    }
)


class LabelValidationError(ValueError):
    """Raised when a label file is structurally invalid."""


@dataclass(frozen=True)
class SessionMetadata:
    """Metadata for a labeled recording session."""

    recording_path: str
    participant_id: str = "caden"
    profile_id: str | None = None
    source_label: str | None = None
    frame_count: int = 0
    hand_frame_count: int = 0
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    camera: dict[str, Any] = field(default_factory=dict)
    tracker: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMetadata:
        return cls(**data)


@dataclass(frozen=True)
class GestureEventLabel:
    """Event-level label for an intended or observed gesture/action."""

    label_id: str
    label_type: str
    gesture: str
    start_time: float
    end_time: float
    commit_time: float | None = None
    intended_command: str | None = None
    success: bool | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GestureEventLabel:
        return cls(**data)


@dataclass(frozen=True)
class GesturePhaseLabel:
    """Frame interval label for phase-aware continuous recognition."""

    label_id: str
    phase: str
    start_time: float
    end_time: float
    gesture: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GesturePhaseLabel:
        return cls(**data)


@dataclass(frozen=True)
class GestureLabelFile:
    """Serializable labels for one continuous recording."""

    schema_version: int
    created_at: float
    session: SessionMetadata
    event_labels: tuple[GestureEventLabel, ...] = ()
    phase_labels: tuple[GesturePhaseLabel, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "session": self.session.to_dict(),
            "event_labels": [label.to_dict() for label in self.event_labels],
            "phase_labels": [label.to_dict() for label in self.phase_labels],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GestureLabelFile:
        return cls(
            schema_version=int(data["schema_version"]),
            created_at=float(data["created_at"]),
            session=SessionMetadata.from_dict(data["session"]),
            event_labels=tuple(
                GestureEventLabel.from_dict(item) for item in data.get("event_labels", [])
            ),
            phase_labels=tuple(
                GesturePhaseLabel.from_dict(item) for item in data.get("phase_labels", [])
            ),
        )


@dataclass(frozen=True)
class LabelValidationResult:
    """Result of label-file validation."""

    ok: bool
    errors: tuple[str, ...] = ()


def init_label_file(
    recording_path: Path,
    *,
    participant_id: str = "caden",
    notes: str = "",
) -> GestureLabelFile:
    """Create a starter label file from a replayable recording."""
    metadata = _metadata_from_recording(recording_path, participant_id=participant_id, notes=notes)
    return GestureLabelFile(
        schema_version=LABEL_SCHEMA_VERSION,
        created_at=utc_timestamp(),
        session=metadata,
        event_labels=(),
        phase_labels=(
            GesturePhaseLabel(
                label_id="phase-001",
                phase="background",
                start_time=metadata.start_timestamp or 0.0,
                end_time=metadata.end_timestamp or 0.0,
                notes="Starter background interval. Replace or split during manual labeling.",
            ),
        ),
    )


def load_label_file(path: Path) -> GestureLabelFile:
    with path.open(encoding="utf-8") as handle:
        return GestureLabelFile.from_dict(json.load(handle))


def save_label_file(label_file: GestureLabelFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(label_file.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_label_file(label_file: GestureLabelFile) -> LabelValidationResult:
    errors: list[str] = []
    if label_file.schema_version != LABEL_SCHEMA_VERSION:
        errors.append(f"unsupported schema_version={label_file.schema_version}")

    start = label_file.session.start_timestamp
    end = label_file.session.end_timestamp
    if start is not None and end is not None and end < start:
        errors.append("session end_timestamp must be >= start_timestamp")

    label_ids: set[str] = set()
    for event in label_file.event_labels:
        _validate_id(event.label_id, label_ids, errors)
        if event.label_type not in EVENT_LABEL_TYPES:
            errors.append(f"{event.label_id}: unsupported label_type={event.label_type}")
        _validate_interval(event.label_id, event.start_time, event.end_time, start, end, errors)
        if event.commit_time is not None and not (
            event.start_time <= event.commit_time <= event.end_time
        ):
            errors.append(f"{event.label_id}: commit_time must be inside interval")

    for phase in label_file.phase_labels:
        _validate_id(phase.label_id, label_ids, errors)
        if phase.phase not in PHASE_LABELS:
            errors.append(f"{phase.label_id}: unsupported phase={phase.phase}")
        _validate_interval(phase.label_id, phase.start_time, phase.end_time, start, end, errors)

    return LabelValidationResult(ok=not errors, errors=tuple(errors))


def _metadata_from_recording(
    recording_path: Path,
    *,
    participant_id: str,
    notes: str,
) -> SessionMetadata:
    frame_count = 0
    hand_frame_count = 0
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    source_label: str | None = None
    profile_id: str | None = None
    camera: dict[str, Any] = {}
    tracker: dict[str, Any] = {}

    for record in iter_recording(recording_path):
        if record.kind == "tracking_frame":
            assert isinstance(record.payload, TrackingFrame)
            frame = record.payload
            frame_count += 1
            hand_frame_count += 1 if frame.hands else 0
            start_timestamp = frame.timestamp if start_timestamp is None else start_timestamp
            end_timestamp = frame.timestamp
        elif record.kind == "event":
            assert isinstance(record.payload, EventLogEntry)
            payload = record.payload.payload
            if source_label is None and isinstance(payload.get("label"), str):
                source_label = payload["label"]
            if profile_id is None and isinstance(payload.get("profile_id"), str):
                profile_id = payload["profile_id"]
            if not camera and isinstance(payload.get("camera_settings"), dict):
                camera = payload["camera_settings"]
            if not tracker and isinstance(payload.get("mediapipe"), dict):
                tracker = {"backend": payload.get("backend"), "mediapipe": payload["mediapipe"]}

    return SessionMetadata(
        recording_path=str(recording_path),
        participant_id=participant_id,
        profile_id=profile_id,
        source_label=source_label,
        frame_count=frame_count,
        hand_frame_count=hand_frame_count,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        camera=camera,
        tracker=tracker,
        notes=notes,
    )


def _validate_id(label_id: str, seen: set[str], errors: list[str]) -> None:
    if not label_id:
        errors.append("label_id cannot be empty")
        return
    if label_id in seen:
        errors.append(f"duplicate label_id={label_id}")
    seen.add(label_id)


def _validate_interval(
    label_id: str,
    start_time: float,
    end_time: float,
    session_start: float | None,
    session_end: float | None,
    errors: list[str],
) -> None:
    if end_time < start_time:
        errors.append(f"{label_id}: end_time must be >= start_time")
    if session_start is not None and start_time < session_start:
        errors.append(f"{label_id}: start_time is before recording start")
    if session_end is not None and end_time > session_end:
        errors.append(f"{label_id}: end_time is after recording end")
