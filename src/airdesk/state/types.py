"""Typed data structures for AirDesk pipeline boundaries."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


def utc_timestamp() -> float:
    """Return a UNIX timestamp for logs and replay records."""
    return time.time()


@dataclass(frozen=True)
class FrameMetadata:
    """Metadata for a captured frame without requiring pixel data."""

    timestamp: float
    source_id: str
    width: int
    height: int
    sequence: int
    color_format: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameMetadata:
        return cls(**data)


@dataclass(frozen=True)
class CapturedFrame:
    """A captured frame boundary; pixel payloads stay out of serialized recordings."""

    metadata: FrameMetadata
    image_ref: str | None = None
    image: Any | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata.to_dict(), "image_ref": self.image_ref}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapturedFrame:
        return cls(
            metadata=FrameMetadata.from_dict(data["metadata"]), image_ref=data.get("image_ref")
        )


@dataclass(frozen=True)
class Landmark:
    """Normalized 2D/3D hand landmark coordinate."""

    x: float
    y: float
    z: float = 0.0
    visibility: float | None = None
    presence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Landmark:
        return cls(**data)


@dataclass(frozen=True)
class HandLandmarks:
    """Backend-independent hand landmark set."""

    landmarks: tuple[Landmark, ...]
    handedness: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "landmarks": [landmark.to_dict() for landmark in self.landmarks],
            "handedness": self.handedness,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandLandmarks:
        return cls(
            landmarks=tuple(Landmark.from_dict(item) for item in data["landmarks"]),
            handedness=data.get("handedness"),
            confidence=data.get("confidence"),
        )


@dataclass(frozen=True)
class NormalizedHand:
    """Hand state normalized for recognition independent of tracker backend."""

    hand_id: str
    landmarks: HandLandmarks
    palm_center: tuple[float, float, float]
    bbox: tuple[float, float, float, float]
    handedness: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "landmarks": self.landmarks.to_dict(),
            "palm_center": list(self.palm_center),
            "bbox": list(self.bbox),
            "handedness": self.handedness,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedHand:
        return cls(
            hand_id=data["hand_id"],
            landmarks=HandLandmarks.from_dict(data["landmarks"]),
            palm_center=tuple(data["palm_center"]),
            bbox=tuple(data["bbox"]),
            handedness=data.get("handedness"),
            confidence=data.get("confidence"),
        )


@dataclass(frozen=True)
class TrackingFrame:
    """Normalized tracking output consumed by recognizers and replay."""

    timestamp: float
    source_id: str
    frame: FrameMetadata
    hands: tuple[NormalizedHand, ...] = ()
    debug_image_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "frame": self.frame.to_dict(),
            "hands": [hand.to_dict() for hand in self.hands],
            "debug_image_ref": self.debug_image_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackingFrame:
        return cls(
            timestamp=data["timestamp"],
            source_id=data["source_id"],
            frame=FrameMetadata.from_dict(data["frame"]),
            hands=tuple(NormalizedHand.from_dict(item) for item in data.get("hands", [])),
            debug_image_ref=data.get("debug_image_ref"),
        )


@dataclass(frozen=True)
class GestureCandidate:
    """Recognizer output before mode/profile policy confirms an action."""

    name: str
    confidence: float
    timestamp: float
    hand_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GestureCandidate:
        return cls(**data)


@dataclass(frozen=True)
class GestureConfirmation:
    """A gesture accepted by mode/profile policy."""

    candidate: GestureCandidate
    confirmed_at: float
    profile_id: str
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "confirmed_at": self.confirmed_at,
            "profile_id": self.profile_id,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class ActionRequest:
    """Typed request passed from gesture policy to an action target."""

    action_type: str
    command: str
    parameters: dict[str, Any] = field(default_factory=dict)
    source: str = "gesture"
    profile_id: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: float = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionRequest:
        return cls(**data)


@dataclass(frozen=True)
class ActionResult:
    """Result returned by an action target."""

    request_id: str
    ok: bool
    target: str
    executed_at: float
    message: str
    command_preview: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionResult:
        return cls(**data)


@dataclass(frozen=True)
class EventLogEntry:
    """Generic event log entry for runtime and study instrumentation."""

    event_type: str
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventLogEntry:
        return cls(**data)
