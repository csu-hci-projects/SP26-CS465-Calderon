"""Optional MediaPipe tracking backend scaffold."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from airdesk.state.types import TrackingFrame


@dataclass
class MediaPipeHandTrackerBackend:
    """Placeholder for the first live webcam hand-tracking backend."""

    name: str = "mediapipe"

    def start(self) -> None:
        try:
            import mediapipe  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Add an optional backend dependency before using "
                "the mediapipe tracker."
            ) from exc

    def stop(self) -> None:
        return None

    def frames(self) -> Iterator[TrackingFrame]:
        raise NotImplementedError("MediaPipe live tracking is intentionally deferred past Sprint 0")
