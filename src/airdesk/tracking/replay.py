"""Mock and replay hand tracking backends."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from airdesk.recording.jsonl import iter_recording
from airdesk.state.types import TrackingFrame


@dataclass
class MockHandTrackerBackend:
    """Deterministic backend for tests and recognizer development."""

    sequence: Iterable[TrackingFrame]
    name: str = "mock"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def frames(self) -> Iterator[TrackingFrame]:
        yield from self.sequence


@dataclass
class ReplayHandTrackerBackend:
    """Feeds JSONL tracking frames back into the recognition pipeline."""

    path: Path
    name: str = "replay"

    def start(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def stop(self) -> None:
        return None

    def frames(self) -> Iterator[TrackingFrame]:
        for record in iter_recording(self.path):
            if record.kind == "tracking_frame":
                payload = record.payload
                if isinstance(payload, TrackingFrame):
                    yield payload
