"""JSONL recording and replay format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from airdesk.state.types import EventLogEntry, TrackingFrame

RECORDING_VERSION = 1
RecordingKind = Literal["tracking_frame", "event"]


@dataclass(frozen=True)
class RecordingRecord:
    """One JSONL record from an AirDesk recording."""

    kind: RecordingKind
    payload: TrackingFrame | EventLogEntry


class JsonlRecordingWriter:
    """Writes AirDesk replay/event records as newline-delimited JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("w", encoding="utf-8")

    def __enter__(self) -> JsonlRecordingWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._handle.close()

    def write_tracking_frame(self, frame: TrackingFrame) -> None:
        self._write(
            {"version": RECORDING_VERSION, "kind": "tracking_frame", "frame": frame.to_dict()}
        )

    def write_event(self, event: EventLogEntry) -> None:
        self._write({"version": RECORDING_VERSION, "kind": "event", "event": event.to_dict()})

    def _write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")


def iter_recording(path: Path) -> list[RecordingRecord]:
    records: list[RecordingRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("version") != RECORDING_VERSION:
                raise ValueError(f"{path}:{line_number}: unsupported recording version")
            kind = data.get("kind")
            if kind == "tracking_frame":
                records.append(
                    RecordingRecord(
                        kind="tracking_frame", payload=TrackingFrame.from_dict(data["frame"])
                    )
                )
            elif kind == "event":
                records.append(
                    RecordingRecord(kind="event", payload=EventLogEntry.from_dict(data["event"]))
                )
            else:
                raise ValueError(f"{path}:{line_number}: unsupported record kind {kind!r}")
    return records
