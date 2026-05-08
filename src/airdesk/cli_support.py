"""Shared helpers for AirDesk CLI command modules."""

from __future__ import annotations

from pathlib import Path

import typer

from airdesk.labels import save_label_file, validate_label_file
from airdesk.recording.jsonl import iter_recording
from airdesk.state.types import TrackingFrame


def _relative_label_time(start_timestamp: float | None, seconds: float) -> float:
    if start_timestamp is None:
        return seconds
    return start_timestamp + seconds


def _save_valid_label_file(label_file: object, path: Path) -> None:
    result = validate_label_file(label_file)
    if not result.ok:
        for error in result.errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
    save_label_file(label_file, path)



def _tracking_frames_from_recording(recording: Path) -> list[TrackingFrame]:
    return [
        record.payload
        for record in iter_recording(recording)
        if record.kind == "tracking_frame" and isinstance(record.payload, TrackingFrame)
    ]


