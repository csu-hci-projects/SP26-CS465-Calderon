"""Feature export helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from airdesk.features.landmarks import FrameFeatureRow, extract_feature_rows
from airdesk.labels import GestureLabelFile
from airdesk.recording.jsonl import iter_recording
from airdesk.state.types import TrackingFrame


def export_features_csv(
    recording_path: Path,
    out_path: Path,
    *,
    labels: GestureLabelFile | None = None,
) -> list[FrameFeatureRow]:
    """Export deterministic frame features from a recording to CSV."""
    frames = [
        record.payload
        for record in iter_recording(recording_path)
        if record.kind == "tracking_frame" and isinstance(record.payload, TrackingFrame)
    ]
    rows = extract_feature_rows(frames, labels=labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return rows
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    return rows
