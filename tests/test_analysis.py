from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from airdesk.analysis.recording import analyze_recording
from airdesk.recording.jsonl import JsonlRecordingWriter
from airdesk.state.types import FrameMetadata, NormalizedHand, TrackingFrame


def frame_at(timestamp: float, sequence: int, hand: NormalizedHand) -> TrackingFrame:
    metadata = FrameMetadata(
        timestamp=timestamp,
        source_id="analysis-test",
        width=640,
        height=480,
        sequence=sequence,
    )
    return TrackingFrame(
        timestamp=timestamp,
        source_id="analysis-test",
        frame=metadata,
        hands=(hand,),
    )


def test_analyze_recording_reports_counts_fps_runs_and_jitter(
    tmp_path: Path,
    make_hand: Callable[[str], NormalizedHand],
) -> None:
    path = tmp_path / "analysis.jsonl"
    with JsonlRecordingWriter(path) as writer:
        writer.write_tracking_frame(frame_at(1.0, 1, make_hand("open_palm")))
        writer.write_tracking_frame(frame_at(1.1, 2, make_hand("open_palm")))
        writer.write_tracking_frame(frame_at(1.2, 3, make_hand("pinch")))

    analysis = analyze_recording(path)

    assert analysis.frames == 3
    assert analysis.hand_frames == 3
    assert analysis.average_fps == pytest.approx(10)
    assert analysis.candidate_counts["open_palm"] == 3
    assert analysis.longest_runs["open_palm"] == 3
    assert analysis.candidate_counts["pinch"] == 1
    assert "landmark_0" in analysis.landmark_jitter
