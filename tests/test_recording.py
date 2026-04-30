from __future__ import annotations

from pathlib import Path

from airdesk.recording.jsonl import JsonlRecordingWriter, iter_recording
from airdesk.state.types import EventLogEntry, TrackingFrame
from airdesk.tracking.replay import ReplayHandTrackerBackend


def test_jsonl_recording_round_trips_tracking_frame_and_event(
    tmp_path: Path,
    make_empty_tracking_frame: TrackingFrame,
) -> None:
    path = tmp_path / "nested" / "recording.jsonl"
    event = EventLogEntry(
        event_type="gesture_candidate", timestamp=1.1, payload={"name": "open_palm"}
    )

    with JsonlRecordingWriter(path) as writer:
        writer.write_tracking_frame(make_empty_tracking_frame)
        writer.write_event(event)

    records = iter_recording(path)

    assert records[0].kind == "tracking_frame"
    assert records[0].payload == make_empty_tracking_frame
    assert records[1].kind == "event"
    assert records[1].payload == event


def test_replay_backend_yields_deterministic_tracking_frames() -> None:
    backend = ReplayHandTrackerBackend(Path("tests/fixtures/replay-one-frame.jsonl"))

    backend.start()
    frames = list(backend.frames())

    assert len(frames) == 1
    assert frames[0].source_id == "fixture"
