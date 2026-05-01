from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from airdesk.actions.dry_run import DryRunActionTarget
from airdesk.profiles.loader import load_profile
from airdesk.runtime import AirdeskRuntime, RuntimeStatus, format_runtime_status
from airdesk.state.types import EventLogEntry, FrameMetadata, NormalizedHand, TrackingFrame
from airdesk.tracking.replay import MockHandTrackerBackend


def frame_at(timestamp: float, sequence: int, hand: NormalizedHand) -> TrackingFrame:
    metadata = FrameMetadata(
        timestamp=timestamp,
        source_id="runtime-test",
        width=640,
        height=480,
        sequence=sequence,
    )
    return TrackingFrame(
        timestamp=timestamp,
        source_id="runtime-test",
        frame=metadata,
        hands=(hand,),
    )


def test_runtime_routes_replayed_gestures_to_dry_run(
    make_hand: Callable[[str], NormalizedHand],
) -> None:
    frames = [
        frame_at(1.0, 1, make_hand("open_palm")),
        frame_at(1.4, 2, make_hand("open_palm")),
        frame_at(1.5, 3, make_hand("pinch")),
    ]
    target = DryRunActionTarget()
    runtime = AirdeskRuntime(
        tracker=MockHandTrackerBackend(frames),
        profile=load_profile(Path("configs/profiles/study-safe.toml")),
        action_target=target,
    )

    summary = runtime.run()

    assert summary.frames == 3
    assert summary.actions == 2
    assert target.executed[-1].command == "pinch-observed"


def test_paused_runtime_suppresses_action_execution(
    make_hand: Callable[[str], NormalizedHand],
) -> None:
    frames = [
        frame_at(1.0, 1, make_hand("open_palm")),
        frame_at(1.4, 2, make_hand("open_palm")),
        frame_at(1.5, 3, make_hand("pinch")),
    ]
    target = DryRunActionTarget()
    writer = CapturingEventWriter()
    runtime = AirdeskRuntime(
        tracker=MockHandTrackerBackend(frames),
        profile=load_profile(Path("configs/profiles/study-safe.toml")),
        action_target=target,
        event_writer=writer,
        paused=True,
    )

    summary = runtime.run()

    assert summary.frames == 3
    assert summary.actions == 0
    assert target.executed == []
    assert "action_blocked" in [event.event_type for event in writer.events]
    assert runtime.status.state == "paused"
    assert runtime.status.detail == "action_suppressed"


def test_runtime_status_formatter_prioritizes_pause() -> None:
    assert format_runtime_status(RuntimeStatus(state="listening")) == "listening"
    assert (
        format_runtime_status(RuntimeStatus(state="action_requested", detail="workspace"))
        == "action_requested: workspace"
    )
    assert format_runtime_status(RuntimeStatus(state="listening", paused=True)) == "paused"


class CapturingEventWriter:
    def __init__(self) -> None:
        self.events: list[EventLogEntry] = []

    def write_event(self, event: EventLogEntry) -> None:
        self.events.append(event)
