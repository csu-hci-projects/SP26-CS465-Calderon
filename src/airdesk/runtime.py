"""Safe AirDesk runtime pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from airdesk.actions.base import ActionTarget
from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.modes.command import CommandModePolicy
from airdesk.profiles.models import Profile
from airdesk.profiles.resolver import BindingResolver
from airdesk.state.types import ActionResult, EventLogEntry, TrackingFrame, utc_timestamp
from airdesk.tracking.interfaces import HandTrackerBackend


@dataclass(frozen=True)
class RuntimeSummary:
    """Counts emitted by a bounded runtime pass."""

    frames: int
    events: int
    actions: int
    session_id: str
    interrupted: bool = False
    duration_seconds: float = 0.0
    results: tuple[ActionResult, ...] = ()


class EventWriter(Protocol):
    """Sink for runtime events."""

    def write_event(self, event: EventLogEntry) -> None:
        """Persist one event."""


@dataclass
class AirdeskRuntime:
    """Composes tracking, recognition, mode policy, binding resolution, and actions."""

    tracker: HandTrackerBackend
    profile: Profile
    action_target: ActionTarget
    recognizer: StaticHandPoseRecognizer = field(default_factory=StaticHandPoseRecognizer)
    policy: CommandModePolicy = field(default_factory=CommandModePolicy)
    event_writer: EventWriter | None = None
    session_metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid4()))

    def run(self) -> RuntimeSummary:
        results: list[ActionResult] = []
        resolver = BindingResolver(self.profile)
        frame_count = 0
        event_count = 0
        started_at = utc_timestamp()
        monotonic_started_at = time.monotonic()
        interrupted = False

        def emit(event: EventLogEntry) -> None:
            nonlocal event_count
            event = _with_session_id(event, self.session_id)
            event_count += 1
            if self.event_writer is not None:
                self.event_writer.write_event(event)

        emit(self._session_start_event(started_at))
        try:
            self.tracker.start()
            for frame in self.tracker.frames():
                frame_count += 1
                for event in self._process_frame(frame, resolver, results):
                    emit(event)
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            self.tracker.stop()
            duration_seconds = time.monotonic() - monotonic_started_at
            emit(
                self._session_finish_event(
                    timestamp=utc_timestamp(),
                    frames=frame_count,
                    events=event_count + 1,
                    actions=len(results),
                    interrupted=interrupted,
                    duration_seconds=duration_seconds,
                )
            )

        return RuntimeSummary(
            frames=frame_count,
            events=event_count,
            actions=len(results),
            session_id=self.session_id,
            interrupted=interrupted,
            duration_seconds=duration_seconds,
            results=tuple(results),
        )

    def _process_frame(
        self,
        frame: TrackingFrame,
        resolver: BindingResolver,
        results: list[ActionResult],
    ) -> list[EventLogEntry]:
        emitted: list[EventLogEntry] = []
        candidates = self.recognizer.recognize(frame)
        for candidate in candidates:
            emitted.append(
                EventLogEntry(
                    event_type="gesture_candidate",
                    timestamp=candidate.timestamp,
                    payload=candidate.to_dict(),
                )
            )
        mode_events, confirmations = self.policy.update(
            candidates,
            timestamp=frame.timestamp,
            profile_id=self.profile.profile_id,
        )
        emitted.extend(mode_events)
        for confirmation in confirmations:
            emitted.append(
                EventLogEntry(
                    event_type="gesture_confirmed",
                    timestamp=confirmation.confirmed_at,
                    payload=confirmation.to_dict(),
                )
            )
            request = resolver.resolve(confirmation)
            if request is None:
                continue
            emitted.append(
                EventLogEntry(
                    event_type="action_requested",
                    timestamp=utc_timestamp(),
                    payload=request.to_dict(),
                )
            )
            result = self.action_target.execute(request)
            results.append(result)
            emitted.append(
                EventLogEntry(
                    event_type="action_executed" if result.ok else "action_failed",
                    timestamp=result.executed_at,
                    payload=result.to_dict(),
                )
            )
        return emitted

    def _session_start_event(self, timestamp: float) -> EventLogEntry:
        return EventLogEntry(
            event_type="session_start",
            timestamp=timestamp,
            payload={
                "backend": self.tracker.name,
                "profile_id": self.profile.profile_id,
                "profile_name": self.profile.name,
                **self.session_metadata,
            },
        )

    def _session_finish_event(
        self,
        *,
        timestamp: float,
        frames: int,
        events: int,
        actions: int,
        interrupted: bool,
        duration_seconds: float,
    ) -> EventLogEntry:
        return EventLogEntry(
            event_type="session_finish",
            timestamp=timestamp,
            payload={
                "frames": frames,
                "events": events,
                "actions": actions,
                "interrupted": interrupted,
                "duration_seconds": duration_seconds,
            },
        )


def format_runtime_summary(summary: RuntimeSummary) -> str:
    """Format a runtime summary for CLI output."""
    return f"frames={summary.frames} events={summary.events} actions={summary.actions}"


def _with_session_id(event: EventLogEntry, session_id: str) -> EventLogEntry:
    if event.session_id == session_id:
        return event
    return EventLogEntry(
        event_type=event.event_type,
        timestamp=event.timestamp,
        payload=event.payload,
        session_id=session_id,
    )
