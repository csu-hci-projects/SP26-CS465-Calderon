"""Safe AirDesk runtime pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from airdesk.actions.base import ActionTarget
from airdesk.gestures.base import CompositeGestureRecognizer, GestureRecognizer
from airdesk.gestures.phrases import IntentGatedSwipeRecognizer
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


@dataclass(frozen=True)
class RuntimeStatus:
    """Small status surface for live preview and tests."""

    state: str = "idle"
    detail: str = ""
    paused: bool = False


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
    recognizer: GestureRecognizer = field(default_factory=lambda: _default_runtime_recognizer())
    policy: CommandModePolicy = field(default_factory=CommandModePolicy)
    event_writer: EventWriter | None = None
    session_metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid4()))
    paused: bool = False
    status: RuntimeStatus = field(default_factory=RuntimeStatus)
    _pending_events: list[EventLogEntry] = field(default_factory=list, init=False)

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
        self._set_status("paused" if self.paused else "idle")
        try:
            self.tracker.start()
            for frame in self.tracker.frames():
                frame_count += 1
                for event in self._process_frame(frame, resolver, results):
                    emit(event)
                for event in self._drain_pending_events():
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

    def set_paused(self, paused: bool) -> EventLogEntry | None:
        """Set paused state and return an event when the state changed."""
        if self.paused == paused:
            return None
        self.paused = paused
        self._set_status("paused" if paused else "idle", "keyboard_toggle")
        event = EventLogEntry(
            event_type="runtime_paused" if paused else "runtime_resumed",
            timestamp=utc_timestamp(),
            payload={"paused": paused, "reason": "keyboard_toggle"},
            session_id=self.session_id,
        )
        self._pending_events.append(event)
        return event

    def toggle_pause(self) -> bool:
        """Toggle paused state for preview key handlers."""
        self.set_paused(not self.paused)
        return self.paused

    def status_text(self) -> str:
        """Return compact live status text."""
        return format_runtime_status(self.status)

    def _drain_pending_events(self) -> list[EventLogEntry]:
        events = self._pending_events
        self._pending_events = []
        return events

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
        self._update_status_from_events(mode_events)
        if self.paused:
            if confirmations:
                self._set_status("paused", "action_suppressed")
                emitted.append(
                    EventLogEntry(
                        event_type="action_blocked",
                        timestamp=utc_timestamp(),
                        payload={
                            "reason": "paused",
                            "confirmations": [
                                confirmation.to_dict() for confirmation in confirmations
                            ],
                        },
                    )
                )
            return emitted
        for confirmation in confirmations:
            self._set_status("confirmed", confirmation.candidate.name)
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
            self._set_status("action_requested", request.command)
            result = self.action_target.execute(request)
            results.append(result)
            self._set_status("action_executed" if result.ok else "blocked", result.message)
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

    def _set_status(self, state: str, detail: str = "") -> None:
        self.status = RuntimeStatus(state=state, detail=detail, paused=self.paused)

    def _update_status_from_events(self, events: list[EventLogEntry]) -> None:
        for event in events:
            if event.event_type != "mode_changed":
                continue
            mode = str(event.payload.get("mode", "idle"))
            reason = str(event.payload.get("reason", ""))
            self._set_status("listening" if mode == "command" else "idle", reason)


def format_runtime_summary(summary: RuntimeSummary) -> str:
    """Format a runtime summary for CLI output."""
    return f"frames={summary.frames} events={summary.events} actions={summary.actions}"


def format_runtime_status(status: RuntimeStatus) -> str:
    """Format runtime status for live preview overlays."""
    prefix = "paused" if status.paused else status.state
    return f"{prefix}: {status.detail}" if status.detail else prefix


def _with_session_id(event: EventLogEntry, session_id: str) -> EventLogEntry:
    if event.session_id == session_id:
        return event
    return EventLogEntry(
        event_type=event.event_type,
        timestamp=event.timestamp,
        payload=event.payload,
        session_id=session_id,
    )


def _default_runtime_recognizer() -> CompositeGestureRecognizer:
    static_recognizer = StaticHandPoseRecognizer()
    return CompositeGestureRecognizer(
        recognizers=(
            static_recognizer,
            IntentGatedSwipeRecognizer(pose_recognizer=static_recognizer),
        )
    )
