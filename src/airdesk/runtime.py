"""Safe AirDesk runtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    results: tuple[ActionResult, ...] = ()


@dataclass
class AirdeskRuntime:
    """Composes tracking, recognition, mode policy, binding resolution, and actions."""

    tracker: HandTrackerBackend
    profile: Profile
    action_target: ActionTarget
    recognizer: StaticHandPoseRecognizer = field(default_factory=StaticHandPoseRecognizer)
    policy: CommandModePolicy = field(default_factory=CommandModePolicy)

    def run(self) -> RuntimeSummary:
        events: list[EventLogEntry] = []
        results: list[ActionResult] = []
        resolver = BindingResolver(self.profile)
        frame_count = 0

        self.tracker.start()
        try:
            for frame in self.tracker.frames():
                frame_count += 1
                events.extend(self._process_frame(frame, resolver, results))
        finally:
            self.tracker.stop()

        return RuntimeSummary(
            frames=frame_count,
            events=len(events),
            actions=len(results),
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


def format_runtime_summary(summary: RuntimeSummary) -> str:
    """Format a runtime summary for CLI output."""
    return f"frames={summary.frames} events={summary.events} actions={summary.actions}"
