"""Command-mode clutch policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from airdesk.state.types import EventLogEntry, GestureCandidate, GestureConfirmation


@dataclass(frozen=True)
class CommandModeConfig:
    """Timing and confidence settings for command-mode gesture policy."""

    activation_gesture: str = "open_palm"
    cancel_gesture: str = "fist"
    activation_hold_ms: int = 300
    listening_timeout_ms: int = 1500
    min_confidence: float = 0.75


@dataclass
class CommandModePolicy:
    """Turns primitive gesture candidates into mode and confirmation events."""

    config: CommandModeConfig = CommandModeConfig()
    listening: bool = False
    _activation_started_at: float | None = None
    _listening_started_at: float | None = None
    _confirmed_names: set[str] = field(default_factory=set)

    def update(
        self,
        candidates: list[GestureCandidate],
        *,
        timestamp: float,
        profile_id: str,
    ) -> tuple[list[EventLogEntry], list[GestureConfirmation]]:
        """Advance policy state using candidates from one frame."""
        events: list[EventLogEntry] = []
        confirmations: list[GestureConfirmation] = []
        eligible = [
            candidate
            for candidate in candidates
            if candidate.confidence >= self.config.min_confidence
        ]
        names = {candidate.name for candidate in eligible}

        if self.listening and self._timed_out(timestamp):
            self._exit_listening()
            events.append(_mode_event("command", "idle", timestamp, "timeout", profile_id))

        if self.config.cancel_gesture in names:
            if self.listening:
                self._exit_listening()
                events.append(_mode_event("command", "idle", timestamp, "cancel", profile_id))
            self._activation_started_at = None
            return events, confirmations

        activation = _first_named(eligible, self.config.activation_gesture)
        if activation is not None and not self.listening:
            if self._activation_started_at is None:
                self._activation_started_at = activation.timestamp
            held_ms = (timestamp - self._activation_started_at) * 1000
            if held_ms >= self.config.activation_hold_ms:
                self.listening = True
                self._listening_started_at = timestamp
                self._confirmed_names.clear()
                events.append(
                    _mode_event("idle", "command", timestamp, "activation_hold", profile_id)
                )
                confirmations.append(
                    GestureConfirmation(
                        candidate=activation,
                        confirmed_at=timestamp,
                        profile_id=profile_id,
                        mode="command",
                    )
                )
            return events, confirmations

        if activation is None and not self.listening:
            self._activation_started_at = None

        if not self.listening:
            return events, confirmations

        for candidate in eligible:
            if candidate.name == self.config.activation_gesture:
                continue
            if candidate.name in self._confirmed_names:
                continue
            self._confirmed_names.add(candidate.name)
            confirmations.append(
                GestureConfirmation(
                    candidate=candidate,
                    confirmed_at=timestamp,
                    profile_id=profile_id,
                    mode="command",
                )
            )

        return events, confirmations

    def _timed_out(self, timestamp: float) -> bool:
        if self._listening_started_at is None:
            return False
        elapsed_ms = (timestamp - self._listening_started_at) * 1000
        return elapsed_ms >= self.config.listening_timeout_ms

    def _exit_listening(self) -> None:
        self.listening = False
        self._activation_started_at = None
        self._listening_started_at = None
        self._confirmed_names.clear()


def _first_named(candidates: list[GestureCandidate], name: str) -> GestureCandidate | None:
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    return None


def _mode_event(
    previous_mode: str,
    mode: str,
    timestamp: float,
    reason: str,
    profile_id: str,
) -> EventLogEntry:
    return EventLogEntry(
        event_type="mode_changed",
        timestamp=timestamp,
        payload={
            "previous_mode": previous_mode,
            "mode": mode,
            "reason": reason,
            "profile_id": profile_id,
        },
    )
