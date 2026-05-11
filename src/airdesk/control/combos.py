"""Short per-hand combo buffer for stable pose events."""

from __future__ import annotations

from dataclasses import dataclass, field

from airdesk.control.debounce import PoseEvent


@dataclass(frozen=True)
class ComboConfig:
    """Combo-buffer limits."""

    max_events: int = 4
    expiry_seconds: float = 2.0


@dataclass
class ComboBuffer:
    """Stores recent stable pose-entry events and consumes matched combos."""

    config: ComboConfig = ComboConfig()
    _events: list[PoseEvent] = field(default_factory=list)

    def add(self, event: PoseEvent) -> None:
        if event.event_type != "entered":
            return
        self._events.append(event)
        self._events = self._events[-self.config.max_events :]
        self.expire(now=event.timestamp)

    def expire(self, *, now: float) -> None:
        self._events = [
            event for event in self._events if now - event.timestamp <= self.config.expiry_seconds
        ]

    def match(self, sequence: tuple[str, ...], *, now: float, hand_id: str | None = None) -> bool:
        """Return true when the latest same-hand event sequence matches."""
        self.expire(now=now)
        if not sequence:
            return False
        candidates = self._events
        if hand_id is not None:
            candidates = [event for event in candidates if event.hand_id == hand_id]
        if len(candidates) < len(sequence):
            return False
        tail = candidates[-len(sequence) :]
        if len({event.hand_id for event in tail}) != 1:
            return False
        if tuple(event.pose for event in tail) != sequence:
            return False
        consumed = set(tail)
        self._events = [event for event in self._events if event not in consumed]
        return True

    def summary(self, *, now: float) -> str:
        self.expire(now=now)
        return " -> ".join(f"{event.hand_id}:{event.pose}" for event in self._events)
