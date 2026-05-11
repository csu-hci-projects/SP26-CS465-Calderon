"""Stable pose debouncing for deterministic control."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PoseEvent:
    """One stable pose transition or hold event."""

    hand_id: str
    pose: str
    event_type: str
    timestamp: float
    duration: float = 0.0


@dataclass(frozen=True)
class PoseDebounceConfig:
    """Timing thresholds for stable pose events."""

    enter_frames: int = 2
    release_frames: int = 2
    held_interval_seconds: float = 0.35


@dataclass
class _PoseState:
    active: bool = False
    seen_frames: int = 0
    missing_frames: int = 0
    entered_at: float = 0.0
    last_held_at: float = 0.0


@dataclass
class PoseDebouncer:
    """Emit enter/held/release events instead of per-frame pose spam."""

    config: PoseDebounceConfig = PoseDebounceConfig()
    _states: dict[tuple[str, str], _PoseState] = field(default_factory=dict)

    def update(
        self,
        *,
        hand_id: str,
        timestamp: float,
        active_poses: frozenset[str],
    ) -> list[PoseEvent]:
        events: list[PoseEvent] = []
        keys = set(self._states)
        keys.update((hand_id, pose) for pose in active_poses)

        for key in sorted(keys):
            state_hand_id, pose = key
            if state_hand_id != hand_id:
                continue
            state = self._states.setdefault(key, _PoseState())
            if pose in active_poses:
                state.seen_frames += 1
                state.missing_frames = 0
                if not state.active and state.seen_frames >= self.config.enter_frames:
                    state.active = True
                    state.entered_at = timestamp
                    state.last_held_at = timestamp
                    events.append(PoseEvent(hand_id, pose, "entered", timestamp))
                elif (
                    state.active
                    and timestamp - state.last_held_at >= self.config.held_interval_seconds
                ):
                    state.last_held_at = timestamp
                    events.append(
                        PoseEvent(
                            hand_id,
                            pose,
                            "held",
                            timestamp,
                            duration=timestamp - state.entered_at,
                        )
                    )
            else:
                state.seen_frames = 0
                if not state.active:
                    continue
                state.missing_frames += 1
                if state.missing_frames >= self.config.release_frames:
                    state.active = False
                    events.append(
                        PoseEvent(
                            hand_id,
                            pose,
                            "released",
                            timestamp,
                            duration=timestamp - state.entered_at,
                        )
                    )
                    state.entered_at = 0.0
                    state.last_held_at = 0.0

        return events
