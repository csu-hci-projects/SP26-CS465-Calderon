"""Stateful rule recognizers for intent-gated gesture phrases."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.state.types import GestureCandidate, TrackingFrame


class GesturePhase(StrEnum):
    """Coarse phases used for continuous gesture spotting and later labels."""

    BACKGROUND = "background"
    ARMED = "armed"
    STROKE = "stroke"
    RELEASE = "release"
    CONFIRMED = "confirmed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class PhraseRecognizerConfig:
    """Thresholds for the first Sprint 3 phrase recognizer."""

    arm_gesture: str = "open_palm"
    min_arm_confidence: float = 0.8
    min_displacement: float = 0.18
    max_vertical_ratio: float = 0.7
    min_duration_seconds: float = 0.08
    max_duration_seconds: float = 0.9
    cooldown_seconds: float = 0.7
    history_seconds: float = 1.0


@dataclass
class PhraseState:
    """Per-hand state for one intent-gated phrase."""

    phase: GesturePhase = GesturePhase.BACKGROUND
    armed_at: float | None = None
    armed_position: tuple[float, float] | None = None
    last_confirmed_at: float | None = None


@dataclass
class IntentGatedSwipeRecognizer:
    """Recognizes left/right swipe phrases after an explicit open-palm arm."""

    config: PhraseRecognizerConfig = field(default_factory=PhraseRecognizerConfig)
    pose_recognizer: StaticHandPoseRecognizer = field(default_factory=StaticHandPoseRecognizer)
    name: str = "intent-gated-swipe"

    def __post_init__(self) -> None:
        self._states: dict[str, PhraseState] = {}
        self._history: dict[str, deque[tuple[float, float, float]]] = {}

    def recognize(self, frame: TrackingFrame) -> list[GestureCandidate]:
        static_candidates = list(self.pose_recognizer.recognize(frame))
        armed_hands = {
            candidate.hand_id
            for candidate in static_candidates
            if candidate.name == self.config.arm_gesture
            and candidate.confidence >= self.config.min_arm_confidence
            and candidate.hand_id is not None
        }
        candidates: list[GestureCandidate] = []
        visible_hands = {hand.hand_id for hand in frame.hands}
        for hand in frame.hands:
            candidates.extend(
                self._recognize_hand(
                    timestamp=frame.timestamp,
                    hand_id=hand.hand_id,
                    position=(hand.palm_center[0], hand.palm_center[1]),
                    is_armed=hand.hand_id in armed_hands,
                )
            )
        for hand_id in set(self._states) - visible_hands:
            self._states[hand_id].phase = GesturePhase.CANCELED
        return candidates

    def _recognize_hand(
        self,
        *,
        timestamp: float,
        hand_id: str,
        position: tuple[float, float],
        is_armed: bool,
    ) -> list[GestureCandidate]:
        state = self._states.setdefault(hand_id, PhraseState())
        history = self._history.setdefault(hand_id, deque())
        history.append((timestamp, position[0], position[1]))
        self._trim_history(history, timestamp)

        if self._in_cooldown(state, timestamp):
            state.phase = GesturePhase.RELEASE
            return []

        if not is_armed:
            if state.phase in {GesturePhase.ARMED, GesturePhase.STROKE}:
                state.phase = GesturePhase.CANCELED
            else:
                state.phase = GesturePhase.BACKGROUND
            state.armed_at = None
            state.armed_position = None
            return []

        if state.phase not in {GesturePhase.ARMED, GesturePhase.STROKE}:
            state.phase = GesturePhase.ARMED
            state.armed_at = timestamp
            state.armed_position = position
            return []

        if state.armed_at is None or state.armed_position is None:
            state.phase = GesturePhase.ARMED
            state.armed_at = timestamp
            state.armed_position = position
            return []

        elapsed = timestamp - state.armed_at
        if elapsed > self.config.max_duration_seconds:
            state.phase = GesturePhase.CANCELED
            state.armed_at = timestamp
            state.armed_position = position
            return []

        dx = position[0] - state.armed_position[0]
        dy = position[1] - state.armed_position[1]
        if abs(dx) < self.config.min_displacement:
            return []
        state.phase = GesturePhase.STROKE
        if elapsed < self.config.min_duration_seconds:
            return []
        if abs(dy) > abs(dx) * self.config.max_vertical_ratio:
            return []

        direction = "right" if dx > 0 else "left"
        state.phase = GesturePhase.CONFIRMED
        state.last_confirmed_at = timestamp
        state.armed_at = None
        state.armed_position = None
        confidence = min(1.0, 0.75 + abs(dx))
        return [
            GestureCandidate(
                name=f"swipe_{direction}",
                confidence=confidence,
                timestamp=timestamp,
                hand_id=hand_id,
                metadata={
                    "phase": GesturePhase.CONFIRMED.value,
                    "duration_seconds": elapsed,
                    "dx": dx,
                    "dy": dy,
                    "history_points": len(history),
                    "recognizer": self.name,
                },
            )
        ]

    def _trim_history(self, history: deque[tuple[float, float, float]], timestamp: float) -> None:
        min_timestamp = timestamp - self.config.history_seconds
        while history and history[0][0] < min_timestamp:
            history.popleft()

    def _in_cooldown(self, state: PhraseState, timestamp: float) -> bool:
        if state.last_confirmed_at is None:
            return False
        return timestamp - state.last_confirmed_at < self.config.cooldown_seconds

