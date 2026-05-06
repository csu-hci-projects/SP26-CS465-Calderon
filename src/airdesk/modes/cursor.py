"""Modeful cursor control from normalized hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

from airdesk.actions.cursor import CursorBounds, CursorPosition
from airdesk.gestures.primitives import INDEX_TIP, THUMB_TIP
from airdesk.state.types import NormalizedHand, TrackingFrame


@dataclass(frozen=True)
class CursorControlConfig:
    """Tuning for pinch-held cursor takeover."""

    pinch_threshold: float = 0.06
    release_threshold: float = 0.08
    gain: float = 1.8
    smoothing_alpha: float = 0.35
    dead_zone_px: int = 3
    max_step_px: int = 140
    mirror_x: bool = True


@dataclass(frozen=True)
class CursorControlUpdate:
    """Result of one cursor-control frame update."""

    active: bool
    moved: bool = False
    position: CursorPosition | None = None
    event: str | None = None
    detail: str = ""


@dataclass
class PinchCursorController:
    """Pinch-hold relative cursor controller.

    A pinch starts cursor mode at the current compositor cursor position. While
    the pinch remains held, hand movement is mapped as relative movement, which
    avoids jumping the cursor to the webcam hand coordinate.
    """

    config: CursorControlConfig = CursorControlConfig()
    active: bool = False
    _anchor_hand: tuple[float, float] | None = None
    _anchor_cursor: CursorPosition | None = None
    _last_position: CursorPosition | None = None

    def update(
        self,
        frame: TrackingFrame,
        *,
        current_cursor: CursorPosition,
        bounds: CursorBounds,
        paused: bool = False,
    ) -> CursorControlUpdate:
        hand = frame.hands[0] if frame.hands else None
        if paused:
            return CursorControlUpdate(active=self.active, event="paused")
        if hand is None:
            return self._release("tracking_lost")

        pinching = self._is_pinching(hand)
        hand_point = self._hand_point(hand)
        if not self.active:
            if not pinching:
                return CursorControlUpdate(active=False)
            self.active = True
            self._anchor_hand = hand_point
            self._anchor_cursor = current_cursor
            self._last_position = current_cursor
            return CursorControlUpdate(
                active=True,
                position=current_cursor,
                event="cursor_activated",
                detail=f"anchor={current_cursor.x},{current_cursor.y}",
            )

        if not pinching:
            return self._release("pinch_released")

        assert self._anchor_hand is not None
        assert self._anchor_cursor is not None
        assert self._last_position is not None

        dx = hand_point[0] - self._anchor_hand[0]
        dy = hand_point[1] - self._anchor_hand[1]
        if self.config.mirror_x:
            dx = -dx

        raw = CursorPosition(
            x=round(self._anchor_cursor.x + dx * bounds.width * self.config.gain),
            y=round(self._anchor_cursor.y + dy * bounds.height * self.config.gain),
        )
        clamped = bounds.clamp(raw)
        limited = self._limit_step(self._smooth(clamped))
        if _distance_px(self._last_position, limited) <= self.config.dead_zone_px:
            return CursorControlUpdate(active=True, position=self._last_position)

        self._last_position = bounds.clamp(limited)
        return CursorControlUpdate(active=True, moved=True, position=self._last_position)

    def status_text(self) -> str:
        if not self.active:
            return "cursor idle | pinch-hold to move"
        if self._last_position is None:
            return "cursor active"
        return f"cursor active | {self._last_position.x},{self._last_position.y}"

    def _release(self, reason: str) -> CursorControlUpdate:
        if not self.active:
            return CursorControlUpdate(active=False)
        self.active = False
        self._anchor_hand = None
        self._anchor_cursor = None
        position = self._last_position
        self._last_position = None
        return CursorControlUpdate(
            active=False,
            position=position,
            event="cursor_released",
            detail=reason,
        )

    def _smooth(self, target: CursorPosition) -> CursorPosition:
        if self._last_position is None:
            return target
        alpha = self.config.smoothing_alpha
        return CursorPosition(
            x=round(self._last_position.x + (target.x - self._last_position.x) * alpha),
            y=round(self._last_position.y + (target.y - self._last_position.y) * alpha),
        )

    def _limit_step(self, target: CursorPosition) -> CursorPosition:
        if self._last_position is None:
            return target
        dx = target.x - self._last_position.x
        dy = target.y - self._last_position.y
        max_step = self.config.max_step_px
        return CursorPosition(
            x=self._last_position.x + min(max_step, max(-max_step, dx)),
            y=self._last_position.y + min(max_step, max(-max_step, dy)),
        )

    def _is_pinching(self, hand: NormalizedHand) -> bool:
        points = hand.landmarks.landmarks
        if len(points) <= max(THUMB_TIP, INDEX_TIP):
            return False
        thumb = points[THUMB_TIP]
        index = points[INDEX_TIP]
        threshold = self.config.release_threshold if self.active else self.config.pinch_threshold
        return dist((thumb.x, thumb.y, thumb.z), (index.x, index.y, index.z)) <= threshold

    @staticmethod
    def _hand_point(hand: NormalizedHand) -> tuple[float, float]:
        palm_x, palm_y, _palm_z = hand.palm_center
        return palm_x, palm_y


def _distance_px(first: CursorPosition, second: CursorPosition) -> float:
    return dist((first.x, first.y), (second.x, second.y))
