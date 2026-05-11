"""Runtime loop for deterministic AirDesk control."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from airdesk.actions.cursor import CursorBounds, CursorPosition, CursorTarget
from airdesk.actions.input import (
    DryRunPointerInputTarget,
    PointerButtonEvent,
    PointerInputTarget,
    PointerScrollEvent,
)
from airdesk.control.debounce import PoseDebouncer, PoseEvent
from airdesk.control.grammar import POINTER_ACTION, ControlGrammar, ControlIntent
from airdesk.control.poses import ControlPoseFeatures, ControlPoseRecognizer
from airdesk.recording.jsonl import JsonlRecordingWriter
from airdesk.state.types import ActionResult, EventLogEntry, utc_timestamp
from airdesk.tracking.interfaces import HandTrackerBackend


@dataclass(frozen=True)
class ControlRuntimeConfig:
    """Tuning for the deterministic control runtime."""

    execute: bool = False
    pause_on_start: bool = False
    cursor_gain: float = 4.5
    cursor_smoothing_alpha: float = 0.25
    cursor_dead_zone_px: int = 1
    mirror_x: bool = True
    scroll_motion_threshold: float = 0.045
    scroll_amount_per_step: int = 1


@dataclass(frozen=True)
class ControlRuntimeSummary:
    """Summary printed after a control session."""

    frames: int
    cursor_moves: int
    action_requests: int
    action_successes: int
    paused: bool


@dataclass
class ControlRuntime:
    """Runs primitive pose logic, combo grammar, and guarded action routing."""

    tracker: HandTrackerBackend
    cursor_target: CursorTarget
    hyprland_target: object
    pointer_target: PointerInputTarget = field(default_factory=DryRunPointerInputTarget)
    pose_recognizer: ControlPoseRecognizer = field(default_factory=ControlPoseRecognizer)
    debouncer: PoseDebouncer = field(default_factory=PoseDebouncer)
    grammar: ControlGrammar = field(default_factory=ControlGrammar)
    event_writer: JsonlRecordingWriter | None = None
    config: ControlRuntimeConfig = ControlRuntimeConfig()
    monitor: str | None = None
    session_id: str = field(default_factory=lambda: str(uuid4()))
    paused: bool = False
    _last_hand_point: tuple[str, float, float] | None = None
    _pinch_scroll_anchor_y: dict[str, float] = field(default_factory=dict)
    _last_status: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.paused = self.config.pause_on_start

    def run(self) -> ControlRuntimeSummary:
        frames = 0
        cursor_moves = 0
        action_requests = 0
        action_successes = 0
        bounds = self.cursor_target.bounds(monitor=self.monitor)
        current_cursor = self.cursor_target.current_position()
        self._emit(
            "control_session_start",
            {
                "execute": self.config.execute,
                "cursor_target": self.cursor_target.name,
                "pointer_target": self.pointer_target.name,
                "hyprland_target": getattr(self.hyprland_target, "name", "unknown"),
                "bounds": {
                    "x": bounds.x,
                    "y": bounds.y,
                    "width": bounds.width,
                    "height": bounds.height,
                    "name": bounds.name,
                },
                "paused_at_start": self.paused,
                "pointer_execute_available": False,
            },
        )

        self.tracker.start()
        try:
            for frame in self.tracker.frames():
                frames += 1
                features = self.pose_recognizer.features_for_frame(frame)
                events: list[PoseEvent] = []
                for hand_features in features:
                    events.extend(
                        self.debouncer.update(
                            hand_id=hand_features.hand_id,
                            timestamp=frame.timestamp,
                            active_poses=hand_features.poses,
                        )
                    )
                scroll_delta_by_hand = self._scroll_delta_by_hand(features)
                self._emit_frame_status(
                    features,
                    events,
                    timestamp=frame.timestamp,
                    scroll_delta_by_hand=scroll_delta_by_hand,
                )

                if self.paused:
                    continue

                moved = self._move_cursor_from_features(
                    features=features,
                    bounds=bounds,
                    current_cursor=current_cursor,
                )
                if moved is not None:
                    result = self.cursor_target.move_to(moved)
                    self._emit(
                        "control_cursor_moved",
                        {"position": moved.__dict__, "result": result.to_dict()},
                    )
                    if result.ok:
                        current_cursor = moved
                        cursor_moves += 1

                for intent in self.grammar.update(
                    features=features,
                    events=events,
                    timestamp=frame.timestamp,
                    scroll_delta_by_hand=scroll_delta_by_hand,
                ):
                    action_requests += 1
                    result = self._execute_intent(intent)
                    if result.ok:
                        action_successes += 1
                        self._last_status["executed"] = intent.name
                        self._last_status["suppressed"] = "none"
                    else:
                        self._last_status["suppressed"] = result.message
                    self._emit(
                        "control_action_result",
                        {
                            "intent": intent.name,
                            "hand_id": intent.hand_id,
                            "reason": intent.reason,
                            "high_risk": intent.high_risk,
                            "result": result.to_dict(),
                        },
                    )
        finally:
            self.tracker.stop()
            close_pointer = getattr(self.pointer_target, "close", None)
            if close_pointer is not None:
                close_pointer()
            self._emit(
                "control_session_finish",
                {
                    "frames": frames,
                    "cursor_moves": cursor_moves,
                    "action_requests": action_requests,
                    "action_successes": action_successes,
                    "paused": self.paused,
                },
            )

        return ControlRuntimeSummary(
            frames=frames,
            cursor_moves=cursor_moves,
            action_requests=action_requests,
            action_successes=action_successes,
            paused=self.paused,
        )

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._emit("control_paused" if self.paused else "control_resumed", {"paused": self.paused})

    def status_text(self) -> str:
        prefix = "paused | " if self.paused else ""
        now = utc_timestamp()
        combo = self.grammar.combo_buffer.summary(now=now)
        seeing = self._last_status.get("seeing", "none")
        armed = self.grammar.armed_summary(now=now)
        executed = self._last_status.get("executed", "none")
        suppressed = self._last_status.get("suppressed", "none")
        target_window = self._last_status.get("target_window", "active")
        return (
            f"{prefix}Seeing: {seeing} | Combo: {combo or 'empty'} | "
            f"Armed: {armed} | Target window: {target_window} | "
            f"Executed: {executed} | Suppressed: {suppressed}"
        )

    def _execute_intent(self, intent: ControlIntent) -> ActionResult:
        if intent.request.command in {"killactive", "movetoworkspace"}:
            self._last_status["target_window"] = self._active_window_title()
        self._emit(
            "control_action_requested",
            {
                "intent": intent.name,
                "request": intent.request.to_dict(),
                "reason": intent.reason,
                "high_risk": intent.high_risk,
            },
        )
        self._last_status["armed"] = intent.name
        if intent.request.action_type == POINTER_ACTION:
            if intent.request.command == "button":
                return self.pointer_target.button(
                    PointerButtonEvent(
                        button=str(intent.request.parameters["button"]),
                        action=str(intent.request.parameters.get("action", "click")),
                    )
                )
            if intent.request.command == "scroll":
                return self.pointer_target.scroll(
                    PointerScrollEvent(amount_y=int(intent.request.parameters["amount_y"]))
                )
        return self.hyprland_target.execute(intent.request)

    def _active_window_title(self) -> str:
        title_getter = getattr(self.hyprland_target, "active_window_title", None)
        if title_getter is None:
            return "active"
        title = title_getter()
        return title or "active"

    def _move_cursor_from_features(
        self,
        *,
        features: list[ControlPoseFeatures],
        bounds: CursorBounds,
        current_cursor: CursorPosition,
    ) -> CursorPosition | None:
        hand_features = next(
            (item for item in features if "open_palm" in item.poses or not item.poses),
            None,
        )
        if hand_features is None:
            self._last_hand_point = None
            return None
        hand_point = (hand_features.hand_id, hand_features.palm_x, hand_features.palm_y)
        if self._last_hand_point is None or self._last_hand_point[0] != hand_features.hand_id:
            self._last_hand_point = hand_point
            return None

        _last_hand_id, last_x, last_y = self._last_hand_point
        dx = hand_features.palm_x - last_x
        dy = hand_features.palm_y - last_y
        if self.config.mirror_x:
            dx = -dx
        target = bounds.clamp(
            CursorPosition(
                x=round(current_cursor.x + dx * bounds.width * self.config.cursor_gain),
                y=round(current_cursor.y + dy * bounds.height * self.config.cursor_gain),
            )
        )
        alpha = self.config.cursor_smoothing_alpha
        smoothed = CursorPosition(
            x=round(current_cursor.x + (target.x - current_cursor.x) * alpha),
            y=round(current_cursor.y + (target.y - current_cursor.y) * alpha),
        )
        self._last_hand_point = hand_point
        if _distance_px(current_cursor, smoothed) <= self.config.cursor_dead_zone_px:
            return None
        return smoothed

    def _scroll_delta_by_hand(self, features: list[ControlPoseFeatures]) -> dict[str, int]:
        deltas: dict[str, int] = {}
        active_hand_ids = {feature.hand_id for feature in features}
        for hand_id in list(self._pinch_scroll_anchor_y):
            if hand_id not in active_hand_ids:
                self._pinch_scroll_anchor_y.pop(hand_id, None)
        for feature in features:
            if "index_pinch" not in feature.poses:
                self._pinch_scroll_anchor_y.pop(feature.hand_id, None)
                continue
            anchor = self._pinch_scroll_anchor_y.setdefault(feature.hand_id, feature.palm_y)
            dy = feature.palm_y - anchor
            if abs(dy) < self.config.scroll_motion_threshold:
                continue
            deltas[feature.hand_id] = (
                -self.config.scroll_amount_per_step
                if dy < 0
                else self.config.scroll_amount_per_step
            )
            self._pinch_scroll_anchor_y[feature.hand_id] = feature.palm_y
        return deltas

    def _emit_frame_status(
        self,
        features: list[ControlPoseFeatures],
        events: list[PoseEvent],
        *,
        timestamp: float,
        scroll_delta_by_hand: dict[str, int],
    ) -> None:
        seeing = [feature.sees() for feature in features]
        event_summaries = [f"{event.hand_id}:{event.pose}:{event.event_type}" for event in events]
        suppressed = "paused" if self.paused else "none"
        if not features:
            suppressed = "no hands"
        self._last_status.update(
            {
                "seeing": "; ".join(seeing) if seeing else "none",
                "suppressed": suppressed,
                "target_window": "active",
                "armed": self.grammar.armed_summary(now=timestamp),
            }
        )
        if self.event_writer is None:
            return
        self._emit(
            "control_seen",
            {
                "seeing": seeing,
                "events": [event.__dict__ for event in events],
                "event_summaries": event_summaries,
                "combo": self.grammar.combo_buffer.summary(now=timestamp),
                "armed": self._last_status.get("armed", "none"),
                "target_window": self._last_status.get("target_window", "active"),
                "executed": self._last_status.get("executed", "none"),
                "suppressed": suppressed,
                "scroll_delta_by_hand": scroll_delta_by_hand,
            },
        )

    def _emit(self, event_type: str, payload: dict[str, object]) -> None:
        if self.event_writer is None:
            return
        self.event_writer.write_event(
            EventLogEntry(
                event_type=event_type,
                timestamp=utc_timestamp(),
                payload=payload,
                session_id=self.session_id,
            )
        )


def format_control_summary(summary: ControlRuntimeSummary) -> str:
    """Return a compact CLI summary."""
    return (
        "control "
        f"frames={summary.frames} "
        f"cursor_moves={summary.cursor_moves} "
        f"actions={summary.action_requests} "
        f"successes={summary.action_successes} "
        f"paused={summary.paused}"
    )


def _distance_px(first: CursorPosition, second: CursorPosition) -> float:
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5
