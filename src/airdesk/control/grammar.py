"""Dry-run-first control grammar over stable pose events."""

from __future__ import annotations

from dataclasses import dataclass, field

from airdesk.actions.hyprland import HYPRLAND_DISPATCH
from airdesk.control.combos import ComboBuffer
from airdesk.control.debounce import PoseEvent
from airdesk.control.poses import ControlPoseFeatures
from airdesk.state.types import ActionRequest

POINTER_ACTION = "pointer.input"


@dataclass(frozen=True)
class ControlIntent:
    """An action the control runtime may route to a target."""

    name: str
    request: ActionRequest
    hand_id: str
    reason: str
    high_risk: bool = False


@dataclass(frozen=True)
class ControlGrammarConfig:
    """Cooldowns and safety timing for the deterministic grammar."""

    command_cooldown_seconds: float = 0.75
    click_cooldown_seconds: float = 0.25
    close_cooldown_seconds: float = 2.0
    tap_max_seconds: float = 0.30
    scroll_cooldown_seconds: float = 0.12
    workspace_arm_seconds: float = 1.50
    window_move_arm_seconds: float = 1.75


@dataclass
class ControlGrammar:
    """Map pose transitions, holds, and combos to guarded action intents."""

    config: ControlGrammarConfig = ControlGrammarConfig()
    combo_buffer: ComboBuffer = field(default_factory=ComboBuffer)
    _last_fired: dict[str, float] = field(default_factory=dict)
    _pending_taps: dict[tuple[str, str], bool] = field(default_factory=dict)
    _workspace_armed_until: dict[str, float] = field(default_factory=dict)
    _window_move_armed_until: dict[str, float] = field(default_factory=dict)

    def update(
        self,
        *,
        features: list[ControlPoseFeatures],
        events: list[PoseEvent],
        timestamp: float,
        scroll_delta_by_hand: dict[str, int] | None = None,
    ) -> list[ControlIntent]:
        intents: list[ControlIntent] = []
        feature_by_hand = {item.hand_id: item for item in features}
        scroll_delta_by_hand = scroll_delta_by_hand or {}
        self._expire_workspace_arms(timestamp)
        self._expire_window_move_arms(timestamp)

        for event in events:
            self.combo_buffer.add(event)

        for event in events:
            hand_features = feature_by_hand.get(event.hand_id)
            if hand_features is None:
                continue
            if event.event_type == "entered":
                if event.pose == "index_pinch":
                    self._pending_taps[(event.hand_id, "index_pinch")] = True
                elif event.pose == "middle_pinch":
                    self._pending_taps[(event.hand_id, "middle_pinch")] = True

            if event.event_type == "held":
                if event.pose == "index_pinch":
                    scroll_delta = scroll_delta_by_hand.get(event.hand_id, 0)
                    if scroll_delta != 0:
                        self._pending_taps[(event.hand_id, "index_pinch")] = False
                        intents.extend(
                            self._intent_if_ready(
                                key=f"{event.hand_id}:scroll",
                                timestamp=timestamp,
                                cooldown=self.config.scroll_cooldown_seconds,
                                intent=ControlIntent(
                                    name="scroll",
                                    request=ActionRequest(
                                        action_type=POINTER_ACTION,
                                        command="scroll",
                                        parameters={"amount_y": scroll_delta},
                                        source="control",
                                    ),
                                    hand_id=event.hand_id,
                                    reason="index pinch hold with vertical motion",
                                ),
                            )
                        )
                if event.pose == "open_palm" and hand_features.palm_zone == "center":
                    self._workspace_armed_until[event.hand_id] = (
                        timestamp + self.config.workspace_arm_seconds
                    )
                elif (
                    event.pose == "open_palm"
                    and hand_features.palm_zone in {"left", "right"}
                    and self._workspace_is_armed(event.hand_id, timestamp)
                ):
                    direction = "+1" if hand_features.palm_zone == "left" else "-1"
                    self._workspace_armed_until.pop(event.hand_id, None)
                    intents.extend(
                        self._hyprland_intent_if_ready(
                            key=f"{event.hand_id}:workspace:{direction}",
                            timestamp=timestamp,
                            name=f"workspace_{direction}",
                            command="workspace",
                            args=[direction],
                            hand_id=event.hand_id,
                            reason="open palm side zone after center arm",
                        )
                    )
                elif event.pose == "fist" and hand_features.palm_zone == "center":
                    self._window_move_armed_until[event.hand_id] = (
                        timestamp + self.config.window_move_arm_seconds
                    )
                elif (
                    event.pose == "fist"
                    and hand_features.palm_zone in {"left", "right"}
                    and self._window_move_is_armed(event.hand_id, timestamp)
                ):
                    direction = "+1" if hand_features.palm_zone == "left" else "-1"
                    self._window_move_armed_until.pop(event.hand_id, None)
                    intents.extend(
                        self._hyprland_intent_if_ready(
                            key=f"{event.hand_id}:move_window:{direction}",
                            timestamp=timestamp,
                            name=f"move_window_{direction}",
                            command="movetoworkspace",
                            args=[direction],
                            hand_id=event.hand_id,
                            reason="fist held in side zone",
                        )
                    )

            if event.event_type == "released":
                if event.pose == "open_palm":
                    self._workspace_armed_until.pop(event.hand_id, None)
                if event.pose == "fist":
                    self._window_move_armed_until.pop(event.hand_id, None)
                if event.pose == "index_pinch":
                    intents.extend(
                        self._click_intent_if_tap(
                            current_features=hand_features,
                            hand_id=event.hand_id,
                            pose="index_pinch",
                            button="left",
                            timestamp=timestamp,
                            duration=event.duration,
                            reason="index pinch tap",
                        )
                    )
                elif event.pose == "middle_pinch":
                    intents.extend(
                        self._click_intent_if_tap(
                            current_features=hand_features,
                            hand_id=event.hand_id,
                            pose="middle_pinch",
                            button="right",
                            timestamp=timestamp,
                            duration=event.duration,
                            reason="thumb/middle pinch tap",
                        )
                    )

        for hand_features in features:
            if self.combo_buffer.match(
                ("open_palm", f"sideways_open_palm_{hand_features.palm_zone}"),
                now=timestamp,
                hand_id=hand_features.hand_id,
            ):
                intents.extend(
                    self._hyprland_intent_if_ready(
                        key=f"{hand_features.hand_id}:launcher",
                        timestamp=timestamp,
                        name="open_launcher",
                        command="global",
                        args=["caelestia:launcher"],
                        hand_id=hand_features.hand_id,
                        reason="open palm to sideways palm combo",
                    )
                )
            if self.combo_buffer.match(
                ("open_palm", "fist", "open_palm"),
                now=timestamp,
                hand_id=hand_features.hand_id,
            ):
                intents.extend(
                    self._hyprland_intent_if_ready(
                        key=f"{hand_features.hand_id}:close_window",
                        timestamp=timestamp,
                        name="close_window",
                        command="killactive",
                        args=[],
                        hand_id=hand_features.hand_id,
                        reason="deliberate open palm/fist/open palm combo",
                        cooldown=self.config.close_cooldown_seconds,
                        high_risk=True,
                    )
                )

        return intents

    def armed_summary(self, *, now: float) -> str:
        """Return a compact runtime summary of currently armed control states."""
        self._expire_window_move_arms(now)
        self._expire_workspace_arms(now)
        armed: list[str] = []
        armed.extend(
            f"{hand_id}:workspace {expires_at - now:.1f}s"
            for hand_id, expires_at in sorted(self._workspace_armed_until.items())
        )
        armed.extend(
            f"{hand_id}:window_move {expires_at - now:.1f}s"
            for hand_id, expires_at in sorted(self._window_move_armed_until.items())
        )
        if not armed:
            return "none"
        return ", ".join(armed)

    def _hyprland_intent_if_ready(
        self,
        *,
        key: str,
        timestamp: float,
        name: str,
        command: str,
        args: list[str],
        hand_id: str,
        reason: str,
        cooldown: float | None = None,
        high_risk: bool = False,
    ) -> list[ControlIntent]:
        return self._intent_if_ready(
            key=key,
            timestamp=timestamp,
            cooldown=cooldown or self.config.command_cooldown_seconds,
            intent=ControlIntent(
                name=name,
                request=ActionRequest(
                    action_type=HYPRLAND_DISPATCH,
                    command=command,
                    parameters={"args": args},
                    source="control",
                ),
                hand_id=hand_id,
                reason=reason,
                high_risk=high_risk,
            ),
        )

    def _click_intent_if_tap(
        self,
        *,
        current_features: ControlPoseFeatures,
        hand_id: str,
        pose: str,
        button: str,
        timestamp: float,
        duration: float,
        reason: str,
    ) -> list[ControlIntent]:
        pending_key = (hand_id, pose)
        pending = self._pending_taps.pop(pending_key, False)
        if not pending or duration > self.config.tap_max_seconds:
            return []
        if not self._is_clean_pinch_release(current_features):
            return []
        return self._intent_if_ready(
            key=f"{hand_id}:{button}_click",
            timestamp=timestamp,
            cooldown=self.config.click_cooldown_seconds,
            intent=ControlIntent(
                name=f"{button}_click",
                request=ActionRequest(
                    action_type=POINTER_ACTION,
                    command="button",
                    parameters={"button": button, "action": "click"},
                    source="control",
                ),
                hand_id=hand_id,
                reason=reason,
            ),
        )

    @staticmethod
    def _is_clean_pinch_release(features: ControlPoseFeatures) -> bool:
        blocked = {"fist", "sideways_open_palm_left", "sideways_open_palm_right"}
        return features.poses.isdisjoint(blocked)

    def _intent_if_ready(
        self,
        *,
        key: str,
        timestamp: float,
        cooldown: float,
        intent: ControlIntent,
    ) -> list[ControlIntent]:
        last = self._last_fired.get(key)
        if last is not None and timestamp - last < cooldown:
            return []
        self._last_fired[key] = timestamp
        return [intent]

    def _window_move_is_armed(self, hand_id: str, timestamp: float) -> bool:
        return self._window_move_armed_until.get(hand_id, 0.0) >= timestamp

    def _workspace_is_armed(self, hand_id: str, timestamp: float) -> bool:
        return self._workspace_armed_until.get(hand_id, 0.0) >= timestamp

    def _expire_workspace_arms(self, timestamp: float) -> None:
        self._workspace_armed_until = {
            hand_id: expires_at
            for hand_id, expires_at in self._workspace_armed_until.items()
            if expires_at >= timestamp
        }

    def _expire_window_move_arms(self, timestamp: float) -> None:
        self._window_move_armed_until = {
            hand_id: expires_at
            for hand_id, expires_at in self._window_move_armed_until.items()
            if expires_at >= timestamp
        }
