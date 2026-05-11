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
    tap_max_seconds: float = 0.45
    scroll_cooldown_seconds: float = 0.12
    fist_command_arm_seconds: float = 1.25
    workspace_motion_threshold: float = 0.10
    move_window_motion_threshold: float = 0.12
    fist_axis_margin: float = 0.04


@dataclass(frozen=True)
class _FistArm:
    anchor_x: float
    anchor_y: float
    anchor_zone: str
    expires_at: float


@dataclass
class ControlGrammar:
    """Map pose transitions, holds, and combos to guarded action intents."""

    config: ControlGrammarConfig = ControlGrammarConfig()
    combo_buffer: ComboBuffer = field(default_factory=ComboBuffer)
    _last_fired: dict[str, float] = field(default_factory=dict)
    _pending_taps: dict[tuple[str, str], bool] = field(default_factory=dict)
    _held_buttons: dict[tuple[str, str], str] = field(default_factory=dict)
    _fist_arms: dict[str, _FistArm] = field(default_factory=dict)
    last_diagnostics: list[str] = field(default_factory=list)

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
        self.last_diagnostics = []
        self._expire_fist_arms(timestamp)

        for event in events:
            self.combo_buffer.add(event)

        for event in events:
            hand_features = feature_by_hand.get(event.hand_id)
            if event.event_type == "entered":
                if event.pose == "index_pinch":
                    self._pending_taps[(event.hand_id, "index_pinch")] = True
                elif event.pose == "middle_pinch":
                    self._pending_taps[(event.hand_id, "middle_pinch")] = True
                elif event.pose == "fist" and hand_features is not None:
                    self._arm_fist(
                        hand_id=event.hand_id,
                        hand_features=hand_features,
                        timestamp=timestamp,
                        reason="fist entered",
                    )

            if hand_features is None:
                if event.event_type == "released":
                    intents.extend(
                        self._click_intents_for_release_without_features(
                            hand_id=event.hand_id,
                            pose=event.pose,
                            timestamp=timestamp,
                            duration=event.duration,
                        )
                    )
                continue

            if event.event_type == "held":
                if event.pose == "index_pinch":
                    if event.duration > self.config.tap_max_seconds:
                        self._pending_taps[(event.hand_id, "index_pinch")] = False
                        intents.extend(
                            self._button_hold_intent_if_needed(
                                hand_id=event.hand_id,
                                pose="index_pinch",
                                button="left",
                                reason="index pinch hold",
                            )
                        )
                elif event.pose == "middle_pinch":
                    scroll_delta = scroll_delta_by_hand.get(event.hand_id, 0)
                    if scroll_delta != 0:
                        self._pending_taps[(event.hand_id, "middle_pinch")] = False
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
                                    reason="middle pinch hold with vertical motion",
                                ),
                            )
                        )
                if event.pose == "fist":
                    self._arm_fist_if_missing(
                        hand_id=event.hand_id,
                        hand_features=hand_features,
                        timestamp=timestamp,
                        reason="fist held without anchor",
                    )
                    intents.extend(
                        self._fist_motion_intent_if_ready(
                            hand_features=hand_features,
                            timestamp=timestamp,
                        )
                    )

            if event.event_type == "released":
                if (
                    event.pose == "fist"
                    and self._fist_arms.pop(event.hand_id, None) is not None
                ):
                    self.last_diagnostics.append(
                        f"{event.hand_id}: fist arm released without firing"
                    )
                if event.pose == "index_pinch":
                    if self._button_is_held(event.hand_id, "index_pinch"):
                        intents.extend(
                            self._button_release_intent(
                                hand_id=event.hand_id,
                                pose="index_pinch",
                                button="left",
                                reason="index pinch hold release",
                            )
                        )
                    else:
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
        self._expire_fist_arms(now)
        armed: list[str] = []
        armed.extend(
            (
                f"{hand_id}:fist_command {arm.expires_at - now:.1f}s "
                f"anchor={arm.anchor_x:.2f},{arm.anchor_y:.2f}"
            )
            for hand_id, arm in sorted(self._fist_arms.items())
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

    def _click_intents_for_release_without_features(
        self,
        *,
        hand_id: str,
        pose: str,
        timestamp: float,
        duration: float,
    ) -> list[ControlIntent]:
        if pose == "index_pinch":
            if self._button_is_held(hand_id, pose):
                return self._button_release_intent(
                    hand_id=hand_id,
                    pose=pose,
                    button="left",
                    reason="index pinch hold release after tracking dropout",
                )
            return self._click_intent_if_tap(
                current_features=None,
                hand_id=hand_id,
                pose=pose,
                button="left",
                timestamp=timestamp,
                duration=duration,
                reason="index pinch tap after tracking dropout",
            )
        if pose == "middle_pinch":
            return self._click_intent_if_tap(
                current_features=None,
                hand_id=hand_id,
                pose=pose,
                button="right",
                timestamp=timestamp,
                duration=duration,
                reason="thumb/middle pinch tap after tracking dropout",
            )
        return []

    def _button_hold_intent_if_needed(
        self,
        *,
        hand_id: str,
        pose: str,
        button: str,
        reason: str,
    ) -> list[ControlIntent]:
        held_key = (hand_id, pose)
        if held_key in self._held_buttons:
            return []
        self._held_buttons[held_key] = button
        return [
            ControlIntent(
                name=f"{button}_button_down",
                request=ActionRequest(
                    action_type=POINTER_ACTION,
                    command="button",
                    parameters={"button": button, "action": "press"},
                    source="control",
                ),
                hand_id=hand_id,
                reason=reason,
            )
        ]

    def _button_release_intent(
        self,
        *,
        hand_id: str,
        pose: str,
        button: str,
        reason: str,
    ) -> list[ControlIntent]:
        held_key = (hand_id, pose)
        if self._held_buttons.pop(held_key, None) is None:
            return []
        return [
            ControlIntent(
                name=f"{button}_button_up",
                request=ActionRequest(
                    action_type=POINTER_ACTION,
                    command="button",
                    parameters={"button": button, "action": "release"},
                    source="control",
                ),
                hand_id=hand_id,
                reason=reason,
            )
        ]

    def _button_is_held(self, hand_id: str, pose: str) -> bool:
        return (hand_id, pose) in self._held_buttons

    def _arm_fist(
        self,
        *,
        hand_id: str,
        hand_features: ControlPoseFeatures,
        timestamp: float,
        reason: str,
    ) -> None:
        self._fist_arms[hand_id] = _FistArm(
            anchor_x=hand_features.palm_x,
            anchor_y=hand_features.palm_y,
            anchor_zone=hand_features.palm_zone,
            expires_at=timestamp + self.config.fist_command_arm_seconds,
        )
        self.last_diagnostics.append(
            f"{hand_id}: fist armed from {reason} "
            f"at x={hand_features.palm_x:.3f} y={hand_features.palm_y:.3f}"
        )

    def _arm_fist_if_missing(
        self,
        *,
        hand_id: str,
        hand_features: ControlPoseFeatures,
        timestamp: float,
        reason: str,
    ) -> None:
        if hand_id in self._fist_arms:
            return
        self._arm_fist(
            hand_id=hand_id,
            hand_features=hand_features,
            timestamp=timestamp,
            reason=reason,
        )

    def _fist_motion_intent_if_ready(
        self, *, hand_features: ControlPoseFeatures, timestamp: float
    ) -> list[ControlIntent]:
        arm = self._fist_arms.get(hand_features.hand_id)
        if arm is None:
            self.last_diagnostics.append(f"{hand_features.hand_id}: fist held with no arm")
            return []
        if arm.expires_at < timestamp:
            self._fist_arms.pop(hand_features.hand_id, None)
            self.last_diagnostics.append(f"{hand_features.hand_id}: fist arm expired")
            return []

        dx = hand_features.palm_x - arm.anchor_x
        dy = hand_features.palm_y - arm.anchor_y
        vertical_ready = abs(dy) >= self.config.workspace_motion_threshold
        zone_crossed = (
            arm.anchor_zone != hand_features.palm_zone
            and hand_features.palm_zone in {"left", "right"}
        )
        horizontal_ready = (
            abs(dx) >= self.config.move_window_motion_threshold
            or (
                zone_crossed
                and abs(dx) >= self.config.move_window_motion_threshold * 0.5
            )
        )

        if vertical_ready and horizontal_ready:
            axis_delta = abs(abs(dy) - abs(dx))
            if axis_delta < self.config.fist_axis_margin:
                self.last_diagnostics.append(
                    f"{hand_features.hand_id}: fist motion ambiguous "
                    f"dx={dx:.3f} dy={dy:.3f}"
                )
                return []
            if abs(dy) > abs(dx):
                horizontal_ready = False
            else:
                vertical_ready = False

        if vertical_ready:
            direction = "-1" if dy < 0 else "+1"
            self._fist_arms.pop(hand_features.hand_id, None)
            return self._hyprland_intent_if_ready(
                key=f"{hand_features.hand_id}:workspace:{direction}",
                timestamp=timestamp,
                name=f"workspace_{direction}",
                command="workspace",
                args=[direction],
                hand_id=hand_features.hand_id,
                reason=f"fist vertical motion from anchor dy={dy:.3f}",
            )

        if horizontal_ready:
            direction = "+1" if dx < 0 or hand_features.palm_zone == "left" else "-1"
            self._fist_arms.pop(hand_features.hand_id, None)
            return self._hyprland_intent_if_ready(
                key=f"{hand_features.hand_id}:move_window:{direction}",
                timestamp=timestamp,
                name=f"move_window_{direction}",
                command="movetoworkspace",
                args=[direction],
                hand_id=hand_features.hand_id,
                reason=(
                    "fist horizontal motion from anchor "
                    f"dx={dx:.3f} zone={hand_features.palm_zone}"
                ),
            )

        self.last_diagnostics.append(
            f"{hand_features.hand_id}: fist armed but below motion thresholds "
            f"dx={dx:.3f} dy={dy:.3f}"
        )
        return []

    def _click_intent_if_tap(
        self,
        *,
        current_features: ControlPoseFeatures | None,
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
        if current_features is not None and not self._is_clean_pinch_release(current_features):
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

    def _expire_fist_arms(self, timestamp: float) -> None:
        self._fist_arms = {
            hand_id: arm
            for hand_id, arm in self._fist_arms.items()
            if arm.expires_at >= timestamp
        }
