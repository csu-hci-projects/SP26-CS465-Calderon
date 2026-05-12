"""Dry-run-first control grammar over stable pose events."""

from __future__ import annotations

from dataclasses import dataclass, field

from airdesk.actions.hyprland import HYPRLAND_DISPATCH
from airdesk.control.combos import ComboBuffer
from airdesk.control.debounce import PoseEvent
from airdesk.control.poses import PINCH_POSES, ControlPoseFeatures
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
    click_cooldown_seconds: float = 0.16
    close_cooldown_seconds: float = 2.0
    tap_max_seconds: float = 0.55
    middle_click_max_seconds: float = 1.25
    middle_click_release_margin: float = 0.02
    index_drag_hold_seconds: float = 0.35
    index_drag_motion_threshold: float = 0.025
    scroll_cooldown_seconds: float = 0.12
    fist_command_arm_seconds: float = 1.25
    fist_repeat_cooldown_seconds: float = 0.75
    workspace_motion_threshold: float = 0.10
    move_window_motion_threshold: float = 0.12
    fist_axis_margin: float = 0.04
    workspace_selector_prefix: str = "r"


@dataclass(frozen=True)
class _FistArm:
    anchor_x: float
    anchor_y: float
    anchor_zone: str
    expires_at: float


@dataclass(frozen=True)
class _PinchStart:
    palm_x: float
    palm_y: float
    timestamp: float


@dataclass
class ControlGrammar:
    """Map pose transitions, holds, and combos to guarded action intents."""

    config: ControlGrammarConfig = ControlGrammarConfig()
    combo_buffer: ComboBuffer = field(default_factory=ComboBuffer)
    _last_fired: dict[str, float] = field(default_factory=dict)
    _pending_taps: dict[tuple[str, str], bool] = field(default_factory=dict)
    _pinch_starts: dict[tuple[str, str], _PinchStart] = field(default_factory=dict)
    _blocked_taps: set[tuple[str, str]] = field(default_factory=set)
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
        self._update_conflict_blocks(features)

        for event in events:
            self.combo_buffer.add(event)

        for event in events:
            hand_features = feature_by_hand.get(event.hand_id)
            if event.event_type == "entered":
                if event.pose == "index_pinch":
                    self._pending_taps[(event.hand_id, "index_pinch")] = True
                    self._blocked_taps.discard((event.hand_id, "index_pinch"))
                    self._remember_pinch_start(event, hand_features)
                elif event.pose == "middle_pinch":
                    self._pending_taps[(event.hand_id, "middle_pinch")] = True
                    self._blocked_taps.discard((event.hand_id, "middle_pinch"))
                    self._remember_pinch_start(event, hand_features)
                elif event.pose == "fist" and hand_features is not None:
                    self._cancel_pending_pinch_taps(
                        hand_id=event.hand_id,
                        reason="fist entered",
                    )
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
                    intents.extend(
                        self._index_drag_intent_if_ready(
                            hand_features=hand_features,
                            timestamp=timestamp,
                            reason="index pinch hold",
                        )
                    )
                elif event.pose == "middle_pinch":
                    intents.extend(
                        self._middle_scroll_intent_if_ready(
                            hand_id=event.hand_id,
                            timestamp=timestamp,
                            scroll_delta=scroll_delta_by_hand.get(event.hand_id, 0),
                        )
                    )
                if event.pose == "fist":
                    self._cancel_pending_pinch_taps(
                        hand_id=event.hand_id,
                        reason="fist held",
                    )
                    self._arm_fist_if_missing(
                        hand_id=event.hand_id,
                        hand_features=hand_features,
                        timestamp=timestamp,
                        reason="fist held without anchor",
                    )
                    self._refresh_fist_arm(
                        hand_id=event.hand_id,
                        timestamp=timestamp,
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
                    self._pinch_starts.pop((event.hand_id, "index_pinch"), None)
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
                    self._pinch_starts.pop((event.hand_id, "middle_pinch"), None)
                    intents.extend(
                        self._click_intent_if_tap(
                            current_features=hand_features,
                            hand_id=event.hand_id,
                            pose="middle_pinch",
                            button="right",
                            timestamp=timestamp,
                            duration=event.duration,
                            reason="thumb/middle pinch tap",
                            max_duration=self.config.middle_click_max_seconds,
                        )
                    )

        for hand_features in features:
            if "index_pinch" in hand_features.poses:
                intents.extend(
                    self._index_drag_intent_if_ready(
                        hand_features=hand_features,
                        timestamp=timestamp,
                        reason="index pinch drag motion",
                    )
                )
            if "middle_pinch" in hand_features.poses:
                intents.extend(
                    self._middle_scroll_intent_if_ready(
                        hand_id=hand_features.hand_id,
                        timestamp=timestamp,
                        scroll_delta=scroll_delta_by_hand.get(hand_features.hand_id, 0),
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

    def _update_conflict_blocks(self, features: list[ControlPoseFeatures]) -> None:
        for hand_features in features:
            pinch_conflict = self._ambiguity_blocks_pending_tap(hand_features)
            closed_hand_conflict = (
                "fist" in hand_features.poses
                or hand_features.ambiguity == "forming_fist_pinch_conflict"
            )
            if pinch_conflict or closed_hand_conflict:
                self._cancel_pending_pinch_taps(
                    hand_id=hand_features.hand_id,
                    reason=hand_features.ambiguity or "fist pose active",
                )

    def _cancel_pending_pinch_taps(self, *, hand_id: str, reason: str) -> None:
        blocked: list[str] = []
        for pose in PINCH_POSES:
            key = (hand_id, pose)
            if self._pending_taps.get(key):
                self._pending_taps[key] = False
                self._blocked_taps.add(key)
                blocked.append(pose)
        if blocked:
            self.last_diagnostics.append(
                f"{hand_id}: canceled pending pinch tap(s) "
                f"{','.join(sorted(blocked))} due to {reason}"
            )
            for pose in blocked:
                self._pinch_starts.pop((hand_id, pose), None)

    @staticmethod
    def _ambiguity_blocks_pending_tap(hand_features: ControlPoseFeatures) -> bool:
        if hand_features.ambiguity is None:
            return False
        if not hand_features.suppressed_poses & PINCH_POSES:
            return False
        if hand_features.ambiguity != "index_middle_pinch_conflict":
            return True
        return _has_closed_hand_conflict(hand_features)

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
            self._pinch_starts.pop((hand_id, pose), None)
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
            self._pinch_starts.pop((hand_id, pose), None)
            self._pending_taps.pop((hand_id, pose), None)
            self._blocked_taps.discard((hand_id, pose))
            self.last_diagnostics.append(
                f"{hand_id}: suppressed middle_pinch tap after tracking dropout"
            )
            return []
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

    def _remember_pinch_start(
        self,
        event: PoseEvent,
        hand_features: ControlPoseFeatures | None,
    ) -> None:
        if hand_features is None:
            return
        self._pinch_starts[(event.hand_id, event.pose)] = _PinchStart(
            palm_x=hand_features.palm_x,
            palm_y=hand_features.palm_y,
            timestamp=event.timestamp,
        )

    def _index_drag_intent_if_ready(
        self,
        *,
        hand_features: ControlPoseFeatures,
        timestamp: float,
        reason: str,
    ) -> list[ControlIntent]:
        hand_id = hand_features.hand_id
        pose = "index_pinch"
        if self._button_is_held(hand_id, pose):
            return []
        pending_key = (hand_id, pose)
        start = self._pinch_starts.get(pending_key)
        if start is None:
            return []
        dx = hand_features.palm_x - start.palm_x
        dy = hand_features.palm_y - start.palm_y
        duration = timestamp - start.timestamp
        moved = (dx * dx + dy * dy) ** 0.5 >= self.config.index_drag_motion_threshold
        held = duration >= self.config.index_drag_hold_seconds
        if not moved and not held:
            return []
        self._pending_taps[pending_key] = False
        return self._button_hold_intent_if_needed(
            hand_id=hand_id,
            pose=pose,
            button="left",
            reason=f"{reason} duration={duration:.3f} dx={dx:.3f} dy={dy:.3f}",
        )

    def _middle_scroll_intent_if_ready(
        self,
        *,
        hand_id: str,
        timestamp: float,
        scroll_delta: int,
    ) -> list[ControlIntent]:
        if scroll_delta == 0:
            return []
        self._pending_taps[(hand_id, "middle_pinch")] = False
        return self._intent_if_ready(
            key=f"{hand_id}:scroll",
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
                hand_id=hand_id,
                reason="middle pinch drag with vertical motion",
            ),
        )

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

    def _refresh_fist_arm(self, *, hand_id: str, timestamp: float) -> None:
        arm = self._fist_arms.get(hand_id)
        if arm is None:
            return
        self._fist_arms[hand_id] = _FistArm(
            anchor_x=arm.anchor_x,
            anchor_y=arm.anchor_y,
            anchor_zone=arm.anchor_zone,
            expires_at=timestamp + self.config.fist_command_arm_seconds,
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
            workspace_arg = self._workspace_arg(direction)
            key = f"{hand_features.hand_id}:workspace:{workspace_arg}"
            cooldown_remaining = self._cooldown_remaining(
                key=key,
                timestamp=timestamp,
                cooldown=self.config.fist_repeat_cooldown_seconds,
            )
            if cooldown_remaining > 0:
                self.last_diagnostics.append(
                    f"{hand_features.hand_id}: holding workspace {workspace_arg} "
                    f"repeat cooldown {cooldown_remaining:.2f}s dy={dy:.3f}"
                )
                return []
            self.last_diagnostics.append(
                f"{hand_features.hand_id}: firing workspace {workspace_arg} "
                f"from fist dy={dy:.3f}"
            )
            return self._hyprland_intent_if_ready(
                key=key,
                timestamp=timestamp,
                name=f"workspace_{workspace_arg}",
                command="workspace",
                args=[workspace_arg],
                hand_id=hand_features.hand_id,
                reason=f"fist vertical motion from anchor dy={dy:.3f}",
                cooldown=self.config.fist_repeat_cooldown_seconds,
            )

        if horizontal_ready:
            direction = "+1" if dx < 0 or hand_features.palm_zone == "left" else "-1"
            workspace_arg = self._workspace_arg(direction)
            key = f"{hand_features.hand_id}:move_window:{workspace_arg}"
            cooldown_remaining = self._cooldown_remaining(
                key=key,
                timestamp=timestamp,
                cooldown=self.config.fist_repeat_cooldown_seconds,
            )
            if cooldown_remaining > 0:
                self.last_diagnostics.append(
                    f"{hand_features.hand_id}: holding movetoworkspace {workspace_arg} "
                    f"repeat cooldown {cooldown_remaining:.2f}s dx={dx:.3f} "
                    f"zone={hand_features.palm_zone}"
                )
                return []
            self.last_diagnostics.append(
                f"{hand_features.hand_id}: firing movetoworkspace {workspace_arg} "
                f"from fist dx={dx:.3f} zone={hand_features.palm_zone}"
            )
            return self._hyprland_intent_if_ready(
                key=f"{hand_features.hand_id}:move_window:{workspace_arg}",
                timestamp=timestamp,
                name=f"move_window_{workspace_arg}",
                command="movetoworkspace",
                args=[workspace_arg],
                hand_id=hand_features.hand_id,
                reason=(
                    "fist horizontal motion from anchor "
                    f"dx={dx:.3f} zone={hand_features.palm_zone}"
                ),
                cooldown=self.config.fist_repeat_cooldown_seconds,
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
        max_duration: float | None = None,
    ) -> list[ControlIntent]:
        pending_key = (hand_id, pose)
        pending = self._pending_taps.pop(pending_key, False)
        blocked = pending_key in self._blocked_taps
        self._blocked_taps.discard(pending_key)
        if blocked:
            self.last_diagnostics.append(
                f"{hand_id}: suppressed {pose} tap because the release was ambiguous"
            )
            return []
        max_click_duration = max_duration or self.config.tap_max_seconds
        if not pending or duration > max_click_duration:
            return []
        if current_features is not None and not self._is_clean_pinch_release(
            current_features,
            pose=pose,
        ):
            self.last_diagnostics.append(
                f"{hand_id}: suppressed {pose} tap on non-clean release"
            )
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

    def _is_clean_pinch_release(self, features: ControlPoseFeatures, *, pose: str) -> bool:
        if pose == "middle_pinch":
            return self._is_clean_middle_pinch_release(features)
        if features.ambiguity == "index_middle_pinch_conflict":
            return (
                pose in features.suppressed_poses
                and features.pose_scores.get(pose, 0.0) > 0.0
                and not _has_closed_hand_conflict(features)
            )
        if features.ambiguity is not None:
            return False
        blocked = {
            "fist",
            "sideways_open_palm_left",
            "sideways_open_palm_right",
            *(PINCH_POSES - {pose}),
        }
        return features.poses.isdisjoint(blocked)

    def _is_clean_middle_pinch_release(self, features: ControlPoseFeatures) -> bool:
        if features.ambiguity is not None:
            return False
        blocked = {
            "fist",
            "index_pinch",
            "middle_pinch",
            "sideways_open_palm_left",
            "sideways_open_palm_right",
        }
        if not features.poses.isdisjoint(blocked):
            return False
        evidence = features.pose_evidence.get("middle_pinch")
        threshold = 0.0
        if isinstance(evidence, dict):
            threshold_value = evidence.get("threshold")
            if isinstance(threshold_value, (int, float)):
                threshold = float(threshold_value)
        return (
            features.middle_pinch_distance
            >= threshold + self.config.middle_click_release_margin
        )

    def _workspace_arg(self, direction: str) -> str:
        prefix = self.config.workspace_selector_prefix
        return f"{prefix}{direction}" if prefix else direction

    def _cooldown_remaining(self, *, key: str, timestamp: float, cooldown: float) -> float:
        last = self._last_fired.get(key)
        if last is None:
            return 0.0
        elapsed = timestamp - last
        if elapsed >= cooldown:
            return 0.0
        return cooldown - elapsed

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


def _has_closed_hand_conflict(features: ControlPoseFeatures) -> bool:
    if "fist" in features.poses or features.ambiguity == "forming_fist_pinch_conflict":
        return True
    fist_evidence = features.pose_evidence.get("fist")
    if not isinstance(fist_evidence, dict):
        return False
    return bool(fist_evidence.get("forming_fist"))
