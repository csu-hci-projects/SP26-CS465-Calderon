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


@dataclass
class ControlGrammar:
    """Map pose transitions, holds, and combos to guarded action intents."""

    config: ControlGrammarConfig = ControlGrammarConfig()
    combo_buffer: ComboBuffer = field(default_factory=ComboBuffer)
    _last_fired: dict[str, float] = field(default_factory=dict)

    def update(
        self,
        *,
        features: list[ControlPoseFeatures],
        events: list[PoseEvent],
        timestamp: float,
    ) -> list[ControlIntent]:
        intents: list[ControlIntent] = []
        feature_by_hand = {item.hand_id: item for item in features}

        for event in events:
            self.combo_buffer.add(event)

        for event in events:
            hand_features = feature_by_hand.get(event.hand_id)
            if hand_features is None:
                continue
            if event.event_type == "entered":
                if event.pose == "index_pinch":
                    intents.extend(
                        self._intent_if_ready(
                            key=f"{event.hand_id}:left_click",
                            timestamp=timestamp,
                            cooldown=self.config.click_cooldown_seconds,
                            intent=ControlIntent(
                                name="left_click",
                                request=ActionRequest(
                                    action_type=POINTER_ACTION,
                                    command="button",
                                    parameters={"button": "left", "action": "click"},
                                    source="control",
                                ),
                                hand_id=event.hand_id,
                                reason="index pinch tap",
                            ),
                        )
                    )
                elif event.pose == "middle_pinch":
                    intents.extend(
                        self._intent_if_ready(
                            key=f"{event.hand_id}:right_click",
                            timestamp=timestamp,
                            cooldown=self.config.click_cooldown_seconds,
                            intent=ControlIntent(
                                name="right_click",
                                request=ActionRequest(
                                    action_type=POINTER_ACTION,
                                    command="button",
                                    parameters={"button": "right", "action": "click"},
                                    source="control",
                                ),
                                hand_id=event.hand_id,
                                reason="thumb/middle pinch tap",
                            ),
                        )
                    )

            if event.event_type == "held":
                if event.pose in {"sideways_open_palm_left", "sideways_open_palm_right"}:
                    direction = "-1" if event.pose.endswith("left") else "+1"
                    intents.extend(
                        self._hyprland_intent_if_ready(
                            key=f"{event.hand_id}:workspace:{direction}",
                            timestamp=timestamp,
                            name=f"workspace_{direction}",
                            command="workspace",
                            args=[direction],
                            hand_id=event.hand_id,
                            reason="sideways open palm hold",
                        )
                    )
                elif event.pose == "fist" and hand_features.palm_zone in {"left", "right"}:
                    direction = "-1" if hand_features.palm_zone == "left" else "+1"
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
