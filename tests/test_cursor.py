from __future__ import annotations

import subprocess
from collections.abc import Sequence

from airdesk.actions.cursor import (
    CursorBounds,
    CursorPosition,
    HyprlandCursorTarget,
    monitor_bounds_from_json,
    parse_cursor_position,
)
from airdesk.modes.cursor import CursorControlConfig, PinchCursorController
from airdesk.state.types import (
    FrameMetadata,
    HandLandmarks,
    Landmark,
    NormalizedHand,
    TrackingFrame,
)


def test_parse_hyprland_cursor_position() -> None:
    assert parse_cursor_position("3232, 1174\n") == CursorPosition(3232, 1174)


def test_monitor_bounds_prefers_focused_monitor() -> None:
    text = """
    [
      {"name": "eDP-1", "x": 0, "y": 0, "width": 2560, "height": 1600, "focused": false},
      {"name": "HDMI-A-1", "x": 2560, "y": 0, "width": 3440, "height": 1440, "focused": true}
    ]
    """

    bounds = monitor_bounds_from_json(text)

    assert bounds == CursorBounds(x=2560, y=0, width=3440, height=1440, name="HDMI-A-1")


def test_hyprland_cursor_target_moves_with_injected_runner() -> None:
    calls: list[list[str]] = []

    def runner(
        command: Sequence[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    target = HyprlandCursorTarget(runner=runner)

    result = target.move_to(CursorPosition(100, 200))

    assert result.ok is True
    assert calls == [["hyprctl", "dispatch", "movecursor", "100", "200"]]
    assert result.command_preview == calls[0]


def test_pinch_cursor_controller_moves_relative_to_activation_anchor() -> None:
    controller = PinchCursorController(
        CursorControlConfig(
            gain=2.0,
            smoothing_alpha=1.0,
            max_step_px=1000,
            dead_zone_px=0,
            mirror_x=True,
        )
    )
    bounds = CursorBounds(x=0, y=0, width=1000, height=1000)
    cursor = CursorPosition(100, 100)

    activated = controller.update(
        _frame(_pinch_hand(0.50, 0.50)),
        current_cursor=cursor,
        bounds=bounds,
    )
    moved = controller.update(_frame(_pinch_hand(0.45, 0.50)), current_cursor=cursor, bounds=bounds)
    released = controller.update(
        _frame(_open_hand(0.45, 0.50)),
        current_cursor=cursor,
        bounds=bounds,
    )

    assert activated.event == "cursor_activated"
    assert moved.moved is True
    assert moved.position == CursorPosition(200, 100)
    assert released.event == "cursor_released"
    assert released.detail == "pinch_released"


def _frame(hand: NormalizedHand) -> TrackingFrame:
    metadata = FrameMetadata(timestamp=1.0, source_id="test", width=640, height=480, sequence=1)
    return TrackingFrame(timestamp=1.0, source_id="test", frame=metadata, hands=(hand,))


def _pinch_hand(palm_x: float, palm_y: float) -> NormalizedHand:
    landmarks = [Landmark(palm_x, palm_y, 0.0) for _ in range(21)]
    landmarks[4] = Landmark(palm_x - 0.01, palm_y, 0.0)
    landmarks[8] = Landmark(palm_x - 0.01, palm_y, 0.0)
    return _hand(landmarks, palm_x, palm_y)


def _open_hand(palm_x: float, palm_y: float) -> NormalizedHand:
    landmarks = [Landmark(palm_x, palm_y, 0.0) for _ in range(21)]
    landmarks[4] = Landmark(palm_x - 0.2, palm_y, 0.0)
    landmarks[8] = Landmark(palm_x + 0.2, palm_y, 0.0)
    return _hand(landmarks, palm_x, palm_y)


def _hand(landmarks: list[Landmark], palm_x: float, palm_y: float) -> NormalizedHand:
    return NormalizedHand(
        hand_id="hand-0",
        landmarks=HandLandmarks(tuple(landmarks), handedness="right", confidence=1.0),
        palm_center=(palm_x, palm_y, 0.0),
        bbox=(palm_x - 0.1, palm_y - 0.1, palm_x + 0.1, palm_y + 0.1),
        handedness="right",
        confidence=1.0,
    )
