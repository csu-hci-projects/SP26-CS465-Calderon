from __future__ import annotations

import subprocess
from collections.abc import Sequence

from airdesk.actions.dry_run import DryRunActionTarget
from airdesk.actions.hyprland import (
    CONTROL_HYPRLAND_DISPATCHERS,
    HYPRLAND_DISPATCH,
    GuardedHyprlandActionTarget,
    HyprlandActionTarget,
)
from airdesk.actions.input import DryRunPointerInputTarget, PointerButtonEvent, PointerScrollEvent
from airdesk.state.types import ActionRequest


def test_dry_run_records_action_without_executing() -> None:
    target = DryRunActionTarget()
    request = ActionRequest(
        action_type=HYPRLAND_DISPATCH, command="workspace", parameters={"args": ["r+1"]}
    )

    result = target.execute(request)

    assert result.ok is True
    assert target.executed == [request]
    assert result.command_preview == [HYPRLAND_DISPATCH, "workspace", "r+1"]


def test_hyprland_builds_dispatch_command_with_injected_runner() -> None:
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

    target = HyprlandActionTarget(runner=runner)
    request = ActionRequest(
        action_type=HYPRLAND_DISPATCH, command="movefocus", parameters={"args": ["l"]}
    )

    result = target.execute(request)

    assert result.ok is True
    assert calls == [["hyprctl", "dispatch", "movefocus", "l"]]
    assert result.command_preview == calls[0]


def test_hyprland_reads_active_window_title_with_injected_runner() -> None:
    def runner(
        command: Sequence[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"title": "Demo Window", "class": "demo"}\n',
            stderr="",
        )

    target = HyprlandActionTarget(runner=runner)

    assert target.active_window_title() == "Demo Window"


def test_guarded_hyprland_blocks_non_allowlisted_dispatcher() -> None:
    target = GuardedHyprlandActionTarget()
    request = ActionRequest(
        action_type=HYPRLAND_DISPATCH,
        command="killactive",
        parameters={"args": []},
    )

    result = target.execute(request)

    assert result.ok is False
    assert "blocked unsafe" in result.message
    assert result.command_preview == ["hyprctl", "dispatch", "killactive"]


def test_guarded_hyprland_allows_safe_dispatcher_with_injected_runner() -> None:
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

    target = GuardedHyprlandActionTarget(inner=HyprlandActionTarget(runner=runner))
    request = ActionRequest(
        action_type=HYPRLAND_DISPATCH,
        command="workspace",
        parameters={"args": ["r+1"]},
    )

    result = target.execute(request)

    assert result.ok is True
    assert calls == [["hyprctl", "dispatch", "workspace", "r+1"]]


def test_control_guard_allows_demo_dispatchers_with_injected_runner() -> None:
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

    target = GuardedHyprlandActionTarget(
        inner=HyprlandActionTarget(runner=runner),
        allowed_dispatchers=CONTROL_HYPRLAND_DISPATCHERS,
    )
    request = ActionRequest(
        action_type=HYPRLAND_DISPATCH,
        command="movetoworkspace",
        parameters={"args": ["+1"]},
    )

    result = target.execute(request)

    assert result.ok is True
    assert calls == [["hyprctl", "dispatch", "movetoworkspace", "+1"]]


def test_dry_run_pointer_input_records_buttons_and_scrolls() -> None:
    target = DryRunPointerInputTarget()

    button = target.button(PointerButtonEvent(button="left"))
    scroll = target.scroll(PointerScrollEvent(amount_y=-1))

    assert button.command_preview == ["pointer.button", "left", "click"]
    assert scroll.command_preview == ["pointer.scroll", "-1"]
    assert target.buttons == [PointerButtonEvent(button="left")]
    assert target.scrolls == [PointerScrollEvent(amount_y=-1)]
