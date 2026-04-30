from __future__ import annotations

import subprocess
from collections.abc import Sequence

from airdesk.actions.dry_run import DryRunActionTarget
from airdesk.actions.hyprland import HYPRLAND_DISPATCH, HyprlandActionTarget
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
