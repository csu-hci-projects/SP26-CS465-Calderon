"""Hyprland action adapter."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from airdesk.state.types import ActionRequest, ActionResult, utc_timestamp

HYPRLAND_DISPATCH = "hyprland.dispatch"
SAFE_HYPRLAND_DISPATCHERS = frozenset({"workspace", "movefocus"})


class CommandRunner(Protocol):
    """Small protocol so tests can verify command mapping without Hyprland."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command and return a completed process."""


@dataclass
class HyprlandActionTarget:
    """Wraps `hyprctl dispatch` behind an action target boundary."""

    runner: CommandRunner = subprocess.run
    name: str = "hyprland"

    def execute(self, request: ActionRequest) -> ActionResult:
        if request.action_type != HYPRLAND_DISPATCH:
            return ActionResult(
                request_id=request.request_id,
                ok=False,
                target=self.name,
                executed_at=utc_timestamp(),
                message=f"unsupported Hyprland action type: {request.action_type}",
            )

        command = self.build_command(request)
        completed = self.runner(command, check=False, capture_output=True, text=True)
        ok = completed.returncode == 0
        output = completed.stdout.strip() if ok else completed.stderr.strip()
        return ActionResult(
            request_id=request.request_id,
            ok=ok,
            target=self.name,
            executed_at=utc_timestamp(),
            message=output or ("ok" if ok else f"hyprctl exited {completed.returncode}"),
            command_preview=command,
        )

    @staticmethod
    def build_command(request: ActionRequest) -> list[str]:
        args = request.parameters.get("args", [])
        if not isinstance(args, list):
            raise TypeError("Hyprland action parameters['args'] must be a list")
        return ["hyprctl", "dispatch", request.command, *[str(arg) for arg in args]]


@dataclass
class GuardedHyprlandActionTarget:
    """Allowlisted Hyprland target for pilot-safe real execution."""

    inner: HyprlandActionTarget = field(default_factory=HyprlandActionTarget)
    allowed_dispatchers: frozenset[str] = SAFE_HYPRLAND_DISPATCHERS
    name: str = "hyprland-guarded"

    def execute(self, request: ActionRequest) -> ActionResult:
        if request.action_type != HYPRLAND_DISPATCH:
            return ActionResult(
                request_id=request.request_id,
                ok=False,
                target=self.name,
                executed_at=utc_timestamp(),
                message=f"unsupported action type for guarded execution: {request.action_type}",
            )
        if request.command not in self.allowed_dispatchers:
            return ActionResult(
                request_id=request.request_id,
                ok=False,
                target=self.name,
                executed_at=utc_timestamp(),
                message=f"blocked unsafe Hyprland dispatcher: {request.command}",
                command_preview=HyprlandActionTarget.build_command(request),
            )
        return self.inner.execute(request)
