"""Hyprland action adapter."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from airdesk.state.types import ActionRequest, ActionResult, utc_timestamp

HYPRLAND_DISPATCH = "hyprland.dispatch"
SAFE_HYPRLAND_DISPATCHERS = frozenset({"workspace", "movefocus"})
CONTROL_HYPRLAND_DISPATCHERS = frozenset(
    {"global", "workspace", "movetoworkspace", "killactive"}
)


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
    verify_after_dispatch: bool = False
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
        before = (
            self._verification_snapshot(request.command)
            if self.verify_after_dispatch
            else None
        )
        completed = self.runner(command, check=False, capture_output=True, text=True)
        ok = completed.returncode == 0
        output = completed.stdout.strip() if ok else completed.stderr.strip()
        message = output or ("ok" if ok else f"hyprctl exited {completed.returncode}")
        if ok and before is not None:
            message = self._verified_message(
                request=request,
                before=before,
                output=output,
            )
        return ActionResult(
            request_id=request.request_id,
            ok=ok,
            target=self.name,
            executed_at=utc_timestamp(),
            message=message,
            command_preview=command,
        )

    def active_window_title(self) -> str | None:
        """Return the active Hyprland window title when available."""
        completed = self.runner(
            ["hyprctl", "activewindow", "-j"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        data = json.loads(completed.stdout)
        if not isinstance(data, dict):
            return None
        title = data.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
        app_class = data.get("class")
        if isinstance(app_class, str) and app_class.strip():
            return app_class.strip()
        return None

    def _verification_snapshot(self, dispatcher: str) -> dict[str, Any] | None:
        if dispatcher == "workspace":
            return {"active_workspace": self._query_json(["hyprctl", "activeworkspace", "-j"])}
        if dispatcher == "movetoworkspace":
            return {"active_window": self._query_json(["hyprctl", "activewindow", "-j"])}
        return None

    def _verified_message(
        self,
        *,
        request: ActionRequest,
        before: dict[str, Any],
        output: str,
    ) -> str:
        base = output or "ok"
        args = request.parameters.get("args", [])
        arg_text = " ".join(str(item) for item in args) if isinstance(args, list) else str(args)
        if request.command == "workspace":
            after = self._query_json(["hyprctl", "activeworkspace", "-j"])
            before_label = _workspace_label(before.get("active_workspace"))
            after_label = _workspace_label(after)
            changed = before_label != after_label
            state = "changed" if changed else "unchanged"
            return (
                f"{base}; verified workspace {state} "
                f"{before_label} -> {after_label} arg={arg_text}"
            )
        if request.command == "movetoworkspace":
            after = self._query_json(["hyprctl", "activewindow", "-j"])
            before_label = _active_window_workspace_label(before.get("active_window"))
            after_label = _active_window_workspace_label(after)
            changed = before_label != after_label
            state = "changed" if changed else "unchanged"
            return (
                f"{base}; verified active window workspace {state} "
                f"{before_label} -> {after_label} arg={arg_text}"
            )
        return base

    def _query_json(self, command: Sequence[str]) -> Any:
        completed = self.runner(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def build_command(request: ActionRequest) -> list[str]:
        args = request.parameters.get("args", [])
        if not isinstance(args, list):
            raise TypeError("Hyprland action parameters['args'] must be a list")
        return ["hyprctl", "dispatch", request.command, *[str(arg) for arg in args]]


def _workspace_label(data: Any) -> str:
    if not isinstance(data, dict):
        return "unknown"
    workspace_id = data.get("id")
    name = data.get("name")
    monitor = data.get("monitor")
    return f"{workspace_id}:{name}@{monitor}"


def _active_window_workspace_label(data: Any) -> str:
    if not isinstance(data, dict):
        return "unknown"
    address = data.get("address") or "unknown-window"
    workspace = data.get("workspace")
    if not isinstance(workspace, dict):
        return f"{address}:unknown-workspace"
    workspace_id = workspace.get("id")
    workspace_name = workspace.get("name")
    return f"{address}:{workspace_id}:{workspace_name}"


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

    def active_window_title(self) -> str | None:
        """Return the guarded target's active window title, if Hyprland reports one."""
        return self.inner.active_window_title()
