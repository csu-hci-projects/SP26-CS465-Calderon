"""Cursor action adapters."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from airdesk.actions.input import PointerInputTarget, PointerMoveEvent
from airdesk.state.types import ActionResult, utc_timestamp


@dataclass(frozen=True)
class CursorPosition:
    """Global compositor cursor position."""

    x: int
    y: int


@dataclass(frozen=True)
class CursorBounds:
    """Global compositor rectangle available for cursor movement."""

    x: int
    y: int
    width: int
    height: int
    name: str = "unknown"

    @property
    def right(self) -> int:
        return self.x + self.width - 1

    @property
    def bottom(self) -> int:
        return self.y + self.height - 1

    def clamp(self, position: CursorPosition) -> CursorPosition:
        return CursorPosition(
            x=min(self.right, max(self.x, position.x)),
            y=min(self.bottom, max(self.y, position.y)),
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


class CursorTarget(Protocol):
    """Cursor movement target."""

    name: str

    def current_position(self) -> CursorPosition:
        """Return current global cursor position."""

    def bounds(self, monitor: str | None = None) -> CursorBounds:
        """Return the movement rectangle."""

    def move_to(self, position: CursorPosition) -> ActionResult:
        """Move cursor to an absolute global position."""


@dataclass
class DryRunCursorTarget:
    """Cursor target for tests and dry-run cursor sessions."""

    initial_position: CursorPosition = field(default_factory=lambda: CursorPosition(0, 0))
    movement_bounds: CursorBounds = field(
        default_factory=lambda: CursorBounds(x=0, y=0, width=1920, height=1080, name="dry-run")
    )
    moved: list[CursorPosition] = field(default_factory=list)
    name: str = "cursor-dry-run"

    def current_position(self) -> CursorPosition:
        if self.moved:
            return self.moved[-1]
        return self.initial_position

    def bounds(self, monitor: str | None = None) -> CursorBounds:
        return self.movement_bounds

    def move_to(self, position: CursorPosition) -> ActionResult:
        clamped = self.movement_bounds.clamp(position)
        self.moved.append(clamped)
        return ActionResult(
            request_id="cursor-dry-run",
            ok=True,
            target=self.name,
            executed_at=utc_timestamp(),
            message=f"dry-run cursor move to {clamped.x},{clamped.y}",
            command_preview=["cursor.move", str(clamped.x), str(clamped.y)],
        )


@dataclass
class HyprlandCursorTarget:
    """Cursor movement through Hyprland's movecursor dispatcher."""

    runner: CommandRunner = subprocess.run
    name: str = "hyprland-cursor"

    def current_position(self) -> CursorPosition:
        completed = self.runner(
            ["hyprctl", "cursorpos"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or f"hyprctl exited {completed.returncode}"
            raise RuntimeError(f"Could not read Hyprland cursor position: {message}")
        return parse_cursor_position(completed.stdout)

    def bounds(self, monitor: str | None = None) -> CursorBounds:
        completed = self.runner(
            ["hyprctl", "monitors", "-j"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or f"hyprctl exited {completed.returncode}"
            raise RuntimeError(f"Could not read Hyprland monitors: {message}")
        return monitor_bounds_from_json(completed.stdout, monitor=monitor)

    def move_to(self, position: CursorPosition) -> ActionResult:
        command = ["hyprctl", "dispatch", "movecursor", str(position.x), str(position.y)]
        completed = self.runner(command, check=False, capture_output=True, text=True)
        ok = completed.returncode == 0
        output = completed.stdout.strip() if ok else completed.stderr.strip()
        return ActionResult(
            request_id="cursor-move",
            ok=ok,
            target=self.name,
            executed_at=utc_timestamp(),
            message=output or ("ok" if ok else f"hyprctl exited {completed.returncode}"),
            command_preview=command,
        )


@dataclass
class UInputRelativeCursorTarget:
    """Cursor movement through a virtual relative mouse."""

    pointer_target: PointerInputTarget
    reference: CursorTarget = field(default_factory=HyprlandCursorTarget)
    name: str = "uinput-cursor"
    _last_position: CursorPosition | None = None

    def current_position(self) -> CursorPosition:
        if self._last_position is None:
            self._last_position = self.reference.current_position()
        return self._last_position

    def bounds(self, monitor: str | None = None) -> CursorBounds:
        return self.reference.bounds(monitor=monitor)

    def move_to(self, position: CursorPosition) -> ActionResult:
        current = self.current_position()
        dx = position.x - current.x
        dy = position.y - current.y
        result = self.pointer_target.move(PointerMoveEvent(dx=dx, dy=dy))
        if result.ok:
            self._last_position = position
        return ActionResult(
            request_id="cursor-uinput",
            ok=result.ok,
            target=self.name,
            executed_at=result.executed_at,
            message=(
                f"uinput cursor relative move {dx},{dy} to {position.x},{position.y}"
                if result.ok
                else result.message
            ),
            command_preview=[
                "uinput.cursor",
                str(dx),
                str(dy),
                "target",
                str(position.x),
                str(position.y),
            ],
        )


def parse_cursor_position(text: str) -> CursorPosition:
    """Parse `hyprctl cursorpos` output."""
    match = text.strip().replace(" ", "").split(",")
    if len(match) != 2:
        raise ValueError(f"Could not parse cursor position: {text!r}")
    return CursorPosition(x=int(match[0]), y=int(match[1]))


def monitor_bounds_from_json(text: str, *, monitor: str | None = None) -> CursorBounds:
    """Return focused or named monitor bounds from `hyprctl monitors -j` output."""
    monitors = json.loads(text)
    if not isinstance(monitors, list) or not monitors:
        raise ValueError("Hyprland reported no monitors")

    selected = None
    if monitor is not None:
        selected = next((item for item in monitors if item.get("name") == monitor), None)
        if selected is None:
            raise ValueError(f"Hyprland monitor not found: {monitor}")
    if selected is None:
        selected = next((item for item in monitors if item.get("focused") is True), monitors[0])

    return CursorBounds(
        x=int(selected["x"]),
        y=int(selected["y"]),
        width=int(selected["width"]),
        height=int(selected["height"]),
        name=str(selected.get("name", "unknown")),
    )
