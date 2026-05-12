"""Pointer button and scroll action adapters."""

from __future__ import annotations

import fcntl
import os
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from airdesk.state.types import ActionResult, utc_timestamp

EV_SYN = 0x00
EV_KEY = 0x01
EV_REL = 0x02
SYN_REPORT = 0x00
REL_X = 0x00
REL_Y = 0x01
REL_WHEEL = 0x08
BTN_LEFT = 0x110
BTN_RIGHT = 0x111
BUS_USB = 0x03
UI_DEV_CREATE = 0x5501
UI_DEV_DESTROY = 0x5502
UI_SET_EVBIT = 0x40045564
UI_SET_KEYBIT = 0x40045565
UI_SET_RELBIT = 0x40045566


@dataclass(frozen=True)
class PointerButtonEvent:
    """Discrete pointer button request."""

    button: str
    action: str = "click"


@dataclass(frozen=True)
class PointerScrollEvent:
    """Discrete pointer scroll request."""

    amount_y: int


@dataclass(frozen=True)
class PointerMoveEvent:
    """Relative pointer movement request."""

    dx: int
    dy: int


class PointerInputTarget(Protocol):
    """Pointer movement/button/scroll target boundary."""

    name: str

    def move(self, event: PointerMoveEvent) -> ActionResult:
        """Send a relative pointer movement request."""

    def button(self, event: PointerButtonEvent) -> ActionResult:
        """Send a pointer button request."""

    def scroll(self, event: PointerScrollEvent) -> ActionResult:
        """Send a pointer scroll request."""


@dataclass
class DryRunPointerInputTarget:
    """Pointer target for tests and dry-run control sessions."""

    name: str = "pointer-dry-run"
    moves: list[PointerMoveEvent] = field(default_factory=list)
    buttons: list[PointerButtonEvent] = field(default_factory=list)
    scrolls: list[PointerScrollEvent] = field(default_factory=list)

    def move(self, event: PointerMoveEvent) -> ActionResult:
        self.moves.append(event)
        return ActionResult(
            request_id="pointer-move-dry-run",
            ok=True,
            target=self.name,
            executed_at=utc_timestamp(),
            message=f"dry-run pointer move {event.dx},{event.dy}",
            command_preview=["pointer.move", str(event.dx), str(event.dy)],
        )

    def button(self, event: PointerButtonEvent) -> ActionResult:
        self.buttons.append(event)
        return ActionResult(
            request_id="pointer-button-dry-run",
            ok=True,
            target=self.name,
            executed_at=utc_timestamp(),
            message=f"dry-run pointer {event.button} {event.action}",
            command_preview=["pointer.button", event.button, event.action],
        )

    def scroll(self, event: PointerScrollEvent) -> ActionResult:
        self.scrolls.append(event)
        return ActionResult(
            request_id="pointer-scroll-dry-run",
            ok=True,
            target=self.name,
            executed_at=utc_timestamp(),
            message=f"dry-run pointer scroll {event.amount_y}",
            command_preview=["pointer.scroll", str(event.amount_y)],
        )


@dataclass
class UInputPointerInputTarget:
    """Linux `/dev/uinput` pointer-button and wheel target."""

    device_path: str = "/dev/uinput"
    name: str = "pointer-uinput"
    opener: Callable[[str, int], int] = os.open
    writer: Callable[[int, bytes], int] = os.write
    ioctl: Callable[[int, int, int], int] = fcntl.ioctl
    closer: Callable[[int], None] = os.close
    _fd: int | None = None

    def move(self, event: PointerMoveEvent) -> ActionResult:
        if event.dx == 0 and event.dy == 0:
            return self._result(
                ok=True,
                message="uinput pointer move 0,0",
                preview=["uinput.move", "0", "0"],
            )
        try:
            fd = self._ensure_device()
            if event.dx != 0:
                self._emit(fd, EV_REL, REL_X, event.dx)
            if event.dy != 0:
                self._emit(fd, EV_REL, REL_Y, event.dy)
            self._emit(fd, EV_SYN, SYN_REPORT, 0)
        except OSError as exc:
            return self._result(
                ok=False,
                message=f"uinput move failed: {exc}",
                preview=["uinput.move", str(event.dx), str(event.dy)],
            )
        return self._result(
            ok=True,
            message=f"uinput pointer move {event.dx},{event.dy}",
            preview=["uinput.move", str(event.dx), str(event.dy)],
        )

    def button(self, event: PointerButtonEvent) -> ActionResult:
        code = _button_code(event.button)
        if code is None:
            return self._result(
                ok=False,
                message=f"unsupported pointer button: {event.button}",
                preview=["uinput.button", event.button, event.action],
            )
        try:
            fd = self._ensure_device()
            for value in _button_values(event.action):
                self._emit(fd, EV_KEY, code, value)
                self._emit(fd, EV_SYN, SYN_REPORT, 0)
        except OSError as exc:
            return self._result(
                ok=False,
                message=f"uinput button failed: {exc}",
                preview=["uinput.button", event.button, event.action],
            )
        return self._result(
            ok=True,
            message=f"uinput pointer {event.button} {event.action}",
            preview=["uinput.button", event.button, event.action],
        )

    def scroll(self, event: PointerScrollEvent) -> ActionResult:
        try:
            fd = self._ensure_device()
            self._emit(fd, EV_REL, REL_WHEEL, event.amount_y)
            self._emit(fd, EV_SYN, SYN_REPORT, 0)
        except OSError as exc:
            return self._result(
                ok=False,
                message=f"uinput scroll failed: {exc}",
                preview=["uinput.scroll", str(event.amount_y)],
            )
        return self._result(
            ok=True,
            message=f"uinput pointer scroll {event.amount_y}",
            preview=["uinput.scroll", str(event.amount_y)],
        )

    def close(self) -> None:
        if self._fd is None:
            return
        try:
            self.ioctl(self._fd, UI_DEV_DESTROY, 0)
        finally:
            self.closer(self._fd)
            self._fd = None

    def _ensure_device(self) -> int:
        if self._fd is not None:
            return self._fd
        fd = self.opener(self.device_path, os.O_WRONLY | os.O_NONBLOCK)
        for event_type in (EV_KEY, EV_REL):
            self.ioctl(fd, UI_SET_EVBIT, event_type)
        for code in (BTN_LEFT, BTN_RIGHT):
            self.ioctl(fd, UI_SET_KEYBIT, code)
        for code in (REL_X, REL_Y, REL_WHEEL):
            self.ioctl(fd, UI_SET_RELBIT, code)
        self.writer(fd, _uinput_user_dev())
        self.ioctl(fd, UI_DEV_CREATE, 0)
        time.sleep(0.05)
        self._fd = fd
        return fd

    def _emit(self, fd: int, event_type: int, code: int, value: int) -> None:
        seconds = int(time.time())
        micros = int((time.time() - seconds) * 1_000_000)
        self.writer(fd, struct.pack("llHHi", seconds, micros, event_type, code, value))

    def _result(self, *, ok: bool, message: str, preview: list[str]) -> ActionResult:
        return ActionResult(
            request_id="pointer-uinput",
            ok=ok,
            target=self.name,
            executed_at=utc_timestamp(),
            message=message,
            command_preview=preview,
        )


def _button_code(button: str) -> int | None:
    return {"left": BTN_LEFT, "right": BTN_RIGHT}.get(button)


def _button_values(action: str) -> tuple[int, ...]:
    if action == "press":
        return (1,)
    if action == "release":
        return (0,)
    return (1, 0)


def _uinput_user_dev() -> bytes:
    name = b"AirDesk virtual pointer"
    return struct.pack(
        "80sHHHHI" + "i" * 256,
        name,
        BUS_USB,
        0xCA4D,
        0xA1DE,
        1,
        0,
        *([0] * 256),
    )
