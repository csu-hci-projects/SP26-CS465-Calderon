"""Pointer button and scroll action adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from airdesk.state.types import ActionResult, utc_timestamp


@dataclass(frozen=True)
class PointerButtonEvent:
    """Discrete pointer button request."""

    button: str
    action: str = "click"


@dataclass(frozen=True)
class PointerScrollEvent:
    """Discrete pointer scroll request."""

    amount_y: int


class PointerInputTarget(Protocol):
    """Pointer button/scroll target boundary."""

    name: str

    def button(self, event: PointerButtonEvent) -> ActionResult:
        """Send a pointer button request."""

    def scroll(self, event: PointerScrollEvent) -> ActionResult:
        """Send a pointer scroll request."""


@dataclass
class DryRunPointerInputTarget:
    """Pointer target for tests and dry-run control sessions."""

    name: str = "pointer-dry-run"
    buttons: list[PointerButtonEvent] = field(default_factory=list)
    scrolls: list[PointerScrollEvent] = field(default_factory=list)

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
