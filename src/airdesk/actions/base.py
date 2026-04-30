"""Action target interfaces."""

from __future__ import annotations

from typing import Protocol

from airdesk.state.types import ActionRequest, ActionResult


class ActionTarget(Protocol):
    """Executes typed action requests."""

    name: str

    def execute(self, request: ActionRequest) -> ActionResult:
        """Execute or simulate an action request."""
