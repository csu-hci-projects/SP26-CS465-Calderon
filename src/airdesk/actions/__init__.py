"""Action targets and desktop adapters."""

from airdesk.actions.base import ActionTarget
from airdesk.actions.cursor import DryRunCursorTarget, HyprlandCursorTarget
from airdesk.actions.dry_run import DryRunActionTarget
from airdesk.actions.hyprland import GuardedHyprlandActionTarget, HyprlandActionTarget
from airdesk.actions.input import DryRunPointerInputTarget, UInputPointerInputTarget

__all__ = [
    "ActionTarget",
    "DryRunCursorTarget",
    "DryRunActionTarget",
    "DryRunPointerInputTarget",
    "GuardedHyprlandActionTarget",
    "HyprlandCursorTarget",
    "HyprlandActionTarget",
    "UInputPointerInputTarget",
]
