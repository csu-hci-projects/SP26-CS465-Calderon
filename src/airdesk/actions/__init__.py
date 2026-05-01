"""Action targets and desktop adapters."""

from airdesk.actions.base import ActionTarget
from airdesk.actions.dry_run import DryRunActionTarget
from airdesk.actions.hyprland import GuardedHyprlandActionTarget, HyprlandActionTarget

__all__ = [
    "ActionTarget",
    "DryRunActionTarget",
    "GuardedHyprlandActionTarget",
    "HyprlandActionTarget",
]
