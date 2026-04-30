"""Profile data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionBinding:
    """Maps a gesture to a typed action request."""

    gesture: str
    action_type: str
    command: str
    parameters: dict[str, Any] = field(default_factory=dict)
    mode: str = "command"
    min_confidence: float = 0.75
    cooldown_ms: int = 500
    allow_destructive: bool = False


@dataclass(frozen=True)
class Profile:
    """Profile-level gesture/action policy."""

    profile_id: str
    name: str
    allowed_modes: tuple[str, ...]
    activation_gesture: str
    bindings: tuple[ActionBinding, ...]
    min_confidence: float = 0.75
    cooldown_ms: int = 500
    destructive_actions: bool = False
    logging_level: str = "events"
    dry_run_default: bool = True
