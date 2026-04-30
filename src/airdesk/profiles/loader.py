"""Profile file loading and validation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from airdesk.profiles.models import ActionBinding, Profile


class ProfileValidationError(ValueError):
    """Raised when a profile config is missing required fields."""


def load_profile(path: Path) -> Profile:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return profile_from_mapping(data, source=str(path))


def profile_from_mapping(data: dict[str, Any], *, source: str = "<mapping>") -> Profile:
    required = ("id", "name", "allowed_modes", "activation_gesture", "bindings")
    missing = [key for key in required if key not in data]
    if missing:
        raise ProfileValidationError(f"{source}: missing required fields: {', '.join(missing)}")

    bindings = data["bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise ProfileValidationError(f"{source}: bindings must be a non-empty list")

    return Profile(
        profile_id=_expect_str(data, "id", source),
        name=_expect_str(data, "name", source),
        allowed_modes=tuple(_expect_str_list(data, "allowed_modes", source)),
        activation_gesture=_expect_str(data, "activation_gesture", source),
        bindings=tuple(_binding_from_mapping(item, source=source) for item in bindings),
        min_confidence=float(data.get("min_confidence", 0.75)),
        cooldown_ms=int(data.get("cooldown_ms", 500)),
        destructive_actions=bool(data.get("destructive_actions", False)),
        logging_level=str(data.get("logging_level", "events")),
        dry_run_default=bool(data.get("dry_run_default", True)),
    )


def _binding_from_mapping(data: dict[str, Any], *, source: str) -> ActionBinding:
    for key in ("gesture", "action_type", "command"):
        if key not in data:
            raise ProfileValidationError(f"{source}: binding missing required field: {key}")
    return ActionBinding(
        gesture=str(data["gesture"]),
        action_type=str(data["action_type"]),
        command=str(data["command"]),
        parameters=dict(data.get("parameters", {})),
        mode=str(data.get("mode", "command")),
        min_confidence=float(data.get("min_confidence", 0.75)),
        cooldown_ms=int(data.get("cooldown_ms", 500)),
        allow_destructive=bool(data.get("allow_destructive", False)),
    )


def _expect_str(data: dict[str, Any], key: str, source: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ProfileValidationError(f"{source}: {key} must be a non-empty string")
    return value


def _expect_str_list(data: dict[str, Any], key: str, source: str) -> list[str]:
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ProfileValidationError(f"{source}: {key} must be a list of strings")
    return value
