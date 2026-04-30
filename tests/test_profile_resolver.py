from __future__ import annotations

from pathlib import Path

from airdesk.profiles.loader import load_profile
from airdesk.profiles.resolver import BindingResolver
from airdesk.state.types import GestureCandidate, GestureConfirmation


def confirmation(name: str, timestamp: float, confidence: float = 1.0) -> GestureConfirmation:
    return GestureConfirmation(
        candidate=GestureCandidate(name=name, confidence=confidence, timestamp=timestamp),
        confirmed_at=timestamp,
        profile_id="study-safe",
        mode="command",
    )


def test_resolves_study_safe_binding_to_action_request() -> None:
    profile = load_profile(Path("configs/profiles/study-safe.toml"))
    resolver = BindingResolver(profile)

    request = resolver.resolve(confirmation("pinch", 1.0))

    assert request is not None
    assert request.action_type == "dry-run.note"
    assert request.command == "pinch-observed"
    assert request.profile_id == "study-safe"


def test_resolver_applies_cooldown() -> None:
    profile = load_profile(Path("configs/profiles/study-safe.toml"))
    resolver = BindingResolver(profile)

    assert resolver.resolve(confirmation("pinch", 1.0)) is not None
    assert resolver.resolve(confirmation("pinch", 1.1)) is None


def test_resolver_rejects_low_confidence() -> None:
    profile = load_profile(Path("configs/profiles/study-safe.toml"))
    resolver = BindingResolver(profile)

    assert resolver.resolve(confirmation("pinch", 1.0, confidence=0.1)) is None
