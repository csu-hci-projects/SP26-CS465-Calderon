from __future__ import annotations

from airdesk.modes.command import CommandModeConfig, CommandModePolicy
from airdesk.state.types import GestureCandidate


def candidate(name: str, timestamp: float, confidence: float = 1.0) -> GestureCandidate:
    return GestureCandidate(name=name, confidence=confidence, timestamp=timestamp)


def test_open_palm_hold_enters_listening() -> None:
    policy = CommandModePolicy(CommandModeConfig(activation_hold_ms=300))

    events, confirmations = policy.update(
        [candidate("open_palm", 1.0)],
        timestamp=1.0,
        profile_id="study-safe",
    )
    assert events == []
    assert confirmations == []

    events, confirmations = policy.update(
        [candidate("open_palm", 1.3)],
        timestamp=1.3,
        profile_id="study-safe",
    )

    assert policy.listening is True
    assert events[0].payload["mode"] == "command"
    assert confirmations[0].candidate.name == "open_palm"


def test_short_open_palm_does_not_enter_listening() -> None:
    policy = CommandModePolicy(CommandModeConfig(activation_hold_ms=300))

    policy.update([candidate("open_palm", 1.0)], timestamp=1.0, profile_id="study-safe")
    events, confirmations = policy.update(
        [candidate("open_palm", 1.2)],
        timestamp=1.2,
        profile_id="study-safe",
    )

    assert policy.listening is False
    assert events == []
    assert confirmations == []


def test_fist_cancels_listening() -> None:
    policy = CommandModePolicy(CommandModeConfig(activation_hold_ms=0))
    policy.update([candidate("open_palm", 1.0)], timestamp=1.0, profile_id="study-safe")

    events, confirmations = policy.update(
        [candidate("fist", 1.1)],
        timestamp=1.1,
        profile_id="study-safe",
    )

    assert policy.listening is False
    assert confirmations == []
    assert events[0].payload["reason"] == "cancel"


def test_timeout_exits_listening() -> None:
    policy = CommandModePolicy(CommandModeConfig(activation_hold_ms=0, listening_timeout_ms=100))
    policy.update([candidate("open_palm", 1.0)], timestamp=1.0, profile_id="study-safe")

    events, _confirmations = policy.update([], timestamp=1.2, profile_id="study-safe")

    assert policy.listening is False
    assert events[0].payload["reason"] == "timeout"
