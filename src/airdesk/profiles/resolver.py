"""Resolve confirmed gestures against profile bindings."""

from __future__ import annotations

from dataclasses import dataclass, field

from airdesk.profiles.models import ActionBinding, Profile
from airdesk.state.types import ActionRequest, GestureConfirmation


@dataclass
class BindingResolver:
    """Applies profile thresholds, safety policy, and cooldowns."""

    profile: Profile
    _last_triggered_at: dict[str, float] = field(default_factory=dict)

    def resolve(self, confirmation: GestureConfirmation) -> ActionRequest | None:
        binding = self._matching_binding(confirmation)
        if binding is None:
            return None
        required_confidence = max(binding.min_confidence, self.profile.min_confidence)
        if confirmation.candidate.confidence < required_confidence:
            return None
        if binding.allow_destructive and not self.profile.destructive_actions:
            return None
        key = f"{binding.mode}:{binding.gesture}:{binding.action_type}:{binding.command}"
        last_triggered_at = self._last_triggered_at.get(key)
        cooldown_ms = max(binding.cooldown_ms, self.profile.cooldown_ms)
        if last_triggered_at is not None:
            elapsed_ms = (confirmation.confirmed_at - last_triggered_at) * 1000
            if elapsed_ms < cooldown_ms:
                return None
        self._last_triggered_at[key] = confirmation.confirmed_at
        return action_request_from_binding(
            binding,
            profile_id=self.profile.profile_id,
            source=f"gesture:{confirmation.candidate.name}",
        )

    def _matching_binding(self, confirmation: GestureConfirmation) -> ActionBinding | None:
        for binding in self.profile.bindings:
            if binding.gesture == confirmation.candidate.name and binding.mode == confirmation.mode:
                return binding
        return None


def action_request_from_binding(
    binding: ActionBinding,
    *,
    profile_id: str,
    source: str,
) -> ActionRequest:
    """Create a typed action request from a resolved profile binding."""
    return ActionRequest(
        action_type=binding.action_type,
        command=binding.command,
        parameters=dict(binding.parameters),
        source=source,
        profile_id=profile_id,
    )
