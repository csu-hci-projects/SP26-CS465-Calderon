"""Safe action target that records intent without touching the desktop."""

from __future__ import annotations

from dataclasses import dataclass, field

from airdesk.state.types import ActionRequest, ActionResult, utc_timestamp


@dataclass
class DryRunActionTarget:
    """Action target for tests, study-safe profiles, and early debugging."""

    name: str = "dry-run"
    executed: list[ActionRequest] = field(default_factory=list)

    def execute(self, request: ActionRequest) -> ActionResult:
        self.executed.append(request)
        return ActionResult(
            request_id=request.request_id,
            ok=True,
            target=self.name,
            executed_at=utc_timestamp(),
            message=f"dry-run: {request.action_type} {request.command}",
            command_preview=self.command_preview(request),
        )

    @staticmethod
    def command_preview(request: ActionRequest) -> list[str]:
        args = request.parameters.get("args", [])
        return [request.action_type, request.command, *[str(arg) for arg in args]]
