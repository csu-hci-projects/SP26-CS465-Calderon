"""AirDesk command-line interface."""

from __future__ import annotations

import glob
import platform
from pathlib import Path
from typing import Annotated

import typer

from airdesk import __version__
from airdesk.actions.dry_run import DryRunActionTarget
from airdesk.actions.hyprland import HYPRLAND_DISPATCH
from airdesk.profiles.loader import load_profile
from airdesk.recording.jsonl import iter_recording
from airdesk.state.types import ActionRequest

app = typer.Typer(no_args_is_help=True, help="AirDesk spatial input prototype CLI.")
camera_app = typer.Typer(help="Camera discovery and probing commands.")
hyprland_app = typer.Typer(help="Hyprland action helpers.")
profile_app = typer.Typer(help="Profile loading and validation commands.")

app.add_typer(camera_app, name="camera")
app.add_typer(hyprland_app, name="hyprland")
app.add_typer(profile_app, name="profile")


@app.command()
def doctor() -> None:
    """Print basic environment information."""
    typer.echo(f"AirDesk {__version__}")
    typer.echo(f"Python {platform.python_version()}")
    typer.echo(f"Platform {platform.platform()}")


@app.command()
def replay(path: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Read a JSONL recording and report replayable frame/event counts."""
    frame_count = 0
    event_count = 0
    for record in iter_recording(path):
        if record.kind == "tracking_frame":
            frame_count += 1
        elif record.kind == "event":
            event_count += 1
    typer.echo(f"frames={frame_count} events={event_count}")


@camera_app.command("list")
def list_cameras() -> None:
    """List Linux video device paths visible to the process."""
    devices = sorted(glob.glob("/dev/video*"))
    if not devices:
        typer.echo("no /dev/video* devices found")
        raise typer.Exit(code=0)
    for device in devices:
        typer.echo(device)


@camera_app.command()
def probe(device: str = "/dev/video0") -> None:
    """Report whether a camera device path exists."""
    path = Path(device)
    status = "present" if path.exists() else "missing"
    typer.echo(f"{device}: {status}")


@hyprland_app.command("dry-run")
def hyprland_dry_run(
    command: Annotated[str, typer.Argument(help="Hyprland dispatcher name.")],
    args: Annotated[list[str] | None, typer.Argument(help="Dispatcher arguments.")] = None,
) -> None:
    """Preview a Hyprland dispatch request without executing it."""
    request = ActionRequest(
        action_type=HYPRLAND_DISPATCH,
        command=command,
        parameters={"args": args or []},
        source="cli",
    )
    result = DryRunActionTarget().execute(request)
    typer.echo(" ".join(result.command_preview or []))


@profile_app.command("validate")
def validate_profile(path: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Load and validate a profile file."""
    profile = load_profile(path)
    typer.echo(f"{profile.profile_id}: {profile.name}")
