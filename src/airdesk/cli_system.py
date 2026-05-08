"""Camera, Hyprland, and profile CLI commands."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Annotated

import typer

from airdesk.actions.dry_run import DryRunActionTarget
from airdesk.actions.hyprland import HYPRLAND_DISPATCH
from airdesk.capture.opencv import CameraSettings, camera_modes, format_probe_result, probe_camera
from airdesk.profiles.loader import load_profile
from airdesk.state.types import ActionRequest


def list_cameras() -> None:
    """List Linux video device paths visible to the process."""
    devices = sorted(glob.glob("/dev/video*"))
    if not devices:
        typer.echo("no /dev/video* devices found")
        raise typer.Exit(code=0)
    for device in devices:
        typer.echo(device)


def probe(
    device: Annotated[str, typer.Option(help="Camera path or numeric index.")] = "/dev/video0",
    width: Annotated[int | None, typer.Option(help="Requested capture width.")] = None,
    height: Annotated[int | None, typer.Option(help="Requested capture height.")] = None,
    fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = None,
    fourcc: Annotated[str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")] = None,
) -> None:
    """Attempt to open a camera device and read one frame."""
    try:
        result = probe_camera(
            device,
            settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_probe_result(result))


def modes(device: Annotated[str, typer.Option(help="Camera path.")] = "/dev/video0") -> None:
    """Report camera modes through v4l2-ctl when available."""
    typer.echo(camera_modes(device))


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


def validate_profile(path: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Load and validate a profile file."""
    profile = load_profile(path)
    typer.echo(f"{profile.profile_id}: {profile.name}")


def register_system_commands(
    camera_app: typer.Typer,
    hyprland_app: typer.Typer,
    profile_app: typer.Typer,
) -> None:
    """Register small system helper command groups."""
    camera_app.command("list")(list_cameras)
    camera_app.command()(probe)
    camera_app.command("modes")(modes)
    hyprland_app.command("dry-run")(hyprland_dry_run)
    profile_app.command("validate")(validate_profile)
