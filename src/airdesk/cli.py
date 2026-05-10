"""AirDesk command-line interface."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Annotated

import typer

from airdesk import __version__
from airdesk.analysis import analyze_recording, format_analysis
from airdesk.cli_gesture_replay import register_gesture_replay_commands
from airdesk.cli_labeling import register_feature_commands, register_label_commands
from airdesk.cli_live_commands import register_live_tracking_commands
from airdesk.cli_public_data import register_public_data_commands
from airdesk.cli_recording import register_recording_commands
from airdesk.cli_runtime import register_runtime_commands
from airdesk.cli_system import register_system_commands
from airdesk.cli_tcn import register_tcn_commands

app = typer.Typer(no_args_is_help=True, help="AirDesk spatial input prototype CLI.")
camera_app = typer.Typer(help="Camera discovery and probing commands.")
hyprland_app = typer.Typer(help="Hyprland action helpers.")
profile_app = typer.Typer(help="Profile loading and validation commands.")
label_app = typer.Typer(help="Continuous gesture labeling commands.")
features_app = typer.Typer(help="Feature extraction commands.")
gesture_app = typer.Typer(help="Gesture recognizer evaluation commands.")
cursor_app = typer.Typer(help="Modeful cursor control commands.")
public_data_app = typer.Typer(help="Public gesture dataset import commands.")

app.add_typer(camera_app, name="camera")
app.add_typer(hyprland_app, name="hyprland")
app.add_typer(profile_app, name="profile")
app.add_typer(label_app, name="label")
app.add_typer(features_app, name="features")
app.add_typer(gesture_app, name="gesture")
app.add_typer(cursor_app, name="cursor")
app.add_typer(public_data_app, name="public-data")
register_label_commands(label_app)
register_feature_commands(features_app)
register_tcn_commands(gesture_app)
register_gesture_replay_commands(gesture_app)
register_system_commands(camera_app, hyprland_app, profile_app)
register_recording_commands(app, gesture_app)
register_runtime_commands(app, cursor_app)
register_public_data_commands(public_data_app)


@app.command()
def doctor() -> None:
    """Print basic environment information."""
    typer.echo(f"AirDesk {__version__}")
    typer.echo(f"Python {platform.python_version()}")
    typer.echo(f"Platform {platform.platform()}")


@app.command()
def analyze(path: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Analyze a JSONL recording for timing, gesture, and stability signals."""
    typer.echo(format_analysis(analyze_recording(path)))


register_live_tracking_commands(app, gesture_app)
