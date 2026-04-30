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
from airdesk.capture.opencv import format_probe_result, probe_camera
from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.profiles.loader import load_profile
from airdesk.recording.jsonl import JsonlRecordingWriter, iter_recording
from airdesk.state.types import ActionRequest, EventLogEntry, TrackingFrame, utc_timestamp
from airdesk.tracking.interfaces import HandTrackerBackend
from airdesk.tracking.mediapipe import DEFAULT_HAND_LANDMARKER_MODEL
from airdesk.tracking.replay import ReplayHandTrackerBackend

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
def replay(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    recognize: Annotated[
        bool,
        typer.Option(help="Run the static primitive recognizer over replayed frames."),
    ] = True,
) -> None:
    """Read a JSONL recording and report replayable frame/event counts."""
    summary = _summarize_records(path, recognize=recognize)
    typer.echo(_format_summary(summary))


@app.command()
def track(
    backend: Annotated[str, typer.Option(help="Tracking backend to run.")] = "mediapipe",
    device: Annotated[str, typer.Option(help="Camera path or numeric index.")] = "/dev/video0",
    model_path: Annotated[
        Path,
        typer.Option(help="MediaPipe Hand Landmarker .task model path."),
    ] = DEFAULT_HAND_LANDMARKER_MODEL,
    auto_download_model: Annotated[
        bool,
        typer.Option(help="Download the MediaPipe model to --model-path if missing."),
    ] = True,
    max_frames: Annotated[int | None, typer.Option(help="Stop after this many frames.")] = None,
    show: Annotated[bool, typer.Option(help="Show an OpenCV landmark debug window.")] = False,
) -> None:
    """Run live tracking and print compact frame summaries without recording or actions."""
    tracker = _make_tracker(
        backend=backend,
        device=device,
        max_frames=max_frames,
        show=show,
        model_path=model_path,
        auto_download_model=auto_download_model,
    )
    recognizer = StaticHandPoseRecognizer()
    try:
        tracker.start()
        for frame in tracker.frames():
            candidates = recognizer.recognize(frame)
            typer.echo(_format_frame_summary(frame, candidates))
    except KeyboardInterrupt:
        typer.echo("interrupted")
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        tracker.stop()


@app.command()
def record(
    out: Annotated[Path, typer.Option(help="Output JSONL recording path.")],
    backend: Annotated[str, typer.Option(help="Tracking backend to record.")] = "mediapipe",
    device: Annotated[str, typer.Option(help="Camera path or numeric index.")] = "/dev/video0",
    model_path: Annotated[
        Path,
        typer.Option(help="MediaPipe Hand Landmarker .task model path."),
    ] = DEFAULT_HAND_LANDMARKER_MODEL,
    auto_download_model: Annotated[
        bool,
        typer.Option(help="Download the MediaPipe model to --model-path if missing."),
    ] = True,
    max_frames: Annotated[int | None, typer.Option(help="Stop after this many frames.")] = None,
    show: Annotated[bool, typer.Option(help="Show an OpenCV landmark debug window.")] = False,
) -> None:
    """Record normalized tracking frames and runtime events as JSONL."""
    tracker = _make_tracker(
        backend=backend,
        device=device,
        max_frames=max_frames,
        show=show,
        model_path=model_path,
        auto_download_model=auto_download_model,
    )
    frame_count = 0
    interrupted = False
    try:
        tracker.start()
        with JsonlRecordingWriter(out) as writer:
            writer.write_event(
                EventLogEntry(
                    event_type="recording_started",
                    timestamp=utc_timestamp(),
                    payload={"backend": backend, "device": device, "max_frames": max_frames},
                )
            )
            try:
                for frame in tracker.frames():
                    writer.write_tracking_frame(frame)
                    frame_count += 1
            except KeyboardInterrupt:
                interrupted = True
                writer.write_event(
                    EventLogEntry(
                        event_type="recording_interrupted",
                        timestamp=utc_timestamp(),
                        payload={"frames": frame_count},
                    )
                )
            writer.write_event(
                EventLogEntry(
                    event_type="recording_finished",
                    timestamp=utc_timestamp(),
                    payload={"frames": frame_count, "interrupted": interrupted},
                )
            )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        tracker.stop()
    typer.echo(f"recorded frames={frame_count} out={out}")


def _summarize_records(path: Path, *, recognize: bool) -> dict[str, int]:
    summary = {
        "frames": 0,
        "events": 0,
        "hands": 0,
        "open_palm": 0,
        "fist": 0,
        "pinch": 0,
    }
    recognizer = StaticHandPoseRecognizer()
    for record in iter_recording(path):
        if record.kind == "tracking_frame":
            assert isinstance(record.payload, TrackingFrame)
            summary["frames"] += 1
            summary["hands"] += len(record.payload.hands)
            if recognize:
                for candidate in recognizer.recognize(record.payload):
                    if candidate.name in summary:
                        summary[candidate.name] += 1
        elif record.kind == "event":
            summary["events"] += 1
    return summary


def _format_summary(summary: dict[str, int]) -> str:
    return " ".join(f"{key}={value}" for key, value in summary.items())


def _format_frame_summary(frame: TrackingFrame, candidates: object) -> str:
    names = ",".join(candidate.name for candidate in candidates) or "none"
    return (
        f"frame={frame.frame.sequence} hands={len(frame.hands)} "
        f"size={frame.frame.width}x{frame.frame.height} candidates={names}"
    )


def _make_tracker(
    *,
    backend: str,
    device: str,
    max_frames: int | None,
    show: bool,
    model_path: Path = DEFAULT_HAND_LANDMARKER_MODEL,
    auto_download_model: bool = True,
) -> HandTrackerBackend:
    if backend == "mediapipe":
        from airdesk.tracking.mediapipe import MediaPipeHandTrackerBackend

        return MediaPipeHandTrackerBackend(
            device=device,
            model_path=model_path,
            auto_download_model=auto_download_model,
            max_frames=max_frames,
            show=show,
        )
    if backend == "replay":
        return ReplayHandTrackerBackend(Path(device))
    raise typer.BadParameter(f"unsupported tracking backend: {backend}")


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
def probe(
    device: Annotated[str, typer.Option(help="Camera path or numeric index.")] = "/dev/video0",
) -> None:
    """Attempt to open a camera device and read one frame."""
    try:
        result = probe_camera(device)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_probe_result(result))


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
