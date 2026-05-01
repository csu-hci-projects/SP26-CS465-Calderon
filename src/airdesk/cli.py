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
from airdesk.analysis.recording import analyze_recording, format_analysis
from airdesk.capture.opencv import CameraSettings, camera_modes, format_probe_result, probe_camera
from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.profiles.loader import load_profile
from airdesk.recording.jsonl import JsonlRecordingWriter, iter_recording
from airdesk.runtime import AirdeskRuntime, format_runtime_summary
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
def analyze(path: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Analyze a JSONL recording for timing, gesture, and stability signals."""
    typer.echo(format_analysis(analyze_recording(path)))


@app.command()
def track(
    backend: Annotated[str, typer.Option(help="Tracking backend to run.")] = "mediapipe",
    device: Annotated[str, typer.Option(help="Camera path or numeric index.")] = "/dev/video0",
    width: Annotated[int | None, typer.Option(help="Requested capture width.")] = None,
    height: Annotated[int | None, typer.Option(help="Requested capture height.")] = None,
    fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = None,
    fourcc: Annotated[str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")] = None,
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
    mirror: Annotated[bool, typer.Option(help="Mirror the preview window.")] = True,
) -> None:
    """Run live tracking and print compact frame summaries without recording or actions."""
    tracker = _make_tracker(
        backend=backend,
        device=device,
        max_frames=max_frames,
        show=show,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=model_path,
        auto_download_model=auto_download_model,
        preview_mirror=mirror,
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
def tune(
    backend: Annotated[str, typer.Option(help="Tracking backend to tune.")] = "mediapipe",
    device: Annotated[str, typer.Option(help="Camera path or numeric index.")] = "/dev/video0",
    width: Annotated[int | None, typer.Option(help="Requested capture width.")] = 640,
    height: Annotated[int | None, typer.Option(help="Requested capture height.")] = 480,
    fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = 30,
    fourcc: Annotated[
        str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")
    ] = "MJPG",
    extended_threshold: Annotated[
        float,
        typer.Option(help="Finger tip-vs-MCP y-distance threshold."),
    ] = 0.08,
    pinch_threshold: Annotated[
        float,
        typer.Option(help="Thumb/index distance threshold for pinch."),
    ] = 0.06,
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
    mirror: Annotated[bool, typer.Option(help="Mirror the preview window.")] = True,
) -> None:
    """Run a live primitive-tuning session with per-frame landmark features."""
    tracker = _make_tracker(
        backend=backend,
        device=device,
        max_frames=max_frames,
        show=show,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=model_path,
        auto_download_model=auto_download_model,
        preview_mirror=mirror,
        preview_extended_threshold=extended_threshold,
        preview_pinch_threshold=pinch_threshold,
    )
    recognizer = StaticHandPoseRecognizer(
        extended_threshold=extended_threshold,
        pinch_threshold=pinch_threshold,
    )
    previous_timestamp: float | None = None
    typer.echo(
        "target: open_palm extended=4 spread>=0.16 | fist folded=4 | "
        f"pinch distance<={pinch_threshold:.3f}"
    )
    try:
        tracker.start()
        for frame in tracker.frames():
            candidates = recognizer.recognize(frame)
            features = recognizer.features_for_frame(frame)
            frame_fps = _instant_fps(previous_timestamp, frame.timestamp)
            previous_timestamp = frame.timestamp
            typer.echo(_format_tune_summary(frame, candidates, features, frame_fps))
    except KeyboardInterrupt:
        typer.echo("interrupted")
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        tracker.stop()


@app.command()
def view(
    backend: Annotated[str, typer.Option(help="Tracking backend to view.")] = "mediapipe",
    device: Annotated[str, typer.Option(help="Camera path or numeric index.")] = "/dev/video0",
    width: Annotated[int | None, typer.Option(help="Requested capture width.")] = 640,
    height: Annotated[int | None, typer.Option(help="Requested capture height.")] = 480,
    fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = 30,
    fourcc: Annotated[
        str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")
    ] = "MJPG",
    model_path: Annotated[
        Path,
        typer.Option(help="MediaPipe Hand Landmarker .task model path."),
    ] = DEFAULT_HAND_LANDMARKER_MODEL,
    auto_download_model: Annotated[
        bool,
        typer.Option(help="Download the MediaPipe model to --model-path if missing."),
    ] = True,
    max_frames: Annotated[int | None, typer.Option(help="Stop after this many frames.")] = None,
    mirror: Annotated[bool, typer.Option(help="Mirror the preview window.")] = True,
) -> None:
    """Open a live webcam preview with MediaPipe hand overlays."""
    tracker = _make_tracker(
        backend=backend,
        device=device,
        max_frames=max_frames,
        show=True,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=model_path,
        auto_download_model=auto_download_model,
        preview_mirror=mirror,
    )
    typer.echo("Opening AirDesk live view. Press q or esc in the preview window to quit.")
    try:
        tracker.start()
        for _frame in tracker.frames():
            pass
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
    width: Annotated[int | None, typer.Option(help="Requested capture width.")] = None,
    height: Annotated[int | None, typer.Option(help="Requested capture height.")] = None,
    fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = None,
    fourcc: Annotated[str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")] = None,
    label: Annotated[str | None, typer.Option(help="Short label for this recording.")] = None,
    duration: Annotated[
        float | None,
        typer.Option(help="Stop after this many seconds based on frame timestamps."),
    ] = None,
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
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=model_path,
        auto_download_model=auto_download_model,
    )
    frame_count = 0
    interrupted = False
    first_frame_timestamp: float | None = None
    try:
        tracker.start()
        with JsonlRecordingWriter(out) as writer:
            writer.write_event(
                EventLogEntry(
                    event_type="recording_started",
                    timestamp=utc_timestamp(),
                    payload={
                        "backend": backend,
                        "device": device,
                        "max_frames": max_frames,
                        "duration": duration,
                        "label": label,
                        "model_path": str(model_path),
                        "camera_settings": CameraSettings(
                            width=width,
                            height=height,
                            fps=fps,
                            fourcc=fourcc,
                        ).to_dict(),
                    },
                )
            )
            try:
                for frame in tracker.frames():
                    if first_frame_timestamp is None:
                        first_frame_timestamp = frame.timestamp
                    if duration is not None and frame.timestamp - first_frame_timestamp >= duration:
                        break
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
                    payload={
                        "frames": frame_count,
                        "interrupted": interrupted,
                        "duration": duration,
                        "label": label,
                    },
                )
            )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        tracker.stop()
    typer.echo(f"recorded frames={frame_count} out={out}")


@app.command()
def run(
    backend: Annotated[str, typer.Option(help="Tracking backend to run.")] = "replay",
    recording: Annotated[Path | None, typer.Option(help="Replay JSONL recording path.")] = None,
    device: Annotated[
        str,
        typer.Option(help="Camera path, numeric index, or replay path."),
    ] = "/dev/video0",
    profile: Annotated[
        Path,
        typer.Option(help="Profile TOML path."),
    ] = Path("configs/profiles/study-safe.toml"),
    dry_run: Annotated[bool, typer.Option(help="Route actions to the dry-run target.")] = True,
    width: Annotated[int | None, typer.Option(help="Requested capture width.")] = None,
    height: Annotated[int | None, typer.Option(help="Requested capture height.")] = None,
    fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = None,
    fourcc: Annotated[str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")] = None,
    max_frames: Annotated[int | None, typer.Option(help="Stop after this many frames.")] = None,
    show: Annotated[bool, typer.Option(help="Show an OpenCV landmark debug window.")] = False,
) -> None:
    """Run the safe recognition/policy/action pipeline."""
    if not dry_run:
        typer.echo("Sprint 2 runtime only supports --dry-run.", err=True)
        raise typer.Exit(code=1)
    source = str(recording) if recording is not None else device
    tracker = _make_tracker(
        backend=backend,
        device=source,
        max_frames=max_frames,
        show=show,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
    )
    runtime = AirdeskRuntime(
        tracker=tracker,
        profile=load_profile(profile),
        action_target=DryRunActionTarget(),
    )
    typer.echo(format_runtime_summary(runtime.run()))


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


def _format_tune_summary(
    frame: TrackingFrame,
    candidates: object,
    features: object,
    frame_fps: float | None,
) -> str:
    names = ",".join(candidate.name for candidate in candidates) or "none"
    fps = f"{frame_fps:.1f}" if frame_fps is not None else "unknown"
    if not features:
        return (
            f"frame={frame.frame.sequence} fps={fps} hands=0 "
            f"size={frame.frame.width}x{frame.frame.height} candidates={names}"
        )
    feature_parts = []
    for feature in features:
        values = feature.to_flat_dict()
        feature_parts.append(
            "hand={hand_id} side={handedness} conf={confidence} extended={extended} "
            "folded={folded} spread={spread} pinch={pinch}".format(**values)
        )
    return (
        f"frame={frame.frame.sequence} fps={fps} hands={len(feature_parts)} "
        f"size={frame.frame.width}x{frame.frame.height} candidates={names} | "
        + " | ".join(feature_parts)
    )


def _instant_fps(previous_timestamp: float | None, timestamp: float) -> float | None:
    if previous_timestamp is None:
        return None
    elapsed = timestamp - previous_timestamp
    if elapsed <= 0:
        return None
    return 1.0 / elapsed


def _make_tracker(
    *,
    backend: str,
    device: str,
    max_frames: int | None,
    show: bool,
    camera_settings: CameraSettings | None = None,
    model_path: Path = DEFAULT_HAND_LANDMARKER_MODEL,
    auto_download_model: bool = True,
    preview_mirror: bool = True,
    preview_extended_threshold: float = 0.08,
    preview_pinch_threshold: float = 0.06,
) -> HandTrackerBackend:
    if backend == "mediapipe":
        from airdesk.tracking.mediapipe import MediaPipeHandTrackerBackend

        return MediaPipeHandTrackerBackend(
            device=device,
            model_path=model_path,
            auto_download_model=auto_download_model,
            camera_settings=camera_settings or CameraSettings(),
            max_frames=max_frames,
            show=show,
            preview_mirror=preview_mirror,
            preview_extended_threshold=preview_extended_threshold,
            preview_pinch_threshold=preview_pinch_threshold,
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


@camera_app.command("modes")
def modes(device: Annotated[str, typer.Option(help="Camera path.")] = "/dev/video0") -> None:
    """Report camera modes through v4l2-ctl when available."""
    typer.echo(camera_modes(device))


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
