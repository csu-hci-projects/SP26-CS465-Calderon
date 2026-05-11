"""Runtime and live-action CLI command surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer

from airdesk.actions.cursor import CursorTarget, DryRunCursorTarget, HyprlandCursorTarget
from airdesk.actions.dry_run import DryRunActionTarget
from airdesk.actions.hyprland import (
    CONTROL_HYPRLAND_DISPATCHERS,
    HYPRLAND_DISPATCH,
    SAFE_HYPRLAND_DISPATCHERS,
    GuardedHyprlandActionTarget,
)
from airdesk.actions.input import DryRunPointerInputTarget
from airdesk.capture.opencv import CameraSettings
from airdesk.cli_tracking import _make_tracker
from airdesk.control.runtime import ControlRuntime, ControlRuntimeConfig, format_control_summary
from airdesk.modes.cursor import CursorControlConfig, PinchCursorController
from airdesk.profiles.loader import load_profile
from airdesk.recording.jsonl import JsonlRecordingWriter
from airdesk.runtime import AirdeskRuntime, format_runtime_summary
from airdesk.state.types import EventLogEntry, utc_timestamp
from airdesk.tracking.interfaces import HandTrackerBackend
from airdesk.tracking.mediapipe import (
    DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
    DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    DEFAULT_HAND_LANDMARKER_MODEL,
    MediaPipeHandTrackerBackend,
)


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
    execute: Annotated[
        bool,
        typer.Option(help="Opt in to guarded real Hyprland execution."),
    ] = False,
    allow_profile_execute: Annotated[
        bool,
        typer.Option(help="Allow --execute even when the profile defaults to dry-run."),
    ] = False,
    width: Annotated[int | None, typer.Option(help="Requested capture width.")] = None,
    height: Annotated[int | None, typer.Option(help="Requested capture height.")] = None,
    fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = None,
    fourcc: Annotated[str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")] = None,
    model_path: Annotated[
        Path,
        typer.Option(help="MediaPipe Hand Landmarker .task model path."),
    ] = DEFAULT_HAND_LANDMARKER_MODEL,
    max_num_hands: Annotated[
        int,
        typer.Option(help="Maximum number of hands for MediaPipe to track."),
    ] = DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
    min_detection_confidence: Annotated[
        float,
        typer.Option(help="Minimum palm detection confidence."),
    ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    min_presence_confidence: Annotated[
        float,
        typer.Option(help="Minimum hand landmark presence confidence."),
    ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    min_tracking_confidence: Annotated[
        float,
        typer.Option(help="Minimum tracking confidence / box IoU threshold."),
    ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    auto_download_model: Annotated[
        bool,
        typer.Option(help="Download the MediaPipe model to --model-path if missing."),
    ] = True,
    events_out: Annotated[
        Path | None,
        typer.Option(help="Write runtime events as JSONL."),
    ] = None,
    pause_on_start: Annotated[
        bool,
        typer.Option(help="Start runtime paused so gestures cannot request actions."),
    ] = False,
    max_frames: Annotated[int | None, typer.Option(help="Stop after this many frames.")] = None,
    show: Annotated[bool, typer.Option(help="Show an OpenCV landmark debug window.")] = False,
) -> None:
    """Run the safe recognition/policy/action pipeline."""
    if not dry_run and not execute:
        typer.echo("Real actions require explicit --execute.", err=True)
        raise typer.Exit(code=1)
    effective_dry_run = not execute
    source = str(recording) if recording is not None else device
    tracker = _make_tracker(
        backend=backend,
        device=source,
        max_frames=max_frames,
        show=show,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=model_path,
        auto_download_model=auto_download_model,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    loaded_profile = load_profile(profile)
    action_target = _make_runtime_action_target(
        profile=loaded_profile,
        execute=execute,
        allow_profile_execute=allow_profile_execute,
    )
    event_writer = JsonlRecordingWriter(events_out) if events_out is not None else None
    try:
        runtime = AirdeskRuntime(
            tracker=tracker,
            profile=loaded_profile,
            action_target=action_target,
            event_writer=event_writer,
            paused=pause_on_start,
            session_metadata={
                "backend": backend,
                "source": source,
                "profile_path": str(profile),
                "dry_run": effective_dry_run,
                "execute": execute,
                "allow_profile_execute": allow_profile_execute,
                "paused_at_start": pause_on_start,
                "show": show,
                "max_frames": max_frames,
                "camera": {
                    "device": device,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "fourcc": fourcc,
                },
                "mediapipe": {
                    "model_path": str(model_path),
                    "auto_download_model": auto_download_model,
                    "max_num_hands": max_num_hands,
                    "min_detection_confidence": min_detection_confidence,
                    "min_presence_confidence": min_presence_confidence,
                    "min_tracking_confidence": min_tracking_confidence,
                },
            },
        )
        _attach_runtime_preview_controls(tracker, runtime)
        typer.echo(format_runtime_summary(runtime.run()))
    finally:
        if event_writer is not None:
            event_writer.close()


def cursor_run(
    backend: Annotated[str, typer.Option(help="Tracking backend to run.")] = "mediapipe",
    recording: Annotated[Path | None, typer.Option(help="Replay JSONL recording path.")] = None,
    device: Annotated[
        str,
        typer.Option(help="Camera path, numeric index, or replay path."),
    ] = "/dev/video0",
    execute: Annotated[
        bool,
        typer.Option(help="Move the real Hyprland cursor. Dry-run is the default."),
    ] = False,
    monitor: Annotated[
        str | None,
        typer.Option(help="Hyprland monitor name to constrain movement."),
    ] = None,
    width: Annotated[int | None, typer.Option(help="Requested capture width.")] = None,
    height: Annotated[int | None, typer.Option(help="Requested capture height.")] = None,
    fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = None,
    fourcc: Annotated[str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")] = None,
    model_path: Annotated[
        Path,
        typer.Option(help="MediaPipe Hand Landmarker .task model path."),
    ] = DEFAULT_HAND_LANDMARKER_MODEL,
    max_num_hands: Annotated[
        int,
        typer.Option(help="Maximum number of hands for MediaPipe to track."),
    ] = DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
    min_detection_confidence: Annotated[
        float,
        typer.Option(help="Minimum palm detection confidence."),
    ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    min_presence_confidence: Annotated[
        float,
        typer.Option(help="Minimum hand landmark presence confidence."),
    ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    min_tracking_confidence: Annotated[
        float,
        typer.Option(help="Minimum tracking confidence / box IoU threshold."),
    ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    auto_download_model: Annotated[
        bool,
        typer.Option(help="Download the MediaPipe model to --model-path if missing."),
    ] = True,
    pinch_threshold: Annotated[
        float,
        typer.Option(help="Thumb/index distance that enters cursor mode."),
    ] = 0.06,
    release_threshold: Annotated[
        float,
        typer.Option(help="Thumb/index distance that exits cursor mode."),
    ] = 0.08,
    gain: Annotated[float, typer.Option(help="Relative hand-to-cursor movement gain.")] = 1.8,
    smoothing_alpha: Annotated[
        float,
        typer.Option(help="Cursor smoothing alpha from 0 to 1."),
    ] = 0.35,
    dead_zone_px: Annotated[int, typer.Option(help="Ignore movements below this pixel size.")] = 3,
    max_step_px: Annotated[int, typer.Option(help="Maximum cursor step per tracking frame.")] = 140,
    mirror_x: Annotated[
        bool,
        typer.Option(help="Mirror hand X movement for webcam control."),
    ] = True,
    events_out: Annotated[
        Path | None,
        typer.Option(help="Write cursor runtime events as JSONL."),
    ] = None,
    pause_on_start: Annotated[
        bool,
        typer.Option(help="Start paused; press p in preview to resume."),
    ] = False,
    max_frames: Annotated[int | None, typer.Option(help="Stop after this many frames.")] = None,
    show: Annotated[bool, typer.Option(help="Show an OpenCV landmark debug window.")] = True,
) -> None:
    """Run pinch-held cursor control.

    Dry-run is default. Real cursor movement requires `--execute`.
    """
    source = str(recording) if recording is not None else device
    tracker = _make_tracker(
        backend=backend,
        device=source,
        max_frames=max_frames,
        show=show,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=model_path,
        auto_download_model=auto_download_model,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    target: CursorTarget = HyprlandCursorTarget() if execute else DryRunCursorTarget()
    controller = PinchCursorController(
        CursorControlConfig(
            pinch_threshold=pinch_threshold,
            release_threshold=release_threshold,
            gain=gain,
            smoothing_alpha=smoothing_alpha,
            dead_zone_px=dead_zone_px,
            max_step_px=max_step_px,
            mirror_x=mirror_x,
        )
    )
    event_writer = JsonlRecordingWriter(events_out) if events_out is not None else None
    session_id = str(uuid4())
    state: dict[str, bool] = {"paused": pause_on_start}
    frame_count = 0
    move_count = 0

    def emit(event_type: str, payload: dict[str, object] | None = None) -> None:
        if event_writer is None:
            return
        event_writer.write_event(
            EventLogEntry(
                event_type=event_type,
                timestamp=utc_timestamp(),
                payload=payload or {},
                session_id=session_id,
            )
        )

    if isinstance(tracker, MediaPipeHandTrackerBackend):
        tracker.preview_status_provider = lambda: (
            ("paused | " if state["paused"] else "") + controller.status_text()
        )

        def handle_key(key: int) -> bool:
            if key == ord("p"):
                state["paused"] = not state["paused"]
                emit(
                    "cursor_runtime_paused" if state["paused"] else "cursor_runtime_resumed",
                    {"paused": state["paused"]},
                )
                return True
            return False

        tracker.preview_key_handler = handle_key

    try:
        bounds = target.bounds(monitor=monitor)
        current_cursor = target.current_position()
        emit(
            "cursor_session_start",
            {
                "backend": backend,
                "source": source,
                "execute": execute,
                "target": target.name,
                "monitor": bounds.name,
                "bounds": {
                    "x": bounds.x,
                    "y": bounds.y,
                    "width": bounds.width,
                    "height": bounds.height,
                },
                "paused_at_start": pause_on_start,
                "click_available": False,
            },
        )
        tracker.start()
        for frame in tracker.frames():
            frame_count += 1
            update = controller.update(
                frame,
                current_cursor=current_cursor,
                bounds=bounds,
                paused=state["paused"],
            )
            if update.event is not None:
                emit(update.event, {"detail": update.detail, "active": update.active})
            if update.moved and update.position is not None:
                result = target.move_to(update.position)
                if result.ok:
                    current_cursor = update.position
                    move_count += 1
                emit(
                    "cursor_moved" if result.ok else "cursor_move_failed",
                    {
                        "x": update.position.x,
                        "y": update.position.y,
                        "result": result.to_dict(),
                    },
                )
    finally:
        tracker.stop()
        emit(
            "cursor_session_finish",
            {"frames": frame_count, "moves": move_count, "paused": state["paused"]},
        )
        if event_writer is not None:
            event_writer.close()

    mode = "execute" if execute else "dry-run"
    typer.echo(f"cursor {mode} complete frames={frame_count} moves={move_count}")


def control_run(
    backend: Annotated[str, typer.Option(help="Tracking backend to run.")] = "mediapipe",
    recording: Annotated[Path | None, typer.Option(help="Replay JSONL recording path.")] = None,
    device: Annotated[
        str,
        typer.Option(help="Camera path, numeric index, or replay path."),
    ] = "/dev/video0",
    execute: Annotated[
        bool,
        typer.Option(help="Opt in to guarded real Hyprland cursor/window actions."),
    ] = False,
    monitor: Annotated[
        str | None,
        typer.Option(help="Hyprland monitor name to constrain cursor movement."),
    ] = None,
    width: Annotated[int | None, typer.Option(help="Requested capture width.")] = None,
    height: Annotated[int | None, typer.Option(help="Requested capture height.")] = None,
    fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = None,
    fourcc: Annotated[str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")] = None,
    model_path: Annotated[
        Path,
        typer.Option(help="MediaPipe Hand Landmarker .task model path."),
    ] = DEFAULT_HAND_LANDMARKER_MODEL,
    max_num_hands: Annotated[
        int,
        typer.Option(help="Maximum number of hands for MediaPipe to track."),
    ] = DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
    min_detection_confidence: Annotated[
        float,
        typer.Option(help="Minimum palm detection confidence."),
    ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    min_presence_confidence: Annotated[
        float,
        typer.Option(help="Minimum hand landmark presence confidence."),
    ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    min_tracking_confidence: Annotated[
        float,
        typer.Option(help="Minimum tracking confidence / box IoU threshold."),
    ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    auto_download_model: Annotated[
        bool,
        typer.Option(help="Download the MediaPipe model to --model-path if missing."),
    ] = True,
    cursor_gain: Annotated[float, typer.Option(help="Open-hand cursor movement gain.")] = 1.0,
    cursor_smoothing_alpha: Annotated[
        float,
        typer.Option(help="Open-hand cursor smoothing alpha from 0 to 1."),
    ] = 0.35,
    cursor_dead_zone_px: Annotated[
        int,
        typer.Option(help="Ignore cursor movements below this pixel size."),
    ] = 3,
    mirror_x: Annotated[
        bool,
        typer.Option(help="Mirror hand X movement for webcam control."),
    ] = True,
    events_out: Annotated[
        Path | None,
        typer.Option(help="Write control runtime events as JSONL."),
    ] = None,
    pause_on_start: Annotated[
        bool,
        typer.Option(help="Start paused; press p in preview to resume."),
    ] = False,
    max_frames: Annotated[int | None, typer.Option(help="Stop after this many frames.")] = None,
    show: Annotated[bool, typer.Option(help="Show an OpenCV landmark debug window.")] = True,
) -> None:
    """Run deterministic landmark-logic control.

    Dry-run is default. `--execute` enables guarded Hyprland cursor/window
    dispatches, while pointer buttons/scroll remain dry-run until a real input
    injection target is installed and tested.
    """
    source = str(recording) if recording is not None else device
    tracker = _make_tracker(
        backend=backend,
        device=source,
        max_frames=max_frames,
        show=show,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=model_path,
        auto_download_model=auto_download_model,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    cursor_target: CursorTarget = HyprlandCursorTarget() if execute else DryRunCursorTarget()
    hyprland_target = (
        GuardedHyprlandActionTarget(allowed_dispatchers=CONTROL_HYPRLAND_DISPATCHERS)
        if execute
        else DryRunActionTarget()
    )
    event_writer = JsonlRecordingWriter(events_out) if events_out is not None else None
    runtime = ControlRuntime(
        tracker=tracker,
        cursor_target=cursor_target,
        hyprland_target=hyprland_target,
        pointer_target=DryRunPointerInputTarget(),
        event_writer=event_writer,
        config=ControlRuntimeConfig(
            execute=execute,
            pause_on_start=pause_on_start,
            cursor_gain=cursor_gain,
            cursor_smoothing_alpha=cursor_smoothing_alpha,
            cursor_dead_zone_px=cursor_dead_zone_px,
            mirror_x=mirror_x,
        ),
        monitor=monitor,
    )
    _attach_control_preview_controls(tracker, runtime)
    try:
        typer.echo(format_control_summary(runtime.run()))
    finally:
        if event_writer is not None:
            event_writer.close()


def _attach_runtime_preview_controls(
    tracker: HandTrackerBackend,
    runtime: AirdeskRuntime,
) -> None:
    if not isinstance(tracker, MediaPipeHandTrackerBackend):
        return
    tracker.preview_status_provider = runtime.status_text

    def handle_key(key: int) -> bool:
        if key == ord("p"):
            runtime.toggle_pause()
            return True
        return False

    tracker.preview_key_handler = handle_key


def _attach_control_preview_controls(
    tracker: HandTrackerBackend,
    runtime: ControlRuntime,
) -> None:
    if not isinstance(tracker, MediaPipeHandTrackerBackend):
        return
    tracker.preview_status_provider = runtime.status_text

    def handle_key(key: int) -> bool:
        if key == ord("p"):
            runtime.toggle_pause()
            return True
        return False

    tracker.preview_key_handler = handle_key


def _make_runtime_action_target(
    *,
    profile: object,
    execute: bool,
    allow_profile_execute: bool,
) -> DryRunActionTarget | GuardedHyprlandActionTarget:
    if not execute:
        return DryRunActionTarget()
    if getattr(profile, "dry_run_default", True) and not allow_profile_execute:
        typer.echo(
            "Profile defaults to dry-run; use --allow-profile-execute to override for "
            "guarded local pilot execution.",
            err=True,
        )
        raise typer.Exit(code=1)
    if getattr(profile, "destructive_actions", False):
        typer.echo("Refusing --execute for a profile that permits destructive actions.", err=True)
        raise typer.Exit(code=1)
    unsafe_bindings = [
        binding
        for binding in getattr(profile, "bindings", ())
        if binding.action_type == HYPRLAND_DISPATCH
        and binding.command not in SAFE_HYPRLAND_DISPATCHERS
    ]
    if unsafe_bindings:
        commands = ", ".join(sorted({binding.command for binding in unsafe_bindings}))
        typer.echo(f"Refusing --execute for unsafe Hyprland dispatchers: {commands}", err=True)
        raise typer.Exit(code=1)
    return GuardedHyprlandActionTarget()


def register_runtime_commands(
    app: typer.Typer, cursor_app: typer.Typer, control_app: typer.Typer
) -> None:
    """Register live runtime commands while keeping `airdesk.cli:app` stable."""
    app.command()(run)
    cursor_app.command("run")(cursor_run)
    control_app.command("run")(control_run)
