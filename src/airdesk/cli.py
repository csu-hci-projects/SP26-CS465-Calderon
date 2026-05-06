"""AirDesk command-line interface."""

from __future__ import annotations

import glob
import platform
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Annotated
from uuid import uuid4

import typer

from airdesk import __version__
from airdesk.actions.cursor import CursorTarget, DryRunCursorTarget, HyprlandCursorTarget
from airdesk.actions.dry_run import DryRunActionTarget
from airdesk.actions.hyprland import (
    HYPRLAND_DISPATCH,
    SAFE_HYPRLAND_DISPATCHERS,
    GuardedHyprlandActionTarget,
)
from airdesk.analysis import (
    analyze_recording,
    evaluate_dtw_holdout,
    evaluate_dtw_recognizer,
    evaluate_rule_recognizer,
    format_analysis,
    format_evaluation,
    format_holdout_evaluation,
    save_evaluation_json,
    save_holdout_json,
)
from airdesk.capture.opencv import CameraSettings, camera_modes, format_probe_result, probe_camera
from airdesk.features import export_features_csv
from airdesk.gestures.base import CompositeGestureRecognizer
from airdesk.gestures.dtw import DtwCalibrationInput, DtwGestureModel, calibrate_dtw_model
from airdesk.gestures.phrases import IntentGatedSwipeRecognizer
from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.labels import (
    add_event_label,
    add_phase_label,
    init_label_file,
    load_label_file,
    save_label_file,
    suggest_stroke_label,
    validate_label_file,
)
from airdesk.modes.cursor import CursorControlConfig, PinchCursorController
from airdesk.profiles.loader import load_profile
from airdesk.recording.jsonl import JsonlRecordingWriter, iter_recording
from airdesk.runtime import AirdeskRuntime, format_runtime_summary
from airdesk.state.types import ActionRequest, EventLogEntry, TrackingFrame, utc_timestamp
from airdesk.tracking.interfaces import HandTrackerBackend
from airdesk.tracking.mediapipe import (
    DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
    DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    DEFAULT_HAND_LANDMARKER_MODEL,
    MediaPipeHandTrackerBackend,
)
from airdesk.tracking.replay import ReplayHandTrackerBackend

DEFAULT_COLLECTION_LABELS = (
    "open-palm-hold",
    "fist-hold",
    "pinch-hold",
    "swipe-left-positive",
    "swipe-right-positive",
    "normal-desk-motion-negative",
)


@dataclass(frozen=True)
class CollectionTakeResult:
    """Result of one prompted collection take."""

    frames: int
    decision: str

app = typer.Typer(no_args_is_help=True, help="AirDesk spatial input prototype CLI.")
camera_app = typer.Typer(help="Camera discovery and probing commands.")
hyprland_app = typer.Typer(help="Hyprland action helpers.")
profile_app = typer.Typer(help="Profile loading and validation commands.")
label_app = typer.Typer(help="Continuous gesture labeling commands.")
features_app = typer.Typer(help="Feature extraction commands.")
gesture_app = typer.Typer(help="Gesture recognizer evaluation commands.")
cursor_app = typer.Typer(help="Modeful cursor control commands.")

app.add_typer(camera_app, name="camera")
app.add_typer(hyprland_app, name="hyprland")
app.add_typer(profile_app, name="profile")
app.add_typer(label_app, name="label")
app.add_typer(features_app, name="features")
app.add_typer(gesture_app, name="gesture")
app.add_typer(cursor_app, name="cursor")


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


@app.command("collection-summary")
def collection_summary(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    pattern: Annotated[
        str,
        typer.Option(help="Recording filename pattern for directories."),
    ] = "*.jsonl",
) -> None:
    """Summarize candidate counts across a collection directory or recording files."""
    paths = _collection_paths(path, pattern=pattern)
    if not paths:
        typer.echo(f"No recordings matched {path}/{pattern}", err=True)
        raise typer.Exit(code=1)

    rows = [_collection_summary_row(recording_path) for recording_path in paths]
    for row in rows:
        typer.echo(_format_collection_row(row))
    typer.echo(_format_collection_totals(rows))


@label_app.command("init")
def label_init(
    recording: Annotated[Path, typer.Argument(exists=True, readable=True)],
    out: Annotated[Path | None, typer.Option(help="Output label JSON path.")] = None,
    participant: Annotated[str, typer.Option(help="Participant/user identifier.")] = "caden",
    notes: Annotated[str, typer.Option(help="Starter notes for this label file.")] = "",
    overwrite: Annotated[bool, typer.Option(help="Overwrite an existing label file.")] = False,
) -> None:
    """Create a starter label file for a continuous recording."""
    output = out or recording.with_suffix(".labels.json")
    if output.exists() and not overwrite:
        typer.echo(f"Label file already exists: {output}. Use --overwrite to replace it.", err=True)
        raise typer.Exit(code=1)
    label_file = init_label_file(recording, participant_id=participant, notes=notes)
    save_label_file(label_file, output)
    typer.echo(
        f"wrote labels={output} frames={label_file.session.frame_count} "
        f"hand_frames={label_file.session.hand_frame_count}"
    )


@label_app.command("validate")
def label_validate(path: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Validate a gesture label file."""
    result = validate_label_file(load_label_file(path))
    if not result.ok:
        for error in result.errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"valid labels={path}")


@label_app.command("add-phase")
def label_add_phase(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    phase: Annotated[str, typer.Option(help="Phase label, e.g. stroke_left.")],
    start: Annotated[float, typer.Option(help="Start seconds relative to recording start.")],
    end: Annotated[float, typer.Option(help="End seconds relative to recording start.")],
    gesture: Annotated[str | None, typer.Option(help="Optional gesture name.")] = None,
    notes: Annotated[str, typer.Option(help="Optional notes.")] = "",
) -> None:
    """Append one phase label using relative seconds from the recording start."""
    label_file = load_label_file(path)
    updated = add_phase_label(
        label_file,
        phase=phase,
        start_time=_relative_label_time(label_file.session.start_timestamp, start),
        end_time=_relative_label_time(label_file.session.start_timestamp, end),
        gesture=gesture,
        notes=notes,
    )
    _save_valid_label_file(updated, path)
    typer.echo(f"added phase={phase} labels={path}")


@label_app.command("add-event")
def label_add_event(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    gesture: Annotated[str, typer.Option(help="Gesture name, e.g. swipe_left.")],
    start: Annotated[float, typer.Option(help="Start seconds relative to recording start.")],
    end: Annotated[float, typer.Option(help="End seconds relative to recording start.")],
    label_type: Annotated[str, typer.Option(help="Event label type.")] = "gesture",
    commit: Annotated[
        float | None,
        typer.Option(help="Optional commit seconds relative to recording start."),
    ] = None,
    intended_command: Annotated[str | None, typer.Option(help="Optional intended command.")] = None,
    success: Annotated[bool | None, typer.Option(help="Optional success flag.")] = None,
    notes: Annotated[str, typer.Option(help="Optional notes.")] = "",
) -> None:
    """Append one event label using relative seconds from the recording start."""
    label_file = load_label_file(path)
    updated = add_event_label(
        label_file,
        gesture=gesture,
        start_time=_relative_label_time(label_file.session.start_timestamp, start),
        end_time=_relative_label_time(label_file.session.start_timestamp, end),
        label_type=label_type,
        commit_time=(
            _relative_label_time(label_file.session.start_timestamp, commit)
            if commit is not None
            else None
        ),
        intended_command=intended_command,
        success=success,
        notes=notes,
    )
    _save_valid_label_file(updated, path)
    typer.echo(f"added event={gesture} labels={path}")


@label_app.command("suggest")
def label_suggest(
    recording: Annotated[Path, typer.Argument(exists=True, readable=True)],
    gesture: Annotated[
        str | None,
        typer.Option(help="Gesture name to label, e.g. swipe_left. Inferred when possible."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Label JSON path to create or update when --apply is used."),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(help="Apply the suggested phase and event labels to --out."),
    ] = False,
    participant: Annotated[str, typer.Option(help="Participant/user identifier.")] = "caden",
    min_duration: Annotated[
        float,
        typer.Option(help="Minimum stroke-window duration in seconds."),
    ] = 0.25,
    max_duration: Annotated[
        float,
        typer.Option(help="Maximum stroke-window duration in seconds."),
    ] = 1.25,
    pad_seconds: Annotated[
        float,
        typer.Option(help="Context padding added before/after the detected stroke."),
    ] = 0.08,
) -> None:
    """Suggest a stroke label from the strongest palm-motion window."""
    suggestion = suggest_stroke_label(
        recording,
        gesture=gesture,
        min_duration=min_duration,
        max_duration=max_duration,
        pad_seconds=pad_seconds,
    )
    typer.echo(
        "suggestion "
        f"gesture={suggestion.gesture} phase={suggestion.phase} "
        f"start={suggestion.start_seconds:.3f} end={suggestion.end_seconds:.3f} "
        f"direction={suggestion.direction} confidence={suggestion.confidence:.2f}"
    )

    output = out or recording.with_suffix(".labels.json")
    if not apply:
        typer.echo(
            "apply with: "
            f"uv run airdesk label suggest {recording} --gesture {suggestion.gesture} "
            f"--out {output} --apply"
        )
        return

    label_file = load_label_file(output) if output.exists() else init_label_file(
        recording,
        participant_id=participant,
        notes="Initialized from label suggest.",
    )
    label_file = add_phase_label(
        label_file,
        phase=suggestion.phase,
        start_time=suggestion.start_time,
        end_time=suggestion.end_time,
        gesture=suggestion.gesture,
        notes=(
            "Auto-suggested from strongest palm-motion window; "
            "review before training/evaluation."
        ),
    )
    label_file = add_event_label(
        label_file,
        gesture=suggestion.gesture,
        start_time=suggestion.start_time,
        end_time=suggestion.end_time,
        label_type="gesture",
        notes=(
            "Auto-suggested from strongest palm-motion window; "
            "review before training/evaluation."
        ),
    )
    _save_valid_label_file(label_file, output)
    typer.echo(f"applied suggestion labels={output}")


@features_app.command("export")
def features_export(
    recording: Annotated[Path, typer.Argument(exists=True, readable=True)],
    out: Annotated[Path, typer.Option(help="Output CSV feature path.")],
    labels: Annotated[
        Path | None,
        typer.Option(help="Optional gesture labels JSON path."),
    ] = None,
) -> None:
    """Export deterministic landmark-derived features as CSV."""
    label_file = load_label_file(labels) if labels is not None else None
    rows = export_features_csv(recording, out, labels=label_file)
    typer.echo(f"exported features={out} rows={len(rows)}")


@gesture_app.command("evaluate")
def gesture_evaluate(
    recording: Annotated[Path, typer.Option(help="Recording JSONL path.")],
    labels: Annotated[Path, typer.Option(help="Gesture labels JSON path.")],
    recognizer: Annotated[str, typer.Option(help="Recognizer to evaluate.")] = "rule",
    model: Annotated[
        Path | None,
        typer.Option(help="Recognizer model path, required for --recognizer dtw."),
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Optional JSON output path.")] = None,
) -> None:
    """Evaluate a recognizer against event labels for one recording."""
    label_file = load_label_file(labels)
    if recognizer == "rule":
        evaluation = evaluate_rule_recognizer(recording, labels, label_file)
    elif recognizer == "dtw":
        if model is None:
            typer.echo("--model is required when --recognizer dtw.", err=True)
            raise typer.Exit(code=1)
        evaluation = evaluate_dtw_recognizer(
            recording,
            labels,
            label_file,
            DtwGestureModel.load(model),
        )
    else:
        typer.echo(f"Unsupported recognizer={recognizer}. Use rule or dtw.", err=True)
        raise typer.Exit(code=1)
    if out is not None:
        save_evaluation_json(evaluation, out)
    typer.echo(format_evaluation(evaluation))


@gesture_app.command("calibrate")
def gesture_calibrate(
    kind: Annotated[str, typer.Option(help="Calibration kind. Only dtw is implemented.")],
    recording: Annotated[
        list[Path] | None,
        typer.Option(help="Recording JSONL path. Repeat for multiple recordings."),
    ] = None,
    labels: Annotated[
        list[Path] | None,
        typer.Option(help="Gesture label JSON path. Repeat in recording order."),
    ] = None,
    out: Annotated[Path, typer.Option(help="Output model JSON path.")] = Path(
        "data/models/gestures/caden-dtw.json"
    ),
    cooldown_seconds: Annotated[
        float,
        typer.Option(help="Candidate suppression cooldown in seconds."),
    ] = 0.5,
    min_window_seconds: Annotated[
        float,
        typer.Option(help="Minimum DTW candidate window duration."),
    ] = 0.25,
    max_window_seconds: Annotated[
        float,
        typer.Option(help="Maximum DTW candidate window duration."),
    ] = 1.25,
    window_step_seconds: Annotated[
        float,
        typer.Option(help="DTW candidate window duration step."),
    ] = 0.1,
    negative_distance_margin: Annotated[
        float,
        typer.Option(help="Multiplier applied to closest negative DTW distance."),
    ] = 0.85,
    min_palm_dx_fraction: Annotated[
        float,
        typer.Option(help="Optional fraction of calibrated horizontal palm motion to require."),
    ] = 0.0,
) -> None:
    """Calibrate a personalized gesture recognizer from labeled recordings."""
    if kind != "dtw":
        typer.echo("Only --kind dtw is implemented.", err=True)
        raise typer.Exit(code=1)
    recording_paths = recording or []
    label_paths = labels or []
    if not recording_paths:
        typer.echo("At least one --recording is required.", err=True)
        raise typer.Exit(code=1)
    if len(recording_paths) != len(label_paths):
        typer.echo("--recording and --labels counts must match.", err=True)
        raise typer.Exit(code=1)
    inputs = [
        DtwCalibrationInput(
            recording=recording_path,
            labels=load_label_file(label_path),
            label_path=label_path,
        )
        for recording_path, label_path in zip(recording_paths, label_paths, strict=True)
    ]
    try:
        model = calibrate_dtw_model(
            inputs,
            cooldown_seconds=cooldown_seconds,
            min_window_seconds=min_window_seconds,
            max_window_seconds=max_window_seconds,
            window_step_seconds=window_step_seconds,
            negative_distance_margin=negative_distance_margin,
            min_palm_dx_fraction=min_palm_dx_fraction,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    model.save(out)
    gestures = ", ".join(sorted(model.thresholds))
    typer.echo(
        f"wrote dtw_model={out} templates={len(model.templates)} gestures={gestures}"
    )


@gesture_app.command("holdout-dtw")
def gesture_holdout_dtw(
    recordings_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Recording directory."),
    ],
    labels_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Label directory."),
    ],
    out: Annotated[Path, typer.Option(help="Output JSON summary path.")],
    model_out: Annotated[
        Path | None,
        typer.Option(help="Optional output path for the calibrated train-only DTW model."),
    ] = None,
    train_per_gesture: Annotated[
        int,
        typer.Option(help="Training recordings per positive gesture."),
    ] = 6,
    test_per_gesture: Annotated[
        int,
        typer.Option(help="Held-out test recordings per positive gesture."),
    ] = 2,
    train_negatives: Annotated[
        int,
        typer.Option(help="Training negative/background recordings."),
    ] = 6,
    test_negatives: Annotated[
        int,
        typer.Option(help="Held-out test negative/background recordings."),
    ] = 2,
    cooldown_seconds: Annotated[
        float,
        typer.Option(help="Candidate suppression cooldown in seconds."),
    ] = 0.5,
    min_window_seconds: Annotated[
        float,
        typer.Option(help="Minimum DTW candidate window duration."),
    ] = 0.25,
    max_window_seconds: Annotated[
        float,
        typer.Option(help="Maximum DTW candidate window duration."),
    ] = 1.25,
    window_step_seconds: Annotated[
        float,
        typer.Option(help="DTW candidate window duration step."),
    ] = 0.1,
    negative_distance_margin: Annotated[
        float,
        typer.Option(help="Multiplier applied to closest negative DTW distance."),
    ] = 0.85,
    min_palm_dx_fraction: Annotated[
        float,
        typer.Option(help="Optional fraction of calibrated horizontal palm motion to require."),
    ] = 0.0,
) -> None:
    """Run deterministic train/test DTW holdout evaluation for a collection batch."""
    try:
        evaluation = evaluate_dtw_holdout(
            recordings_dir=recordings_dir,
            labels_dir=labels_dir,
            model_path=model_out,
            train_per_gesture=train_per_gesture,
            test_per_gesture=test_per_gesture,
            train_negatives=train_negatives,
            test_negatives=test_negatives,
            cooldown_seconds=cooldown_seconds,
            min_window_seconds=min_window_seconds,
            max_window_seconds=max_window_seconds,
            window_step_seconds=window_step_seconds,
            negative_distance_margin=negative_distance_margin,
            min_palm_dx_fraction=min_palm_dx_fraction,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    save_holdout_json(evaluation, out)
    typer.echo(format_holdout_evaluation(evaluation))
    typer.echo(f"wrote holdout={out}")


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
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
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
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
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
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
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
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
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
                        "mediapipe": {
                            "max_num_hands": max_num_hands,
                            "min_detection_confidence": min_detection_confidence,
                            "min_presence_confidence": min_presence_confidence,
                            "min_tracking_confidence": min_tracking_confidence,
                        },
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
def collect(
    out_dir: Annotated[Path, typer.Option(help="Directory for collected JSONL takes.")] = Path(
        "data/recordings/collection"
    ),
    label: Annotated[
        list[str] | None,
        typer.Option("--label", "-l", help="Gesture/session label to collect. Repeatable."),
    ] = None,
    reps: Annotated[int, typer.Option(help="Kept repetitions per label.")] = 3,
    duration: Annotated[float, typer.Option(help="Recording duration per take in seconds.")] = 5.0,
    countdown: Annotated[float, typer.Option(help="Countdown duration before recording.")] = 3.0,
    backend: Annotated[str, typer.Option(help="Tracking backend to collect from.")] = "mediapipe",
    device: Annotated[
        str,
        typer.Option(help="Camera path, numeric index, or replay path."),
    ] = "/dev/video0",
    width: Annotated[int | None, typer.Option(help="Requested capture width.")] = 640,
    height: Annotated[int | None, typer.Option(help="Requested capture height.")] = 480,
    fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = 30,
    fourcc: Annotated[
        str | None,
        typer.Option(help="Requested camera FOURCC, e.g. MJPG."),
    ] = "MJPG",
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
    show: Annotated[bool, typer.Option(help="Show webcam preview with collection status.")] = True,
    auto_keep: Annotated[
        bool,
        typer.Option(help="Keep every take without prompting."),
    ] = False,
) -> None:
    """Collect prompted gesture recordings with countdown and keep/redo flow."""
    labels = tuple(label) if label else DEFAULT_COLLECTION_LABELS
    if reps < 1:
        typer.echo("--reps must be at least 1.", err=True)
        raise typer.Exit(code=1)
    if duration <= 0:
        typer.echo("--duration must be positive.", err=True)
        raise typer.Exit(code=1)
    if countdown < 0:
        typer.echo("--countdown cannot be negative.", err=True)
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    kept_paths: list[Path] = []
    for current_label in labels:
        kept = 0
        while kept < reps:
            repetition = kept + 1
            output = _next_collection_path(out_dir, current_label, repetition)
            typer.echo(f"\n{current_label} rep {repetition}/{reps}")
            preview_driven = show and not auto_keep and backend == "mediapipe"
            if preview_driven:
                typer.echo("Use the preview: space=start, k=keep, r=redo, s=skip, q=quit.")
            elif not auto_keep:
                typer.echo("Press Enter to start, or type s to skip this rep, q to quit.")
                response = input("> ").strip().lower()
                if response == "q":
                    raise typer.Exit()
                if response == "s":
                    kept += 1
                    continue

            take = _record_collection_take(
                output=output,
                label=current_label,
                repetition=repetition,
                backend=backend,
                device=device,
                width=width,
                height=height,
                fps=fps,
                fourcc=fourcc,
                duration=duration,
                countdown=countdown,
                model_path=model_path,
                auto_download_model=auto_download_model,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
                show=show,
                preview_driven=preview_driven,
            )

            if auto_keep:
                keep = take.decision == "keep"
            else:
                if preview_driven:
                    if take.decision == "quit":
                        raise typer.Exit()
                    if take.decision == "redo":
                        output.unlink(missing_ok=True)
                        continue
                    if take.decision == "skip":
                        output.unlink(missing_ok=True)
                        kept += 1
                        continue
                    keep = take.decision == "keep"
                    if keep:
                        typer.echo(f"Kept frames={take.frames} out={output}")
                else:
                    typer.echo(f"Recorded frames={take.frames} out={output}")
                    response = (
                        input("Keep, redo, skip, or quit? [k/r/s/q] ").strip().lower() or "k"
                    )
                    if response == "q":
                        raise typer.Exit()
                    keep = response == "k"
                    if response == "r":
                        output.unlink(missing_ok=True)
                        continue
                    if response == "s":
                        output.unlink(missing_ok=True)
                        kept += 1
                        continue
            if keep:
                kept_paths.append(output)
                kept += 1

    typer.echo(f"collection complete kept={len(kept_paths)} out_dir={out_dir}")


@app.command()
def benchmark(
    backend: Annotated[str, typer.Option(help="Tracking backend to benchmark.")] = "mediapipe",
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
    max_frames: Annotated[int, typer.Option(help="Frames to benchmark.")] = 120,
) -> None:
    """Benchmark live tracking FPS and hand-present frames for a configuration."""
    tracker = _make_tracker(
        backend=backend,
        device=device,
        max_frames=max_frames,
        show=False,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=model_path,
        auto_download_model=auto_download_model,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    timestamps: list[float] = []
    hand_frames = 0
    try:
        tracker.start()
        for frame in tracker.frames():
            timestamps.append(frame.timestamp)
            if frame.hands:
                hand_frames += 1
    except KeyboardInterrupt:
        typer.echo("interrupted")
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        tracker.stop()

    average_fps = _average_fps_from_timestamps(timestamps)
    fps_text = f"{average_fps:.2f}" if average_fps is not None else "unknown"
    typer.echo(
        f"frames={len(timestamps)} hand_frames={hand_frames} average_fps={fps_text} "
        f"model_path={model_path} max_num_hands={max_num_hands} "
        f"min_detection={min_detection_confidence:.2f} "
        f"min_presence={min_presence_confidence:.2f} "
        f"min_tracking={min_tracking_confidence:.2f}"
    )


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


@cursor_app.command("run")
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


def _record_collection_take(
    *,
    output: Path,
    label: str,
    repetition: int,
    backend: str,
    device: str,
    width: int | None,
    height: int | None,
    fps: float | None,
    fourcc: str | None,
    duration: float,
    countdown: float,
    model_path: Path,
    auto_download_model: bool,
    max_num_hands: int,
    min_detection_confidence: float,
    min_presence_confidence: float,
    min_tracking_confidence: float,
    show: bool,
    preview_driven: bool,
) -> CollectionTakeResult:
    status = {"text": f"ready: {label} rep {repetition}"}
    state: dict[str, str | None] = {
        "phase": "waiting" if preview_driven else "countdown",
        "decision": None,
    }
    tracker = _make_tracker(
        backend=backend,
        device=device,
        max_frames=None,
        show=show,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=model_path,
        auto_download_model=auto_download_model,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    _attach_collection_preview_controls(
        tracker,
        status_provider=lambda: status["text"],
        key_handler=lambda key: _handle_collection_preview_key(key, state),
    )
    frame_count = 0
    interrupted = False
    first_frame_timestamp: float | None = None
    recording_started_at: float | None = None

    try:
        tracker.start()
        with JsonlRecordingWriter(output) as writer:
            writer.write_event(
                EventLogEntry(
                    event_type="collection_take_started",
                    timestamp=utc_timestamp(),
                    payload={
                        "backend": backend,
                        "device": device,
                        "duration": duration,
                        "countdown": countdown,
                        "label": label,
                        "repetition": repetition,
                        "model_path": str(model_path),
                        "mediapipe": {
                            "max_num_hands": max_num_hands,
                            "min_detection_confidence": min_detection_confidence,
                            "min_presence_confidence": min_presence_confidence,
                            "min_tracking_confidence": min_tracking_confidence,
                        },
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
                    if state["phase"] == "done":
                        break
                    if state["phase"] == "waiting":
                        status["text"] = (
                            f"{label} rep {repetition} | position hand | "
                            "space=start s=skip q=quit"
                        )
                        continue
                    if state["phase"] == "review":
                        if state["decision"] is not None:
                            break
                        status["text"] = (
                            f"done frames={frame_count} | k=keep r=redo s=skip q=quit"
                        )
                        continue
                    if state["phase"] == "countdown":
                        if first_frame_timestamp is None:
                            first_frame_timestamp = frame.timestamp
                        elapsed = frame.timestamp - first_frame_timestamp
                        if elapsed < countdown:
                            remaining = countdown - elapsed
                            status["text"] = (
                                f"countdown {remaining:.1f}s | {label} rep {repetition}"
                            )
                            continue
                        state["phase"] = "recording"
                    if recording_started_at is None:
                        recording_started_at = frame.timestamp
                        writer.write_event(
                            EventLogEntry(
                                event_type="collection_recording_started",
                                timestamp=utc_timestamp(),
                                payload={"label": label, "repetition": repetition},
                            )
                        )
                    recorded_elapsed = frame.timestamp - recording_started_at
                    if recorded_elapsed >= duration:
                        state["phase"] = "review"
                        if not preview_driven:
                            state["decision"] = "keep"
                            break
                        status["text"] = (
                            f"done frames={frame_count} | k=keep r=redo s=skip q=quit"
                        )
                        continue
                    status["text"] = (
                        f"recording {recorded_elapsed:.1f}/{duration:.1f}s | "
                        f"{label} rep {repetition}"
                    )
                    writer.write_tracking_frame(frame)
                    frame_count += 1
            except KeyboardInterrupt:
                interrupted = True
                writer.write_event(
                    EventLogEntry(
                        event_type="collection_take_interrupted",
                        timestamp=utc_timestamp(),
                        payload={"frames": frame_count, "label": label, "repetition": repetition},
                    )
                )
            status["text"] = f"done: {label} rep {repetition} frames={frame_count}"
            writer.write_event(
                EventLogEntry(
                    event_type="collection_take_finished",
                    timestamp=utc_timestamp(),
                    payload={
                        "frames": frame_count,
                        "interrupted": interrupted,
                        "duration": duration,
                        "countdown": countdown,
                        "label": label,
                        "repetition": repetition,
                    },
                )
            )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        tracker.stop()
    return CollectionTakeResult(frames=frame_count, decision=state["decision"] or "keep")


def _attach_collection_preview_controls(
    tracker: HandTrackerBackend,
    status_provider: Callable[[], str],
    key_handler: Callable[[int], bool] | None = None,
) -> None:
    if isinstance(tracker, MediaPipeHandTrackerBackend):
        tracker.preview_status_provider = status_provider
        tracker.preview_key_handler = key_handler


def _handle_collection_preview_key(key: int, state: dict[str, str | None]) -> bool:
    phase = state["phase"]
    if phase == "waiting":
        if key == ord(" "):
            state["phase"] = "countdown"
            return True
        if key == ord("s"):
            state["decision"] = "skip"
            state["phase"] = "done"
            return True
        if key == ord("q"):
            state["decision"] = "quit"
            state["phase"] = "done"
            return True
    if phase == "review":
        decisions = {
            ord("k"): "keep",
            ord("r"): "redo",
            ord("s"): "skip",
            ord("q"): "quit",
        }
        if key in decisions:
            state["decision"] = decisions[key]
            state["phase"] = "done"
            return True
    return False


def _next_collection_path(out_dir: Path, label: str, repetition: int) -> Path:
    slug = _slugify_label(label)
    candidate = out_dir / f"{slug}-{repetition:03d}.jsonl"
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = out_dir / f"{slug}-{repetition:03d}-{suffix}.jsonl"
        if not candidate.exists():
            return candidate
        suffix += 1


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "take"


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


def _summarize_records(path: Path, *, recognize: bool) -> dict[str, int]:
    summary = {
        "frames": 0,
        "events": 0,
        "hands": 0,
        "open_palm": 0,
        "fist": 0,
        "pinch": 0,
        "swipe_left": 0,
        "swipe_right": 0,
        "point_left": 0,
        "point_right": 0,
    }
    static_recognizer = StaticHandPoseRecognizer()
    recognizer = CompositeGestureRecognizer(
        recognizers=(
            static_recognizer,
            IntentGatedSwipeRecognizer(pose_recognizer=static_recognizer),
        )
    )
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


def _collection_paths(path: Path, *, pattern: str) -> list[Path]:
    if path.is_dir():
        return sorted(recording for recording in path.glob(pattern) if recording.is_file())
    return [path]


def _collection_summary_row(path: Path) -> dict[str, str | int | float]:
    analysis = analyze_recording(path)
    return {
        "file": path.name,
        "label": _recording_label(path),
        **analysis.to_flat_dict(),
    }


def _recording_label(path: Path) -> str:
    for record in iter_recording(path):
        if record.kind != "event":
            continue
        assert isinstance(record.payload, EventLogEntry)
        label = record.payload.payload.get("label")
        if isinstance(label, str) and label:
            return label
    return re.sub(r"-\d{3}(?:-\d+)?$", "", path.stem)


def _format_collection_row(row: dict[str, str | int | float]) -> str:
    keys = (
        "file",
        "label",
        "frames",
        "hand_frames",
        "average_fps",
        "open_palm_count",
        "swipe_left_count",
        "swipe_right_count",
        "point_left_count",
        "point_right_count",
        "pinch_count",
        "fist_count",
    )
    return " ".join(f"{key}={row.get(key, 0)}" for key in keys)


def _format_collection_totals(rows: list[dict[str, str | int | float]]) -> str:
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        label = str(row["label"])
        label_totals = totals.setdefault(
            label,
            {
                "files": 0,
                "frames": 0,
                "hand_frames": 0,
                "swipe_left_count": 0,
                "swipe_right_count": 0,
                "point_left_count": 0,
                "point_right_count": 0,
                "pinch_count": 0,
                "fist_count": 0,
            },
        )
        label_totals["files"] += 1
        for key in tuple(label_totals):
            if key == "files":
                continue
            label_totals[key] += int(row.get(key, 0))

    parts = []
    for label, label_totals in sorted(totals.items()):
        values = " ".join(f"{key}={value}" for key, value in label_totals.items())
        parts.append(f"label={label} {values}")
    return "totals | " + " | ".join(parts)


def _relative_label_time(start_timestamp: float | None, seconds: float) -> float:
    if start_timestamp is None:
        return seconds
    return start_timestamp + seconds


def _save_valid_label_file(label_file: object, path: Path) -> None:
    result = validate_label_file(label_file)
    if not result.ok:
        for error in result.errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
    save_label_file(label_file, path)


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


def _average_fps_from_timestamps(timestamps: list[float]) -> float | None:
    if len(timestamps) < 2:
        return None
    intervals = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:], strict=False)
        if later > earlier
    ]
    if not intervals:
        return None
    return 1.0 / fmean(intervals)


def _make_tracker(
    *,
    backend: str,
    device: str,
    max_frames: int | None,
    show: bool,
    camera_settings: CameraSettings | None = None,
    model_path: Path = DEFAULT_HAND_LANDMARKER_MODEL,
    auto_download_model: bool = True,
    max_num_hands: int = DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
    min_detection_confidence: float = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    min_presence_confidence: float = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    min_tracking_confidence: float = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
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
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
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
