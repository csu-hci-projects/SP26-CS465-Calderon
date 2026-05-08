"""AirDesk command-line interface."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from statistics import fmean
from time import monotonic
from typing import Annotated, cast
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
    evaluate_motion_recognizer,
    evaluate_rule_recognizer,
    format_analysis,
    format_evaluation,
    format_holdout_evaluation,
    save_evaluation_json,
    save_holdout_json,
)
from airdesk.capture.opencv import CameraSettings
from airdesk.cli_labeling import register_feature_commands, register_label_commands
from airdesk.cli_live import (
    _format_live_dtw_candidate,
    _format_live_tcn_prediction,
    _format_live_tcn_preview_predictions,
    _format_tracker_timing,
    _is_live_tcn_gesture_target,
    _live_dtw_preview_status,
    _live_feature_streams,
    _live_tcn_preview_status,
    _show_live_tcn_prediction,
)
from airdesk.cli_recording import register_recording_commands
from airdesk.cli_support import _save_valid_label_file, _tracking_frames_from_recording
from airdesk.cli_system import register_system_commands
from airdesk.cli_tcn import register_tcn_commands
from airdesk.cli_tracking import _make_tracker
from airdesk.features import FeatureRowStream, FrameFeatureRow, extract_feature_rows
from airdesk.gestures.decoder import EventDecoder, EventDecoderConfig, frames_from_candidates
from airdesk.gestures.dtw import (
    DtwCalibrationInput,
    DtwGestureModel,
    DtwTemplateRecognizer,
    calibrate_dtw_model,
)
from airdesk.gestures.motion import (
    MotionEventConfig,
    MotionEventRecognizer,
    SwipeGesture,
    diagnose_motion_rows,
)
from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.labels import load_label_file
from airdesk.ml import (
    CausalTcnLivePrediction,
    CausalTcnLivePredictor,
    MissingMlDependencyError,
    refine_motion_aligned_label_file,
)
from airdesk.modes.cursor import CursorControlConfig, PinchCursorController
from airdesk.profiles.loader import load_profile
from airdesk.recording.jsonl import JsonlRecordingWriter, iter_recording
from airdesk.runtime import AirdeskRuntime, format_runtime_summary
from airdesk.state.types import (
    EventLogEntry,
    GestureCandidate,
    TrackingFrame,
    utc_timestamp,
)
from airdesk.tracking.interfaces import HandTrackerBackend
from airdesk.tracking.mediapipe import (
    DEFAULT_HAND_LANDMARKER_DELEGATE,
    DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
    DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    DEFAULT_HAND_LANDMARKER_MODEL,
    MediaPipeHandTrackerBackend,
)

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
register_label_commands(label_app)
register_feature_commands(features_app)
register_tcn_commands(gesture_app)
register_system_commands(camera_app, hyprland_app, profile_app)
register_recording_commands(app, gesture_app)


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
    elif recognizer == "motion":
        evaluation = evaluate_motion_recognizer(recording, labels, label_file)
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
        typer.echo(f"Unsupported recognizer={recognizer}. Use rule, motion, or dtw.", err=True)
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


@gesture_app.command("spot-dtw")
def gesture_spot_dtw(
    recording: Annotated[Path, typer.Option(exists=True, readable=True, help="Recording JSONL.")],
    model: Annotated[Path, typer.Option(exists=True, readable=True, help="DTW model JSON.")],
    out: Annotated[Path | None, typer.Option(help="Optional JSON candidate output path.")] = None,
) -> None:
    """Spot DTW gesture candidates in an unlabeled continuous recording."""
    frames = [
        record.payload
        for record in iter_recording(recording)
        if record.kind == "tracking_frame" and isinstance(record.payload, TrackingFrame)
    ]
    rows = extract_feature_rows(frames)
    recognizer = DtwTemplateRecognizer(DtwGestureModel.load(model))
    candidates = recognizer.recognize_rows(rows)
    first_timestamp = frames[0].timestamp if frames else None
    payload = {
        "recording": str(recording),
        "model": str(model),
        "candidate_count": len(candidates),
        "candidates": [
            {
                "index": index,
                "gesture": candidate.name,
                "timestamp": candidate.timestamp,
                "timestamp_relative": (
                    candidate.timestamp - first_timestamp
                    if first_timestamp is not None
                    else None
                ),
                "confidence": candidate.confidence,
                "metadata": candidate.metadata,
            }
            for index, candidate in enumerate(candidates, start=1)
        ],
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    typer.echo(f"recording={Path(recording).name} recognizer=dtw candidates={len(candidates)}")
    for item in payload["candidates"]:
        typer.echo(
            f"{item['index']:02d} gesture={item['gesture']} "
            f"t={item['timestamp_relative']:.3f} confidence={item['confidence']:.3f}"
        )
    if out is not None:
        typer.echo(f"wrote candidates={out}")


@gesture_app.command("spot-motion")
def gesture_spot_motion(
    recording: Annotated[Path, typer.Option(exists=True, readable=True, help="Recording JSONL.")],
    out: Annotated[Path | None, typer.Option(help="Optional JSON candidate output path.")] = None,
    labels: Annotated[
        Path | None,
        typer.Option(exists=True, readable=True, help="Optional labels JSON for diagnostics."),
    ] = None,
    diagnostic_limit: Annotated[
        int,
        typer.Option(help="Number of strongest motion rows to include per hand stream."),
    ] = 8,
    min_dx_per_hand_scale: Annotated[
        float,
        typer.Option(help="Minimum absolute hand-normalized horizontal displacement."),
    ] = 0.65,
    min_peak_velocity: Annotated[
        float,
        typer.Option(help="Minimum peak absolute horizontal velocity."),
    ] = 0.45,
    min_direction_consistency: Annotated[
        float,
        typer.Option(help="Minimum same-direction motion consistency in [0, 1]."),
    ] = 0.60,
    release_velocity: Annotated[
        float,
        typer.Option(help="Commit an active motion when immediate velocity drops below this."),
    ] = 0.20,
    min_duration_seconds: Annotated[
        float,
        typer.Option(help="Minimum seconds from activation to peak."),
    ] = 0.08,
    max_duration_seconds: Annotated[
        float,
        typer.Option(help="Maximum seconds from activation to peak."),
    ] = 1.25,
    min_event_separation: Annotated[
        float,
        typer.Option(help="Minimum same-gesture separation per hand stream."),
    ] = 0.20,
    positive_dx_gesture: Annotated[
        str,
        typer.Option(help="Gesture name for positive raw camera dx: swipe_right or swipe_left."),
    ] = "swipe_right",
) -> None:
    """Spot deterministic per-hand motion events in an unlabeled recording."""
    try:
        config = _motion_event_config(
            min_dx_per_hand_scale=min_dx_per_hand_scale,
            min_peak_velocity=min_peak_velocity,
            min_direction_consistency=min_direction_consistency,
            release_velocity=release_velocity,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            min_event_separation=min_event_separation,
            positive_dx_gesture=positive_dx_gesture,
        )
        frames = _tracking_frames_from_recording(recording)
        label_file = load_label_file(labels) if labels is not None else None
        rows = extract_feature_rows(frames, labels=label_file)
        candidates = MotionEventRecognizer(config).recognize_rows(rows)
        diagnostics = diagnose_motion_rows(rows, config, limit_per_hand=diagnostic_limit)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    payload = _candidate_payload(
        recording=recording,
        recognizer="motion",
        candidates=candidates,
        first_timestamp=frames[0].timestamp if frames else None,
        extra={
            "labels": str(labels) if labels is not None else None,
            "motion_config": config.to_dict(),
            "motion_diagnostics": [item.to_dict() for item in diagnostics],
        },
    )
    if out is not None:
        _write_json(out, payload)
    typer.echo(
        f"recording={Path(recording).name} recognizer=motion candidates={len(candidates)}"
    )
    for item in payload["candidates"]:
        typer.echo(
            f"{item['index']:02d} gesture={item['gesture']} "
            f"hand={item['hand_id'] or 'unknown'} "
            f"t={item['timestamp_relative']:.3f} confidence={item['confidence']:.3f}"
        )
    if out is not None:
        typer.echo(f"wrote candidates={out}")


@gesture_app.command("evaluate-motion")
def gesture_evaluate_motion(
    recording: Annotated[Path, typer.Option(exists=True, readable=True, help="Recording JSONL.")],
    labels: Annotated[Path, typer.Option(exists=True, readable=True, help="Gesture labels JSON.")],
    out: Annotated[Path | None, typer.Option(help="Optional JSON output path.")] = None,
    min_dx_per_hand_scale: Annotated[
        float,
        typer.Option(help="Minimum absolute hand-normalized horizontal displacement."),
    ] = 0.65,
    min_peak_velocity: Annotated[
        float,
        typer.Option(help="Minimum peak absolute horizontal velocity."),
    ] = 0.45,
    min_direction_consistency: Annotated[
        float,
        typer.Option(help="Minimum same-direction motion consistency in [0, 1]."),
    ] = 0.60,
    release_velocity: Annotated[
        float,
        typer.Option(help="Commit an active motion when immediate velocity drops below this."),
    ] = 0.20,
    min_duration_seconds: Annotated[
        float,
        typer.Option(help="Minimum seconds from activation to peak."),
    ] = 0.08,
    max_duration_seconds: Annotated[
        float,
        typer.Option(help="Maximum seconds from activation to peak."),
    ] = 1.25,
    min_event_separation: Annotated[
        float,
        typer.Option(help="Minimum same-gesture separation per hand stream."),
    ] = 0.20,
    positive_dx_gesture: Annotated[
        str,
        typer.Option(help="Gesture name for positive raw camera dx: swipe_right or swipe_left."),
    ] = "swipe_right",
) -> None:
    """Evaluate deterministic per-hand motion events against labels."""
    label_file = load_label_file(labels)
    try:
        config = _motion_event_config(
            min_dx_per_hand_scale=min_dx_per_hand_scale,
            min_peak_velocity=min_peak_velocity,
            min_direction_consistency=min_direction_consistency,
            release_velocity=release_velocity,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            min_event_separation=min_event_separation,
            positive_dx_gesture=positive_dx_gesture,
        )
        evaluation = evaluate_motion_recognizer(recording, labels, label_file, config)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if out is not None:
        save_evaluation_json(evaluation, out)
    typer.echo(format_evaluation(evaluation))


@gesture_app.command("refine-chart-labels")
def gesture_refine_chart_labels(
    features_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Feature CSV directory."),
    ],
    labels_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Source label directory."),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(help="Output directory for experimental refined label copies."),
    ],
    report: Annotated[
        Path | None,
        typer.Option(help="Optional JSON report path for refinement diagnostics."),
    ] = None,
    pattern: Annotated[str, typer.Option(help="Feature filename glob pattern.")] = "*.csv",
    search_padding_seconds: Annotated[
        float,
        typer.Option(help="Seconds before/after each prompt label to search for motion peaks."),
    ] = 1.5,
    min_motion_score: Annotated[
        float,
        typer.Option(help="Minimum absolute hand-normalized window displacement to refine."),
    ] = 0.35,
    min_direction_consistency: Annotated[
        float,
        typer.Option(help="Minimum same-direction motion consistency for refinement peaks."),
    ] = 0.35,
    stroke_seconds: Annotated[
        float | None,
        typer.Option(help="Override refined stroke duration; defaults to original label duration."),
    ] = None,
    recovery_seconds: Annotated[
        float,
        typer.Option(help="Recovery phase duration to write after each refined stroke."),
    ] = 0.35,
) -> None:
    """Write non-destructive experimental motion-aligned label copies for chart data."""
    feature_paths = sorted(features_dir.glob(pattern))
    if not feature_paths:
        typer.echo(f"No feature CSV files matched {features_dir}/{pattern}", err=True)
        raise typer.Exit(code=1)
    reports: list[dict[str, object]] = []
    refined_count = 0
    changed_count = 0
    for feature_path in feature_paths:
        label_path = labels_dir / f"{feature_path.stem}.labels.json"
        if not label_path.exists():
            typer.echo(f"Missing label file for features={feature_path}: {label_path}", err=True)
            raise typer.Exit(code=1)
        try:
            result = refine_motion_aligned_label_file(
                feature_path=feature_path,
                label_path=label_path,
                search_padding_seconds=search_padding_seconds,
                min_motion_score=min_motion_score,
                min_direction_consistency=min_direction_consistency,
                stroke_seconds=stroke_seconds,
                recovery_seconds=recovery_seconds,
            )
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        output = out_dir / label_path.name
        _save_valid_label_file(result.label_file, output)
        reports.append({**result.to_dict(), "output_label_path": str(output)})
        refined_count += len(result.refined_events)
        changed_count += sum(1 for item in result.refined_events if item.changed)
    payload = {
        "status": "diagnostic_only",
        "warning": (
            "Motion-peak refined labels are experimental weak labels. "
            "Do not treat them as training truth until they beat prompt labels on holdout replay."
        ),
        "features_dir": str(features_dir),
        "labels_dir": str(labels_dir),
        "out_dir": str(out_dir),
        "pattern": pattern,
        "search_padding_seconds": search_padding_seconds,
        "min_motion_score": min_motion_score,
        "min_direction_consistency": min_direction_consistency,
        "stroke_seconds": stroke_seconds,
        "recovery_seconds": recovery_seconds,
        "files": reports,
        "summary": {
            "files": len(reports),
            "refined_events": refined_count,
            "changed_events": changed_count,
        },
    }
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    typer.echo(
        f"refined_chart_labels files={len(reports)} events={refined_count} "
        f"changed={changed_count} out_dir={out_dir} status=diagnostic_only"
    )
    if report is not None:
        typer.echo(f"wrote refinement_report={report}")


@gesture_app.command("decode-candidates")
def gesture_decode_candidates(
    candidates: Annotated[Path, typer.Option(exists=True, readable=True, help="Candidate JSON.")],
    out: Annotated[Path | None, typer.Option(help="Optional decoded JSON output path.")] = None,
    activation_threshold: Annotated[
        float,
        typer.Option(help="Confidence needed to start an event."),
    ] = 0.55,
    release_threshold: Annotated[
        float,
        typer.Option(help="Confidence below this starts recovery/commit."),
    ] = 0.35,
    min_peak_confidence: Annotated[
        float,
        typer.Option(help="Minimum peak confidence required to commit."),
    ] = 0.60,
    min_event_separation: Annotated[
        float,
        typer.Option(help="Minimum seconds between same-gesture commits."),
    ] = 0.50,
    recovery_seconds: Annotated[
        float,
        typer.Option(help="Seconds below release threshold before committing."),
    ] = 0.25,
    cooldown_seconds: Annotated[
        float,
        typer.Option(help="Same-gesture cooldown seconds after a commit."),
    ] = 0.50,
) -> None:
    """Apply hysteresis/peak/cooldown event decoding to candidate JSON."""
    with candidates.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_candidates = [
        _candidate_from_json(item)
        for item in payload.get("candidates", [])
        if isinstance(item, dict)
    ]
    config = EventDecoderConfig(
        activation_threshold=activation_threshold,
        release_threshold=release_threshold,
        min_peak_confidence=min_peak_confidence,
        min_event_separation_seconds=min_event_separation,
        recovery_seconds=recovery_seconds,
        cooldown_seconds=cooldown_seconds,
    )
    try:
        decoded = EventDecoder(config).decode(frames_from_candidates(raw_candidates))
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    result = {
        "source_candidates": str(candidates),
        "decoder_config": config.to_dict(),
        "candidate_count": len(raw_candidates),
        "decoded_count": len(decoded),
        "candidates": [candidate.to_dict() for candidate in decoded],
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    typer.echo(
        f"decoded candidates={len(raw_candidates)} events={len(decoded)} "
        f"source={Path(candidates).name}"
    )
    for index, candidate in enumerate(decoded, start=1):
        typer.echo(
            f"{index:02d} gesture={candidate.name} "
            f"t={candidate.timestamp:.3f} confidence={candidate.confidence:.3f}"
        )
    if out is not None:
        typer.echo(f"wrote decoded={out}")


@gesture_app.command("score-sequence")
def gesture_score_sequence(
    candidates: Annotated[Path, typer.Option(exists=True, readable=True, help="Candidate JSON.")],
    expected_sequence: Annotated[
        str,
        typer.Option(help="Expected sequence, e.g. 'R L R R L L'."),
    ],
    out: Annotated[Path | None, typer.Option(help="Optional JSON score output path.")] = None,
) -> None:
    """Compare spotted candidates against an expected gesture order."""
    with candidates.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    detected = [
        _sequence_token(item["gesture"])
        for item in payload.get("candidates", [])
        if "gesture" in item
    ]
    expected = [_sequence_token(token) for token in expected_sequence.split()]
    matched = _lcs_length(expected, detected)
    result = {
        "candidates": str(candidates),
        "expected": expected,
        "detected": detected,
        "expected_count": len(expected),
        "detected_count": len(detected),
        "matched_in_order": matched,
        "missed_or_wrong_order": len(expected) - matched,
        "extra_or_wrong_order": len(detected) - matched,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    typer.echo(
        "expected="
        f"{' '.join(expected)} detected={' '.join(detected)} "
        f"matched_in_order={matched}/{len(expected)} "
        f"missed_or_wrong_order={result['missed_or_wrong_order']} "
        f"extra_or_wrong_order={result['extra_or_wrong_order']}"
    )
    if out is not None:
        typer.echo(f"wrote score={out}")


@gesture_app.command("watch-tcn")
def gesture_watch_tcn(
    tcn_model: Annotated[
        Path,
        typer.Option("--model", exists=True, readable=True, help="TCN checkpoint path."),
    ],
    backend: Annotated[str, typer.Option(help="Tracking backend to watch.")] = "mediapipe",
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
    hand_model_path: Annotated[
        Path,
        typer.Option("--hand-model-path", help="MediaPipe Hand Landmarker .task model path."),
    ] = DEFAULT_HAND_LANDMARKER_MODEL,
    max_num_hands: Annotated[
        int,
        typer.Option(help="Maximum number of hands for MediaPipe to track."),
    ] = 2,
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
    hand_delegate: Annotated[
        str,
        typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
    ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
    auto_download_model: Annotated[
        bool,
        typer.Option(help="Download the MediaPipe model to --hand-model-path if missing."),
    ] = True,
    max_frames: Annotated[int | None, typer.Option(help="Stop after this many frames.")] = None,
    show: Annotated[bool, typer.Option(help="Show an OpenCV live preview.")] = True,
    mirror: Annotated[bool, typer.Option(help="Mirror the preview window.")] = True,
    confidence_threshold: Annotated[
        float,
        typer.Option(help="Minimum confidence before printing a prediction line."),
    ] = 0.0,
    min_rows: Annotated[
        int,
        typer.Option(help="Minimum feature rows before the first TCN prediction."),
    ] = 4,
    include_background: Annotated[
        bool,
        typer.Option(help="Print background predictions as well as gestures."),
    ] = False,
    include_recovery: Annotated[
        bool,
        typer.Option(help="Print recovery/reset phase predictions."),
    ] = False,
    show_motion: Annotated[
        bool,
        typer.Option(help="Show hand-normalized horizontal motion in the HUD."),
    ] = True,
    profile_timing: Annotated[
        bool,
        typer.Option(help="Print per-prediction TCN timing diagnostics."),
    ] = False,
) -> None:
    """Watch live/replay TCN classification without triggering desktop actions."""
    if not 0 <= confidence_threshold <= 1:
        typer.echo("confidence-threshold must be in [0, 1]", err=True)
        raise typer.Exit(code=1)
    try:
        predictor = CausalTcnLivePredictor.load(tcn_model)
    except (MissingMlDependencyError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    tracker = _make_tracker(
        backend=backend,
        device=device,
        max_frames=max_frames,
        show=show,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=hand_model_path,
        auto_download_model=auto_download_model,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        delegate=hand_delegate,
        preview_mirror=mirror,
        preview_gestures=False,
    )
    stream = FeatureRowStream()
    rows: list[FrameFeatureRow] = []
    latest_predictions: dict[str, CausalTcnLivePrediction] = {}
    latest_rows_by_hand: dict[str, object] = {}
    state: dict[str, object] = {
        "status": "warming up",
        "alert": "",
        "alert_until": 0.0,
        "predictions": latest_predictions,
        "rows_by_hand": latest_rows_by_hand,
        "show_motion": show_motion,
        "stream_count": 0,
        "row_count": 0,
    }
    first_timestamp: float | None = None
    next_prediction_time_by_hand: dict[str, float] = {}

    if hasattr(tracker, "preview_status_provider"):
        tracker.preview_status_provider = lambda: _live_tcn_preview_status(state)  # type: ignore[attr-defined]

    typer.echo(
        "watching tcn "
        f"model={tcn_model} backend={backend} window={predictor.window_seconds:.2f}s "
        f"stride={predictor.stride_seconds:.2f}s targets={','.join(predictor.targets)}"
    )
    try:
        tracker.start()
        for frame in tracker.frames():
            first_timestamp = frame.timestamp if first_timestamp is None else first_timestamp
            rows.extend(stream.append_rows(frame))
            cutoff = frame.timestamp - predictor.window_seconds
            rows = [item for item in rows if item.timestamp >= cutoff]
            hand_streams = _live_feature_streams(rows)
            if not hand_streams:
                state["status"] = "warming rows=0"
                state["stream_count"] = 0
                state["row_count"] = 0
                continue
            state["stream_count"] = len(hand_streams)
            state["row_count"] = len(rows)
            state["status"] = _format_live_tcn_preview_predictions(state)
            for hand_id, hand_rows in hand_streams.items():
                latest_rows_by_hand[hand_id] = hand_rows[-1]
                if len(hand_rows) < min_rows:
                    continue
                next_prediction_time = next_prediction_time_by_hand.get(hand_id)
                if (
                    next_prediction_time is not None
                    and hand_rows[-1].timestamp < next_prediction_time
                ):
                    continue
                prediction_started_at = monotonic()
                prediction = predictor.predict_rows(hand_rows)
                prediction_ms = (monotonic() - prediction_started_at) * 1000
                latest_predictions[hand_id] = prediction
                state["status"] = _format_live_tcn_preview_predictions(state)
                visible_prediction = _show_live_tcn_prediction(
                    prediction,
                    include_background=include_background,
                    include_recovery=include_recovery,
                    confidence_threshold=confidence_threshold,
                )
                if profile_timing and visible_prediction:
                    typer.echo(
                        f"tcn_predict_ms={prediction_ms:.2f} hand={hand_id} "
                        f"rows={len(hand_rows)} target={prediction.target} "
                        f"confidence={prediction.confidence:.3f}"
                    )
                if _is_live_tcn_gesture_target(prediction.target) and visible_prediction:
                    state["alert"] = (
                        f"{hand_id} {prediction.target} {prediction.confidence:.2f}"
                    )
                    state["alert_until"] = monotonic() + 1.25
                next_prediction_time_by_hand[hand_id] = (
                    hand_rows[-1].timestamp + predictor.stride_seconds
                )
                if visible_prediction:
                    typer.echo(
                        _format_live_tcn_prediction(
                            prediction,
                            first_timestamp=first_timestamp,
                        )
                    )
    except KeyboardInterrupt:
        typer.echo("interrupted")
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        tracker.stop()


@gesture_app.command("watch-dtw")
def gesture_watch_dtw(
    model: Annotated[
        Path,
        typer.Option("--model", exists=True, readable=True, help="DTW model JSON path."),
    ],
    backend: Annotated[str, typer.Option(help="Tracking backend to watch.")] = "mediapipe",
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
    hand_model_path: Annotated[
        Path,
        typer.Option("--hand-model-path", help="MediaPipe Hand Landmarker .task model path."),
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
    hand_delegate: Annotated[
        str,
        typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
    ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
    auto_download_model: Annotated[
        bool,
        typer.Option(help="Download the MediaPipe model to --hand-model-path if missing."),
    ] = True,
    max_frames: Annotated[int | None, typer.Option(help="Stop after this many frames.")] = None,
    show: Annotated[bool, typer.Option(help="Show an OpenCV live preview.")] = True,
    mirror: Annotated[bool, typer.Option(help="Mirror the preview window.")] = True,
    confidence_threshold: Annotated[
        float,
        typer.Option(help="Minimum DTW confidence before printing a candidate."),
    ] = 0.0,
    watch_stride_seconds: Annotated[
        float,
        typer.Option(help="Minimum seconds between DTW scans over the rolling buffer."),
    ] = 0.08,
    profile_timing: Annotated[
        bool,
        typer.Option(help="Print per-scan DTW timing diagnostics."),
    ] = False,
) -> None:
    """Watch live/replay DTW candidate spotting without triggering desktop actions."""
    if not 0 <= confidence_threshold <= 1:
        typer.echo("confidence-threshold must be in [0, 1]", err=True)
        raise typer.Exit(code=1)
    if watch_stride_seconds <= 0:
        typer.echo("watch-stride-seconds must be positive", err=True)
        raise typer.Exit(code=1)
    try:
        dtw_model = DtwGestureModel.load(model)
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(f"Failed to load DTW model: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    tracker = _make_tracker(
        backend=backend,
        device=device,
        max_frames=max_frames,
        show=show,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=hand_model_path,
        auto_download_model=auto_download_model,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        delegate=hand_delegate,
        preview_mirror=mirror,
    )
    recognizer = DtwTemplateRecognizer(dtw_model)
    stream = FeatureRowStream()
    rows: list[FrameFeatureRow] = []
    state = {"status": "warming up", "alert": "", "alert_until": 0.0}
    first_timestamp: float | None = None
    next_scan_time: float | None = None
    last_reported_timestamp: float | None = None

    if hasattr(tracker, "preview_status_provider"):
        tracker.preview_status_provider = lambda: _live_dtw_preview_status(state)  # type: ignore[attr-defined]

    typer.echo(
        "watching dtw "
        f"model={model} backend={backend} "
        f"window={dtw_model.min_window_seconds:.2f}-{dtw_model.max_window_seconds:.2f}s "
        f"step={dtw_model.window_step_seconds:.2f}s"
    )
    try:
        tracker.start()
        for frame in tracker.frames():
            first_timestamp = frame.timestamp if first_timestamp is None else first_timestamp
            rows.extend(stream.append_rows(frame))
            cutoff = frame.timestamp - dtw_model.max_window_seconds - watch_stride_seconds
            rows = [item for item in rows if item.timestamp >= cutoff]
            state["status"] = f"DTW rows={len(rows)} hands={len(frame.hands)}"
            if next_scan_time is not None and frame.timestamp < next_scan_time:
                continue
            scan_started_at = monotonic()
            candidates = [
                candidate
                for candidate in recognizer.recognize_latest_rows(rows)
                if candidate.confidence >= confidence_threshold
            ]
            scan_ms = (monotonic() - scan_started_at) * 1000
            if profile_timing:
                typer.echo(
                    f"dtw_scan_ms={scan_ms:.2f} rows={len(rows)} "
                    f"candidates={len(candidates)} hands={len(frame.hands)}"
                )
            fresh = [
                candidate
                for candidate in candidates
                if last_reported_timestamp is None
                or candidate.timestamp > last_reported_timestamp + 1e-6
            ]
            if fresh:
                for candidate in fresh:
                    typer.echo(
                        _format_live_dtw_candidate(
                            candidate,
                            first_timestamp=first_timestamp,
                        )
                    )
                best = max(fresh, key=lambda item: item.confidence)
                state["alert"] = f"{best.name} {best.confidence:.2f}"
                state["alert_until"] = monotonic() + 1.25
                last_reported_timestamp = max(candidate.timestamp for candidate in fresh)
            next_scan_time = frame.timestamp + watch_stride_seconds
    except KeyboardInterrupt:
        typer.echo("interrupted")
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        tracker.stop()


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
    hand_delegate: Annotated[
        str,
        typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
    ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
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
        delegate=hand_delegate,
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
    hand_delegate: Annotated[
        str,
        typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
    ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
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
        delegate=hand_delegate,
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
    hand_delegate: Annotated[
        str,
        typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
    ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
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
        delegate=hand_delegate,
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
    hand_delegate: Annotated[
        str,
        typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
    ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
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
        delegate=hand_delegate,
    )
    timestamps: list[float] = []
    hand_frames = 0
    hand_confidences: list[float] = []
    current_missing_streak = 0
    longest_missing_streak = 0
    try:
        tracker.start()
        for frame in tracker.frames():
            timestamps.append(frame.timestamp)
            if frame.hands:
                hand_frames += 1
                current_missing_streak = 0
                hand_confidences.extend(
                    hand.confidence for hand in frame.hands if hand.confidence is not None
                )
            else:
                current_missing_streak += 1
                longest_missing_streak = max(longest_missing_streak, current_missing_streak)
    except KeyboardInterrupt:
        typer.echo("interrupted")
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        tracker.stop()

    average_fps = _average_fps_from_timestamps(timestamps)
    fps_text = f"{average_fps:.2f}" if average_fps is not None else "unknown"
    hand_ratio = hand_frames / len(timestamps) if timestamps else 0.0
    mean_confidence = f"{fmean(hand_confidences):.3f}" if hand_confidences else "unknown"
    typer.echo(
        f"frames={len(timestamps)} hand_frames={hand_frames} average_fps={fps_text} "
        f"hand_present_ratio={hand_ratio:.3f} "
        f"longest_no_hand_streak={longest_missing_streak} "
        f"mean_hand_confidence={mean_confidence} "
        f"model_path={model_path} max_num_hands={max_num_hands} "
        f"delegate={hand_delegate} "
        f"min_detection={min_detection_confidence:.2f} "
        f"min_presence={min_presence_confidence:.2f} "
        f"min_tracking={min_tracking_confidence:.2f}"
    )
    timing_text = _format_tracker_timing(tracker)
    if timing_text:
        typer.echo(timing_text)


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


def _sequence_token(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"r", "right", "swipe_right"}:
        return "R"
    if normalized in {"l", "left", "swipe_left"}:
        return "L"
    raise typer.BadParameter(f"unsupported sequence token: {value}")


def _candidate_from_json(item: dict[str, object]) -> GestureCandidate:
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    name = item.get("gesture", item.get("name", ""))
    timestamp = item.get("timestamp", item.get("timestamp_relative", 0.0))
    return GestureCandidate(
        name=str(name),
        confidence=float(item.get("confidence", 0.0) or 0.0),
        timestamp=float(timestamp or 0.0),
        hand_id=str(item["hand_id"]) if isinstance(item.get("hand_id"), str) else None,
        metadata=metadata,
    )


def _motion_event_config(
    *,
    min_dx_per_hand_scale: float,
    min_peak_velocity: float,
    min_direction_consistency: float,
    release_velocity: float,
    min_duration_seconds: float,
    max_duration_seconds: float,
    min_event_separation: float,
    positive_dx_gesture: str,
) -> MotionEventConfig:
    if positive_dx_gesture not in {"swipe_left", "swipe_right"}:
        raise ValueError("positive_dx_gesture must be swipe_left or swipe_right")
    return MotionEventConfig(
        min_dx_per_hand_scale=min_dx_per_hand_scale,
        min_peak_velocity=min_peak_velocity,
        min_direction_consistency=min_direction_consistency,
        release_velocity=release_velocity,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
        min_event_separation_seconds=min_event_separation,
        positive_dx_gesture=cast(SwipeGesture, positive_dx_gesture),
    )


def _candidate_payload(
    *,
    recording: Path,
    recognizer: str,
    candidates: list[GestureCandidate],
    first_timestamp: float | None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "recording": str(recording),
        "recognizer": recognizer,
        "candidate_count": len(candidates),
        "candidates": [
            {
                "index": index,
                "gesture": candidate.name,
                "timestamp": candidate.timestamp,
                "timestamp_relative": (
                    candidate.timestamp - first_timestamp
                    if first_timestamp is not None
                    else None
                ),
                "hand_id": candidate.hand_id,
                "confidence": candidate.confidence,
                "metadata": candidate.metadata,
            }
            for index, candidate in enumerate(candidates, start=1)
        ],
    }
    if extra:
        payload.update(extra)
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0] * (len(right) + 1)
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]
