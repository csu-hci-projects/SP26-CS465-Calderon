"""Replay/offline gesture diagnostic CLI command surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import typer

from airdesk.analysis import (
    evaluate_dtw_holdout,
    evaluate_dtw_recognizer,
    evaluate_motion_recognizer,
    evaluate_rule_recognizer,
    format_evaluation,
    format_holdout_evaluation,
    save_evaluation_json,
    save_holdout_json,
)
from airdesk.cli_support import _save_valid_label_file, _tracking_frames_from_recording
from airdesk.features import extract_feature_rows
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
from airdesk.labels import load_label_file
from airdesk.ml import refine_motion_aligned_label_file
from airdesk.recording.jsonl import iter_recording
from airdesk.state.types import GestureCandidate, TrackingFrame


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


def register_gesture_replay_commands(gesture_app: typer.Typer) -> None:
    """Register replay/offline gesture diagnostics on the gesture app."""
    gesture_app.command("evaluate")(gesture_evaluate)
    gesture_app.command("calibrate")(gesture_calibrate)
    gesture_app.command("holdout-dtw")(gesture_holdout_dtw)
    gesture_app.command("spot-dtw")(gesture_spot_dtw)
    gesture_app.command("spot-motion")(gesture_spot_motion)
    gesture_app.command("evaluate-motion")(gesture_evaluate_motion)
    gesture_app.command("refine-chart-labels")(gesture_refine_chart_labels)
    gesture_app.command("decode-candidates")(gesture_decode_candidates)
    gesture_app.command("score-sequence")(gesture_score_sequence)
