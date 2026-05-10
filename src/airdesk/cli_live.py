"""Live preview/status helpers for CLI commands."""

from __future__ import annotations

from statistics import fmean
from time import monotonic

from airdesk.features import FrameFeatureRow, group_feature_rows_by_stream
from airdesk.ml import CausalTcnLivePrediction, CausalTcnV2LivePrediction
from airdesk.state.types import GestureCandidate
from airdesk.tracking.interfaces import HandTrackerBackend


def _format_tracker_timing(tracker: HandTrackerBackend) -> str:
    samples = getattr(tracker, "timing_samples", None)
    if not samples:
        return ""
    fields = (
        ("capture_read", "capture_read_ms"),
        ("color_convert", "color_convert_ms"),
        ("mediapipe_inference", "inference_ms"),
        ("normalize", "normalize_ms"),
        ("preview_draw", "preview_draw_ms"),
        ("tracker_total", "total_ms"),
    )
    parts = [f"timing_frames={len(samples)}"]
    for label, attr in fields:
        values = [
            float(value)
            for sample in samples
            if (value := getattr(sample, attr, None)) is not None
        ]
        if not values:
            continue
        parts.append(f"{label}_mean_ms={fmean(values):.2f}")
        parts.append(f"{label}_p95_ms={_percentile(values, 0.95):.2f}")
    return " ".join(parts)


def _format_live_tcn_status(prediction: CausalTcnLivePrediction) -> str:
    probabilities = _compact_probabilities(prediction.probabilities)
    target = _short_tcn_target(prediction.target)
    hand = f"{prediction.hand_id} " if prediction.hand_id else ""
    return f"TCN {hand}{target} {prediction.confidence:.2f} | {probabilities}"


def _format_live_tcn_preview_predictions(state: dict[str, object]) -> str:
    stream_count = int(state.get("stream_count", 0))
    row_count = int(state.get("row_count", 0))
    show_motion = bool(state.get("show_motion", False))
    predictions = state.get("predictions", {})
    rows_by_hand = state.get("rows_by_hand", {})
    if not isinstance(predictions, dict) or not predictions:
        return f"TCN streams={stream_count} rows={row_count} warming"
    parts = [f"TCN streams={stream_count} rows={row_count}"]
    for hand_id, prediction in sorted(predictions.items()):
        if not isinstance(prediction, CausalTcnLivePrediction):
            continue
        left = prediction.probabilities.get("stroke_left", 0.0)
        right = prediction.probabilities.get("stroke_right", 0.0)
        motion = ""
        if show_motion and isinstance(rows_by_hand, dict):
            row = rows_by_hand.get(hand_id)
            dx = getattr(row, "palm_window_dx_per_hand_scale", None)
            if isinstance(dx, float):
                motion = f" dx={dx:.2f}"
        parts.append(
            f"{hand_id}:{_short_tcn_target(prediction.target)} "
            f"{prediction.confidence:.2f} L={left:.2f} R={right:.2f}{motion}"
        )
    return " | ".join(parts)


def _format_live_tcn_v2_preview_predictions(state: dict[str, object]) -> str:
    stream_count = int(state.get("stream_count", 0))
    row_count = int(state.get("row_count", 0))
    show_motion = bool(state.get("show_motion", False))
    predictions = state.get("predictions", {})
    rows_by_hand = state.get("rows_by_hand", {})
    if not isinstance(predictions, dict) or not predictions:
        return f"TCN v2 streams={stream_count} rows={row_count} warming"
    parts = [f"TCN v2 streams={stream_count} rows={row_count}"]
    for hand_id, prediction in sorted(predictions.items()):
        if not isinstance(prediction, CausalTcnV2LivePrediction):
            continue
        evidence = prediction.evidence
        motion = ""
        if show_motion and isinstance(rows_by_hand, dict):
            row = rows_by_hand.get(hand_id)
            dx = getattr(row, "palm_window_dx_per_hand_scale", None)
            if isinstance(dx, float):
                motion = f" dx={dx:.2f}"
        parts.append(
            f"{hand_id}:intent={evidence.get('intentional_motion', 0.0):.2f} "
            f"L={evidence.get('stroke_left', 0.0):.2f} "
            f"R={evidence.get('stroke_right', 0.0):.2f} "
            f"S={evidence.get('start', 0.0):.2f} "
            f"E={evidence.get('end', 0.0):.2f}{motion}"
        )
    return " | ".join(parts)


def _format_live_tcn_prediction(
    prediction: CausalTcnLivePrediction,
    *,
    first_timestamp: float | None,
) -> str:
    relative = prediction.end_time
    if first_timestamp is not None:
        relative = prediction.end_time - first_timestamp
    probabilities = _compact_probabilities(prediction.probabilities)
    hand = f" hand={prediction.hand_id}" if prediction.hand_id else ""
    return (
        f"t={relative:7.3f}s{hand} target={prediction.target} "
        f"confidence={prediction.confidence:.3f} {probabilities}"
    )


def _format_live_tcn_v2_prediction(
    prediction: CausalTcnV2LivePrediction,
    *,
    first_timestamp: float | None,
    decoder_scores: dict[str, float],
) -> str:
    relative = prediction.end_time
    if first_timestamp is not None:
        relative = prediction.end_time - first_timestamp
    hand = f" hand={prediction.hand_id}" if prediction.hand_id else ""
    evidence = _compact_probabilities(prediction.evidence)
    scores = _compact_probabilities(decoder_scores)
    return f"t={relative:7.3f}s{hand} evidence=({evidence}) decoder=({scores})"


def _format_live_tcn_v2_candidate(
    candidate: GestureCandidate,
    *,
    first_timestamp: float | None,
) -> str:
    relative = candidate.timestamp
    if first_timestamp is not None:
        relative = candidate.timestamp - first_timestamp
    hand = f" hand={candidate.hand_id}" if candidate.hand_id else ""
    return (
        f"t={relative:7.3f}s{hand} decoded={candidate.name} "
        f"confidence={candidate.confidence:.3f}"
    )


def _show_live_tcn_prediction(
    prediction: CausalTcnLivePrediction,
    *,
    include_background: bool,
    include_recovery: bool,
    confidence_threshold: float,
) -> bool:
    if prediction.confidence < confidence_threshold:
        return False
    if prediction.target == "background":
        return include_background
    if _is_live_tcn_recovery_target(prediction.target):
        return include_recovery
    return True


def _is_live_tcn_gesture_target(target: str) -> bool:
    return target in {"swipe_left", "swipe_right", "stroke_left", "stroke_right"}


def _is_live_tcn_recovery_target(target: str) -> bool:
    return target in {"recovery", "reset", "release", "cooldown"}


def _format_live_dtw_candidate(
    candidate: GestureCandidate,
    *,
    first_timestamp: float | None,
) -> str:
    relative = candidate.timestamp
    if first_timestamp is not None:
        relative = candidate.timestamp - first_timestamp
    distance = candidate.metadata.get("distance", "unknown")
    threshold = candidate.metadata.get("threshold", "unknown")
    if isinstance(distance, float):
        distance = f"{distance:.3f}"
    if isinstance(threshold, float):
        threshold = f"{threshold:.3f}"
    hand = f" hand={candidate.hand_id}" if candidate.hand_id else ""
    return (
        f"t={relative:7.3f}s{hand} target={candidate.name} "
        f"confidence={candidate.confidence:.3f} distance={distance} threshold={threshold}"
    )


def _live_feature_streams(rows: list[FrameFeatureRow]) -> dict[str, list[FrameFeatureRow]]:
    return {
        stream[0].hand_id: stream
        for stream in group_feature_rows_by_stream(rows)
        if stream and stream[0].hand_id
    }


def _live_tcn_preview_status(state: dict[str, object]) -> str:
    status = str(state["status"])
    alert_until = float(state.get("alert_until", 0.0))
    alert = str(state.get("alert", ""))
    if alert and monotonic() <= alert_until:
        return f"{status} | GESTURE {alert}"
    return status


def _live_tcn_v2_preview_status(state: dict[str, object]) -> str:
    status = str(state["status"])
    alert_until = float(state.get("alert_until", 0.0))
    alert = str(state.get("alert", ""))
    if alert and monotonic() <= alert_until:
        return f"{status} | DECODED {alert}"
    return status


def _live_dtw_preview_status(state: dict[str, object]) -> str:
    status = str(state["status"])
    alert_until = float(state.get("alert_until", 0.0))
    alert = str(state.get("alert", ""))
    if alert and monotonic() <= alert_until:
        return f"{status} | GESTURE {alert}"
    return status


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _compact_probabilities(probabilities: dict[str, float]) -> str:
    return " ".join(
        f"{_short_tcn_target(target)}={probability:.2f}"
        for target, probability in sorted(probabilities.items())
    )


def _short_tcn_target(target: str) -> str:
    return {
        "background": "bg",
        "intentional_motion": "intent",
        "swipe_left": "left",
        "swipe_right": "right",
        "stroke_left": "left",
        "stroke_right": "right",
    }.get(target, target)
