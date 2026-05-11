"""TCN v2 event-decoder evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from airdesk.analysis.evaluation import (
    GestureEvaluation,
    diagnose_candidate_events,
    evaluate_candidates,
    holdout_totals,
)
from airdesk.gestures.decoder import DecoderFrame, EventDecoder, EventDecoderConfig
from airdesk.labels import load_label_file
from airdesk.ml import (
    CausalTcnEvidencePrediction,
    feature_window_frame_targets,
    load_tcn_dataset_manifest,
    predict_causal_tcn_v2_manifest,
)
from airdesk.ml.train import _require_torch
from airdesk.state.types import GestureCandidate

TCN_V2_CONTROL_HEADS = {"intentional_motion", "start", "end"}


def evaluate_tcn_v2_manifest(
    *,
    manifest_path: Path,
    model_path: Path,
    event_decoder_config: EventDecoderConfig,
    match_tolerance_seconds: float = 0.5,
    early_match_tolerance_seconds: float = 0.0,
    device: str = "auto",
) -> tuple[GestureEvaluation, ...]:
    """Evaluate TCN v2 decoder-facing evidence against labeled manifest sources."""
    manifest = load_tcn_dataset_manifest(manifest_path)
    predictions = dedupe_tcn_v2_predictions(
        predict_causal_tcn_v2_manifest(
            model_path=model_path,
            manifest_path=manifest_path,
            emit_all_rows=True,
            device=device,
        )
    )
    predictions_by_source: dict[tuple[str, str], list[CausalTcnEvidencePrediction]] = {}
    for prediction in predictions:
        if prediction.label_path is None:
            continue
        predictions_by_source.setdefault(
            (prediction.feature_path, prediction.label_path),
            [],
        ).append(prediction)

    evaluations: list[GestureEvaluation] = []
    for source in manifest.sources:
        if source.label_path is None:
            continue
        labels = load_label_file(Path(source.label_path))
        source_predictions = predictions_by_source.get((source.feature_path, source.label_path), [])
        candidates = decode_tcn_v2_predictions(source_predictions, event_decoder_config)
        evaluations.append(
            evaluate_candidates(
                recording_path=Path(source.feature_path),
                label_path=Path(source.label_path),
                labels=labels,
                recognizer="tcn_v2_event_decoder",
                candidates=candidates,
                match_tolerance_seconds=match_tolerance_seconds,
                early_match_tolerance_seconds=early_match_tolerance_seconds,
            )
        )
    return tuple(evaluations)


def diagnose_tcn_v2_manifest_events(
    *,
    manifest_path: Path,
    model_path: Path,
    event_decoder_config: EventDecoderConfig,
    match_tolerance_seconds: float = 0.5,
    early_match_tolerance_seconds: float = 0.0,
    device: str = "auto",
) -> dict[str, object]:
    """Write detailed TCN v2 decoded-event diagnostics for replay review."""
    manifest = load_tcn_dataset_manifest(manifest_path)
    predictions = dedupe_tcn_v2_predictions(
        predict_causal_tcn_v2_manifest(
            model_path=model_path,
            manifest_path=manifest_path,
            emit_all_rows=True,
            device=device,
        )
    )
    predictions_by_source: dict[tuple[str, str], list[CausalTcnEvidencePrediction]] = {}
    for prediction in predictions:
        if prediction.label_path is None:
            continue
        predictions_by_source.setdefault(
            (prediction.feature_path, prediction.label_path),
            [],
        ).append(prediction)

    evaluations: list[GestureEvaluation] = []
    diagnostics: list[dict[str, object]] = []
    for source in manifest.sources:
        if source.label_path is None:
            continue
        labels = load_label_file(Path(source.label_path))
        source_predictions = predictions_by_source.get((source.feature_path, source.label_path), [])
        candidates = decode_tcn_v2_predictions(source_predictions, event_decoder_config)
        evaluation = evaluate_candidates(
            recording_path=Path(source.feature_path),
            label_path=Path(source.label_path),
            labels=labels,
            recognizer="tcn_v2_event_decoder",
            candidates=candidates,
            match_tolerance_seconds=match_tolerance_seconds,
            early_match_tolerance_seconds=early_match_tolerance_seconds,
        )
        evaluations.append(evaluation)
        diagnostics.append(
            {
                "recording": source.feature_path,
                "labels": source.label_path,
                **diagnose_candidate_events(
                    labels=labels,
                    candidates=candidates,
                    match_tolerance_seconds=match_tolerance_seconds,
                    early_match_tolerance_seconds=early_match_tolerance_seconds,
                ),
            }
        )

    return {
        "recognizer": "tcn_v2_event_decoder",
        "manifest": str(manifest_path),
        "model": str(model_path),
        "match_tolerance_seconds": match_tolerance_seconds,
        "early_match_tolerance_seconds": early_match_tolerance_seconds,
        "event_decoder": event_decoder_config.to_dict(),
        "device": device,
        "summary": holdout_totals(tuple(evaluations)),
        "evaluations": [evaluation.to_dict() for evaluation in evaluations],
        "sources": diagnostics,
    }


def evaluate_tcn_v2_head_manifest(
    *,
    manifest_path: Path,
    model_path: Path,
    threshold: float | None = None,
    batch_size: int = 64,
    device: str = "auto",
) -> dict[str, object]:
    """Evaluate TCN v2 evidence heads on each causal window's final frame."""
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    manifest = load_tcn_dataset_manifest(manifest_path)
    if manifest.target_mode != "v2-evidence":
        raise ValueError("TCN v2 head evaluation requires target_mode='v2-evidence'")
    if not manifest.evidence_targets:
        raise ValueError("TCN v2 manifest is missing evidence_targets")
    predictions = predict_causal_tcn_v2_manifest(
        model_path=model_path,
        manifest_path=manifest_path,
        emit_all_rows=False,
        batch_size=batch_size,
        device=device,
    )
    predictions_by_sample = _predictions_by_sample_id(predictions)
    thresholds, threshold_source = _head_evaluation_thresholds(
        model_path=model_path,
        evidence_targets=manifest.evidence_targets,
        threshold=threshold,
    )
    counts = _empty_head_counts(manifest.evidence_targets)
    gesture_heads = tuple(
        target for target in manifest.evidence_targets if target not in TCN_V2_CONTROL_HEADS
    )
    confusion_labels = ("background",) + gesture_heads
    confusion = {
        actual: {predicted: 0 for predicted in confusion_labels}
        for actual in confusion_labels
    }

    for window in manifest.windows:
        prediction = predictions_by_sample.get(window.sample_id)
        if prediction is None:
            raise ValueError(f"Missing TCN v2 prediction for sample_id={window.sample_id}")
        expected_rows = feature_window_frame_targets(
            window,
            evidence_targets=manifest.evidence_targets,
        )
        if not expected_rows:
            raise ValueError(f"TCN v2 window has no final-frame targets: {window.sample_id}")
        expected_final = expected_rows[-1]
        for index, target in enumerate(manifest.evidence_targets):
            actual = expected_final[index] >= 0.5
            predicted = prediction.evidence.get(target, 0.0) >= thresholds[target]
            _update_binary_counts(counts[target], actual=actual, predicted=predicted)
        actual_gesture = _actual_final_gesture_label(
            expected_final,
            evidence_targets=manifest.evidence_targets,
            gesture_heads=gesture_heads,
        )
        predicted_gesture = _predicted_final_gesture_label(
            prediction.evidence,
            gesture_heads=gesture_heads,
            thresholds=thresholds,
        )
        confusion[actual_gesture][predicted_gesture] += 1

    per_head = {
        target: _head_metrics(counts[target], threshold=thresholds[target])
        for target in manifest.evidence_targets
    }
    return {
        "recognizer": "tcn_v2_final_frame_heads",
        "manifest": str(manifest_path),
        "model": str(model_path),
        "device": device,
        "batch_size": batch_size,
        "threshold_source": threshold_source,
        "thresholds": thresholds,
        "source_count": len(manifest.sources),
        "window_count": len(manifest.windows),
        "evidence_targets": list(manifest.evidence_targets),
        "per_head": per_head,
        "macro": _aggregate_head_metrics(per_head, manifest.evidence_targets),
        "micro": _micro_head_metrics(counts, manifest.evidence_targets),
        "gesture_macro": _aggregate_head_metrics(per_head, gesture_heads),
        "gesture_micro": _micro_head_metrics(counts, gesture_heads),
        "gesture_confusion": {
            "labels": list(confusion_labels),
            "matrix": confusion,
            "top_confusions": _top_gesture_confusions(confusion),
        },
    }


def decode_tcn_v2_predictions(
    predictions: list[CausalTcnEvidencePrediction],
    config: EventDecoderConfig,
) -> list[GestureCandidate]:
    """Convert framewise TCN v2 evidence heads into gesture candidates."""
    frames = [
        DecoderFrame(
            timestamp=prediction.timestamp,
            scores=tcn_v2_decoder_scores(prediction.evidence),
            source_id=prediction.feature_path,
            hand_id=prediction.hand_id or None,
            window_start=prediction.window_start,
            window_end=prediction.window_end,
            metadata={
                "recognizer": "tcn_v2",
                "sample_id": prediction.sample_id,
                "intentional_motion": prediction.evidence.get("intentional_motion", 0.0),
                "start": prediction.evidence.get("start", 0.0),
                "end": prediction.evidence.get("end", 0.0),
                "raw_evidence": prediction.evidence,
                "decoder_scores": tcn_v2_decoder_scores(prediction.evidence),
            },
        )
        for prediction in predictions
    ]
    return EventDecoder(config).decode(frames)


def tcn_v2_decoder_scores(evidence: dict[str, float]) -> dict[str, float]:
    """Map v2 evidence heads into decoder scores that use start/end boundaries."""
    intentional_motion = _bounded_score(evidence.get("intentional_motion", 0.0))
    start = _bounded_score(evidence.get("start", 0.0))
    end = _bounded_score(evidence.get("end", 0.0))
    return {
        "background": max(1.0 - intentional_motion, end),
        "swipe_left": _boundary_adjusted_stroke_score(
            evidence.get("stroke_left", 0.0),
            start=start,
            end=end,
        ),
        "swipe_right": _boundary_adjusted_stroke_score(
            evidence.get("stroke_right", 0.0),
            start=start,
            end=end,
        ),
    }


def _boundary_adjusted_stroke_score(
    stroke_score: float,
    *,
    start: float,
    end: float,
) -> float:
    stroke = _bounded_score(stroke_score)
    start_boosted = min(1.0, stroke * (1.0 + 0.5 * start))
    return start_boosted * max(0.0, 1.0 - 0.85 * end)


def _bounded_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def dedupe_tcn_v2_predictions(
    predictions: list[CausalTcnEvidencePrediction],
) -> list[CausalTcnEvidencePrediction]:
    """Keep one causal-context prediction for each source/hand/timestamp frame."""
    selected: dict[tuple[str, str | None, str, float], CausalTcnEvidencePrediction] = {}
    for prediction in predictions:
        key = (
            prediction.feature_path,
            prediction.label_path,
            prediction.hand_id,
            prediction.timestamp,
        )
        existing = selected.get(key)
        if existing is None or context_seconds(prediction) > context_seconds(existing):
            selected[key] = prediction
    return sorted(
        selected.values(),
        key=lambda item: (item.feature_path, item.timestamp, item.hand_id, item.sample_id),
    )


def context_seconds(prediction: CausalTcnEvidencePrediction) -> float:
    """Return the amount of causal context available for a prediction frame."""
    return max(0.0, prediction.timestamp - prediction.window_start)


def _predictions_by_sample_id(
    predictions: list[CausalTcnEvidencePrediction],
) -> dict[str, CausalTcnEvidencePrediction]:
    selected: dict[str, CausalTcnEvidencePrediction] = {}
    for prediction in predictions:
        if prediction.sample_id in selected:
            raise ValueError(f"Duplicate TCN v2 prediction for sample_id={prediction.sample_id}")
        selected[prediction.sample_id] = prediction
    return selected


def _head_evaluation_thresholds(
    *,
    model_path: Path,
    evidence_targets: tuple[str, ...],
    threshold: float | None,
) -> tuple[dict[str, float], str]:
    if threshold is not None:
        return {target: threshold for target in evidence_targets}, "fixed"
    torch, _nn, _functional = _require_torch()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    calibration = checkpoint.get("metadata", {}).get("calibration", {})
    checkpoint_thresholds = calibration.get("evidence_thresholds", {})
    return (
        {
            target: float(checkpoint_thresholds.get(target, 0.5))
            for target in evidence_targets
        },
        "checkpoint_calibration",
    )


def _empty_head_counts(evidence_targets: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {
        target: {
            "positive": 0,
            "negative": 0,
            "predicted_positive": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 0,
        }
        for target in evidence_targets
    }


def _update_binary_counts(
    counts: dict[str, int],
    *,
    actual: bool,
    predicted: bool,
) -> None:
    if actual:
        counts["positive"] += 1
    else:
        counts["negative"] += 1
    if predicted:
        counts["predicted_positive"] += 1
    if actual and predicted:
        counts["true_positive"] += 1
    elif not actual and predicted:
        counts["false_positive"] += 1
    elif actual and not predicted:
        counts["false_negative"] += 1
    else:
        counts["true_negative"] += 1


def _head_metrics(counts: dict[str, int], *, threshold: float) -> dict[str, float | int]:
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "threshold": threshold,
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _aggregate_head_metrics(
    per_head: dict[str, dict[str, float | int]],
    heads: tuple[str, ...],
) -> dict[str, float]:
    if not heads:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": sum(float(per_head[head]["precision"]) for head in heads) / len(heads),
        "recall": sum(float(per_head[head]["recall"]) for head in heads) / len(heads),
        "f1": sum(float(per_head[head]["f1"]) for head in heads) / len(heads),
    }


def _micro_head_metrics(
    counts: dict[str, dict[str, int]],
    heads: tuple[str, ...],
) -> dict[str, float | int]:
    true_positive = sum(counts[head]["true_positive"] for head in heads)
    false_positive = sum(counts[head]["false_positive"] for head in heads)
    false_negative = sum(counts[head]["false_negative"] for head in heads)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2 * precision * recall, precision + recall),
    }


def _actual_final_gesture_label(
    expected_final: list[float],
    *,
    evidence_targets: tuple[str, ...],
    gesture_heads: tuple[str, ...],
) -> str:
    target_values = dict(zip(evidence_targets, expected_final, strict=True))
    for target in gesture_heads:
        if target_values.get(target, 0.0) >= 0.5:
            return target
    return "background"


def _predicted_final_gesture_label(
    evidence: dict[str, float],
    *,
    gesture_heads: tuple[str, ...],
    thresholds: dict[str, float],
) -> str:
    if not gesture_heads:
        return "background"
    best = max(gesture_heads, key=lambda target: evidence.get(target, 0.0))
    if evidence.get(best, 0.0) >= thresholds[best]:
        return best
    return "background"


def _top_gesture_confusions(
    confusion: dict[str, dict[str, int]],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for actual, row in confusion.items():
        for predicted, count in row.items():
            if actual == predicted or count <= 0:
                continue
            items.append({"actual": actual, "predicted": predicted, "count": count})
    return sorted(items, key=lambda item: (-int(item["count"]), item["actual"], item["predicted"]))[
        :limit
    ]


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
