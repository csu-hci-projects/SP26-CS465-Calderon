"""TCN v2 event-decoder evaluation helpers."""

from __future__ import annotations

from pathlib import Path

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
    load_tcn_dataset_manifest,
    predict_causal_tcn_v2_manifest,
)
from airdesk.state.types import GestureCandidate


def evaluate_tcn_v2_manifest(
    *,
    manifest_path: Path,
    model_path: Path,
    event_decoder_config: EventDecoderConfig,
    match_tolerance_seconds: float = 0.5,
    early_match_tolerance_seconds: float = 0.0,
) -> tuple[GestureEvaluation, ...]:
    """Evaluate TCN v2 decoder-facing evidence against labeled manifest sources."""
    manifest = load_tcn_dataset_manifest(manifest_path)
    predictions = dedupe_tcn_v2_predictions(
        predict_causal_tcn_v2_manifest(
            model_path=model_path,
            manifest_path=manifest_path,
            emit_all_rows=True,
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
) -> dict[str, object]:
    """Write detailed TCN v2 decoded-event diagnostics for replay review."""
    manifest = load_tcn_dataset_manifest(manifest_path)
    predictions = dedupe_tcn_v2_predictions(
        predict_causal_tcn_v2_manifest(
            model_path=model_path,
            manifest_path=manifest_path,
            emit_all_rows=True,
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
        "summary": holdout_totals(tuple(evaluations)),
        "evaluations": [evaluation.to_dict() for evaluation in evaluations],
        "sources": diagnostics,
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
