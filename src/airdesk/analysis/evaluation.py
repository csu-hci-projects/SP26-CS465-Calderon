"""Continuous gesture evaluation utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from airdesk.features import extract_feature_rows
from airdesk.gestures.base import CompositeGestureRecognizer
from airdesk.gestures.dtw import (
    DtwCalibrationInput,
    DtwGestureModel,
    DtwTemplateRecognizer,
    calibrate_dtw_model,
)
from airdesk.gestures.phrases import IntentGatedSwipeRecognizer
from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.labels import GestureEventLabel, GestureLabelFile, load_label_file
from airdesk.recording.jsonl import iter_recording
from airdesk.state.types import GestureCandidate, TrackingFrame


@dataclass(frozen=True)
class GestureEvaluation:
    """Evaluation summary for one recording/label pair."""

    recording: str
    labels: str
    recognizer: str
    intended_events: int
    matched_events: int
    missed_events: int
    candidate_count: int
    false_activations: int
    repeated_fires: int
    latencies_seconds: tuple[float, ...] = ()
    per_gesture: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LabeledRecording:
    """Recording and label file pair used for offline evaluation."""

    recording: Path
    label_path: Path
    labels: GestureLabelFile
    group: str
    gesture: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "recording": str(self.recording),
            "labels": str(self.label_path),
            "group": self.group,
            "gesture": self.gesture,
        }


@dataclass(frozen=True)
class DtwHoldoutEvaluation:
    """Train/test DTW holdout summary across multiple recordings."""

    recognizer: str
    train_recordings: tuple[LabeledRecording, ...]
    test_recordings: tuple[LabeledRecording, ...]
    evaluations: tuple[GestureEvaluation, ...]
    model_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "recognizer": self.recognizer,
            "model_path": self.model_path,
            "split": {
                "train": [item.to_dict() for item in self.train_recordings],
                "test": [item.to_dict() for item in self.test_recordings],
            },
            "summary": holdout_totals(self.evaluations),
            "evaluations": [item.to_dict() for item in self.evaluations],
        }


def evaluate_rule_recognizer(
    recording_path: Path,
    label_path: Path,
    labels: GestureLabelFile,
) -> GestureEvaluation:
    """Evaluate the current rule recognizer against event labels."""
    frames = [
        record.payload
        for record in iter_recording(recording_path)
        if record.kind == "tracking_frame" and isinstance(record.payload, TrackingFrame)
    ]
    recognizer = _rule_recognizer()
    candidates: list[GestureCandidate] = []
    for frame in frames:
        candidates.extend(recognizer.recognize(frame))

    return evaluate_candidates(
        recording_path=recording_path,
        label_path=label_path,
        labels=labels,
        recognizer="rule",
        candidates=candidates,
        match_tolerance_seconds=0.0,
    )


def evaluate_dtw_recognizer(
    recording_path: Path,
    label_path: Path,
    labels: GestureLabelFile,
    model: DtwGestureModel,
) -> GestureEvaluation:
    """Evaluate a calibrated DTW recognizer against event labels."""
    frames = [
        record.payload
        for record in iter_recording(recording_path)
        if record.kind == "tracking_frame" and isinstance(record.payload, TrackingFrame)
    ]
    rows = extract_feature_rows(frames, labels=labels)
    candidates = DtwTemplateRecognizer(model).recognize_rows(rows)
    return evaluate_candidates(
        recording_path=recording_path,
        label_path=label_path,
        labels=labels,
        recognizer="dtw",
        candidates=candidates,
        match_tolerance_seconds=0.5,
    )


def evaluate_candidates(
    *,
    recording_path: Path,
    label_path: Path,
    labels: GestureLabelFile,
    recognizer: str,
    candidates: list[GestureCandidate],
    match_tolerance_seconds: float = 0.0,
) -> GestureEvaluation:
    """Evaluate a candidate stream against gesture event labels."""
    intended = [event for event in labels.event_labels if event.label_type == "gesture"]
    matched_events = 0
    repeated_fires = 0
    latencies: list[float] = []
    per_gesture: dict[str, dict[str, int]] = {}
    matched_candidate_ids: set[int] = set()

    for event in intended:
        bucket = per_gesture.setdefault(
            event.gesture,
            {"intended": 0, "matched": 0, "missed": 0, "false_activations": 0},
        )
        bucket["intended"] += 1
        event_candidates = [
            (index, candidate)
            for index, candidate in enumerate(candidates)
            if candidate.name == event.gesture
            and event.start_time <= candidate.timestamp <= event.end_time + match_tolerance_seconds
        ]
        if not event_candidates:
            bucket["missed"] += 1
            continue
        matched_events += 1
        bucket["matched"] += 1
        repeated_fires += max(0, len(event_candidates) - 1)
        first_index, first_candidate = event_candidates[0]
        matched_candidate_ids.add(first_index)
        latencies.append(first_candidate.timestamp - event.start_time)
        for index, _candidate in event_candidates[1:]:
            matched_candidate_ids.add(index)

    false_activations = 0
    for index, candidate in enumerate(candidates):
        if index in matched_candidate_ids:
            continue
        if _inside_any_event(candidate, intended, match_tolerance_seconds=match_tolerance_seconds):
            continue
        false_activations += 1
        bucket = per_gesture.setdefault(
            candidate.name,
            {"intended": 0, "matched": 0, "missed": 0, "false_activations": 0},
        )
        bucket["false_activations"] += 1

    missed_events = len(intended) - matched_events
    return GestureEvaluation(
        recording=str(recording_path),
        labels=str(label_path),
        recognizer=recognizer,
        intended_events=len(intended),
        matched_events=matched_events,
        missed_events=missed_events,
        candidate_count=len(candidates),
        false_activations=false_activations,
        repeated_fires=repeated_fires,
        latencies_seconds=tuple(latencies),
        per_gesture=per_gesture,
    )


def evaluate_dtw_holdout(
    *,
    recordings_dir: Path,
    labels_dir: Path,
    model_path: Path | None = None,
    train_per_gesture: int = 6,
    test_per_gesture: int = 2,
    train_negatives: int = 6,
    test_negatives: int = 2,
    cooldown_seconds: float = 0.5,
    min_window_seconds: float = 0.25,
    max_window_seconds: float = 1.25,
    window_step_seconds: float = 0.1,
) -> DtwHoldoutEvaluation:
    """Calibrate DTW on a deterministic train split and evaluate a held-out test split."""
    labeled = load_labeled_recordings(recordings_dir, labels_dir)
    train, test = split_holdout_recordings(
        labeled,
        train_per_gesture=train_per_gesture,
        test_per_gesture=test_per_gesture,
        train_negatives=train_negatives,
        test_negatives=test_negatives,
    )
    if not train:
        raise ValueError("DTW holdout requires at least one training recording")
    if not test:
        raise ValueError("DTW holdout requires at least one test recording")

    inputs = [
        DtwCalibrationInput(
            recording=item.recording,
            labels=item.labels,
            label_path=item.label_path,
        )
        for item in train
    ]
    model = calibrate_dtw_model(
        inputs,
        cooldown_seconds=cooldown_seconds,
        min_window_seconds=min_window_seconds,
        max_window_seconds=max_window_seconds,
        window_step_seconds=window_step_seconds,
    )
    if model_path is not None:
        model.save(model_path)

    evaluations = tuple(
        evaluate_dtw_recognizer(item.recording, item.label_path, item.labels, model)
        for item in test
    )
    return DtwHoldoutEvaluation(
        recognizer="dtw",
        train_recordings=tuple(train),
        test_recordings=tuple(test),
        evaluations=evaluations,
        model_path=str(model_path) if model_path is not None else None,
    )


def load_labeled_recordings(recordings_dir: Path, labels_dir: Path) -> list[LabeledRecording]:
    """Load matching recording/label pairs from collection directories."""
    labeled: list[LabeledRecording] = []
    for recording in sorted(recordings_dir.glob("*.jsonl")):
        label_path = labels_dir / f"{recording.stem}.labels.json"
        if not label_path.exists():
            raise ValueError(f"Missing label file for recording={recording}: {label_path}")
        labels = load_label_file(label_path)
        events = [event for event in labels.event_labels if event.label_type == "gesture"]
        gesture = events[0].gesture if events else None
        group = gesture or _recording_group(recording)
        labeled.append(
            LabeledRecording(
                recording=recording,
                label_path=label_path,
                labels=labels,
                group=group,
                gesture=gesture,
            )
        )
    return labeled


def split_holdout_recordings(
    recordings: list[LabeledRecording],
    *,
    train_per_gesture: int,
    test_per_gesture: int,
    train_negatives: int,
    test_negatives: int,
) -> tuple[list[LabeledRecording], list[LabeledRecording]]:
    """Deterministically split positive gesture groups and negatives by filename order."""
    positives: dict[str, list[LabeledRecording]] = {}
    negatives: list[LabeledRecording] = []
    for item in sorted(recordings, key=lambda value: str(value.recording)):
        if item.gesture is None:
            negatives.append(item)
        else:
            positives.setdefault(item.gesture, []).append(item)

    train: list[LabeledRecording] = []
    test: list[LabeledRecording] = []
    for gesture in sorted(positives):
        items = positives[gesture]
        if len(items) < train_per_gesture + test_per_gesture:
            raise ValueError(
                f"Not enough recordings for gesture={gesture}: "
                f"need {train_per_gesture + test_per_gesture}, found {len(items)}"
            )
        train.extend(items[:train_per_gesture])
        test.extend(items[train_per_gesture : train_per_gesture + test_per_gesture])

    if train_negatives + test_negatives > 0:
        if len(negatives) < train_negatives + test_negatives:
            raise ValueError(
                "Not enough negative recordings: "
                f"need {train_negatives + test_negatives}, found {len(negatives)}"
            )
        train.extend(negatives[:train_negatives])
        test.extend(negatives[train_negatives : train_negatives + test_negatives])

    return train, test


def holdout_totals(evaluations: tuple[GestureEvaluation, ...]) -> dict[str, object]:
    """Aggregate existing per-recording metrics for a holdout report."""
    latencies = [
        latency for evaluation in evaluations for latency in evaluation.latencies_seconds
    ]
    per_gesture: dict[str, dict[str, int]] = {}
    for evaluation in evaluations:
        for gesture, metrics in evaluation.per_gesture.items():
            bucket = per_gesture.setdefault(
                gesture,
                {"intended": 0, "matched": 0, "missed": 0, "false_activations": 0},
            )
            for key in bucket:
                bucket[key] += metrics.get(key, 0)
    return {
        "recordings": len(evaluations),
        "intended_events": sum(item.intended_events for item in evaluations),
        "matched_events": sum(item.matched_events for item in evaluations),
        "missed_events": sum(item.missed_events for item in evaluations),
        "candidate_count": sum(item.candidate_count for item in evaluations),
        "false_activations": sum(item.false_activations for item in evaluations),
        "repeated_fires": sum(item.repeated_fires for item in evaluations),
        "mean_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "per_gesture": per_gesture,
    }


def save_evaluation_json(evaluation: GestureEvaluation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
            json.dump(evaluation.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


def save_holdout_json(evaluation: DtwHoldoutEvaluation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def format_evaluation(evaluation: GestureEvaluation) -> str:
    latency = (
        round(sum(evaluation.latencies_seconds) / len(evaluation.latencies_seconds), 4)
        if evaluation.latencies_seconds
        else "unknown"
    )
    return (
        f"recording={Path(evaluation.recording).name} recognizer={evaluation.recognizer} "
        f"intended={evaluation.intended_events} matched={evaluation.matched_events} "
        f"missed={evaluation.missed_events} candidates={evaluation.candidate_count} "
        f"false_activations={evaluation.false_activations} "
        f"repeated_fires={evaluation.repeated_fires} mean_latency={latency}"
    )


def format_holdout_evaluation(evaluation: DtwHoldoutEvaluation) -> str:
    summary = holdout_totals(evaluation.evaluations)
    latency = summary["mean_latency_seconds"]
    formatted_latency = round(latency, 4) if isinstance(latency, float) else "unknown"
    return (
        f"recognizer={evaluation.recognizer} "
        f"train={len(evaluation.train_recordings)} test={len(evaluation.test_recordings)} "
        f"intended={summary['intended_events']} matched={summary['matched_events']} "
        f"missed={summary['missed_events']} candidates={summary['candidate_count']} "
        f"false_activations={summary['false_activations']} "
        f"repeated_fires={summary['repeated_fires']} mean_latency={formatted_latency}"
    )


def _recording_group(recording: Path) -> str:
    stem = recording.stem
    parts = stem.rsplit("-", maxsplit=1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def _inside_any_event(
    candidate: GestureCandidate,
    events: list[GestureEventLabel],
    *,
    match_tolerance_seconds: float,
) -> bool:
    return any(
        event.start_time <= candidate.timestamp <= event.end_time + match_tolerance_seconds
        for event in events
    )


def _rule_recognizer() -> CompositeGestureRecognizer:
    static_recognizer = StaticHandPoseRecognizer()
    return CompositeGestureRecognizer(
        recognizers=(
            static_recognizer,
            IntentGatedSwipeRecognizer(pose_recognizer=static_recognizer),
        )
    )
