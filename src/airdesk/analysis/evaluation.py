"""Continuous gesture evaluation utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from airdesk.gestures.base import CompositeGestureRecognizer
from airdesk.gestures.phrases import IntentGatedSwipeRecognizer
from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.labels import GestureEventLabel, GestureLabelFile
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
            and event.start_time <= candidate.timestamp <= event.end_time
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
        if _inside_any_event(candidate, intended):
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
        recognizer="rule",
        intended_events=len(intended),
        matched_events=matched_events,
        missed_events=missed_events,
        candidate_count=len(candidates),
        false_activations=false_activations,
        repeated_fires=repeated_fires,
        latencies_seconds=tuple(latencies),
        per_gesture=per_gesture,
    )


def save_evaluation_json(evaluation: GestureEvaluation, path: Path) -> None:
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


def _inside_any_event(candidate: GestureCandidate, events: list[GestureEventLabel]) -> bool:
    return any(
        event.start_time <= candidate.timestamp <= event.end_time
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
