from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from airdesk.analysis import evaluate_rule_recognizer, format_evaluation
from airdesk.labels import add_event_label, init_label_file
from airdesk.recording.jsonl import JsonlRecordingWriter
from airdesk.state.types import FrameMetadata, NormalizedHand, TrackingFrame


def frame_at(timestamp: float, sequence: int, hand: NormalizedHand) -> TrackingFrame:
    metadata = FrameMetadata(
        timestamp=timestamp,
        source_id="evaluation-test",
        width=640,
        height=480,
        sequence=sequence,
    )
    return TrackingFrame(
        timestamp=timestamp,
        source_id="evaluation-test",
        frame=metadata,
        hands=(hand,),
    )


def test_evaluate_rule_recognizer_matches_labeled_event(
    tmp_path: Path,
    make_hand: Callable[[str], NormalizedHand],
) -> None:
    recording = tmp_path / "recording.jsonl"
    with JsonlRecordingWriter(recording) as writer:
        writer.write_tracking_frame(frame_at(1.0, 1, make_hand("open_palm")))
        writer.write_tracking_frame(frame_at(1.1, 2, make_hand("open_palm")))
    label_file = init_label_file(recording)
    label_file = add_event_label(
        label_file,
        gesture="open_palm",
        start_time=1.0,
        end_time=1.2,
    )

    evaluation = evaluate_rule_recognizer(recording, tmp_path / "labels.json", label_file)

    assert evaluation.intended_events == 1
    assert evaluation.matched_events == 1
    assert evaluation.missed_events == 0
    assert evaluation.repeated_fires == 1
    assert evaluation.false_activations == 0
    assert "matched=1" in format_evaluation(evaluation)


def test_evaluate_rule_recognizer_counts_false_activations(
    tmp_path: Path,
    make_hand: Callable[[str], NormalizedHand],
) -> None:
    recording = tmp_path / "recording.jsonl"
    with JsonlRecordingWriter(recording) as writer:
        writer.write_tracking_frame(frame_at(1.0, 1, make_hand("pinch")))
    label_file = init_label_file(recording)

    evaluation = evaluate_rule_recognizer(recording, tmp_path / "labels.json", label_file)

    assert evaluation.intended_events == 0
    assert evaluation.false_activations >= 1
