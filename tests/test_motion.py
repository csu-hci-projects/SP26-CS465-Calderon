from __future__ import annotations

from pathlib import Path

from airdesk.analysis import evaluate_motion_recognizer
from airdesk.features import extract_feature_rows
from airdesk.gestures.motion import MotionEventConfig, MotionEventRecognizer, diagnose_motion_rows
from airdesk.labels import add_event_label, add_phase_label, init_label_file, save_label_file
from airdesk.recording.jsonl import JsonlRecordingWriter, iter_recording
from airdesk.state.types import (
    FrameMetadata,
    HandLandmarks,
    Landmark,
    NormalizedHand,
    TrackingFrame,
)


def test_motion_recognizer_keeps_hand_streams_separate() -> None:
    rows = extract_feature_rows(
        _two_hand_frames(
            primary_xs=(0.5, 0.5, 0.5, 0.5, 0.5),
            secondary_xs=(0.30, 0.30, 0.42, 0.58, 0.60),
        )
    )

    candidates = MotionEventRecognizer().recognize_rows(rows)

    assert [candidate.name for candidate in candidates] == ["swipe_right"]
    assert candidates[0].hand_id == "hand-1"
    assert candidates[0].metadata["dx_per_hand_scale"] > 1.0


def test_motion_recognizer_allows_repeated_same_direction_events() -> None:
    rows = extract_feature_rows(
        _one_hand_frames((0.30, 0.30, 0.44, 0.60, 0.60, 0.76, 0.92, 0.92))
    )

    candidates = MotionEventRecognizer().recognize_rows(rows)

    assert [candidate.name for candidate in candidates] == ["swipe_right", "swipe_right"]
    assert candidates[1].timestamp > candidates[0].timestamp
    assert candidates[0].metadata["evidence_id"] != candidates[1].metadata["evidence_id"]


def test_motion_recognizer_rejects_idle_background_motion() -> None:
    rows = extract_feature_rows(_one_hand_frames((0.50, 0.51, 0.50, 0.51, 0.50)))

    assert MotionEventRecognizer().recognize_rows(rows) == []


def test_motion_recognizer_preserves_merged_event_order_across_hands() -> None:
    rows = extract_feature_rows(
        _two_hand_frames(
            primary_xs=(0.30, 0.30, 0.44, 0.60, 0.60, 0.60),
            secondary_xs=(0.70, 0.70, 0.70, 0.70, 0.54, 0.38),
        )
    )

    candidates = MotionEventRecognizer().recognize_rows(rows)

    assert [(candidate.name, candidate.hand_id) for candidate in candidates] == [
        ("swipe_right", "hand-0"),
        ("swipe_left", "hand-1"),
    ]
    assert candidates == sorted(candidates, key=lambda candidate: candidate.timestamp)


def test_motion_recognizer_can_flip_raw_dx_direction_mapping() -> None:
    rows = extract_feature_rows(_one_hand_frames((0.30, 0.30, 0.44, 0.60, 0.60)))
    recognizer = MotionEventRecognizer(
        MotionEventConfig(positive_dx_gesture="swipe_left")
    )

    candidates = recognizer.recognize_rows(rows)

    assert [candidate.name for candidate in candidates] == ["swipe_left"]
    assert candidates[0].metadata["positive_dx_gesture"] == "swipe_left"


def test_motion_diagnostics_explain_rejected_weak_motion() -> None:
    rows = extract_feature_rows(_one_hand_frames((0.30, 0.30, 0.34, 0.36, 0.36)))

    diagnostics = diagnose_motion_rows(rows, limit_per_hand=3)

    assert diagnostics
    assert diagnostics[0].would_emit is False
    assert "dx_per_hand_scale_below_min" in diagnostics[0].rejection_reasons


def test_motion_diagnostics_include_label_phase_context(tmp_path: Path) -> None:
    recording = tmp_path / "swipe-right.jsonl"
    frames = _one_hand_frames((0.30, 0.30, 0.44, 0.60, 0.60))
    _write_recording(recording, (0.30, 0.30, 0.44, 0.60, 0.60))
    labels = init_label_file(recording)
    labels = add_phase_label(
        labels,
        phase="stroke_right",
        start_time=frames[2].timestamp,
        end_time=frames[3].timestamp,
        gesture="swipe_right",
    )
    labels = add_event_label(
        labels,
        gesture="swipe_right",
        start_time=frames[2].timestamp,
        end_time=frames[3].timestamp,
    )

    rows = extract_feature_rows(frames, labels=labels)
    candidates = MotionEventRecognizer().recognize_rows(rows)
    diagnostics = diagnose_motion_rows(rows, limit_per_hand=1)

    assert candidates[0].metadata["peak_phase"] == "stroke_right"
    assert candidates[0].metadata["peak_event"] == "swipe_right"
    assert diagnostics[0].phase == "stroke_right"
    assert diagnostics[0].event == "swipe_right"


def test_evaluate_motion_recognizer_reports_replay_metrics(tmp_path: Path) -> None:
    recording = tmp_path / "swipe-right.jsonl"
    _write_recording(recording, (0.30, 0.30, 0.44, 0.60, 0.60))
    labels = init_label_file(recording)
    labels = add_event_label(
        labels,
        gesture="swipe_right",
        start_time=100.0,
        end_time=100.8,
    )
    label_path = tmp_path / "swipe-right.labels.json"
    save_label_file(labels, label_path)

    evaluation = evaluate_motion_recognizer(recording, label_path, labels)

    assert evaluation.recognizer == "motion"
    assert evaluation.intended_events == 1
    assert evaluation.matched_events == 1
    assert evaluation.false_activations == 0


def _one_hand_frames(xs: tuple[float, ...]) -> list[TrackingFrame]:
    return [
        TrackingFrame(
            timestamp=100.0 + index * 0.2,
            source_id="motion-test",
            frame=FrameMetadata(
                timestamp=100.0 + index * 0.2,
                source_id="motion-test",
                width=640,
                height=480,
                sequence=index,
            ),
            hands=(_hand_at(x, hand_id="hand-0"),),
        )
        for index, x in enumerate(xs)
    ]


def _two_hand_frames(
    *,
    primary_xs: tuple[float, ...],
    secondary_xs: tuple[float, ...],
) -> list[TrackingFrame]:
    frames: list[TrackingFrame] = []
    for index, (primary_x, secondary_x) in enumerate(
        zip(primary_xs, secondary_xs, strict=True)
    ):
        timestamp = 100.0 + index * 0.2
        frames.append(
            TrackingFrame(
                timestamp=timestamp,
                source_id="motion-test",
                frame=FrameMetadata(
                    timestamp=timestamp,
                    source_id="motion-test",
                    width=640,
                    height=480,
                    sequence=index,
                ),
                hands=(
                    _hand_at(primary_x, hand_id="hand-0"),
                    _hand_at(secondary_x, hand_id="hand-1"),
                ),
            )
        )
    return frames


def _write_recording(path: Path, xs: tuple[float, ...]) -> None:
    with JsonlRecordingWriter(path) as writer:
        for frame in _one_hand_frames(xs):
            writer.write_tracking_frame(frame)

    # Sanity check the helper keeps absolute timestamps, matching real recordings.
    assert [record.payload for record in iter_recording(path) if record.kind == "tracking_frame"]


def _hand_at(palm_x: float, *, hand_id: str) -> NormalizedHand:
    landmarks = [Landmark(palm_x, 0.5, 0.0) for _ in range(21)]
    landmarks[4] = Landmark(palm_x - 0.08, 0.5, 0.0)
    landmarks[8] = Landmark(palm_x + 0.06, 0.45, 0.0)
    return NormalizedHand(
        hand_id=hand_id,
        landmarks=HandLandmarks(tuple(landmarks), handedness="right", confidence=1.0),
        palm_center=(palm_x, 0.5, 0.0),
        bbox=(palm_x - 0.1, 0.4, palm_x + 0.1, 0.6),
        handedness="right",
        confidence=1.0,
    )
