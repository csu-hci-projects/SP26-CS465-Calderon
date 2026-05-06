from __future__ import annotations

from pathlib import Path

import pytest

from airdesk.analysis import evaluate_dtw_recognizer
from airdesk.gestures.dtw import (
    DtwCalibrationInput,
    DtwGestureModel,
    DtwTemplateRecognizer,
    calibrate_dtw_model,
    dtw_distance,
)
from airdesk.labels import add_event_label, init_label_file, save_label_file
from airdesk.recording.jsonl import JsonlRecordingWriter
from airdesk.state.types import (
    FrameMetadata,
    HandLandmarks,
    Landmark,
    NormalizedHand,
    TrackingFrame,
)


def test_dtw_distance_handles_identical_and_variable_speed_sequences() -> None:
    template = ((0.0,), (0.5,), (1.0,))
    slower = ((0.0,), (0.25,), (0.5,), (0.75,), (1.0,))
    opposite = ((1.0,), (0.5,), (0.0,))

    assert dtw_distance(template, template) == pytest.approx(0.0)
    assert dtw_distance(template, slower) < dtw_distance(template, opposite)


def test_dtw_calibration_writes_templates_and_round_trips(tmp_path: Path) -> None:
    left_recording = _write_motion_recording(tmp_path / "left.jsonl", (0.7, 0.6, 0.45, 0.3))
    right_recording = _write_motion_recording(tmp_path / "right.jsonl", (0.3, 0.45, 0.6, 0.7))
    left_labels = _label_recording(left_recording, "swipe_left")
    right_labels = _label_recording(right_recording, "swipe_right")

    model = calibrate_dtw_model(
        [
            DtwCalibrationInput(left_recording, left_labels, tmp_path / "left.labels.json"),
            DtwCalibrationInput(right_recording, right_labels, tmp_path / "right.labels.json"),
        ]
    )
    output = tmp_path / "dtw.json"
    model.save(output)

    loaded = DtwGestureModel.load(output)

    assert len(loaded.templates) == 2
    assert set(loaded.thresholds) == {"swipe_left", "swipe_right"}
    assert loaded == model


def test_dtw_recognizer_matches_synthetic_swipe_and_rejects_stationary_negative(
    tmp_path: Path,
) -> None:
    left_a = _write_motion_recording(tmp_path / "left-a.jsonl", (0.7, 0.6, 0.45, 0.3))
    left_b = _write_motion_recording(tmp_path / "left-b.jsonl", (0.72, 0.58, 0.44, 0.31))
    negative = _write_motion_recording(tmp_path / "negative.jsonl", (0.5, 0.5, 0.5, 0.5))
    left_a_labels = _label_recording(left_a, "swipe_left")
    left_b_labels = _label_recording(left_b, "swipe_left")
    negative_labels = init_label_file(negative)

    model = calibrate_dtw_model(
        [
            DtwCalibrationInput(left_a, left_a_labels, tmp_path / "left-a.labels.json"),
            DtwCalibrationInput(left_b, left_b_labels, tmp_path / "left-b.labels.json"),
            DtwCalibrationInput(negative, negative_labels, tmp_path / "negative.labels.json"),
        ],
        min_window_seconds=0.2,
        max_window_seconds=0.5,
        window_step_seconds=0.1,
    )
    recognizer = DtwTemplateRecognizer(model)

    left_rows = _feature_rows(left_b, left_b_labels)
    negative_rows = _feature_rows(negative, negative_labels)

    assert any(candidate.name == "swipe_left" for candidate in recognizer.recognize_rows(left_rows))
    assert recognizer.recognize_rows(negative_rows) == []


def test_evaluate_dtw_recognizer_matches_labeled_event(tmp_path: Path) -> None:
    left_a = _write_motion_recording(tmp_path / "left-a.jsonl", (0.7, 0.6, 0.45, 0.3))
    left_b = _write_motion_recording(tmp_path / "left-b.jsonl", (0.72, 0.58, 0.44, 0.31))
    left_a_labels = _label_recording(left_a, "swipe_left")
    left_b_labels = _label_recording(left_b, "swipe_left")
    label_path = tmp_path / "left-b.labels.json"
    save_label_file(left_b_labels, label_path)
    model = calibrate_dtw_model(
        [
            DtwCalibrationInput(left_a, left_a_labels, tmp_path / "left-a.labels.json"),
            DtwCalibrationInput(left_b, left_b_labels, label_path),
        ],
        min_window_seconds=0.2,
        max_window_seconds=0.5,
        window_step_seconds=0.1,
    )

    evaluation = evaluate_dtw_recognizer(left_b, label_path, left_b_labels, model)

    assert evaluation.recognizer == "dtw"
    assert evaluation.intended_events == 1
    assert evaluation.matched_events == 1
    assert evaluation.missed_events == 0


def _feature_rows(recording: Path, labels: object) -> object:
    from airdesk.features import extract_feature_rows
    from airdesk.recording.jsonl import iter_recording

    frames = [
        record.payload
        for record in iter_recording(recording)
        if record.kind == "tracking_frame" and isinstance(record.payload, TrackingFrame)
    ]
    return extract_feature_rows(frames, labels=labels)


def _label_recording(recording: Path, gesture: str) -> object:
    label_file = init_label_file(recording)
    start = label_file.session.start_timestamp or 0.0
    end = label_file.session.end_timestamp or start
    return add_event_label(label_file, gesture=gesture, start_time=start, end_time=end)


def _write_motion_recording(path: Path, xs: tuple[float, ...]) -> Path:
    with JsonlRecordingWriter(path) as writer:
        timestamp = 1.0
        for sequence, x in enumerate(xs):
            frame = FrameMetadata(
                timestamp=timestamp,
                source_id="dtw-test",
                width=640,
                height=480,
                sequence=sequence,
            )
            writer.write_tracking_frame(
                TrackingFrame(
                    timestamp=timestamp,
                    source_id="dtw-test",
                    frame=frame,
                    hands=(_hand_at(x),),
                )
            )
            timestamp += 0.1
    return path


def _hand_at(palm_x: float) -> NormalizedHand:
    landmarks = [Landmark(palm_x, 0.5, 0.0) for _ in range(21)]
    landmarks[4] = Landmark(palm_x - 0.08, 0.5, 0.0)
    landmarks[8] = Landmark(palm_x + 0.06, 0.45, 0.0)
    return NormalizedHand(
        hand_id="hand-0",
        landmarks=HandLandmarks(tuple(landmarks), handedness="right", confidence=1.0),
        palm_center=(palm_x, 0.5, 0.0),
        bbox=(palm_x - 0.1, 0.4, palm_x + 0.1, 0.6),
        handedness="right",
        confidence=1.0,
    )
