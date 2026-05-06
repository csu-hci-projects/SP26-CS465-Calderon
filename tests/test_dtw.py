from __future__ import annotations

from pathlib import Path

import pytest

from airdesk.analysis import (
    evaluate_dtw_holdout,
    evaluate_dtw_recognizer,
    format_holdout_evaluation,
    save_holdout_json,
)
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


def test_dtw_holdout_trains_only_on_split_and_exports_summary(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    labels_dir = tmp_path / "labels"
    recordings_dir.mkdir()
    labels_dir.mkdir()
    _write_labeled_take(
        recordings_dir,
        labels_dir,
        "swipe-left-positive-001",
        (0.7, 0.6, 0.45, 0.3),
        "swipe_left",
    )
    _write_labeled_take(
        recordings_dir,
        labels_dir,
        "swipe-left-positive-002",
        (0.72, 0.58, 0.44, 0.31),
        "swipe_left",
    )
    _write_labeled_take(
        recordings_dir,
        labels_dir,
        "swipe-right-positive-001",
        (0.3, 0.45, 0.6, 0.7),
        "swipe_right",
    )
    _write_labeled_take(
        recordings_dir,
        labels_dir,
        "swipe-right-positive-002",
        (0.31, 0.44, 0.58, 0.72),
        "swipe_right",
    )
    _write_labeled_take(
        recordings_dir,
        labels_dir,
        "normal-desk-motion-negative-001",
        (0.5, 0.5, 0.5, 0.5),
        None,
    )
    _write_labeled_take(
        recordings_dir,
        labels_dir,
        "normal-desk-motion-negative-002",
        (0.52, 0.5, 0.51, 0.5),
        None,
    )
    model_path = tmp_path / "holdout-model.json"
    summary_path = tmp_path / "holdout-summary.json"

    holdout = evaluate_dtw_holdout(
        recordings_dir=recordings_dir,
        labels_dir=labels_dir,
        model_path=model_path,
        train_per_gesture=1,
        test_per_gesture=1,
        train_negatives=1,
        test_negatives=1,
        min_window_seconds=0.2,
        max_window_seconds=0.5,
        window_step_seconds=0.1,
    )
    save_holdout_json(holdout, summary_path)

    assert model_path.exists()
    assert summary_path.exists()
    assert len(holdout.train_recordings) == 3
    assert len(holdout.test_recordings) == 3
    assert len(holdout.diagnostics) == 3
    assert {Path(item.recording).stem for item in holdout.train_recordings} == {
        "normal-desk-motion-negative-001",
        "swipe-left-positive-001",
        "swipe-right-positive-001",
    }
    assert "intended=2" in format_holdout_evaluation(holdout)
    assert holdout.to_dict()["summary"]["intended_events"] == 2
    assert "swipe_left" in holdout.to_dict()["diagnostics"][0]["best_by_gesture"]


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


def _write_labeled_take(
    recordings_dir: Path,
    labels_dir: Path,
    stem: str,
    xs: tuple[float, ...],
    gesture: str | None,
) -> None:
    recording = _write_motion_recording(recordings_dir / f"{stem}.jsonl", xs)
    label_file = init_label_file(recording)
    if gesture is not None:
        start = label_file.session.start_timestamp or 0.0
        end = label_file.session.end_timestamp or start
        label_file = add_event_label(
            label_file,
            gesture=gesture,
            start_time=start,
            end_time=end,
        )
    save_label_file(label_file, labels_dir / f"{stem}.labels.json")


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
