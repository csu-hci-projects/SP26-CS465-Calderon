from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from airdesk.analysis import evaluate_tcn_manifest
from airdesk.features import FrameFeatureRow
from airdesk.labels import (
    GestureEventLabel,
    GestureLabelFile,
    GesturePhaseLabel,
    SessionMetadata,
    save_label_file,
)
from airdesk.ml import (
    TCN_STREAM_INVARIANT_FEATURE_COLUMNS,
    CausalTcnLivePredictor,
    CausalTcnTrainingConfig,
    build_feature_diagnostics_report,
    build_tcn_dataset_manifest,
    feature_window_matrix,
    load_feature_rows_csv,
    load_tcn_dataset_manifest,
    predict_causal_tcn_manifest,
    prepare_tcn_training_arrays,
    save_tcn_dataset_manifest,
    train_causal_tcn,
)


def test_load_feature_rows_csv_round_trips_export_shape(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    _write_features(
        features,
        [
            _row(timestamp=1.0, frame_index=0, event=""),
            _row(timestamp=1.1, frame_index=1, event="swipe_left"),
        ],
    )

    rows = load_feature_rows_csv(features)

    assert len(rows) == 2
    assert rows[0].timestamp == 1.0
    assert rows[1].event == "swipe_left"
    assert rows[1].tracking_present == 1


def test_build_tcn_manifest_assigns_background_and_swipe_windows(tmp_path: Path) -> None:
    features = tmp_path / "swipe-left-positive-001.csv"
    rows = [
        _row(timestamp=1.0, frame_index=0, event=""),
        _row(timestamp=1.1, frame_index=1, event="swipe_left"),
        _row(timestamp=1.2, frame_index=2, event="swipe_left"),
        _row(timestamp=1.3, frame_index=3, event=""),
        _row(timestamp=1.4, frame_index=4, event=""),
        _row(timestamp=1.5, frame_index=5, event=""),
    ]
    _write_features(features, rows)

    manifest = build_tcn_dataset_manifest(
        [features],
        window_seconds=0.2,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
    )

    assert len(manifest.sources) == 1
    assert manifest.sources[0].target_frame_counts == {
        "background": 4,
        "swipe_left": 2,
        "swipe_right": 0,
    }
    assert [window.target for window in manifest.windows] == ["swipe_left", "background"]
    assert manifest.windows[0].start_row == 0
    assert manifest.windows[0].end_row == 3
    assert manifest.windows[0].target_index == 1


def test_build_tcn_manifest_can_assign_labels_from_matching_label_file(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    labels_dir = tmp_path / "labels"
    features_dir.mkdir()
    labels_dir.mkdir()
    features = features_dir / "swipe-right-positive-001.csv"
    _write_features(
        features,
        [
            _row(timestamp=10.0, frame_index=0, event=""),
            _row(timestamp=10.1, frame_index=1, event=""),
            _row(timestamp=10.2, frame_index=2, event=""),
            _row(timestamp=10.3, frame_index=3, event=""),
        ],
    )
    save_label_file(
        GestureLabelFile(
            schema_version=1,
            created_at=1.0,
            session=SessionMetadata(
                recording_path="recording.jsonl",
                start_timestamp=10.0,
                end_timestamp=10.3,
            ),
            event_labels=(
                GestureEventLabel(
                    label_id="event-001",
                    label_type="gesture",
                    gesture="swipe_right",
                    start_time=10.1,
                    end_time=10.2,
                ),
            ),
        ),
        labels_dir / "swipe-right-positive-001.labels.json",
    )

    manifest = build_tcn_dataset_manifest(
        [features],
        labels_dir=labels_dir,
        window_seconds=0.3,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
    )

    assert len(manifest.windows) == 1
    assert manifest.windows[0].target == "swipe_right"
    assert manifest.sources[0].label_path == str(
        labels_dir / "swipe-right-positive-001.labels.json"
    )


def test_build_tcn_manifest_supports_stream_invariant_phase_targets(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    labels_dir = tmp_path / "labels"
    features_dir.mkdir()
    labels_dir.mkdir()
    features = features_dir / "chained.csv"
    _write_features(
        features,
        [
            _row(timestamp=1.0, frame_index=0, event="", phase="background"),
            _row(timestamp=1.1, frame_index=1, event="", phase="stroke_left"),
            _row(timestamp=1.2, frame_index=2, event="", phase="stroke_left"),
            _row(timestamp=1.3, frame_index=3, event="", phase="recovery"),
        ],
    )
    save_label_file(
        GestureLabelFile(
            schema_version=1,
            created_at=1.0,
            session=SessionMetadata(
                recording_path="recording.jsonl",
                start_timestamp=1.0,
                end_timestamp=1.3,
            ),
            phase_labels=(
                GesturePhaseLabel(
                    label_id="phase-001",
                    phase="stroke_left",
                    start_time=1.1,
                    end_time=1.2,
                ),
                GesturePhaseLabel(
                    label_id="phase-002",
                    phase="recovery",
                    start_time=1.3,
                    end_time=1.3,
                ),
            ),
        ),
        labels_dir / "chained.labels.json",
    )

    manifest = build_tcn_dataset_manifest(
        [features],
        labels_dir=labels_dir,
        window_seconds=0.3,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
        feature_preset="stream-invariant",
        target_mode="phase",
    )

    assert manifest.targets == ("background", "stroke_left", "stroke_right", "recovery")
    assert manifest.feature_columns == TCN_STREAM_INVARIANT_FEATURE_COLUMNS
    assert "palm_x" not in manifest.feature_columns
    assert "palm_y" not in manifest.feature_columns
    assert "palm_z" not in manifest.feature_columns
    assert manifest.feature_preset == "stream-invariant"
    assert manifest.target_mode == "phase"
    assert manifest.windows[0].target == "stroke_left"


def test_save_tcn_dataset_manifest_writes_summary(tmp_path: Path) -> None:
    features = tmp_path / "normal-desk-motion-negative-001.csv"
    _write_features(
        features,
        [
            _row(timestamp=1.0, frame_index=0, event=""),
            _row(timestamp=1.1, frame_index=1, event=""),
            _row(timestamp=1.2, frame_index=2, event=""),
        ],
    )
    manifest = build_tcn_dataset_manifest(
        [features],
        window_seconds=0.2,
        stride_seconds=0.2,
        min_rows=2,
    )
    output = tmp_path / "manifest.json"

    save_tcn_dataset_manifest(manifest, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["targets"] == ["background", "swipe_left", "swipe_right"]
    assert payload["feature_columns"][0] == "dt"
    assert payload["summary"]["window_counts"]["background"] == 1


def test_manifest_loads_and_extracts_window_matrix(tmp_path: Path) -> None:
    features = tmp_path / "swipe-left-positive-001.csv"
    _write_features(
        features,
        [
            _row(timestamp=1.0, frame_index=0, event=""),
            _row(timestamp=1.1, frame_index=1, event="swipe_left"),
            _row(timestamp=1.2, frame_index=2, event="swipe_left"),
        ],
    )
    manifest = build_tcn_dataset_manifest(
        [features],
        window_seconds=0.2,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
    )
    output = tmp_path / "manifest.json"
    save_tcn_dataset_manifest(manifest, output)

    loaded = load_tcn_dataset_manifest(output)
    matrix = feature_window_matrix(loaded.windows[0], feature_columns=("dt", "confidence"))
    arrays = prepare_tcn_training_arrays(loaded)

    assert loaded == manifest
    assert matrix == [[0.0, 1.0], [0.1, 1.0], [0.1, 1.0]]
    assert arrays.labels == (1,)
    assert arrays.lengths == (3,)
    assert len(arrays.feature_mean) == len(loaded.feature_columns)


def test_feature_diagnostics_report_compares_holdout_motion(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    labels_dir = tmp_path / "labels"
    features_dir.mkdir()
    labels_dir.mkdir()
    _write_labeled_motion_features(
        features_dir,
        labels_dir,
        "swipe-left-positive-001",
        gesture="swipe_left",
        xs=(0.10, 0.20, 0.30, 0.40, 0.50),
    )
    _write_labeled_motion_features(
        features_dir,
        labels_dir,
        "swipe-left-positive-002",
        gesture="swipe_left",
        xs=(0.10, 0.14, 0.18, 0.22, 0.26),
    )
    _write_labeled_motion_features(
        features_dir,
        labels_dir,
        "swipe-right-positive-001",
        gesture="swipe_right",
        xs=(0.50, 0.40, 0.30, 0.20, 0.10),
    )
    _write_labeled_motion_features(
        features_dir,
        labels_dir,
        "swipe-right-positive-002",
        gesture="swipe_right",
        xs=(0.50, 0.45, 0.40, 0.35, 0.30),
    )
    _write_labeled_motion_features(
        features_dir,
        labels_dir,
        "normal-desk-motion-negative-001",
        gesture=None,
        xs=(0.30, 0.30, 0.30, 0.30, 0.30),
    )
    _write_labeled_motion_features(
        features_dir,
        labels_dir,
        "normal-desk-motion-negative-002",
        gesture=None,
        xs=(0.31, 0.31, 0.31, 0.31, 0.31),
    )

    report = build_feature_diagnostics_report(
        features_dir=features_dir,
        labels_dir=labels_dir,
        train_per_gesture=1,
        test_per_gesture=1,
        train_negatives=1,
        test_negatives=1,
    )

    train_left = report.aggregates["train:swipe_left"]["metrics"]
    test_left = report.aggregates["test:swipe_left"]["metrics"]
    assert len(report.files) == 6
    assert train_left["palm_dx"]["mean"] == pytest.approx(0.2)
    assert test_left["palm_dx"]["mean"] == pytest.approx(0.08)
    assert test_left["event_first_row_offset_seconds"]["mean"] == pytest.approx(0.0)
    assert report.aggregates["test:normal-desk-motion-negative"]["count"] == 1


def test_train_causal_tcn_smoke_when_torch_is_installed(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("optional PyTorch dependency is not installed")
    features = tmp_path / "features.csv"
    _write_features(
        features,
        [
            _row(timestamp=1.0, frame_index=0, event=""),
            _row(timestamp=1.1, frame_index=1, event=""),
            _row(timestamp=1.2, frame_index=2, event="swipe_left"),
            _row(timestamp=1.3, frame_index=3, event="swipe_left"),
            _row(timestamp=1.4, frame_index=4, event=""),
            _row(timestamp=1.5, frame_index=5, event=""),
        ],
    )
    manifest = build_tcn_dataset_manifest(
        [features],
        window_seconds=0.2,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
    )
    manifest_path = tmp_path / "manifest.json"
    model_path = tmp_path / "model.pt"
    save_tcn_dataset_manifest(manifest, manifest_path)

    result = train_causal_tcn(
        manifest_path=manifest_path,
        out_path=model_path,
        config=CausalTcnTrainingConfig(epochs=1, batch_size=2, validation_fraction=0.0, seed=1),
    )

    assert model_path.exists()
    assert result.samples == len(manifest.windows)
    assert result.validation_accuracy is None


def test_live_tcn_predictor_classifies_in_memory_rows_when_torch_is_installed(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("optional PyTorch dependency is not installed")
    features = tmp_path / "features.csv"
    rows = [
        _row(timestamp=1.0, frame_index=0, event=""),
        _row(timestamp=1.1, frame_index=1, event="swipe_left"),
        _row(timestamp=1.2, frame_index=2, event="swipe_left"),
        _row(timestamp=1.3, frame_index=3, event=""),
    ]
    _write_features(features, rows)
    manifest = build_tcn_dataset_manifest(
        [features],
        window_seconds=0.3,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
    )
    manifest_path = tmp_path / "manifest.json"
    model_path = tmp_path / "model.pt"
    save_tcn_dataset_manifest(manifest, manifest_path)
    train_causal_tcn(
        manifest_path=manifest_path,
        out_path=model_path,
        config=CausalTcnTrainingConfig(epochs=1, batch_size=1, validation_fraction=0.0, seed=1),
    )

    predictor = CausalTcnLivePredictor.load(model_path)
    prediction = predictor.predict_rows(rows)

    assert predictor.window_seconds == manifest.window_seconds
    assert set(prediction.probabilities) == set(manifest.targets)
    assert 0.0 <= prediction.confidence <= 1.0


def test_tcn_prediction_and_evaluation_smoke_when_torch_is_installed(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("optional PyTorch dependency is not installed")
    features_dir = tmp_path / "features"
    labels_dir = tmp_path / "labels"
    features_dir.mkdir()
    labels_dir.mkdir()
    features = features_dir / "swipe-left-positive-001.csv"
    _write_features(
        features,
        [
            _row(timestamp=1.0, frame_index=0, event=""),
            _row(timestamp=1.1, frame_index=1, event=""),
            _row(timestamp=1.2, frame_index=2, event="swipe_left"),
            _row(timestamp=1.3, frame_index=3, event="swipe_left"),
            _row(timestamp=1.4, frame_index=4, event=""),
            _row(timestamp=1.5, frame_index=5, event=""),
        ],
    )
    save_label_file(
        GestureLabelFile(
            schema_version=1,
            created_at=1.0,
            session=SessionMetadata(
                recording_path="recording.jsonl",
                start_timestamp=1.0,
                end_timestamp=1.5,
            ),
            event_labels=(
                GestureEventLabel(
                    label_id="event-001",
                    label_type="gesture",
                    gesture="swipe_left",
                    start_time=1.2,
                    end_time=1.3,
                ),
            ),
        ),
        labels_dir / "swipe-left-positive-001.labels.json",
    )
    manifest = build_tcn_dataset_manifest(
        [features],
        labels_dir=labels_dir,
        window_seconds=0.2,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
    )
    manifest_path = tmp_path / "manifest.json"
    model_path = tmp_path / "model.pt"
    save_tcn_dataset_manifest(manifest, manifest_path)
    train_causal_tcn(
        manifest_path=manifest_path,
        out_path=model_path,
        config=CausalTcnTrainingConfig(epochs=1, batch_size=2, validation_fraction=0.0, seed=1),
    )

    predictions = predict_causal_tcn_manifest(
        model_path=model_path,
        manifest_path=manifest_path,
        confidence_threshold=0.0,
        include_background=True,
    )
    evaluations = evaluate_tcn_manifest(
        manifest_path=manifest_path,
        model_path=model_path,
        confidence_threshold=0.0,
    )

    assert len(predictions) == len(manifest.windows)
    assert len(evaluations) == 1
    assert evaluations[0].recognizer == "tcn"
    assert evaluations[0].intended_events == 1


def _write_features(path: Path, rows: list[FrameFeatureRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _write_labeled_motion_features(
    features_dir: Path,
    labels_dir: Path,
    stem: str,
    *,
    gesture: str | None,
    xs: tuple[float, ...],
) -> None:
    events = ("", gesture or "", gesture or "", gesture or "", "")
    _write_features(
        features_dir / f"{stem}.csv",
        [
            _row(
                timestamp=10.0 + index * 0.1,
                frame_index=index,
                event=events[index],
                palm_x=x,
            )
            for index, x in enumerate(xs)
        ],
    )
    event_labels = ()
    if gesture is not None:
        event_labels = (
            GestureEventLabel(
                label_id="event-001",
                label_type="gesture",
                gesture=gesture,
                start_time=10.1,
                end_time=10.3,
            ),
        )
    save_label_file(
        GestureLabelFile(
            schema_version=1,
            created_at=1.0,
            session=SessionMetadata(
                recording_path=f"{stem}.jsonl",
                start_timestamp=10.0,
                end_timestamp=10.4,
            ),
            event_labels=event_labels,
        ),
        labels_dir / f"{stem}.labels.json",
    )


def _row(
    *,
    timestamp: float,
    frame_index: int,
    event: str,
    phase: str = "",
    palm_x: float = 0.5,
) -> FrameFeatureRow:
    return FrameFeatureRow(
        frame_index=frame_index,
        timestamp=timestamp,
        dt=0.1 if frame_index else 0.0,
        tracking_present=1,
        hand_count=1,
        hand_id="hand-0",
        confidence=1.0,
        palm_x=palm_x,
        palm_y=0.5,
        palm_z=0.0,
        palm_vx=1.0 if frame_index else 0.0,
        palm_vy=0.0,
        palm_speed=1.0 if frame_index else 0.0,
        palm_ax=0.0,
        palm_ay=0.0,
        palm_window_dx=0.1 * frame_index,
        palm_window_dx_per_hand_scale=0.5 * frame_index,
        palm_window_peak_abs_vx=1.0 if frame_index else 0.0,
        palm_window_direction_consistency=1.0 if frame_index else 0.0,
        index_rel_x=0.0,
        index_rel_y=0.0,
        index_rel_vx=0.0,
        index_rel_vy=0.0,
        pinch_distance=0.1,
        pinch_velocity=0.0,
        hand_scale=0.2,
        extended_fingers=4,
        folded_fingers=0,
        phase=phase,
        event=event,
    )
