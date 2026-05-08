from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from airdesk.analysis import (
    diagnose_candidate_events,
    evaluate_tcn_manifest,
    evaluate_tcn_v2_manifest,
)
from airdesk.analysis.evaluation import _dedupe_tcn_v2_predictions
from airdesk.features import FrameFeatureRow
from airdesk.gestures.decoder import EventDecoderConfig
from airdesk.labels import (
    GestureEventLabel,
    GestureLabelFile,
    GesturePhaseLabel,
    SessionMetadata,
    save_label_file,
)
from airdesk.ml import (
    NO_HAND_STREAM_ID,
    TCN_STREAM_INVARIANT_FEATURE_COLUMNS,
    TCN_V2_EVIDENCE_TARGETS,
    CausalTcnEvidencePrediction,
    CausalTcnLivePredictor,
    CausalTcnTrainingConfig,
    build_feature_diagnostics_report,
    build_tcn_dataset_manifest,
    feature_window_frame_targets,
    feature_window_matrix,
    load_feature_rows_csv,
    load_tcn_dataset_manifest,
    predict_causal_tcn_manifest,
    predict_causal_tcn_v2_manifest,
    prepare_tcn_training_arrays,
    prepare_tcn_v2_training_arrays,
    refine_motion_aligned_label_file,
    save_tcn_dataset_manifest,
    train_causal_tcn,
    train_causal_tcn_v2,
)
from airdesk.state.types import GestureCandidate


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


def test_build_tcn_manifest_keeps_two_hand_windows_separate(tmp_path: Path) -> None:
    features = tmp_path / "two-hand.csv"
    _write_features(
        features,
        [
            _row(timestamp=1.0, frame_index=0, event="", hand_id="hand-0", palm_x=0.5),
            _row(timestamp=1.0, frame_index=0, event="", hand_id="hand-1", palm_x=0.7),
            _row(timestamp=1.1, frame_index=1, event="swipe_left", hand_id="hand-0", palm_x=0.5),
            _row(timestamp=1.1, frame_index=1, event="swipe_left", hand_id="hand-1", palm_x=0.6),
            _row(timestamp=1.2, frame_index=2, event="swipe_left", hand_id="hand-0", palm_x=0.5),
            _row(timestamp=1.2, frame_index=2, event="swipe_left", hand_id="hand-1", palm_x=0.4),
        ],
    )

    manifest = build_tcn_dataset_manifest(
        [features],
        window_seconds=0.2,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
    )

    assert {window.hand_id for window in manifest.windows} == {"hand-0", "hand-1"}
    assert all(window.row_count == 3 for window in manifest.windows)
    for window in manifest.windows:
        matrix = feature_window_matrix(window, feature_columns=("palm_x",))
        assert len(matrix) == 3


def test_build_tcn_manifest_keeps_no_hand_windows_separate(tmp_path: Path) -> None:
    features = tmp_path / "tracking-drop.csv"
    _write_features(
        features,
        [
            _row(
                timestamp=1.0,
                frame_index=0,
                event="",
                hand_id="",
                tracking_present=0,
                hand_count=0,
            ),
            _row(
                timestamp=1.1,
                frame_index=1,
                event="",
                hand_id="hand-0",
                tracking_present=1,
                hand_count=1,
            ),
            _row(
                timestamp=1.2,
                frame_index=2,
                event="",
                hand_id="",
                tracking_present=0,
                hand_count=0,
            ),
            _row(
                timestamp=1.3,
                frame_index=3,
                event="",
                hand_id="",
                tracking_present=0,
                hand_count=0,
            ),
        ],
    )

    manifest = build_tcn_dataset_manifest(
        [features],
        window_seconds=0.2,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
    )

    no_hand_window = next(
        window for window in manifest.windows if window.hand_id == NO_HAND_STREAM_ID
    )
    matrix = feature_window_matrix(no_hand_window, feature_columns=manifest.feature_columns)
    assert len(matrix) == no_hand_window.row_count
    assert no_hand_window.target_frame_counts["background"] == no_hand_window.row_count


def test_build_tcn_manifest_motion_gates_stationary_second_hand(tmp_path: Path) -> None:
    features = tmp_path / "two-hand.csv"
    _write_features(
        features,
        [
            _row(timestamp=1.0, frame_index=0, event="", hand_id="hand-0", palm_x=0.5),
            _row(timestamp=1.0, frame_index=0, event="", hand_id="hand-1", palm_x=0.7),
            _row(
                timestamp=1.1,
                frame_index=1,
                event="swipe_left",
                hand_id="hand-0",
                palm_x=0.6,
                palm_window_dx_per_hand_scale=0.6,
            ),
            _row(
                timestamp=1.1,
                frame_index=1,
                event="swipe_left",
                hand_id="hand-1",
                palm_x=0.7,
                palm_window_dx_per_hand_scale=0.0,
            ),
            _row(
                timestamp=1.2,
                frame_index=2,
                event="swipe_left",
                hand_id="hand-0",
                palm_x=0.7,
                palm_window_dx_per_hand_scale=0.8,
            ),
            _row(
                timestamp=1.2,
                frame_index=2,
                event="swipe_left",
                hand_id="hand-1",
                palm_x=0.7,
                palm_window_dx_per_hand_scale=0.0,
            ),
        ],
    )

    manifest = build_tcn_dataset_manifest(
        [features],
        window_seconds=0.2,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
        target_assignment="motion-gated",
    )

    targets_by_hand = {window.hand_id: window.target for window in manifest.windows}
    assert targets_by_hand == {"hand-0": "swipe_left", "hand-1": "background"}
    assert manifest.target_assignment == "motion-gated"


def test_diagnose_candidate_events_lists_matches_misses_false_and_repeats() -> None:
    labels = GestureLabelFile(
        schema_version=1,
        created_at=1.0,
        session=SessionMetadata(
            recording_path="recording.jsonl",
            start_timestamp=1.0,
            end_timestamp=3.0,
        ),
        event_labels=(
            GestureEventLabel(
                label_id="event-001",
                label_type="gesture",
                gesture="swipe_left",
                start_time=1.0,
                end_time=1.2,
            ),
            GestureEventLabel(
                label_id="event-002",
                label_type="gesture",
                gesture="swipe_right",
                start_time=2.0,
                end_time=2.2,
            ),
        ),
    )
    candidates = [
        GestureCandidate(
            name="swipe_left",
            confidence=0.8,
            timestamp=1.1,
            hand_id="hand-0",
            metadata={"window_start": 0.8, "window_end": 1.1},
        ),
        GestureCandidate(name="swipe_left", confidence=0.7, timestamp=1.15),
        GestureCandidate(name="swipe_right", confidence=0.9, timestamp=3.0),
    ]

    diagnostics = diagnose_candidate_events(
        labels=labels,
        candidates=candidates,
        match_tolerance_seconds=0.1,
    )

    assert len(diagnostics["matches"]) == 1
    assert len(diagnostics["repeated_fires"]) == 1
    assert len(diagnostics["missed_events"]) == 1
    assert len(diagnostics["false_activations"]) == 1
    missed = diagnostics["missed_events"][0]
    assert missed["event"]["gesture"] == "swipe_right"
    assert missed["nearest_same_gesture_candidate"]["seconds_from_event_end"] == pytest.approx(0.8)
    false_activation = diagnostics["false_activations"][0]
    assert false_activation["candidate"]["name"] == "swipe_right"
    nearest = false_activation["nearest_same_gesture_event"]
    assert nearest["seconds_from_event_end"] == pytest.approx(0.8)


def test_refine_motion_aligned_label_file_shifts_chart_event_to_motion_peak(
    tmp_path: Path,
) -> None:
    features = tmp_path / "chart.csv"
    labels = tmp_path / "chart.labels.json"
    _write_features(
        features,
        [
            _row(timestamp=9.8, frame_index=0, event="", palm_window_dx_per_hand_scale=0.0),
            _row(
                timestamp=10.0,
                frame_index=1,
                event="swipe_right",
                palm_window_dx_per_hand_scale=0.1,
            ),
            _row(
                timestamp=10.3,
                frame_index=2,
                event="swipe_right",
                palm_window_dx_per_hand_scale=0.2,
            ),
            _row(
                timestamp=10.8,
                frame_index=3,
                event="swipe_right",
                hand_id="hand-1",
                palm_window_dx_per_hand_scale=1.4,
            ),
            _row(timestamp=11.2, frame_index=4, event="", palm_window_dx_per_hand_scale=0.0),
        ],
    )
    save_label_file(
        GestureLabelFile(
            schema_version=1,
            created_at=1.0,
            session=SessionMetadata(
                recording_path="chart.jsonl",
                start_timestamp=9.8,
                end_timestamp=11.2,
            ),
            event_labels=(
                GestureEventLabel(
                    label_id="event-001",
                    label_type="gesture",
                    gesture="swipe_right",
                    start_time=10.0,
                    end_time=10.4,
                ),
            ),
            phase_labels=(
                GesturePhaseLabel(
                    label_id="phase-001",
                    phase="background",
                    start_time=9.8,
                    end_time=11.2,
                ),
                GesturePhaseLabel(
                    label_id="phase-002",
                    phase="stroke_right",
                    start_time=10.0,
                    end_time=10.4,
                    gesture="swipe_right",
                ),
            ),
        ),
        labels,
    )

    result = refine_motion_aligned_label_file(
        feature_path=features,
        label_path=labels,
        search_padding_seconds=1.0,
        min_motion_score=0.35,
    )

    refined = result.label_file.event_labels[0]
    assert refined.start_time == pytest.approx(10.4)
    assert refined.end_time == pytest.approx(10.8)
    assert refined.commit_time == pytest.approx(10.8)
    assert "hand_id=hand-1" in refined.notes
    assert result.refined_events[0].changed is True
    assert result.refined_events[0].hand_id == "hand-1"
    assert [phase.phase for phase in result.label_file.phase_labels] == [
        "background",
        "stroke_right",
        "recovery",
    ]


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
        window_seconds=0.2,
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
        window_seconds=0.2,
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


def test_build_tcn_manifest_phase_stroke_targets_treat_recovery_as_background(
    tmp_path: Path,
) -> None:
    features = tmp_path / "chained.csv"
    _write_features(
        features,
        [
            _row(timestamp=1.0, frame_index=0, event="", phase="stroke_left"),
            _row(timestamp=1.1, frame_index=1, event="", phase="stroke_left"),
            _row(timestamp=1.2, frame_index=2, event="", phase="recovery"),
            _row(timestamp=1.3, frame_index=3, event="", phase="recovery"),
            _row(timestamp=1.4, frame_index=4, event="", phase="recovery"),
        ],
    )

    manifest = build_tcn_dataset_manifest(
        [features],
        window_seconds=0.2,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
        target_mode="phase-stroke",
    )

    assert manifest.targets == ("background", "stroke_left", "stroke_right")
    assert manifest.target_mode == "phase-stroke"
    assert [window.target for window in manifest.windows] == ["stroke_left", "background"]


def test_build_tcn_manifest_v2_evidence_targets_framewise_heads(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    labels_dir = tmp_path / "labels"
    features_dir.mkdir()
    labels_dir.mkdir()
    features = features_dir / "continuous.csv"
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
            event_labels=(
                GestureEventLabel(
                    label_id="event-001",
                    label_type="gesture",
                    gesture="swipe_left",
                    start_time=1.1,
                    end_time=1.2,
                ),
            ),
            phase_labels=(
                GesturePhaseLabel(
                    label_id="phase-001",
                    phase="stroke_left",
                    start_time=1.1,
                    end_time=1.2,
                    gesture="swipe_left",
                ),
                GesturePhaseLabel(
                    label_id="phase-002",
                    phase="recovery",
                    start_time=1.3,
                    end_time=1.3,
                    gesture="swipe_left",
                ),
            ),
        ),
        labels_dir / "continuous.labels.json",
    )

    manifest = build_tcn_dataset_manifest(
        [features],
        labels_dir=labels_dir,
        window_seconds=0.2,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
        target_mode="v2-evidence",
        feature_preset="stream-invariant",
    )

    assert manifest.target_mode == "v2-evidence"
    assert manifest.evidence_targets == TCN_V2_EVIDENCE_TARGETS
    assert manifest.targets == (
        "background",
        "intentional_motion",
        "stroke_left",
        "stroke_right",
        "start",
        "end",
    )
    assert manifest.sources[0].evidence_frame_counts == {
        "intentional_motion": 3,
        "stroke_left": 2,
        "stroke_right": 0,
        "start": 1,
        "end": 1,
    }
    assert manifest.windows[0].target == "stroke_left"
    frame_targets = feature_window_frame_targets(
        manifest.windows[0],
        evidence_targets=manifest.evidence_targets,
    )
    assert frame_targets[0] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert frame_targets[1] == [1.0, 1.0, 0.0, 1.0, 0.0]
    assert frame_targets[2] == [1.0, 1.0, 0.0, 0.0, 1.0]
    arrays = prepare_tcn_v2_training_arrays(manifest)
    assert arrays.lengths == (3,)
    assert arrays.frame_targets[0][1][1] == 1.0
    assert manifest.to_dict()["summary"]["evidence_frame_counts"] == {
        "intentional_motion": 3,
        "stroke_left": 2,
        "stroke_right": 0,
        "start": 1,
        "end": 1,
    }


def test_build_tcn_manifest_v2_motion_gates_resting_hand_evidence(tmp_path: Path) -> None:
    features = tmp_path / "two-hand.csv"
    _write_features(
        features,
        [
            _row(timestamp=1.0, frame_index=0, event="", hand_id="hand-0"),
            _row(timestamp=1.0, frame_index=0, event="", hand_id="hand-1"),
            _row(
                timestamp=1.1,
                frame_index=1,
                event="swipe_right",
                hand_id="hand-0",
                palm_window_dx_per_hand_scale=0.7,
            ),
            _row(
                timestamp=1.1,
                frame_index=1,
                event="swipe_right",
                hand_id="hand-1",
                palm_window_dx_per_hand_scale=0.0,
            ),
            _row(
                timestamp=1.2,
                frame_index=2,
                event="swipe_right",
                hand_id="hand-0",
                palm_window_dx_per_hand_scale=0.8,
            ),
            _row(
                timestamp=1.2,
                frame_index=2,
                event="swipe_right",
                hand_id="hand-1",
                palm_window_dx_per_hand_scale=0.0,
            ),
        ],
    )

    manifest = build_tcn_dataset_manifest(
        [features],
        window_seconds=0.2,
        stride_seconds=0.2,
        min_rows=2,
        min_gesture_fraction=0.5,
        target_mode="v2-evidence",
        target_assignment="motion-gated",
    )

    windows_by_hand = {window.hand_id: window for window in manifest.windows}
    moving_targets = feature_window_frame_targets(
        windows_by_hand["hand-0"],
        evidence_targets=manifest.evidence_targets,
    )
    resting_targets = feature_window_frame_targets(
        windows_by_hand["hand-1"],
        evidence_targets=manifest.evidence_targets,
    )
    assert moving_targets[-1][2] == 1.0
    assert all(frame == [0.0, 0.0, 0.0, 0.0, 0.0] for frame in resting_targets)


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


def test_tcn_v2_training_prediction_and_evaluation_smoke_when_torch_is_installed(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("optional PyTorch dependency is not installed")
    features_dir = tmp_path / "features"
    labels_dir = tmp_path / "labels"
    features_dir.mkdir()
    labels_dir.mkdir()
    features = features_dir / "continuous.csv"
    _write_features(
        features,
        [
            _row(timestamp=1.0, frame_index=0, event="", phase="background"),
            _row(timestamp=1.1, frame_index=1, event="", phase="stroke_right"),
            _row(timestamp=1.2, frame_index=2, event="", phase="stroke_right"),
            _row(timestamp=1.3, frame_index=3, event="", phase="background"),
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
            event_labels=(
                GestureEventLabel(
                    label_id="event-001",
                    label_type="gesture",
                    gesture="swipe_right",
                    start_time=1.1,
                    end_time=1.2,
                ),
            ),
            phase_labels=(
                GesturePhaseLabel(
                    label_id="phase-001",
                    phase="stroke_right",
                    start_time=1.1,
                    end_time=1.2,
                    gesture="swipe_right",
                ),
            ),
        ),
        labels_dir / "continuous.labels.json",
    )
    manifest = build_tcn_dataset_manifest(
        [features],
        labels_dir=labels_dir,
        window_seconds=0.2,
        stride_seconds=0.1,
        min_rows=2,
        min_gesture_fraction=0.25,
        target_mode="v2-evidence",
        feature_preset="stream-invariant",
    )
    manifest_path = tmp_path / "manifest.json"
    model_path = tmp_path / "model.pt"
    save_tcn_dataset_manifest(manifest, manifest_path)

    result = train_causal_tcn_v2(
        manifest_path=manifest_path,
        out_path=model_path,
        config=CausalTcnTrainingConfig(epochs=1, batch_size=2, validation_fraction=0.0, seed=1),
    )
    predictions = predict_causal_tcn_v2_manifest(
        model_path=model_path,
        manifest_path=manifest_path,
    )
    evaluations = evaluate_tcn_v2_manifest(
        manifest_path=manifest_path,
        model_path=model_path,
        event_decoder_config=EventDecoderConfig(
            activation_threshold=0.0,
            release_threshold=0.0,
            min_peak_confidence=0.0,
            cooldown_seconds=0.0,
            min_event_separation_seconds=0.0,
        ),
    )

    assert result.targets == TCN_V2_EVIDENCE_TARGETS
    assert len(predictions) == len(manifest.windows)
    assert set(predictions[0].evidence) == set(TCN_V2_EVIDENCE_TARGETS)
    assert evaluations[0].recognizer == "tcn_v2_event_decoder"
    assert evaluations[0].intended_events == 1


def test_tcn_v2_prediction_dedupe_keeps_fullest_causal_context() -> None:
    short_context = CausalTcnEvidencePrediction(
        sample_id="window-002",
        feature_path="features.csv",
        label_path="labels.json",
        hand_id="hand-0",
        timestamp=1.0,
        window_start=0.9,
        window_end=1.2,
        evidence={"stroke_left": 0.2},
    )
    long_context = CausalTcnEvidencePrediction(
        sample_id="window-001",
        feature_path="features.csv",
        label_path="labels.json",
        hand_id="hand-0",
        timestamp=1.0,
        window_start=0.5,
        window_end=1.1,
        evidence={"stroke_left": 0.8},
    )

    deduped = _dedupe_tcn_v2_predictions([short_context, long_context])

    assert deduped == [long_context]


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
    hand_id: str = "hand-0",
    tracking_present: int = 1,
    hand_count: int = 1,
    palm_window_dx_per_hand_scale: float | None = None,
) -> FrameFeatureRow:
    return FrameFeatureRow(
        frame_index=frame_index,
        timestamp=timestamp,
        dt=0.1 if frame_index else 0.0,
        tracking_present=tracking_present,
        hand_count=hand_count,
        hand_id=hand_id,
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
        palm_window_dx_per_hand_scale=(
            0.5 * frame_index
            if palm_window_dx_per_hand_scale is None
            else palm_window_dx_per_hand_scale
        ),
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
