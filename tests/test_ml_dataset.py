from __future__ import annotations

import csv
import json
from pathlib import Path

from airdesk.features import FrameFeatureRow
from airdesk.labels import (
    GestureEventLabel,
    GestureLabelFile,
    SessionMetadata,
    save_label_file,
)
from airdesk.ml import (
    build_tcn_dataset_manifest,
    load_feature_rows_csv,
    save_tcn_dataset_manifest,
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


def _write_features(path: Path, rows: list[FrameFeatureRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _row(*, timestamp: float, frame_index: int, event: str) -> FrameFeatureRow:
    return FrameFeatureRow(
        frame_index=frame_index,
        timestamp=timestamp,
        dt=0.1 if frame_index else 0.0,
        tracking_present=1,
        hand_count=1,
        hand_id="hand-0",
        confidence=1.0,
        palm_x=0.5,
        palm_y=0.5,
        palm_z=0.0,
        palm_vx=0.0,
        palm_vy=0.0,
        palm_speed=0.0,
        palm_ax=0.0,
        palm_ay=0.0,
        index_rel_x=0.0,
        index_rel_y=0.0,
        index_rel_vx=0.0,
        index_rel_vy=0.0,
        pinch_distance=0.1,
        pinch_velocity=0.0,
        hand_scale=0.2,
        extended_fingers=4,
        folded_fingers=0,
        phase="",
        event=event,
    )
