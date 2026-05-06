from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from airdesk.features import FeatureRowStream, export_features_csv, extract_feature_rows
from airdesk.labels import GesturePhaseLabel, init_label_file
from airdesk.recording.jsonl import JsonlRecordingWriter
from airdesk.state.types import (
    FrameMetadata,
    HandLandmarks,
    Landmark,
    NormalizedHand,
    TrackingFrame,
)


def frame_at(timestamp: float, sequence: int, hand: NormalizedHand) -> TrackingFrame:
    metadata = FrameMetadata(
        timestamp=timestamp,
        source_id="feature-test",
        width=640,
        height=480,
        sequence=sequence,
    )
    return TrackingFrame(
        timestamp=timestamp,
        source_id="feature-test",
        frame=metadata,
        hands=(hand,),
    )


def test_extract_feature_rows_include_motion_and_pose_features(
    make_hand: Callable[[str], NormalizedHand],
) -> None:
    first = frame_at(1.0, 1, _move_hand(make_hand("open_palm"), x=0.4))
    second = frame_at(1.1, 2, _move_hand(make_hand("open_palm"), x=0.5))

    rows = extract_feature_rows([first, second])

    assert len(rows) == 2
    assert rows[0].tracking_present == 1
    assert rows[1].dt == pytest.approx(0.1)
    assert rows[1].palm_vx > 0
    assert rows[1].palm_window_dx == pytest.approx(0.1)
    assert rows[1].palm_window_dx_per_hand_scale == pytest.approx(0.2)
    assert rows[1].palm_window_peak_abs_vx == pytest.approx(1.0)
    assert rows[1].palm_window_direction_consistency == pytest.approx(1.0)
    assert rows[1].extended_fingers == 4
    assert rows[1].phase == ""


def test_extract_feature_rows_attach_phase_labels(
    make_hand: Callable[[str], NormalizedHand],
) -> None:
    frame = frame_at(1.0, 1, make_hand("open_palm"))
    label_file = init_label_file(Path("tests/fixtures/replay-one-frame.jsonl"))
    label_file = type(label_file)(
        schema_version=label_file.schema_version,
        created_at=label_file.created_at,
        session=label_file.session,
        event_labels=(),
        phase_labels=(
            GesturePhaseLabel(
                label_id="phase-test",
                phase="stroke_left",
                start_time=0.9,
                end_time=1.1,
            ),
        ),
    )

    rows = extract_feature_rows([frame], labels=label_file)

    assert rows[0].phase == "stroke_left"


def test_feature_row_stream_matches_batch_extraction(
    make_hand: Callable[[str], NormalizedHand],
) -> None:
    frames = [
        frame_at(1.0, 1, _move_hand(make_hand("open_palm"), x=0.4)),
        frame_at(1.1, 2, _move_hand(make_hand("open_palm"), x=0.5)),
        frame_at(1.2, 3, _move_hand(make_hand("open_palm"), x=0.6)),
    ]
    stream = FeatureRowStream()

    streamed = [stream.append(frame) for frame in frames]

    assert streamed == extract_feature_rows(frames)


def test_export_features_csv_writes_rows(
    tmp_path: Path,
    make_hand: Callable[[str], NormalizedHand],
) -> None:
    recording = tmp_path / "recording.jsonl"
    output = tmp_path / "features.csv"
    with JsonlRecordingWriter(recording) as writer:
        writer.write_tracking_frame(frame_at(1.0, 1, make_hand("pinch")))

    rows = export_features_csv(recording, output)

    assert len(rows) == 1
    text = output.read_text(encoding="utf-8")
    assert "frame_index,timestamp" in text
    assert "pinch_distance" in text


def _move_hand(hand: NormalizedHand, *, x: float) -> NormalizedHand:
    dx = x - hand.palm_center[0]
    landmarks = tuple(
        Landmark(
            landmark.x + dx,
            landmark.y,
            landmark.z,
            visibility=landmark.visibility,
            presence=landmark.presence,
        )
        for landmark in hand.landmarks.landmarks
    )
    return NormalizedHand(
        hand_id=hand.hand_id,
        landmarks=HandLandmarks(
            landmarks=landmarks,
            handedness=hand.landmarks.handedness,
            confidence=hand.landmarks.confidence,
        ),
        palm_center=(x, hand.palm_center[1], hand.palm_center[2]),
        bbox=(hand.bbox[0] + dx, hand.bbox[1], hand.bbox[2] + dx, hand.bbox[3]),
        handedness=hand.handedness,
        confidence=hand.confidence,
    )
