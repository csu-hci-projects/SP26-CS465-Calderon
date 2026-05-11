from __future__ import annotations

import json
from pathlib import Path

from airdesk.labels import validate_label_file
from airdesk.public_datasets.ipn import (
    IPN_AIRDESK_ATOMIC_MAP,
    IPN_EVIDENCE_HEADS,
    IPN_EVIDENCE_TARGETS,
    _ipn_label_file_for_recording,
    class_file_for_ipn_annotations,
    find_ipn_video_path,
    load_ipn_class_index,
    load_ipn_split_segments,
    split_file_for_ipn_annotations,
    write_ipn_mapping_csv,
)
from airdesk.recording.jsonl import JsonlRecordingWriter
from airdesk.state.types import FrameMetadata, TrackingFrame


def test_load_ipn_split_segments_maps_class_labels(tmp_path: Path) -> None:
    class_path = tmp_path / "classIndAll.txt"
    class_path.write_text("1 D0X\n8 G05\n9 G06\n", encoding="utf-8")
    split_path = tmp_path / "trainlistall.txt"
    split_path.write_text(
        "./frames/1CM1_4_R_#229 8 503 544\n"
        "./frames/1CM1_4_R_#229 9 2351 2391\n",
        encoding="utf-8",
    )

    classes = load_ipn_class_index(class_path)
    segments = load_ipn_split_segments(split_path, class_index=classes)

    assert [segment.video_id for segment in segments] == ["1CM1_4_R_#229"] * 2
    assert [segment.label for segment in segments] == ["G05", "G06"]
    assert all(segment.maps_to_airdesk for segment in segments)
    assert IPN_AIRDESK_ATOMIC_MAP["G05"] == ("swipe_left", "stroke_left")


def test_load_ipn_official_drive_annotations(tmp_path: Path) -> None:
    class_path = tmp_path / "classIdx.txt"
    class_path.write_text("id,label\n1,D0X\n8,G05\n9,G06\n", encoding="utf-8")
    split_path = tmp_path / "Annot_TrainList.txt"
    split_path.write_text(
        "1CM1_4_R_#229,D0X,1,1,17,17\n"
        "1CM1_4_R_#229,G05,8,503,544,42\n"
        "1CM1_4_R_#229,G06,9,2351,2391,41\n",
        encoding="utf-8",
    )

    classes = load_ipn_class_index(class_path)
    segments = load_ipn_split_segments(split_path, class_index=classes)

    assert class_file_for_ipn_annotations(tmp_path) == class_path
    assert split_file_for_ipn_annotations(tmp_path, "train") == split_path
    assert [segment.label for segment in segments] == ["D0X", "G05", "G06"]
    assert [segment.class_index for segment in segments] == [1, 8, 9]
    assert [segment.start_frame for segment in segments] == [1, 503, 2351]
    assert [segment.end_frame for segment in segments] == [17, 544, 2391]


def test_ipn_label_file_maps_only_atomic_left_right_swipes(tmp_path: Path) -> None:
    recording_path = tmp_path / "ipn.jsonl"
    with JsonlRecordingWriter(recording_path) as writer:
        for sequence in range(1, 61):
            timestamp = (sequence - 1) / 30.0
            metadata = FrameMetadata(
                timestamp=timestamp,
                source_id="ipn-test",
                width=640,
                height=480,
                sequence=sequence,
            )
            writer.write_tracking_frame(
                TrackingFrame(
                    timestamp=timestamp,
                    source_id="ipn-test",
                    frame=metadata,
                    hands=(),
                )
            )
    split_path = tmp_path / "trainlistall.txt"
    split_path.write_text(
        "./frames/ipn-test 8 10 20\n"
        "./frames/ipn-test 4 21 25\n"
        "./frames/ipn-test 9 30 40\n",
        encoding="utf-8",
    )
    segments = load_ipn_split_segments(
        split_path,
        class_index={8: "G05", 4: "G01", 9: "G06"},
    )

    label_file = _ipn_label_file_for_recording(
        recording_path=recording_path,
        segments=segments,
        fps=30.0,
        observed_frame_count=60,
    )

    assert validate_label_file(label_file).ok
    assert [event.gesture for event in label_file.event_labels] == [
        "swipe_left",
        "swipe_right",
    ]
    assert [phase.phase for phase in label_file.phase_labels if phase.phase != "background"] == [
        "stroke_left",
        "stroke_right",
    ]


def test_ipn_label_file_can_map_all_gesture_classes_as_evidence(tmp_path: Path) -> None:
    recording_path = tmp_path / "ipn.jsonl"
    with JsonlRecordingWriter(recording_path) as writer:
        for sequence in range(1, 61):
            timestamp = (sequence - 1) / 30.0
            metadata = FrameMetadata(
                timestamp=timestamp,
                source_id="ipn-test",
                width=640,
                height=480,
                sequence=sequence,
            )
            writer.write_tracking_frame(
                TrackingFrame(
                    timestamp=timestamp,
                    source_id="ipn-test",
                    frame=metadata,
                    hands=(),
                )
            )
    split_path = tmp_path / "trainlistall.txt"
    split_path.write_text(
        "./frames/ipn-test 1 1 5\n"
        "./frames/ipn-test 4 10 20\n"
        "./frames/ipn-test 8 21 30\n"
        "./frames/ipn-test 14 31 40\n",
        encoding="utf-8",
    )
    segments = load_ipn_split_segments(
        split_path,
        class_index={1: "D0X", 4: "G01", 8: "G05", 14: "G11"},
    )

    label_file = _ipn_label_file_for_recording(
        recording_path=recording_path,
        segments=segments,
        fps=30.0,
        observed_frame_count=60,
        label_mode="ipn-all",
    )

    assert validate_label_file(label_file).ok
    assert [event.gesture for event in label_file.event_labels] == [
        "ipn_g01",
        "ipn_g05",
        "ipn_g11",
    ]
    assert [phase.phase for phase in label_file.phase_labels] == ["background"]
    assert IPN_EVIDENCE_HEADS["G05"] == "ipn_g05"
    assert "ipn_g11" in IPN_EVIDENCE_TARGETS


def test_find_ipn_video_path_supports_nested_download_parts(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos" / "part-1"
    video_dir.mkdir(parents=True)
    expected = video_dir / "1CM1_4_R_#229.mp4"
    expected.write_bytes(b"not really a video")

    assert find_ipn_video_path(tmp_path / "videos", "1CM1_4_R_#229") == expected


def test_write_ipn_mapping_csv_records_reviewable_mapping(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.csv"

    write_ipn_mapping_csv(mapping)

    text = mapping.read_text(encoding="utf-8")
    assert "G05,Throw left,swipe_left,stroke_left" in text
    assert "G01,Click with one finger,," in text


def test_generated_ipn_labels_are_stable_json(tmp_path: Path) -> None:
    recording_path = tmp_path / "ipn.jsonl"
    with JsonlRecordingWriter(recording_path) as writer:
        metadata = FrameMetadata(
            timestamp=0.0,
            source_id="ipn-json",
            width=640,
            height=480,
            sequence=1,
        )
        writer.write_tracking_frame(
            TrackingFrame(timestamp=0.0, source_id="ipn-json", frame=metadata, hands=())
        )
    label_file = _ipn_label_file_for_recording(
        recording_path=recording_path,
        segments=[],
        fps=30.0,
        observed_frame_count=1,
    )

    payload = label_file.to_dict()

    assert json.loads(json.dumps(payload))["session"]["participant_id"] == "ipn-public"
