from __future__ import annotations

from pathlib import Path

from airdesk.labels import (
    GestureLabelFile,
    GesturePhaseLabel,
    add_event_label,
    add_phase_label,
    init_label_file,
    load_label_file,
    save_label_file,
    validate_label_file,
)


def test_init_label_file_extracts_recording_metadata() -> None:
    label_file = init_label_file(
        Path("tests/fixtures/replay-one-frame.jsonl"),
        participant_id="caden",
    )

    assert label_file.schema_version == 1
    assert label_file.session.recording_path == "tests/fixtures/replay-one-frame.jsonl"
    assert label_file.session.participant_id == "caden"
    assert label_file.session.frame_count == 1
    assert label_file.phase_labels[0].phase == "background"
    assert validate_label_file(label_file).ok is True


def test_label_file_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    label_file = init_label_file(Path("tests/fixtures/replay-one-frame.jsonl"))

    save_label_file(label_file, path)

    assert load_label_file(path) == label_file


def test_validate_label_file_rejects_unknown_phase() -> None:
    label_file = init_label_file(Path("tests/fixtures/replay-one-frame.jsonl"))
    invalid = GestureLabelFile(
        schema_version=label_file.schema_version,
        created_at=label_file.created_at,
        session=label_file.session,
        event_labels=label_file.event_labels,
        phase_labels=(
            GesturePhaseLabel(
                label_id="phase-bad",
                phase="mystery",
                start_time=label_file.session.start_timestamp or 0.0,
                end_time=label_file.session.end_timestamp or 0.0,
            ),
        ),
    )

    result = validate_label_file(invalid)

    assert result.ok is False
    assert "unsupported phase=mystery" in result.errors[0]


def test_add_phase_and_event_labels_generate_ids() -> None:
    label_file = init_label_file(Path("tests/fixtures/replay-one-frame.jsonl"))

    updated = add_phase_label(
        label_file,
        phase="stroke_left",
        start_time=label_file.session.start_timestamp or 0.0,
        end_time=label_file.session.end_timestamp or 0.0,
        gesture="swipe_left",
    )
    updated = add_event_label(
        updated,
        gesture="swipe_left",
        start_time=label_file.session.start_timestamp or 0.0,
        end_time=label_file.session.end_timestamp or 0.0,
    )

    assert updated.phase_labels[-1].label_id == "phase-002"
    assert updated.event_labels[-1].label_id == "event-001"
    assert validate_label_file(updated).ok is True
