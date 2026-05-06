from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from airdesk.cli import _handle_collection_preview_key, app
from airdesk.recording.jsonl import JsonlRecordingWriter, iter_recording
from airdesk.state.types import (
    FrameMetadata,
    HandLandmarks,
    Landmark,
    NormalizedHand,
    TrackingFrame,
)


def test_cli_help_works() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "AirDesk spatial input prototype CLI" in result.stdout


def test_tune_help_exposes_threshold_options() -> None:
    result = CliRunner().invoke(app, ["tune", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "--extended-threshold" in result.stdout
    assert "--pinch-threshold" in result.stdout
    assert "--max-num-hands" in result.stdout
    assert "--min-tracking-confidence" in result.stdout


def test_view_help_describes_live_preview() -> None:
    result = CliRunner().invoke(app, ["view", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "live webcam preview" in result.stdout
    assert "--model-path" in result.stdout
    assert "--max-num-hands" in result.stdout


def test_benchmark_help_exposes_mediapipe_tuning_options() -> None:
    result = CliRunner().invoke(app, ["benchmark", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "--model-path" in result.stdout
    assert "--max-num-hands" in result.stdout
    assert "--min-detection-confidence" in result.stdout
    assert "--min-presence-confidence" in result.stdout
    assert "--min-tracking-confidence" in result.stdout


def test_benchmark_replay_reports_frame_counts() -> None:
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--backend",
            "replay",
            "--device",
            "tests/fixtures/replay-one-frame.jsonl",
        ],
    )

    assert result.exit_code == 0
    assert "frames=1" in result.stdout
    assert "hand_frames=0" in result.stdout
    assert "average_fps=unknown" in result.stdout


def test_run_help_exposes_live_tuning_options() -> None:
    result = CliRunner().invoke(app, ["run", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "--model-path" in result.stdout
    assert "--max-num-hands" in result.stdout
    assert "--min-tracking-confidence" in result.stdout
    assert "--events-out" in result.stdout
    assert "--pause-on-start" in result.stdout
    assert "--execute" in result.stdout
    assert "--allow-profile-execute" in result.stdout


def test_cursor_run_help_exposes_cursor_controls() -> None:
    result = CliRunner().invoke(app, ["cursor", "run", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "--execute" in result.stdout
    assert "--pinch-threshold" in result.stdout
    assert "--release-threshold" in result.stdout
    assert "--events-out" in result.stdout


def test_collect_help_describes_prompted_collection() -> None:
    result = CliRunner().invoke(app, ["collect", "--help"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "--label" in result.stdout
    assert "--reps" in result.stdout
    assert "--countdown" in result.stdout
    assert "--auto-keep" in result.stdout


def test_collect_replay_auto_keep_writes_prompted_takes(tmp_path: Path) -> None:
    output_dir = tmp_path / "collection"

    result = CliRunner().invoke(
        app,
        [
            "collect",
            "--backend",
            "replay",
            "--device",
            "tests/fixtures/replay-one-frame.jsonl",
            "--out-dir",
            str(output_dir),
            "--label",
            "swipe-left-positive",
            "--reps",
            "2",
            "--duration",
            "1",
            "--countdown",
            "0",
            "--no-show",
            "--auto-keep",
        ],
    )

    assert result.exit_code == 0
    assert "collection complete kept=2" in result.stdout
    first = output_dir / "swipe-left-positive-001.jsonl"
    second = output_dir / "swipe-left-positive-002.jsonl"
    assert first.exists()
    assert second.exists()
    first_records = iter_recording(first)
    events = [record.payload for record in first_records if record.kind == "event"]
    frames = [record.payload for record in first_records if record.kind == "tracking_frame"]
    assert [event.event_type for event in events] == [
        "collection_take_started",
        "collection_recording_started",
        "collection_take_finished",
    ]
    assert events[0].payload["label"] == "swipe-left-positive"
    assert events[0].payload["repetition"] == 1
    assert events[-1].payload["frames"] == 1
    assert len(frames) == 1


def test_collection_summary_reports_directory_totals(tmp_path: Path) -> None:
    output_dir = tmp_path / "collection"
    collect_result = CliRunner().invoke(
        app,
        [
            "collect",
            "--backend",
            "replay",
            "--device",
            "tests/fixtures/replay-one-frame.jsonl",
            "--out-dir",
            str(output_dir),
            "--label",
            "normal-desk-motion-negative",
            "--reps",
            "1",
            "--duration",
            "1",
            "--countdown",
            "0",
            "--no-show",
            "--auto-keep",
        ],
    )

    result = CliRunner().invoke(app, ["collection-summary", str(output_dir)])

    assert collect_result.exit_code == 0
    assert result.exit_code == 0
    assert "file=normal-desk-motion-negative-001.jsonl" in result.stdout
    assert "label=normal-desk-motion-negative" in result.stdout
    assert "totals | label=normal-desk-motion-negative files=1" in result.stdout


def test_label_init_and_validate_cli(tmp_path: Path) -> None:
    output = tmp_path / "replay-one-frame.labels.json"

    init_result = CliRunner().invoke(
        app,
        [
            "label",
            "init",
            "tests/fixtures/replay-one-frame.jsonl",
            "--out",
            str(output),
        ],
    )
    validate_result = CliRunner().invoke(app, ["label", "validate", str(output)])

    assert init_result.exit_code == 0
    assert "frames=1" in init_result.stdout
    assert validate_result.exit_code == 0
    assert "valid labels=" in validate_result.stdout


def test_label_add_phase_and_event_cli(tmp_path: Path) -> None:
    output = tmp_path / "replay-one-frame.labels.json"
    CliRunner().invoke(
        app,
        [
            "label",
            "init",
            "tests/fixtures/replay-one-frame.jsonl",
            "--out",
            str(output),
        ],
    )

    phase_result = CliRunner().invoke(
        app,
        [
            "label",
            "add-phase",
            str(output),
            "--phase",
            "stroke_left",
            "--start",
            "0",
            "--end",
            "0",
            "--gesture",
            "swipe_left",
        ],
    )
    event_result = CliRunner().invoke(
        app,
        [
            "label",
            "add-event",
            str(output),
            "--gesture",
            "swipe_left",
            "--start",
            "0",
            "--end",
            "0",
        ],
    )

    assert phase_result.exit_code == 0
    assert event_result.exit_code == 0
    assert "stroke_left" in output.read_text(encoding="utf-8")
    assert "swipe_left" in output.read_text(encoding="utf-8")


def test_label_suggest_cli_applies_motion_label(tmp_path: Path) -> None:
    recording = tmp_path / "swipe-right-positive-001.jsonl"
    output = tmp_path / "swipe-right-positive-001.labels.json"
    _write_cli_motion_recording(recording)

    result = CliRunner().invoke(
        app,
        [
            "label",
            "suggest",
            str(recording),
            "--gesture",
            "swipe_right",
            "--out",
            str(output),
            "--apply",
        ],
    )

    assert result.exit_code == 0
    assert "applied suggestion labels=" in result.stdout
    text = output.read_text(encoding="utf-8")
    assert "stroke_right" in text
    assert "swipe_right" in text


def test_features_export_cli_writes_csv(tmp_path: Path) -> None:
    output = tmp_path / "features.csv"

    result = CliRunner().invoke(
        app,
        [
            "features",
            "export",
            "tests/fixtures/replay-one-frame.jsonl",
            "--out",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "rows=1" in result.stdout
    assert "tracking_present" in output.read_text(encoding="utf-8")


def test_gesture_evaluate_cli_writes_json(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    output = tmp_path / "evaluation.json"
    CliRunner().invoke(
        app,
        [
            "label",
            "init",
            "tests/fixtures/replay-one-frame.jsonl",
            "--out",
            str(labels),
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "gesture",
            "evaluate",
            "--recording",
            "tests/fixtures/replay-one-frame.jsonl",
            "--labels",
            str(labels),
            "--out",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "recognizer=rule" in result.stdout
    assert "intended=0" in result.stdout
    assert output.exists()


def test_collection_preview_keys_drive_start_and_review_decisions() -> None:
    state: dict[str, str | None] = {"phase": "waiting", "decision": None}

    assert _handle_collection_preview_key(ord(" "), state) is True
    assert state == {"phase": "countdown", "decision": None}

    state = {"phase": "review", "decision": None}

    assert _handle_collection_preview_key(ord("r"), state) is True
    assert state == {"phase": "done", "decision": "redo"}


def test_run_writes_events_out_for_replay_backend(tmp_path: Path) -> None:
    output = tmp_path / "runtime-events.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--backend",
            "replay",
            "--recording",
            "tests/fixtures/replay-one-frame.jsonl",
            "--profile",
            "configs/profiles/study-safe.toml",
            "--events-out",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "frames=1" in result.stdout
    records = iter_recording(output)
    events = [record.payload for record in records if record.kind == "event"]
    assert [event.event_type for event in events] == ["session_start", "session_finish"]
    assert {event.session_id for event in events} == {events[0].session_id}
    assert events[0].payload["backend"] == "replay"
    assert events[0].payload["profile_id"] == "study-safe"
    assert events[0].payload["dry_run"] is True
    assert events[1].payload["frames"] == 1
    assert events[1].payload["events"] == 2
    assert events[1].payload["actions"] == 0
    assert events[1].payload["interrupted"] is False


def test_run_refuses_no_dry_run_without_execute() -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--backend",
            "replay",
            "--recording",
            "tests/fixtures/replay-one-frame.jsonl",
            "--no-dry-run",
        ],
    )

    assert result.exit_code == 1
    assert "Real actions require explicit --execute" in result.stderr


def test_run_execute_requires_profile_override_for_dry_run_profiles() -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--backend",
            "replay",
            "--recording",
            "tests/fixtures/replay-one-frame.jsonl",
            "--profile",
            "configs/profiles/window-manager.toml",
            "--execute",
        ],
    )

    assert result.exit_code == 1
    assert "Profile defaults to dry-run" in result.stderr


def test_run_execute_allows_replay_when_profile_override_is_explicit() -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--backend",
            "replay",
            "--recording",
            "tests/fixtures/replay-one-frame.jsonl",
            "--profile",
            "configs/profiles/window-manager.toml",
            "--execute",
            "--allow-profile-execute",
        ],
    )

    assert result.exit_code == 0
    assert "frames=1" in result.stdout


def test_replay_reports_frame_event_and_recognizer_counts() -> None:
    result = CliRunner().invoke(app, ["replay", "tests/fixtures/replay-one-frame.jsonl"])

    assert result.exit_code == 0
    assert "frames=1" in result.stdout
    assert "events=0" in result.stdout
    assert "open_palm=0" in result.stdout


def test_record_command_writes_metadata_events_with_label(tmp_path: Path) -> None:
    output = tmp_path / "recording.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "record",
            "--backend",
            "replay",
            "--device",
            "tests/fixtures/replay-one-frame.jsonl",
            "--label",
            "cli-test",
            "--out",
            str(output),
        ],
    )

    assert result.exit_code == 0
    records = iter_recording(output)
    events = [record.payload for record in records if record.kind == "event"]
    assert events[0].payload["label"] == "cli-test"
    assert events[0].payload["mediapipe"]["max_num_hands"] == 1
    assert events[-1].payload["frames"] == 1


def _write_cli_motion_recording(path: Path) -> None:
    with JsonlRecordingWriter(path) as writer:
        timestamp = 100.0
        for sequence, x in enumerate((0.30, 0.30, 0.34, 0.48, 0.60, 0.62)):
            frame = FrameMetadata(
                timestamp=timestamp,
                source_id="test",
                width=640,
                height=480,
                sequence=sequence,
            )
            writer.write_tracking_frame(
                TrackingFrame(
                    timestamp=timestamp,
                    source_id="test",
                    frame=frame,
                    hands=(_cli_hand_at(x),),
                )
            )
            timestamp += 0.2


def _cli_hand_at(palm_x: float) -> NormalizedHand:
    landmarks = [Landmark(palm_x, 0.5, 0.0) for _ in range(21)]
    return NormalizedHand(
        hand_id="hand-0",
        landmarks=HandLandmarks(tuple(landmarks), handedness="right", confidence=1.0),
        palm_center=(palm_x, 0.5, 0.0),
        bbox=(palm_x - 0.1, 0.4, palm_x + 0.1, 0.6),
        handedness="right",
        confidence=1.0,
    )
