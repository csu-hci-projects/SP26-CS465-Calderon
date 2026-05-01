from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from airdesk.cli import app
from airdesk.recording.jsonl import iter_recording


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
