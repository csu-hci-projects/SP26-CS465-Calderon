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
    result = CliRunner().invoke(app, ["tune", "--help"])

    assert result.exit_code == 0
    assert "--extended-threshold" in result.stdout
    assert "--pinch-threshold" in result.stdout


def test_view_help_describes_live_preview() -> None:
    result = CliRunner().invoke(app, ["view", "--help"])

    assert result.exit_code == 0
    assert "live webcam preview" in result.stdout


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
    assert events[-1].payload["frames"] == 1
