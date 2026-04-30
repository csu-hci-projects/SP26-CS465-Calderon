from __future__ import annotations

from typer.testing import CliRunner

from airdesk.cli import app


def test_cli_help_works() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "AirDesk spatial input prototype CLI" in result.stdout


def test_replay_reports_frame_event_and_recognizer_counts() -> None:
    result = CliRunner().invoke(app, ["replay", "tests/fixtures/replay-one-frame.jsonl"])

    assert result.exit_code == 0
    assert "frames=1" in result.stdout
    assert "events=0" in result.stdout
    assert "open_palm=0" in result.stdout
