from __future__ import annotations

from typer.testing import CliRunner

from airdesk.cli import app


def test_cli_help_works() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "AirDesk spatial input prototype CLI" in result.stdout
