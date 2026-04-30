from __future__ import annotations

from pathlib import Path

import pytest

from airdesk.profiles.loader import ProfileValidationError, load_profile


def test_loads_sample_study_safe_profile() -> None:
    profile = load_profile(Path("configs/profiles/study-safe.toml"))

    assert profile.profile_id == "study-safe"
    assert profile.dry_run_default is True
    assert profile.activation_gesture == "open_palm"
    assert {binding.gesture for binding in profile.bindings} >= {"open_palm", "fist", "pinch"}


def test_loads_sample_window_manager_profile() -> None:
    profile = load_profile(Path("configs/profiles/window-manager.toml"))

    workspace_bindings = [binding for binding in profile.bindings if binding.command == "workspace"]
    assert profile.profile_id == "window-manager"
    assert workspace_bindings[0].parameters == {"args": ["r-1"]}


def test_rejects_bad_profile_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('id = "bad"\n', encoding="utf-8")

    with pytest.raises(ProfileValidationError):
        load_profile(path)
