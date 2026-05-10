from __future__ import annotations

from airdesk.capture.opencv import (
    CameraProbeResult,
    CameraSettings,
    format_probe_result,
    parse_camera_device,
)


def test_parse_camera_device_accepts_indices_and_paths() -> None:
    assert parse_camera_device("0") == 0
    assert parse_camera_device(2) == 2
    assert parse_camera_device("/dev/video0") == 0


def test_format_probe_result_includes_core_camera_facts() -> None:
    result = CameraProbeResult(
        device="/dev/video0",
        backend="opencv",
        opened=True,
        requested=CameraSettings(width=640, height=480, fps=30, fourcc="MJPG"),
        frame_read=True,
        width=1280,
        height=720,
        fps=30.0,
    )

    assert (
        format_probe_result(result) == "/dev/video0: opened frame=yes requested=640x480@30.00/MJPG "
        "size=1280x720 fps=30.00 backend=opencv"
    )


def test_camera_settings_can_request_small_capture_buffer() -> None:
    assert CameraSettings(buffer_size=1).to_dict()["buffer_size"] == 1
