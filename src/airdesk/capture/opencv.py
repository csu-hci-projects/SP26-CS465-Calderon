"""OpenCV camera capture and probing."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from airdesk.state.types import CapturedFrame, FrameMetadata, utc_timestamp


@dataclass(frozen=True)
class CameraSettings:
    """Requested camera capture settings."""

    width: int | None = None
    height: int | None = None
    fps: float | None = None
    fourcc: str | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "fourcc": self.fourcc,
        }


@dataclass(frozen=True)
class CameraProbeResult:
    """Result of attempting to open and read one frame from a camera."""

    device: str
    backend: str
    opened: bool
    requested: CameraSettings = field(default_factory=CameraSettings)
    frame_read: bool = False
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    message: str = ""


def parse_camera_device(device: str | int) -> str | int:
    """Return an OpenCV-compatible camera identifier."""
    if isinstance(device, int):
        return device
    if device.isdigit():
        return int(device)
    video_match = re.fullmatch(r"/dev/video(\d+)", device)
    if video_match:
        return int(video_match.group(1))
    return device


def format_probe_result(result: CameraProbeResult) -> str:
    """Render probe output as a compact CLI-friendly line."""
    status = "opened" if result.opened else "failed"
    frame = "frame=yes" if result.frame_read else "frame=no"
    size = f"{result.width}x{result.height}" if result.width and result.height else "unknown"
    fps = f"{result.fps:.2f}" if result.fps and result.fps > 0 else "unknown"
    requested_size = (
        f"{result.requested.width}x{result.requested.height}"
        if result.requested.width and result.requested.height
        else "default"
    )
    requested_fps = (
        f"{result.requested.fps:.2f}"
        if result.requested.fps and result.requested.fps > 0
        else "default"
    )
    requested_format = result.requested.fourcc or "default"
    parts = [
        f"{result.device}: {status}",
        frame,
        f"requested={requested_size}@{requested_fps}/{requested_format}",
        f"size={size}",
        f"fps={fps}",
        f"backend={result.backend}",
    ]
    if result.message:
        parts.append(result.message)
    return " ".join(parts)


def probe_camera(
    device: str | int = "/dev/video0",
    *,
    settings: CameraSettings | None = None,
) -> CameraProbeResult:
    """Attempt to open a camera and read one frame."""
    settings = settings or CameraSettings()
    cv2 = _import_cv2()
    device_arg = parse_camera_device(device)
    capture = cv2.VideoCapture(device_arg)
    try:
        _apply_capture_settings(capture, cv2, settings)
        opened = bool(capture.isOpened())
        if not opened:
            return CameraProbeResult(
                device=str(device),
                backend="opencv",
                opened=False,
                requested=settings,
                message="VideoCapture did not open",
            )

        frame_read, image = capture.read()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or None
        if frame_read and image is not None:
            height, width = image.shape[:2]
        return CameraProbeResult(
            device=str(device),
            backend="opencv",
            opened=True,
            requested=settings,
            frame_read=bool(frame_read),
            width=width,
            height=height,
            fps=fps,
            message="" if frame_read else "opened but could not read a frame",
        )
    finally:
        capture.release()


@dataclass
class OpenCVCaptureBackend:
    """OpenCV-backed frame source."""

    device: str | int = "/dev/video0"
    settings: CameraSettings = field(default_factory=CameraSettings)
    max_frames: int | None = None
    source_id: str | None = None
    color_format: str = "bgr"
    name: str = "opencv"

    def __post_init__(self) -> None:
        self._cv2: Any | None = None
        self._capture: Any | None = None

    def start(self) -> None:
        self._cv2 = _import_cv2()
        self._capture = self._cv2.VideoCapture(parse_camera_device(self.device))
        _apply_capture_settings(self._capture, self._cv2, self.settings)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open camera device: {self.device}")

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def frames(self) -> Iterator[CapturedFrame]:
        if self._capture is None:
            self.start()

        assert self._capture is not None
        sequence = 0
        while self.max_frames is None or sequence < self.max_frames:
            ok, image = self._capture.read()
            if not ok or image is None:
                break
            sequence += 1
            height, width = image.shape[:2]
            metadata = FrameMetadata(
                timestamp=utc_timestamp(),
                source_id=self.source_id or str(self.device),
                width=width,
                height=height,
                sequence=sequence,
                color_format=self.color_format,
            )
            yield CapturedFrame(metadata=metadata, image=image)


def camera_exists(device: str) -> bool:
    """Best-effort non-opening existence check for camera list commands."""
    if device.isdigit():
        return True
    return Path(device).exists()


def camera_modes(device: str = "/dev/video0") -> str:
    """Return camera modes reported by v4l2-ctl when available."""
    if shutil.which("v4l2-ctl") is None:
        return "v4l2-ctl is not installed; install v4l-utils to list camera modes outside OpenCV."
    completed = subprocess.run(
        ["v4l2-ctl", "--device", device, "--list-formats-ext"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return completed.stderr.strip() or f"v4l2-ctl exited {completed.returncode}"
    return completed.stdout.strip()


def _apply_capture_settings(capture: Any, cv2: Any, settings: CameraSettings) -> None:
    if settings.fourcc:
        fourcc = settings.fourcc.upper()
        if len(fourcc) != 4:
            raise RuntimeError(f"camera FOURCC must be four characters, got {settings.fourcc!r}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if settings.width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    if settings.height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    if settings.fps is not None:
        capture.set(cv2.CAP_PROP_FPS, settings.fps)


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run `uv sync --dev --extra live` before using camera "
            "capture commands."
        ) from exc
    return cv2
