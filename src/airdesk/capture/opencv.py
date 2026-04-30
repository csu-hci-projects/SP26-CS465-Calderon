"""OpenCV camera capture and probing."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from airdesk.state.types import CapturedFrame, FrameMetadata, utc_timestamp


@dataclass(frozen=True)
class CameraProbeResult:
    """Result of attempting to open and read one frame from a camera."""

    device: str
    backend: str
    opened: bool
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
    return device


def format_probe_result(result: CameraProbeResult) -> str:
    """Render probe output as a compact CLI-friendly line."""
    status = "opened" if result.opened else "failed"
    frame = "frame=yes" if result.frame_read else "frame=no"
    size = f"{result.width}x{result.height}" if result.width and result.height else "unknown"
    fps = f"{result.fps:.2f}" if result.fps and result.fps > 0 else "unknown"
    parts = [
        f"{result.device}: {status}",
        frame,
        f"size={size}",
        f"fps={fps}",
        f"backend={result.backend}",
    ]
    if result.message:
        parts.append(result.message)
    return " ".join(parts)


def probe_camera(device: str | int = "/dev/video0") -> CameraProbeResult:
    """Attempt to open a camera and read one frame."""
    cv2 = _import_cv2()
    device_arg = parse_camera_device(device)
    capture = cv2.VideoCapture(device_arg)
    try:
        opened = bool(capture.isOpened())
        if not opened:
            return CameraProbeResult(
                device=str(device),
                backend="opencv",
                opened=False,
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


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run `uv sync --dev --extra live` before using camera "
            "capture commands."
        ) from exc
    return cv2
