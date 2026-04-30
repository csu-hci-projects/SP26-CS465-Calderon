"""Capture backend interfaces."""

from airdesk.capture.interfaces import CaptureBackend
from airdesk.capture.opencv import (
    CameraProbeResult,
    CameraSettings,
    OpenCVCaptureBackend,
    camera_modes,
    probe_camera,
)

__all__ = [
    "CameraProbeResult",
    "CameraSettings",
    "CaptureBackend",
    "OpenCVCaptureBackend",
    "camera_modes",
    "probe_camera",
]
