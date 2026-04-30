"""Capture backend interfaces."""

from airdesk.capture.interfaces import CaptureBackend
from airdesk.capture.opencv import CameraProbeResult, OpenCVCaptureBackend, probe_camera

__all__ = ["CameraProbeResult", "CaptureBackend", "OpenCVCaptureBackend", "probe_camera"]
