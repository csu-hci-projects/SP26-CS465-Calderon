"""Shared tracker construction for CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from airdesk.capture.opencv import CameraSettings
from airdesk.tracking.interfaces import HandTrackerBackend
from airdesk.tracking.mediapipe import (
    DEFAULT_HAND_LANDMARKER_DELEGATE,
    DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
    DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    DEFAULT_HAND_LANDMARKER_MODEL,
    MediaPipeHandTrackerBackend,
)
from airdesk.tracking.replay import ReplayHandTrackerBackend


def _make_tracker(
    *,
    backend: str,
    device: str,
    max_frames: int | None,
    show: bool,
    camera_settings: CameraSettings | None = None,
    model_path: Path = DEFAULT_HAND_LANDMARKER_MODEL,
    auto_download_model: bool = True,
    max_num_hands: int = DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
    min_detection_confidence: float = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    min_presence_confidence: float = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    min_tracking_confidence: float = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    delegate: str = DEFAULT_HAND_LANDMARKER_DELEGATE,
    preview_mirror: bool = True,
    preview_gestures: bool = True,
    preview_extended_threshold: float = 0.08,
    preview_pinch_threshold: float = 0.06,
) -> HandTrackerBackend:
    """Build the requested tracker backend from common CLI options."""
    if backend == "mediapipe":
        return MediaPipeHandTrackerBackend(
            device=device,
            model_path=model_path,
            auto_download_model=auto_download_model,
            camera_settings=camera_settings or CameraSettings(),
            max_frames=max_frames,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            delegate=delegate,
            show=show,
            preview_mirror=preview_mirror,
            preview_gestures=preview_gestures,
            preview_extended_threshold=preview_extended_threshold,
            preview_pinch_threshold=preview_pinch_threshold,
        )
    if backend == "replay":
        return ReplayHandTrackerBackend(Path(device))
    raise typer.BadParameter(f"unsupported tracking backend: {backend}")
