"""MediaPipe hand tracking backend."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from airdesk.capture.opencv import CameraSettings, OpenCVCaptureBackend
from airdesk.state.types import HandLandmarks, Landmark, NormalizedHand, TrackingFrame

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
    "float16/1/hand_landmarker.task"
)
DEFAULT_HAND_LANDMARKER_MODEL = Path("data/models/hand_landmarker.task")
PREVIEW_WINDOW_NAME = "AirDesk live view"

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass
class MediaPipeHandTrackerBackend:
    """Live hand tracker backed by MediaPipe Hands."""

    device: str | int = "/dev/video0"
    model_path: Path = DEFAULT_HAND_LANDMARKER_MODEL
    auto_download_model: bool = True
    camera_settings: CameraSettings = field(default_factory=CameraSettings)
    max_frames: int | None = None
    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    show: bool = False
    name: str = "mediapipe"

    def __post_init__(self) -> None:
        self._cv2: Any | None = None
        self._mp: Any | None = None
        self._landmarker: Any | None = None
        self._capture: OpenCVCaptureBackend | None = None

    def start(self) -> None:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe/OpenCV live dependencies are not installed. Run "
                "`uv sync --dev --extra live` before using the mediapipe tracker."
            ) from exc

        model_path = ensure_hand_landmarker_model(
            self.model_path, download=self.auto_download_model
        )
        self._cv2 = cv2
        self._mp = mp
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._capture = OpenCVCaptureBackend(
            device=self.device,
            settings=self.camera_settings,
            max_frames=self.max_frames,
        )
        self._capture.start()

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self.show and self._cv2 is not None:
            self._cv2.destroyAllWindows()

    def frames(self) -> Iterator[TrackingFrame]:
        if self._landmarker is None or self._capture is None:
            self.start()

        assert self._landmarker is not None
        assert self._capture is not None
        assert self._cv2 is not None
        assert self._mp is not None

        try:
            for captured in self._capture.frames():
                if captured.image is None:
                    yield TrackingFrame(
                        timestamp=captured.metadata.timestamp,
                        source_id=captured.metadata.source_id,
                        frame=captured.metadata,
                    )
                    continue

                image = captured.image
                rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
                media_image = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB,
                    data=rgb,
                )
                results = self._landmarker.detect_for_video(
                    media_image,
                    int(captured.metadata.timestamp * 1000),
                )
                hands = normalized_hands_from_mediapipe_results(results)

                if self.show and not self._draw_debug_image(image, hands):
                    break

                yield TrackingFrame(
                    timestamp=captured.metadata.timestamp,
                    source_id=captured.metadata.source_id,
                    frame=captured.metadata,
                    hands=hands,
                )
        finally:
            self.stop()

    def _draw_debug_image(self, image: Any, hands: tuple[NormalizedHand, ...]) -> bool:
        assert self._cv2 is not None
        height, width = image.shape[:2]
        self._draw_header(image, hands)
        for hand in hands:
            self._draw_hand_overlay(image, hand, width=width, height=height)
        self._cv2.imshow(PREVIEW_WINDOW_NAME, image)
        key = self._cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))

    def _draw_header(self, image: Any, hands: tuple[NormalizedHand, ...]) -> None:
        assert self._cv2 is not None
        text = f"AirDesk live view | hands={len(hands)} | q/esc quits"
        self._cv2.rectangle(image, (0, 0), (image.shape[1], 34), (20, 20, 20), -1)
        self._cv2.putText(
            image,
            text,
            (10, 23),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2,
            self._cv2.LINE_AA,
        )

    def _draw_hand_overlay(
        self,
        image: Any,
        hand: NormalizedHand,
        *,
        width: int,
        height: int,
    ) -> None:
        assert self._cv2 is not None
        landmarks = hand.landmarks.landmarks
        color = (0, 255, 0)
        muted = (0, 150, 255)
        for start, end in HAND_CONNECTIONS:
            if start >= len(landmarks) or end >= len(landmarks):
                continue
            self._cv2.line(
                image,
                pixel_point(landmarks[start], width=width, height=height),
                pixel_point(landmarks[end], width=width, height=height),
                muted,
                2,
            )
        for landmark in landmarks:
            self._cv2.circle(image, pixel_point(landmark, width=width, height=height), 3, color, -1)

        x_min, y_min, x_max, y_max = bbox_pixels(hand.bbox, width=width, height=height)
        self._cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        label = hand_label(hand)
        self._cv2.putText(
            image,
            label,
            (x_min, max(55, y_min - 8)),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            self._cv2.LINE_AA,
        )


def normalized_hands_from_mediapipe_results(results: Any) -> tuple[NormalizedHand, ...]:
    """Convert MediaPipe result objects into AirDesk normalized hand state."""
    hand_landmarks = (
        getattr(results, "hand_landmarks", None)
        or getattr(results, "multi_hand_landmarks", None)
        or []
    )
    handedness = (
        getattr(results, "handedness", None) or getattr(results, "multi_handedness", None) or []
    )
    normalized: list[NormalizedHand] = []

    for index, landmarks in enumerate(hand_landmarks):
        label, confidence = _handedness_for_index(handedness, index)
        points = tuple(
            Landmark(x=float(point.x), y=float(point.y), z=float(getattr(point, "z", 0.0)))
            for point in _landmark_points(landmarks)
        )
        if not points:
            continue
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        palm_points = [points[i] for i in (0, 5, 9, 13, 17) if i < len(points)]
        palm_center = (
            sum(point.x for point in palm_points) / len(palm_points),
            sum(point.y for point in palm_points) / len(palm_points),
            sum(point.z for point in palm_points) / len(palm_points),
        )
        hand_state = NormalizedHand(
            hand_id=f"hand-{index}",
            landmarks=HandLandmarks(
                landmarks=points,
                handedness=label,
                confidence=confidence,
            ),
            palm_center=palm_center,
            bbox=(min(xs), min(ys), max(xs), max(ys)),
            handedness=label,
            confidence=confidence,
        )
        normalized.append(hand_state)

    return tuple(normalized)


def ensure_hand_landmarker_model(
    path: Path = DEFAULT_HAND_LANDMARKER_MODEL,
    *,
    download: bool,
) -> Path:
    """Return a local Hand Landmarker model path, downloading it when allowed."""
    if path.exists():
        return path
    if not download:
        raise RuntimeError(
            f"Hand Landmarker model is missing at {path}. Download it from "
            f"{HAND_LANDMARKER_MODEL_URL} or run with model auto-download enabled."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(HAND_LANDMARKER_MODEL_URL, path)
    return path


def pixel_point(landmark: Landmark, *, width: int, height: int) -> tuple[int, int]:
    """Convert a normalized landmark to image pixel coordinates."""
    x = min(width - 1, max(0, round(landmark.x * width)))
    y = min(height - 1, max(0, round(landmark.y * height)))
    return x, y


def bbox_pixels(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert a normalized hand bounding box to pixel coordinates."""
    x_min = min(width - 1, max(0, round(bbox[0] * width)))
    y_min = min(height - 1, max(0, round(bbox[1] * height)))
    x_max = min(width - 1, max(0, round(bbox[2] * width)))
    y_max = min(height - 1, max(0, round(bbox[3] * height)))
    return x_min, y_min, x_max, y_max


def hand_label(hand: NormalizedHand) -> str:
    """Return a compact label for preview overlays."""
    confidence = f"{hand.confidence:.2f}" if hand.confidence is not None else "?"
    handedness = hand.handedness or "hand"
    return f"{handedness} {confidence}"


def _landmark_points(landmarks: Any) -> list[Any]:
    return list(getattr(landmarks, "landmark", landmarks))


def _handedness_for_index(handedness: list[Any], index: int) -> tuple[str | None, float | None]:
    if index >= len(handedness):
        return None, None
    classifications = getattr(handedness[index], "classification", handedness[index])
    if not classifications:
        return None, None
    classification = classifications[0]
    label = (
        getattr(classification, "label", None)
        or getattr(classification, "category_name", None)
        or getattr(classification, "display_name", None)
    )
    score = getattr(classification, "score", None)
    return label, score
