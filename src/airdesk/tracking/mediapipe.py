"""MediaPipe hand tracking backend."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.request import urlretrieve

from airdesk.capture.opencv import CameraSettings, OpenCVCaptureBackend
from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.overlay.live_dashboard import LiveDashboardRenderer
from airdesk.state.types import (
    FrameMetadata,
    HandLandmarks,
    Landmark,
    NormalizedHand,
    TrackingFrame,
)

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
    "float16/1/hand_landmarker.task"
)
DEFAULT_HAND_LANDMARKER_MODEL = Path("data/models/hand_landmarker.task")
DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS = 1
DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE = 0.5
DEFAULT_HAND_LANDMARKER_DELEGATE = "cpu"
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


@dataclass(frozen=True)
class MediaPipeTimingSample:
    """Per-frame live tracking timing sample in milliseconds."""

    capture_read_ms: float | None
    color_convert_ms: float
    inference_ms: float
    normalize_ms: float
    preview_draw_ms: float
    total_ms: float


@dataclass
class MediaPipeHandTrackerBackend:
    """Live hand tracker backed by MediaPipe Hands."""

    device: str | int = "/dev/video0"
    model_path: Path = DEFAULT_HAND_LANDMARKER_MODEL
    auto_download_model: bool = True
    camera_settings: CameraSettings = field(default_factory=CameraSettings)
    max_frames: int | None = None
    max_num_hands: int = DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS
    min_detection_confidence: float = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE
    min_hand_presence_confidence: float = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE
    min_tracking_confidence: float = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE
    delegate: str = DEFAULT_HAND_LANDMARKER_DELEGATE
    show: bool = False
    preview_mirror: bool = True
    preview_layout: str = "camera"
    preview_canvas_width: int = 1180
    preview_canvas_height: int = 720
    preview_gestures: bool = True
    preview_extended_threshold: float = 0.08
    preview_pinch_threshold: float = 0.06
    preview_status_provider: Callable[[], str] | None = None
    preview_dashboard_provider: Callable[[], dict[str, Any] | None] | None = None
    preview_chart_provider: Callable[[], dict[str, Any] | None] | None = None
    preview_key_handler: Callable[[int], bool] | None = None
    name: str = "mediapipe"

    def __post_init__(self) -> None:
        self._cv2: Any | None = None
        self._mp: Any | None = None
        self._landmarker: Any | None = None
        self._capture: OpenCVCaptureBackend | None = None
        self.timing_samples: list[MediaPipeTimingSample] = []

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
        delegate = _base_options_delegate(BaseOptions, self.delegate)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=delegate),
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
        if self.show:
            self._cv2.namedWindow(PREVIEW_WINDOW_NAME, self._cv2.WINDOW_NORMAL)
            if self.preview_layout == "dashboard":
                self._cv2.resizeWindow(
                    PREVIEW_WINDOW_NAME,
                    int(self.preview_canvas_width),
                    int(self.preview_canvas_height),
                )

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
                frame_started_at = perf_counter()
                if captured.image is None:
                    yield TrackingFrame(
                        timestamp=captured.metadata.timestamp,
                        source_id=captured.metadata.source_id,
                        frame=captured.metadata,
                    )
                    continue

                image = captured.image
                color_started_at = perf_counter()
                rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
                color_convert_ms = (perf_counter() - color_started_at) * 1000
                media_image = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB,
                    data=rgb,
                )
                inference_started_at = perf_counter()
                results = self._landmarker.detect_for_video(
                    media_image,
                    int(captured.metadata.timestamp * 1000),
                )
                inference_ms = (perf_counter() - inference_started_at) * 1000
                normalize_started_at = perf_counter()
                hands = normalized_hands_from_mediapipe_results(results)
                normalize_ms = (perf_counter() - normalize_started_at) * 1000

                preview_draw_ms = 0.0
                if self.show and not self._draw_debug_image(image, hands):
                    break
                if self.show:
                    preview_draw_ms = (perf_counter() - normalize_started_at) * 1000 - normalize_ms
                self.timing_samples.append(
                    MediaPipeTimingSample(
                        capture_read_ms=self._capture.last_read_duration_ms,
                        color_convert_ms=color_convert_ms,
                        inference_ms=inference_ms,
                        normalize_ms=normalize_ms,
                        preview_draw_ms=preview_draw_ms,
                        total_ms=(
                            (self._capture.last_read_duration_ms or 0.0)
                            + (perf_counter() - frame_started_at) * 1000
                        ),
                    )
                )

                yield TrackingFrame(
                    timestamp=captured.metadata.timestamp,
                    source_id=captured.metadata.source_id,
                    frame=captured.metadata,
                    hands=hands,
                )
        finally:
            self.stop()

    def _draw_debug_image(self, image: Any, hands: tuple[NormalizedHand, ...]) -> bool:
        if self.preview_layout == "dashboard":
            return self._draw_dashboard_image(image, hands)

        assert self._cv2 is not None
        display_image = self._cv2.flip(image, 1) if self.preview_mirror else image
        height, width = display_image.shape[:2]
        candidates = self._preview_candidates(hands, width=width, height=height)
        self._draw_header(display_image, hands, candidates)
        for hand in hands:
            self._draw_hand_overlay(
                display_image,
                hand,
                width=width,
                height=height,
                candidate_names=candidates.get(hand.hand_id, ()),
            )
        if self.preview_chart_provider is not None:
            chart = self.preview_chart_provider()
            if chart is not None:
                self._draw_chart_overlay(display_image, chart)
        if self.preview_status_provider is not None:
            self._draw_alert_banner(display_image, self.preview_status_provider())
        self._draw_gesture_strip(display_image, candidates)
        self._cv2.imshow(PREVIEW_WINDOW_NAME, display_image)
        key = self._cv2.waitKey(1) & 0xFF
        if self.preview_key_handler is not None and key not in (255, -1):
            self.preview_key_handler(key)
        return key not in (27, ord("q"))

    def _draw_dashboard_image(self, image: Any, hands: tuple[NormalizedHand, ...]) -> bool:
        assert self._cv2 is not None
        dashboard = (
            self.preview_dashboard_provider()
            if self.preview_dashboard_provider is not None
            else {}
        ) or {}
        status = self.preview_status_provider() if self.preview_status_provider is not None else ""
        candidates = self._preview_candidates(
            hands,
            width=image.shape[1],
            height=image.shape[0],
        )
        renderer = LiveDashboardRenderer(
            cv2=self._cv2,
            mirror=self.preview_mirror,
            canvas_width=self.preview_canvas_width,
            canvas_height=self.preview_canvas_height,
        )
        canvas = renderer.render(
            image=image,
            hands=hands,
            dashboard=dashboard,
            status=status,
            candidates=candidates,
            draw_hand_overlay=lambda target, hand, width, height, names: self._draw_hand_overlay(
                target,
                hand,
                width=width,
                height=height,
                candidate_names=names,
            ),
            draw_alert_banner=self._draw_alert_banner,
        )

        self._cv2.imshow(PREVIEW_WINDOW_NAME, canvas)
        key = self._cv2.waitKey(1) & 0xFF
        if self.preview_key_handler is not None and key not in (255, -1):
            self.preview_key_handler(key)
        return key not in (27, ord("q"))

    def _draw_header(
        self,
        image: Any,
        hands: tuple[NormalizedHand, ...],
        candidates: dict[str, tuple[str, ...]],
    ) -> None:
        assert self._cv2 is not None
        gesture_count = sum(len(names) for names in candidates.values())
        mirror = "mirror" if self.preview_mirror else "camera"
        text = (
            f"AirDesk live view | {mirror} | hands={len(hands)} | "
            f"gestures={gesture_count} | q/esc quits"
        )
        self._cv2.rectangle(image, (0, 0), (image.shape[1], 64), (20, 20, 20), -1)
        self._put_text_fit(
            image=image,
            text=text,
            x=10,
            y=23,
            max_width=image.shape[1] - 20,
            scale=0.6,
            color=(240, 240, 240),
        )
        if self.preview_status_provider is None or self.preview_chart_provider is not None:
            return
        status = self.preview_status_provider()
        self._put_text_fit(
            image=image,
            text=status,
            x=10,
            y=52,
            max_width=image.shape[1] - 20,
            scale=0.52,
            color=(
                (0, 255, 255)
                if "GESTURE" in status or "paused" in status
                else (230, 230, 230)
            ),
        )

    def _draw_chart_overlay(self, image: Any, chart: dict[str, Any]) -> None:
        assert self._cv2 is not None
        height, width = image.shape[:2]
        phase = str(chart.get("phase", "recording"))
        label = str(chart.get("label", "chart"))
        elapsed = float(chart.get("elapsed", 0.0))
        duration = chart.get("duration")
        segments = chart.get("segments", [])

        panel_x = max(18, int(width * 0.06))
        panel_y = max(82, int(height * 0.18))
        panel_w = min(width - panel_x * 2, max(420, int(width * 0.88)))
        panel_h = min(230, max(170, int(height * 0.38)))
        self._cv2.rectangle(
            image,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (18, 22, 28),
            -1,
        )
        self._cv2.rectangle(
            image,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (82, 92, 110),
            2,
        )

        if phase == "waiting":
            headline = "READY"
            subline = f"{label} | press space"
            accent = (255, 190, 60)
        elif phase == "countdown":
            remaining = float(chart.get("countdown_remaining", 0.0))
            headline = f"{ceil(max(0.0, remaining))}"
            subline = f"{label} starts next"
            accent = (0, 220, 255)
        else:
            total = (
                f"{elapsed:.1f}s"
                if duration is None
                else f"{elapsed:.1f}/{float(duration):.1f}s"
            )
            headline = str(chart.get("current_text", label))
            subline = total
            accent = _chart_segment_color(str(chart.get("current_kind", "prompt")))

        self._cv2.rectangle(
            image,
            (panel_x, panel_y),
            (panel_x + 10, panel_y + panel_h),
            accent,
            -1,
        )
        self._put_text_fit(
            image=image,
            text=headline,
            x=panel_x + 24,
            y=panel_y + 54,
            max_width=panel_w - 48,
            scale=1.25,
            color=(255, 255, 255),
            thickness=3,
        )
        self._put_text_fit(
            image=image,
            text=subline,
            x=panel_x + 26,
            y=panel_y + 88,
            max_width=panel_w - 52,
            scale=0.62,
            color=(215, 220, 230),
            thickness=2,
        )
        chart_elapsed = elapsed if phase == "recording" else 0.0
        self._draw_chart_progress(
            image=image,
            elapsed=chart_elapsed,
            segments=segments,
            x=panel_x + 26,
            y=panel_y + 108,
            width=panel_w - 52,
            color=accent,
        )
        self._draw_chart_queue(
            image=image,
            elapsed=chart_elapsed,
            segments=segments,
            x=panel_x + 26,
            y=panel_y + 142,
            width=panel_w - 52,
        )

    def _draw_chart_progress(
        self,
        *,
        image: Any,
        elapsed: float,
        segments: list[Any],
        x: int,
        y: int,
        width: int,
        color: tuple[int, int, int],
    ) -> None:
        assert self._cv2 is not None
        segment = _chart_segment_at(elapsed, segments)
        progress = _chart_segment_progress(elapsed, segment)
        height = 14
        self._cv2.rectangle(image, (x, y), (x + width, y + height), (55, 58, 66), -1)
        fill_width = max(4, int(width * progress)) if segment is not None else 0
        if fill_width > 0:
            self._cv2.rectangle(image, (x, y), (x + fill_width, y + height), color, -1)
        self._cv2.rectangle(image, (x, y), (x + width, y + height), (210, 215, 225), 1)

    def _draw_chart_queue(
        self,
        *,
        image: Any,
        elapsed: float,
        segments: list[Any],
        x: int,
        y: int,
        width: int,
    ) -> None:
        upcoming = _chart_visible_queue(elapsed, segments, limit=3)
        if not upcoming:
            return
        gap = 12
        card_count = 3
        card_width = max(80, (width - gap * (card_count - 1)) // card_count)
        card_height = 52
        for index, item in enumerate(upcoming):
            if not isinstance(item, dict):
                continue
            x0 = x + index * (card_width + gap)
            x1 = x0 + card_width
            y0 = y
            y1 = y + card_height
            kind = str(item.get("kind", "prompt"))
            text = str(item.get("text", ""))
            color = _chart_segment_color(kind)
            self._cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
            self._cv2.rectangle(image, (x0, y0), (x1, y1), (250, 250, 250), 1)
            self._put_text_fit(
                image=image,
                text=text,
                x=x0 + 10,
                y=y0 + 33,
                max_width=max(20, card_width - 20),
                scale=0.58,
                color=(255, 255, 255),
                thickness=2,
            )

    def _draw_alert_banner(self, image: Any, status: str) -> None:
        assert self._cv2 is not None
        marker = "GESTURE "
        if marker not in status:
            return
        alert = status.split(marker, maxsplit=1)[1].split("|", maxsplit=1)[0].strip()
        if not alert:
            return
        label = f"Gesture: {alert}"
        width = image.shape[1]
        y_center = max(92, image.shape[0] // 6)
        self._cv2.rectangle(
            image,
            (0, y_center - 36),
            (width, y_center + 28),
            (0, 90, 220),
            -1,
        )
        self._put_text_fit(
            image=image,
            text=label,
            x=18,
            y=y_center + 10,
            max_width=width - 36,
            scale=1.1,
            color=(255, 255, 255),
            thickness=3,
        )

    def _draw_hand_overlay(
        self,
        image: Any,
        hand: NormalizedHand,
        *,
        width: int,
        height: int,
        candidate_names: tuple[str, ...],
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
                pixel_point(
                    landmarks[start],
                    width=width,
                    height=height,
                    mirror=self.preview_mirror,
                ),
                pixel_point(
                    landmarks[end],
                    width=width,
                    height=height,
                    mirror=self.preview_mirror,
                ),
                muted,
                2,
            )
        for landmark in landmarks:
            self._cv2.circle(
                image,
                pixel_point(
                    landmark,
                    width=width,
                    height=height,
                    mirror=self.preview_mirror,
                ),
                3,
                color,
                -1,
            )

        x_min, y_min, x_max, y_max = bbox_pixels(
            hand.bbox,
            width=width,
            height=height,
            mirror=self.preview_mirror,
        )
        self._cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        label = hand_label(hand)
        if candidate_names:
            label = f"{label} | {', '.join(candidate_names)}"
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

    def _draw_gesture_strip(self, image: Any, candidates: dict[str, tuple[str, ...]]) -> None:
        assert self._cv2 is not None
        names = sorted({name for hand_names in candidates.values() for name in hand_names})
        text = "gestures: " + (", ".join(names) if names else "none")
        y = image.shape[0] - 14
        self._cv2.rectangle(
            image, (0, image.shape[0] - 42), (image.shape[1], image.shape[0]), (20, 20, 20), -1
        )
        self._put_text_fit(
            image=image,
            text=text,
            x=10,
            y=y,
            max_width=image.shape[1] - 20,
            scale=0.65,
            color=(0, 255, 255) if names else (190, 190, 190),
        )

    def _put_text_fit(
        self,
        *,
        image: Any,
        text: str,
        x: int,
        y: int,
        max_width: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        assert self._cv2 is not None
        fitted = self._fit_text(text, max_width=max_width, scale=scale, thickness=thickness)
        self._cv2.putText(
            image,
            fitted,
            (x, y),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            self._cv2.LINE_AA,
        )

    def _fit_text(self, text: str, *, max_width: int, scale: float, thickness: int) -> str:
        assert self._cv2 is not None
        if self._text_width(text, scale=scale, thickness=thickness) <= max_width:
            return text
        suffix = "..."
        low = 0
        high = max(0, len(text) - len(suffix))
        best = suffix
        while low <= high:
            middle = (low + high) // 2
            candidate = text[:middle].rstrip() + suffix
            if self._text_width(candidate, scale=scale, thickness=thickness) <= max_width:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    def _text_width(self, text: str, *, scale: float, thickness: int) -> int:
        assert self._cv2 is not None
        size, _baseline = self._cv2.getTextSize(
            text,
            self._cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )
        return int(size[0])

    def _preview_candidates(
        self,
        hands: tuple[NormalizedHand, ...],
        *,
        width: int,
        height: int,
    ) -> dict[str, tuple[str, ...]]:
        if not self.preview_gestures or not hands:
            return {}
        recognizer = StaticHandPoseRecognizer(
            extended_threshold=self.preview_extended_threshold,
            pinch_threshold=self.preview_pinch_threshold,
        )
        frame = TrackingFrame(
            timestamp=0,
            source_id="preview",
            frame=FrameMetadata(
                timestamp=0,
                source_id="preview",
                width=width,
                height=height,
                sequence=0,
            ),
            hands=hands,
        )
        candidates: dict[str, list[str]] = {}
        for candidate in recognizer.recognize(frame):
            if candidate.hand_id is None:
                continue
            candidates.setdefault(candidate.hand_id, []).append(candidate.name)
        return {hand_id: tuple(names) for hand_id, names in candidates.items()}


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


def _base_options_delegate(base_options: Any, delegate: str) -> Any:
    normalized = delegate.strip().lower()
    if normalized == "cpu":
        return base_options.Delegate.CPU
    if normalized == "gpu":
        return base_options.Delegate.GPU
    raise RuntimeError("MediaPipe delegate must be 'cpu' or 'gpu'")


def pixel_point(
    landmark: Landmark,
    *,
    width: int,
    height: int,
    mirror: bool = False,
) -> tuple[int, int]:
    """Convert a normalized landmark to image pixel coordinates."""
    x_value = 1.0 - landmark.x if mirror else landmark.x
    x = min(width - 1, max(0, round(x_value * width)))
    y = min(height - 1, max(0, round(landmark.y * height)))
    return x, y


def _chart_segment_color(kind: str) -> tuple[int, int, int]:
    if kind == "cue":
        return (0, 165, 255)
    if kind == "lead_in":
        return (0, 210, 255)
    if kind == "stroke":
        return (40, 190, 80)
    if kind == "recovery":
        return (210, 130, 45)
    if kind == "rest":
        return (95, 100, 112)
    return (90, 120, 210)


def _fit_interval_inside(
    *,
    center: int,
    width: int,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    available = max(1, maximum - minimum)
    fitted_width = min(width, available)
    x0 = center - fitted_width // 2
    x1 = x0 + fitted_width
    if x0 < minimum:
        x0 = minimum
        x1 = x0 + fitted_width
    if x1 > maximum:
        x1 = maximum
        x0 = x1 - fitted_width
    return x0, x1


def _chart_segment_at(elapsed: float, segments: list[Any]) -> dict[str, Any] | None:
    for item in segments:
        if not isinstance(item, dict):
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if start <= elapsed < end:
            return item
    return None


def _chart_segment_progress(elapsed: float, segment: dict[str, Any] | None) -> float:
    if segment is None:
        return 0.0
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    if end <= start:
        return 1.0
    return min(1.0, max(0.0, (elapsed - start) / (end - start)))


def _chart_visible_queue(
    elapsed: float,
    segments: list[Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    current = _chart_segment_at(elapsed, segments)
    start_after = float(current.get("end", elapsed)) if current is not None else elapsed
    upcoming: list[dict[str, Any]] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        start = float(item.get("start", 0.0))
        if start >= start_after - 1e-6:
            upcoming.append(item)
        if len(upcoming) >= limit:
            break
    return upcoming


def bbox_pixels(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    mirror: bool = False,
) -> tuple[int, int, int, int]:
    """Convert a normalized hand bounding box to pixel coordinates."""
    x0, y0, x1, y1 = bbox
    if mirror:
        x0, x1 = 1.0 - x1, 1.0 - x0
    x_min = min(width - 1, max(0, round(x0 * width)))
    y_min = min(height - 1, max(0, round(bbox[1] * height)))
    x_max = min(width - 1, max(0, round(x1 * width)))
    y_max = min(height - 1, max(0, round(y1 * height)))
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
