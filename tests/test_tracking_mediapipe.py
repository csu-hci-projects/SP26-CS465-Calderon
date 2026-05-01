from __future__ import annotations

from dataclasses import dataclass

from airdesk.tracking.mediapipe import (
    bbox_pixels,
    hand_label,
    normalized_hands_from_mediapipe_results,
    pixel_point,
)


@dataclass
class FakePoint:
    x: float
    y: float
    z: float = 0.0


@dataclass
class FakeLandmarks:
    landmark: list[FakePoint]


@dataclass
class FakeClassification:
    label: str
    score: float


@dataclass
class FakeHandedness:
    classification: list[FakeClassification]


@dataclass
class FakeResults:
    multi_hand_landmarks: list[FakeLandmarks] | None
    multi_handedness: list[FakeHandedness] | None


def test_mediapipe_conversion_maps_landmarks_handedness_and_bounds() -> None:
    points = [FakePoint(x=index / 100, y=(20 - index) / 100, z=0.01) for index in range(21)]
    results = FakeResults(
        multi_hand_landmarks=[FakeLandmarks(points)],
        multi_handedness=[FakeHandedness([FakeClassification(label="Right", score=0.91)])],
    )

    hands = normalized_hands_from_mediapipe_results(results)

    assert len(hands) == 1
    assert hands[0].hand_id == "hand-0"
    assert hands[0].handedness == "Right"
    assert hands[0].confidence == 0.91
    assert len(hands[0].landmarks.landmarks) == 21
    assert hands[0].bbox == (0.0, 0.0, 0.2, 0.2)


def test_mediapipe_conversion_handles_zero_hand_frames() -> None:
    results = FakeResults(multi_hand_landmarks=None, multi_handedness=None)

    assert normalized_hands_from_mediapipe_results(results) == ()


def test_preview_coordinate_helpers_clamp_to_image_bounds() -> None:
    point = FakePoint(x=1.5, y=-0.2)

    assert pixel_point(point, width=640, height=480) == (639, 0)
    assert pixel_point(FakePoint(x=0.25, y=0.5), width=640, height=480, mirror=True) == (
        480,
        240,
    )
    assert bbox_pixels((-0.1, 0.2, 1.2, 0.8), width=640, height=480) == (
        0,
        96,
        639,
        384,
    )
    assert bbox_pixels((0.2, 0.2, 0.6, 0.8), width=640, height=480, mirror=True) == (
        256,
        96,
        512,
        384,
    )


def test_hand_label_includes_handedness_and_confidence() -> None:
    points = [FakePoint(x=index / 100, y=(20 - index) / 100, z=0.01) for index in range(21)]
    hands = normalized_hands_from_mediapipe_results(
        FakeResults(
            multi_hand_landmarks=[FakeLandmarks(points)],
            multi_handedness=[FakeHandedness([FakeClassification(label="Left", score=0.876)])],
        )
    )

    assert hand_label(hands[0]) == "Left 0.88"
