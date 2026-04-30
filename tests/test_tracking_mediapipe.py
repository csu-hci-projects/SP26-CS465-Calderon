from __future__ import annotations

from dataclasses import dataclass

from airdesk.tracking.mediapipe import normalized_hands_from_mediapipe_results


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
