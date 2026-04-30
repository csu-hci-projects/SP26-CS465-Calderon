from __future__ import annotations

from collections.abc import Callable

import pytest

from airdesk.state.types import (
    FrameMetadata,
    HandLandmarks,
    Landmark,
    NormalizedHand,
    TrackingFrame,
)


@pytest.fixture
def make_hand() -> Callable[[str], NormalizedHand]:
    def _make_hand(pose: str) -> NormalizedHand:
        landmarks = [Landmark(0.5, 0.8, 0.0) for _ in range(21)]
        mcp_xs = {5: 0.38, 9: 0.47, 13: 0.56, 17: 0.65}
        tip_xs = {8: 0.34, 12: 0.45, 16: 0.58, 20: 0.70}
        for index, x in mcp_xs.items():
            landmarks[index] = Landmark(x, 0.55, 0.0)

        if pose == "open_palm":
            for index, x in tip_xs.items():
                landmarks[index] = Landmark(x, 0.25, 0.0)
            landmarks[4] = Landmark(0.26, 0.43, 0.0)
        elif pose == "fist":
            for index, x in tip_xs.items():
                landmarks[index] = Landmark(x, 0.66, 0.0)
            landmarks[4] = Landmark(0.42, 0.63, 0.0)
        elif pose == "pinch":
            for index, x in tip_xs.items():
                landmarks[index] = Landmark(x, 0.25, 0.0)
            landmarks[4] = Landmark(0.36, 0.26, 0.0)
            landmarks[8] = Landmark(0.36, 0.26, 0.0)
        else:
            raise ValueError(f"unknown synthetic pose: {pose}")

        hand_landmarks = HandLandmarks(tuple(landmarks), handedness="right", confidence=1.0)
        return NormalizedHand(
            hand_id="hand-0",
            landmarks=hand_landmarks,
            palm_center=(0.5, 0.55, 0.0),
            bbox=(0.2, 0.2, 0.7, 0.8),
            handedness="right",
            confidence=1.0,
        )

    return _make_hand


@pytest.fixture
def make_tracking_frame() -> Callable[..., TrackingFrame]:
    def _make_tracking_frame(*hands: NormalizedHand) -> TrackingFrame:
        frame = FrameMetadata(
            timestamp=1.0,
            source_id="test",
            width=640,
            height=480,
            sequence=1,
        )
        return TrackingFrame(timestamp=1.0, source_id="test", frame=frame, hands=hands)

    return _make_tracking_frame


@pytest.fixture
def make_empty_tracking_frame() -> TrackingFrame:
    frame = FrameMetadata(
        timestamp=1.0,
        source_id="test",
        width=640,
        height=480,
        sequence=1,
    )
    return TrackingFrame(timestamp=1.0, source_id="test", frame=frame, hands=())
