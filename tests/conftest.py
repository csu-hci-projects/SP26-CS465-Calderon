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
        finger_joints = {
            (5, 6, 7, 8): (0.38, 0.34),
            (9, 10, 11, 12): (0.47, 0.45),
            (13, 14, 15, 16): (0.56, 0.58),
            (17, 18, 19, 20): (0.65, 0.70),
        }
        mcp_xs = {
            mcp: mcp_x
            for (mcp, _pip, _dip, _tip), (mcp_x, _tip_x) in finger_joints.items()
        }
        for index, x in mcp_xs.items():
            landmarks[index] = Landmark(x, 0.55, 0.0)

        if pose == "open_palm":
            for (mcp, pip, dip, tip), (mcp_x, tip_x) in finger_joints.items():
                landmarks[mcp] = Landmark(mcp_x, 0.55, 0.0)
                landmarks[pip] = Landmark((mcp_x + tip_x) / 2, 0.45, 0.0)
                landmarks[dip] = Landmark((mcp_x + tip_x) / 2, 0.35, 0.0)
                landmarks[tip] = Landmark(tip_x, 0.25, 0.0)
            landmarks[4] = Landmark(0.26, 0.43, 0.0)
        elif pose == "fist":
            curled_tip_xs = {8: 0.43, 12: 0.47, 16: 0.51, 20: 0.55}
            for mcp, pip, dip, tip in finger_joints:
                landmarks[pip] = Landmark(landmarks[mcp].x, 0.58, 0.0)
                landmarks[dip] = Landmark(curled_tip_xs[tip], 0.62, 0.0)
                landmarks[tip] = Landmark(curled_tip_xs[tip], 0.66, 0.0)
            landmarks[4] = Landmark(0.45, 0.64, 0.0)
        elif pose == "pinch":
            for (mcp, pip, dip, tip), (mcp_x, tip_x) in finger_joints.items():
                landmarks[mcp] = Landmark(mcp_x, 0.55, 0.0)
                landmarks[pip] = Landmark((mcp_x + tip_x) / 2, 0.45, 0.0)
                landmarks[dip] = Landmark((mcp_x + tip_x) / 2, 0.35, 0.0)
                landmarks[tip] = Landmark(tip_x, 0.25, 0.0)
            landmarks[4] = Landmark(0.36, 0.26, 0.0)
            landmarks[8] = Landmark(0.36, 0.26, 0.0)
        elif pose == "point_left":
            landmarks[8] = Landmark(0.18, 0.55, 0.0)
            landmarks[12] = Landmark(0.48, 0.66, 0.0)
            landmarks[16] = Landmark(0.57, 0.66, 0.0)
            landmarks[20] = Landmark(0.66, 0.66, 0.0)
            landmarks[4] = Landmark(0.42, 0.63, 0.0)
        elif pose == "point_right":
            landmarks[8] = Landmark(0.72, 0.55, 0.0)
            landmarks[12] = Landmark(0.48, 0.66, 0.0)
            landmarks[16] = Landmark(0.57, 0.66, 0.0)
            landmarks[20] = Landmark(0.66, 0.66, 0.0)
            landmarks[4] = Landmark(0.42, 0.63, 0.0)
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
