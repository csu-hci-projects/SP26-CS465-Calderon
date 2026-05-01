from __future__ import annotations

from collections.abc import Callable

from airdesk.gestures.base import CompositeGestureRecognizer
from airdesk.gestures.phrases import GesturePhase, IntentGatedSwipeRecognizer
from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.state.types import (
    FrameMetadata,
    HandLandmarks,
    Landmark,
    NormalizedHand,
    TrackingFrame,
)


def names(candidates: object) -> set[str]:
    return {candidate.name for candidate in candidates}


def test_open_palm_primitive(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    frame = make_tracking_frame(make_hand("open_palm"))

    candidates = StaticHandPoseRecognizer().recognize(frame)

    assert "open_palm" in names(candidates)
    assert "fist" not in names(candidates)


def test_fist_primitive(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    frame = make_tracking_frame(make_hand("fist"))

    candidates = StaticHandPoseRecognizer().recognize(frame)

    assert "fist" in names(candidates)
    assert "open_palm" not in names(candidates)


def test_pinch_primitive(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    frame = make_tracking_frame(make_hand("pinch"))

    candidates = StaticHandPoseRecognizer().recognize(frame)

    assert "pinch" in names(candidates)


def test_primitive_features_expose_live_tuning_values(
    make_hand: Callable[[str], NormalizedHand],
) -> None:
    recognizer = StaticHandPoseRecognizer()

    features = recognizer.features_for_hand(make_hand("pinch"))

    assert features is not None
    assert features.extended_fingers == 4
    assert features.folded_fingers == 0
    assert features.pinch_distance == 0
    assert features.finger_spread > 0


def test_point_left_and_right_primitives(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    left_frame = make_tracking_frame(make_hand("point_left"))
    right_frame = make_tracking_frame(make_hand("point_right"))

    left_candidates = StaticHandPoseRecognizer().recognize(left_frame)
    right_candidates = StaticHandPoseRecognizer().recognize(right_frame)

    assert "point_left" in names(left_candidates)
    assert "point_right" not in names(left_candidates)
    assert "point_right" in names(right_candidates)
    assert "point_left" not in names(right_candidates)


def test_intent_gated_swipe_left_and_right_from_open_palm_history(
    make_hand: Callable[[str], NormalizedHand],
) -> None:
    left_recognizer = IntentGatedSwipeRecognizer()
    right_recognizer = IntentGatedSwipeRecognizer()

    left_candidates = _run_sequence(
        left_recognizer,
        [
            _frame_at(1.0, 1, _move_hand(make_hand("open_palm"), x=0.65)),
            _frame_at(1.16, 2, _move_hand(make_hand("open_palm"), x=0.42)),
        ],
    )
    right_candidates = _run_sequence(
        right_recognizer,
        [
            _frame_at(2.0, 1, _move_hand(make_hand("open_palm"), x=0.35)),
            _frame_at(2.16, 2, _move_hand(make_hand("open_palm"), x=0.59)),
        ],
    )

    assert "swipe_left" in names(left_candidates)
    assert "swipe_right" in names(right_candidates)
    assert left_candidates[-1].metadata["phase"] == GesturePhase.CONFIRMED.value


def test_composite_recognizer_combines_static_and_phrase_candidates(
    make_hand: Callable[[str], NormalizedHand],
) -> None:
    static = StaticHandPoseRecognizer()
    recognizer = CompositeGestureRecognizer(
        recognizers=(static, IntentGatedSwipeRecognizer(pose_recognizer=static))
    )

    first_candidates = recognizer.recognize(
        _frame_at(1.0, 1, _move_hand(make_hand("open_palm"), x=0.35))
    )
    second_candidates = recognizer.recognize(
        _frame_at(1.2, 2, _move_hand(make_hand("open_palm"), x=0.6))
    )

    assert "open_palm" in names(first_candidates)
    assert "swipe_right" in names(second_candidates)


def _run_sequence(
    recognizer: IntentGatedSwipeRecognizer,
    frames: list[TrackingFrame],
) -> list[object]:
    candidates = []
    for frame in frames:
        candidates.extend(recognizer.recognize(frame))
    return candidates


def _frame_at(timestamp: float, sequence: int, hand: NormalizedHand) -> TrackingFrame:
    metadata = FrameMetadata(
        timestamp=timestamp,
        source_id="gesture-test",
        width=640,
        height=480,
        sequence=sequence,
    )
    return TrackingFrame(
        timestamp=timestamp,
        source_id="gesture-test",
        frame=metadata,
        hands=(hand,),
    )


def _move_hand(hand: NormalizedHand, *, x: float) -> NormalizedHand:
    dx = x - hand.palm_center[0]
    landmarks = tuple(
        Landmark(
            landmark.x + dx,
            landmark.y,
            landmark.z,
            visibility=landmark.visibility,
            presence=landmark.presence,
        )
        for landmark in hand.landmarks.landmarks
    )
    return NormalizedHand(
        hand_id=hand.hand_id,
        landmarks=HandLandmarks(
            landmarks=landmarks,
            handedness=hand.landmarks.handedness,
            confidence=hand.landmarks.confidence,
        ),
        palm_center=(x, hand.palm_center[1], hand.palm_center[2]),
        bbox=(hand.bbox[0] + dx, hand.bbox[1], hand.bbox[2] + dx, hand.bbox[3]),
        handedness=hand.handedness,
        confidence=hand.confidence,
    )
