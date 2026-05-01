from __future__ import annotations

from collections.abc import Callable

from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.state.types import NormalizedHand, TrackingFrame


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
