from __future__ import annotations

from collections.abc import Callable

from airdesk.control.combos import ComboBuffer
from airdesk.control.debounce import PoseDebounceConfig, PoseDebouncer, PoseEvent
from airdesk.control.grammar import POINTER_ACTION, ControlGrammar, ControlGrammarConfig
from airdesk.control.poses import ControlPoseRecognizer
from airdesk.state.types import Landmark, NormalizedHand, TrackingFrame


def test_control_pose_features_prioritize_conflicting_landmark_facts(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()

    open_features = recognizer.features_for_frame(
        make_tracking_frame(_move_hand(make_hand("open_palm"), x=0.30))
    )[0]
    fist_features = recognizer.features_for_frame(make_tracking_frame(make_hand("fist")))[0]
    middle_features = recognizer.features_for_frame(
        make_tracking_frame(_middle_pinch_hand(make_hand("open_palm")))
    )[0]

    assert "open_palm" in open_features.poses
    assert "sideways_open_palm_left" in open_features.poses
    assert open_features.palm_zone == "left"
    assert "fist" in fist_features.poses
    assert "index_pinch" not in fist_features.poses
    assert "middle_pinch" in middle_features.poses
    assert "open_palm" not in middle_features.poses


def test_control_pose_sideways_palm_suppresses_pinch_artifacts(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    noisy_sideways = _middle_pinch_hand(_move_hand(make_hand("open_palm"), x=0.30))

    features = recognizer.features_for_frame(make_tracking_frame(noisy_sideways))[0]

    assert features.poses == frozenset({"open_palm", "sideways_open_palm_left"})
    assert "middle_pinch" in features.suppressed_poses


def test_pose_debouncer_emits_enter_held_and_release_events() -> None:
    debouncer = PoseDebouncer(PoseDebounceConfig(enter_frames=2, release_frames=2))

    first = debouncer.update(hand_id="hand-0", timestamp=1.0, active_poses=frozenset({"fist"}))
    entered = debouncer.update(hand_id="hand-0", timestamp=1.1, active_poses=frozenset({"fist"}))
    held = debouncer.update(hand_id="hand-0", timestamp=1.6, active_poses=frozenset({"fist"}))
    missing = debouncer.update(hand_id="hand-0", timestamp=1.7, active_poses=frozenset())
    released = debouncer.update(hand_id="hand-0", timestamp=1.8, active_poses=frozenset())

    assert first == []
    assert entered == [PoseEvent("hand-0", "fist", "entered", 1.1)]
    assert held[0].event_type == "held"
    assert missing == []
    assert released[0].event_type == "released"


def test_combo_buffer_matches_same_hand_and_consumes_events() -> None:
    buffer = ComboBuffer()
    buffer.add(PoseEvent("hand-0", "open_palm", "entered", 1.0))
    buffer.add(PoseEvent("hand-1", "fist", "entered", 1.1))
    buffer.add(PoseEvent("hand-0", "fist", "entered", 1.2))
    buffer.add(PoseEvent("hand-0", "open_palm", "entered", 1.3))

    assert buffer.match(("open_palm", "fist", "open_palm"), now=1.3, hand_id="hand-0")
    assert not buffer.match(("open_palm", "fist", "open_palm"), now=1.3, hand_id="hand-0")


def test_control_grammar_routes_clicks_workspace_and_close_combo(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar(ControlGrammarConfig(command_cooldown_seconds=0.0))
    features = recognizer.features_for_frame(make_tracking_frame(make_hand("pinch")))

    pressed = grammar.update(
        features=features,
        events=[PoseEvent("hand-0", "index_pinch", "entered", 1.0)],
        timestamp=1.0,
    )
    click = grammar.update(
        features=features,
        events=[PoseEvent("hand-0", "index_pinch", "released", 1.1, duration=0.1)],
        timestamp=1.1,
    )

    assert pressed == []
    assert click[0].request.action_type == POINTER_ACTION
    assert click[0].request.parameters["button"] == "left"

    side_features = recognizer.features_for_frame(
        make_tracking_frame(_move_hand(make_hand("open_palm"), x=0.70))
    )
    workspace = grammar.update(
        features=side_features,
        events=[PoseEvent("hand-0", "sideways_open_palm_right", "held", 2.0, duration=0.4)],
        timestamp=2.0,
    )

    assert workspace[0].request.command == "workspace"
    assert workspace[0].request.parameters["args"] == ["-1"]

    move_window = grammar.update(
        features=side_features,
        events=[PoseEvent("hand-0", "fist", "held", 2.8, duration=0.4)],
        timestamp=2.8,
    )

    assert move_window[0].request.command == "movetoworkspace"
    assert move_window[0].request.parameters["args"] == ["-1"]

    close = grammar.update(
        features=side_features,
        events=[
            PoseEvent("hand-0", "open_palm", "entered", 3.0),
            PoseEvent("hand-0", "fist", "entered", 3.1),
            PoseEvent("hand-0", "open_palm", "entered", 3.2),
        ],
        timestamp=3.2,
    )

    assert close[0].name == "close_window"
    assert close[0].request.command == "killactive"
    assert close[0].high_risk is True


def test_control_grammar_scrolls_on_index_pinch_hold_and_suppresses_tap(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar(ControlGrammarConfig(scroll_cooldown_seconds=0.0))
    features = recognizer.features_for_frame(make_tracking_frame(make_hand("pinch")))

    grammar.update(
        features=features,
        events=[PoseEvent("hand-0", "index_pinch", "entered", 1.0)],
        timestamp=1.0,
    )
    scroll = grammar.update(
        features=features,
        events=[PoseEvent("hand-0", "index_pinch", "held", 1.4, duration=0.4)],
        timestamp=1.4,
        scroll_delta_by_hand={"hand-0": -1},
    )
    click = grammar.update(
        features=features,
        events=[PoseEvent("hand-0", "index_pinch", "released", 1.5, duration=0.5)],
        timestamp=1.5,
    )

    assert scroll[0].name == "scroll"
    assert scroll[0].request.parameters["amount_y"] == -1
    assert click == []


def test_control_grammar_cancels_pinch_tap_when_release_becomes_fist(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar()
    pinch_features = recognizer.features_for_frame(make_tracking_frame(make_hand("pinch")))
    fist_features = recognizer.features_for_frame(make_tracking_frame(make_hand("fist")))

    grammar.update(
        features=pinch_features,
        events=[PoseEvent("hand-0", "index_pinch", "entered", 1.0)],
        timestamp=1.0,
    )
    click = grammar.update(
        features=fist_features,
        events=[PoseEvent("hand-0", "index_pinch", "released", 1.1, duration=0.1)],
        timestamp=1.1,
    )

    assert click == []


def _move_hand(hand: NormalizedHand, *, x: float) -> NormalizedHand:
    dx = x - hand.palm_center[0]
    landmarks = tuple(
        Landmark(point.x + dx, point.y, point.z, point.visibility, point.presence)
        for point in hand.landmarks.landmarks
    )
    return NormalizedHand(
        hand_id=hand.hand_id,
        landmarks=type(hand.landmarks)(
            landmarks,
            handedness=hand.landmarks.handedness,
            confidence=hand.landmarks.confidence,
        ),
        palm_center=(x, hand.palm_center[1], hand.palm_center[2]),
        bbox=tuple(
            value + dx if index % 2 == 0 else value for index, value in enumerate(hand.bbox)
        ),
        handedness=hand.handedness,
        confidence=hand.confidence,
    )


def _middle_pinch_hand(hand: NormalizedHand) -> NormalizedHand:
    landmarks = list(hand.landmarks.landmarks)
    landmarks[4] = Landmark(0.47, 0.26, 0.0)
    landmarks[12] = Landmark(0.47, 0.26, 0.0)
    return NormalizedHand(
        hand_id=hand.hand_id,
        landmarks=type(hand.landmarks)(
            tuple(landmarks),
            handedness=hand.landmarks.handedness,
            confidence=hand.landmarks.confidence,
        ),
        palm_center=hand.palm_center,
        bbox=hand.bbox,
        handedness=hand.handedness,
        confidence=hand.confidence,
    )
