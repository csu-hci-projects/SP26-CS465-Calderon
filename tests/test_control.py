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
    assert fist_features.pose_scores["fist"] >= 0.72
    assert fist_features.pose_evidence["fist"]["strong_folded_fingers"] == 4
    assert "middle_pinch" in middle_features.poses
    assert "open_palm" not in middle_features.poses


def test_control_pose_sideways_palm_suppresses_weak_pinch_artifacts(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    noisy_sideways = _weak_middle_pinch_hand(_move_hand(make_hand("open_palm"), x=0.30))

    features = recognizer.features_for_frame(make_tracking_frame(noisy_sideways))[0]

    assert features.poses == frozenset({"open_palm", "sideways_open_palm_left"})
    assert "middle_pinch" in features.suppressed_poses
    assert features.ambiguity is None


def test_control_pose_suppresses_ambiguous_sideways_palm_pinch(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    ambiguous_sideways = _middle_pinch_hand(_move_hand(make_hand("open_palm"), x=0.30))

    features = recognizer.features_for_frame(make_tracking_frame(ambiguous_sideways))[0]

    assert features.poses == frozenset()
    assert "middle_pinch" in features.suppressed_poses
    assert features.ambiguity == "sideways_palm_pinch_conflict"


def test_control_pose_requires_strong_finger_fold_for_fist(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    relaxed_curl = _relaxed_curled_hand(make_hand("open_palm"))

    features = recognizer.features_for_frame(make_tracking_frame(relaxed_curl))[0]

    assert "fist" not in features.poses
    assert features.poses == frozenset()


def test_control_pose_accepts_sideways_closed_fist_without_vertical_fold(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    sideways_fist = _sideways_closed_fist(make_hand("open_palm"))

    features = recognizer.features_for_frame(make_tracking_frame(sideways_fist))[0]

    assert "fist" in features.poses
    assert features.pose_evidence["fist"]["closed_fingers"] == 4
    assert features.pose_evidence["fist"]["strong_folded_fingers"] == 0


def test_control_pose_rejects_partial_curl_as_command_pose(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    partial_curl = _partial_curled_hand(make_hand("open_palm"))

    features = recognizer.features_for_frame(make_tracking_frame(partial_curl))[0]

    assert "fist" not in features.poses
    assert features.poses == frozenset()
    assert features.pose_evidence["fist"]["strong_folded_fingers"] == 2


def test_control_pose_rejects_pinch_like_fist_artifact(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    artifact = _pinch_like_fist_artifact(make_hand("fist"))

    features = recognizer.features_for_frame(make_tracking_frame(artifact))[0]

    assert features.poses == frozenset()
    assert "fist" not in features.poses
    assert features.ambiguity == "forming_fist_pinch_conflict"


def test_control_pose_rejects_noisy_sideways_curl(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    noisy_sideways = _move_hand(_pinch_like_fist_artifact(make_hand("fist")), x=0.28)

    features = recognizer.features_for_frame(make_tracking_frame(noisy_sideways))[0]

    assert features.poses == frozenset()
    assert features.ambiguity == "forming_fist_pinch_conflict"


def test_control_pose_log_dict_exposes_evidence_and_ambiguity(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    artifact = _pinch_like_fist_artifact(make_hand("fist"))

    features = recognizer.features_for_frame(make_tracking_frame(artifact))[0]
    payload = features.to_log_dict()

    assert payload["ambiguity"] == "forming_fist_pinch_conflict"
    assert "fist" in payload["pose_evidence"]
    assert payload["pose_scores"]["fist"] > 0
    assert payload["suppressed_poses"]


def test_control_pose_blocks_forming_fist_pinch_artifact(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    forming = _forming_fist_pinch_artifact(make_hand("open_palm"))

    features = recognizer.features_for_frame(make_tracking_frame(forming))[0]

    assert features.poses == frozenset()
    assert features.ambiguity == "forming_fist_pinch_conflict"
    assert {"index_pinch", "middle_pinch"}.issubset(features.suppressed_poses)


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


def test_pose_debouncer_reports_tracked_hand_ids() -> None:
    debouncer = PoseDebouncer(PoseDebounceConfig(enter_frames=1))

    debouncer.update(hand_id="hand-0", timestamp=1.0, active_poses=frozenset({"index_pinch"}))

    assert debouncer.hand_ids() == {"hand-0"}


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

    side_open_features = recognizer.features_for_frame(
        make_tracking_frame(_move_hand(make_hand("open_palm"), x=0.70))
    )
    open_palm_workspace = grammar.update(
        features=side_open_features,
        events=[PoseEvent("hand-0", "open_palm", "held", 2.0, duration=0.4)],
        timestamp=2.0,
    )

    assert open_palm_workspace == []

    side_fist_features = recognizer.features_for_frame(
        make_tracking_frame(_move_hand(make_hand("fist"), x=0.70))
    )
    unarmed_move = grammar.update(
        features=side_fist_features,
        events=[PoseEvent("hand-0", "fist", "held", 2.8, duration=0.4)],
        timestamp=2.8,
    )
    grammar.update(
        features=side_fist_features,
        events=[PoseEvent("hand-0", "fist", "released", 2.9, duration=0.5)],
        timestamp=2.9,
    )

    assert unarmed_move == []

    center_features = recognizer.features_for_frame(make_tracking_frame(make_hand("fist")))
    armed = grammar.update(
        features=center_features,
        events=[PoseEvent("hand-0", "fist", "held", 3.0, duration=0.4)],
        timestamp=3.0,
    )
    move_window = grammar.update(
        features=side_fist_features,
        events=[PoseEvent("hand-0", "fist", "held", 3.2, duration=0.6)],
        timestamp=3.2,
    )

    assert armed == []
    assert move_window[0].request.command == "movetoworkspace"
    assert move_window[0].request.parameters["args"] == ["r-1"]

    grammar.update(
        features=center_features,
        events=[PoseEvent("hand-0", "fist", "entered", 3.4)],
        timestamp=3.4,
    )
    workspace = grammar.update(
        features=recognizer.features_for_frame(
            make_tracking_frame(_move_hand(make_hand("fist"), y=0.44))
        ),
        events=[PoseEvent("hand-0", "fist", "held", 3.6, duration=1.0)],
        timestamp=3.6,
    )

    assert workspace[0].request.command == "workspace"
    assert workspace[0].request.parameters["args"] == ["r-1"]

    close = grammar.update(
        features=side_open_features,
        events=[
            PoseEvent("hand-0", "open_palm", "entered", 4.0),
            PoseEvent("hand-0", "fist", "entered", 4.1),
            PoseEvent("hand-0", "open_palm", "entered", 4.2),
        ],
        timestamp=4.2,
    )

    assert close[0].name == "close_window"
    assert close[0].request.command == "killactive"
    assert close[0].high_risk is True


def test_window_move_arm_expires_and_clears_on_fist_release(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar(ControlGrammarConfig(fist_command_arm_seconds=0.5))
    center_features = recognizer.features_for_frame(make_tracking_frame(make_hand("fist")))
    side_features = recognizer.features_for_frame(
        make_tracking_frame(_move_hand(make_hand("fist"), x=0.70))
    )

    grammar.update(
        features=center_features,
        events=[PoseEvent("hand-0", "fist", "held", 1.0, duration=0.4)],
        timestamp=1.0,
    )
    expired = grammar.update(
        features=side_features,
        events=[PoseEvent("hand-0", "fist", "held", 1.7, duration=0.7)],
        timestamp=1.7,
    )
    grammar.update(
        features=center_features,
        events=[PoseEvent("hand-0", "fist", "held", 2.0, duration=0.4)],
        timestamp=2.0,
    )
    grammar.update(
        features=center_features,
        events=[PoseEvent("hand-0", "fist", "released", 2.1, duration=0.5)],
        timestamp=2.1,
    )
    cleared = grammar.update(
        features=side_features,
        events=[PoseEvent("hand-0", "fist", "held", 2.2, duration=0.6)],
        timestamp=2.2,
    )

    assert expired == []
    assert cleared == []


def test_fist_move_window_uses_horizontal_motion_from_anchor(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar(
        ControlGrammarConfig(
            command_cooldown_seconds=0.0,
            move_window_motion_threshold=0.12,
        )
    )
    center_features = recognizer.features_for_frame(make_tracking_frame(make_hand("fist")))
    side_features = recognizer.features_for_frame(
        make_tracking_frame(_move_hand(make_hand("fist"), x=0.70))
    )

    grammar.update(
        features=center_features,
        events=[PoseEvent("hand-0", "fist", "entered", 1.0)],
        timestamp=1.0,
    )
    move_window = grammar.update(
        features=side_features,
        events=[PoseEvent("hand-0", "fist", "held", 1.2, duration=0.2)],
        timestamp=1.2,
    )
    after_consumed = grammar.update(
        features=side_features,
        events=[PoseEvent("hand-0", "fist", "held", 1.3, duration=0.3)],
        timestamp=1.3,
    )

    assert move_window[0].name == "move_window_r-1"
    assert move_window[0].request.command == "movetoworkspace"
    assert move_window[0].request.parameters["args"] == ["r-1"]
    assert after_consumed == []


def test_fist_motion_ambiguity_does_not_fire_command(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar(
        ControlGrammarConfig(
            command_cooldown_seconds=0.0,
            workspace_motion_threshold=0.10,
            move_window_motion_threshold=0.10,
            fist_axis_margin=0.05,
        )
    )
    center_features = recognizer.features_for_frame(make_tracking_frame(make_hand("fist")))
    diagonal_features = recognizer.features_for_frame(
        make_tracking_frame(_move_hand(make_hand("fist"), x=0.62, y=0.43))
    )

    grammar.update(
        features=center_features,
        events=[PoseEvent("hand-0", "fist", "entered", 1.0)],
        timestamp=1.0,
    )
    intents = grammar.update(
        features=diagonal_features,
        events=[PoseEvent("hand-0", "fist", "held", 1.2, duration=0.2)],
        timestamp=1.2,
    )

    assert intents == []
    assert any("fist motion ambiguous" in item for item in grammar.last_diagnostics)


def test_fist_workspace_motion_fires_once_and_release_clears_anchor(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar(
        ControlGrammarConfig(
            command_cooldown_seconds=0.0,
            fist_command_arm_seconds=0.5,
        )
    )
    center_features = recognizer.features_for_frame(make_tracking_frame(make_hand("fist")))
    top_features = recognizer.features_for_frame(
        make_tracking_frame(_move_hand(make_hand("fist"), y=0.30))
    )

    grammar.update(
        features=center_features,
        events=[PoseEvent("hand-0", "fist", "entered", 1.0)],
        timestamp=1.0,
    )
    workspace = grammar.update(
        features=top_features,
        events=[PoseEvent("hand-0", "fist", "held", 1.2, duration=0.2)],
        timestamp=1.2,
    )
    grammar.update(
        features=center_features,
        events=[PoseEvent("hand-0", "fist", "released", 1.3, duration=0.3)],
        timestamp=1.3,
    )
    grammar.update(
        features=center_features,
        events=[PoseEvent("hand-0", "fist", "entered", 2.0)],
        timestamp=2.0,
    )
    reanchored = grammar.update(
        features=top_features,
        events=[PoseEvent("hand-0", "fist", "held", 2.2, duration=0.6)],
        timestamp=2.2,
    )

    assert workspace[0].request.command == "workspace"
    assert workspace[0].request.parameters["args"] == ["r-1"]
    assert reanchored[0].request.command == "workspace"


def test_control_grammar_holds_left_button_on_index_pinch_hold(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar()
    features = recognizer.features_for_frame(make_tracking_frame(make_hand("pinch")))

    grammar.update(
        features=features,
        events=[PoseEvent("hand-0", "index_pinch", "entered", 1.0)],
        timestamp=1.0,
    )
    button_down = grammar.update(
        features=features,
        events=[PoseEvent("hand-0", "index_pinch", "held", 1.5, duration=0.5)],
        timestamp=1.5,
    )
    button_up = grammar.update(
        features=features,
        events=[PoseEvent("hand-0", "index_pinch", "released", 1.6, duration=0.6)],
        timestamp=1.6,
    )

    assert button_down[0].name == "left_button_down"
    assert button_down[0].request.parameters == {"button": "left", "action": "press"}
    assert button_up[0].name == "left_button_up"
    assert button_up[0].request.parameters == {"button": "left", "action": "release"}


def test_control_grammar_scrolls_on_middle_pinch_hold_and_suppresses_tap(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar(ControlGrammarConfig(scroll_cooldown_seconds=0.0))
    features = recognizer.features_for_frame(
        make_tracking_frame(_middle_pinch_hand(make_hand("open_palm")))
    )

    grammar.update(
        features=features,
        events=[PoseEvent("hand-0", "middle_pinch", "entered", 1.0)],
        timestamp=1.0,
    )
    scroll = grammar.update(
        features=features,
        events=[PoseEvent("hand-0", "middle_pinch", "held", 1.4, duration=0.4)],
        timestamp=1.4,
        scroll_delta_by_hand={"hand-0": -1},
    )
    click = grammar.update(
        features=features,
        events=[PoseEvent("hand-0", "middle_pinch", "released", 1.5, duration=0.5)],
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


def test_control_grammar_cancels_pinch_tap_on_ambiguous_release(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar()
    pinch_features = recognizer.features_for_frame(make_tracking_frame(make_hand("pinch")))
    ambiguous_features = recognizer.features_for_frame(
        make_tracking_frame(_ambiguous_double_pinch_hand(make_hand("open_palm")))
    )

    grammar.update(
        features=pinch_features,
        events=[PoseEvent("hand-0", "index_pinch", "entered", 1.0)],
        timestamp=1.0,
    )
    grammar.update(features=ambiguous_features, events=[], timestamp=1.05)
    click = grammar.update(
        features=ambiguous_features,
        events=[PoseEvent("hand-0", "index_pinch", "released", 1.1, duration=0.1)],
        timestamp=1.1,
    )

    assert click == []
    assert any("suppressed index_pinch tap" in item for item in grammar.last_diagnostics)


def test_control_grammar_cancels_pinch_tap_after_ambiguous_tracking_dropout(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar()
    pinch_features = recognizer.features_for_frame(make_tracking_frame(make_hand("pinch")))
    ambiguous_features = recognizer.features_for_frame(
        make_tracking_frame(_ambiguous_double_pinch_hand(make_hand("open_palm")))
    )

    grammar.update(
        features=pinch_features,
        events=[PoseEvent("hand-0", "index_pinch", "entered", 1.0)],
        timestamp=1.0,
    )
    grammar.update(features=ambiguous_features, events=[], timestamp=1.05)
    click = grammar.update(
        features=[],
        events=[PoseEvent("hand-0", "index_pinch", "released", 1.1, duration=0.1)],
        timestamp=1.1,
    )

    assert click == []


def test_control_grammar_rejects_release_onto_other_pinch(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar()
    pinch_features = recognizer.features_for_frame(make_tracking_frame(make_hand("pinch")))
    middle_features = recognizer.features_for_frame(
        make_tracking_frame(_clean_middle_pinch_hand(make_hand("open_palm")))
    )

    grammar.update(
        features=pinch_features,
        events=[PoseEvent("hand-0", "index_pinch", "entered", 1.0)],
        timestamp=1.0,
    )
    click = grammar.update(
        features=middle_features,
        events=[PoseEvent("hand-0", "index_pinch", "released", 1.1, duration=0.1)],
        timestamp=1.1,
    )

    assert click == []


def test_control_grammar_allows_short_pinch_tap_after_tracking_dropout(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar()
    pinch_features = recognizer.features_for_frame(make_tracking_frame(make_hand("pinch")))

    grammar.update(
        features=pinch_features,
        events=[PoseEvent("hand-0", "index_pinch", "entered", 1.0)],
        timestamp=1.0,
    )
    click = grammar.update(
        features=[],
        events=[PoseEvent("hand-0", "index_pinch", "released", 1.35, duration=0.35)],
        timestamp=1.35,
    )

    assert click[0].name == "left_click"
    assert click[0].request.parameters["button"] == "left"


def test_control_grammar_releases_held_button_after_tracking_dropout(
    make_hand: Callable[[str], NormalizedHand],
    make_tracking_frame: Callable[..., TrackingFrame],
) -> None:
    recognizer = ControlPoseRecognizer()
    grammar = ControlGrammar()
    pinch_features = recognizer.features_for_frame(make_tracking_frame(make_hand("pinch")))

    grammar.update(
        features=pinch_features,
        events=[PoseEvent("hand-0", "index_pinch", "entered", 1.0)],
        timestamp=1.0,
    )
    button_down = grammar.update(
        features=pinch_features,
        events=[PoseEvent("hand-0", "index_pinch", "held", 1.5, duration=0.5)],
        timestamp=1.5,
    )
    button_up = grammar.update(
        features=[],
        events=[PoseEvent("hand-0", "index_pinch", "released", 1.6, duration=0.6)],
        timestamp=1.6,
    )

    assert button_down[0].name == "left_button_down"
    assert button_up[0].name == "left_button_up"
    assert button_up[0].request.parameters == {"button": "left", "action": "release"}


def _move_hand(
    hand: NormalizedHand, *, x: float | None = None, y: float | None = None
) -> NormalizedHand:
    x = hand.palm_center[0] if x is None else x
    y = hand.palm_center[1] if y is None else y
    dx = x - hand.palm_center[0]
    dy = y - hand.palm_center[1]
    landmarks = tuple(
        Landmark(point.x + dx, point.y + dy, point.z, point.visibility, point.presence)
        for point in hand.landmarks.landmarks
    )
    return NormalizedHand(
        hand_id=hand.hand_id,
        landmarks=type(hand.landmarks)(
            landmarks,
            handedness=hand.landmarks.handedness,
            confidence=hand.landmarks.confidence,
        ),
        palm_center=(x, y, hand.palm_center[2]),
        bbox=tuple(
            value + dx if index % 2 == 0 else value + dy
            for index, value in enumerate(hand.bbox)
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


def _clean_middle_pinch_hand(hand: NormalizedHand) -> NormalizedHand:
    landmarks = list(hand.landmarks.landmarks)
    for (mcp, pip, dip, tip), (mcp_x, tip_x) in {
        (5, 6, 7, 8): (0.38, 0.34),
        (9, 10, 11, 12): (0.47, 0.45),
        (13, 14, 15, 16): (0.56, 0.58),
        (17, 18, 19, 20): (0.65, 0.70),
    }.items():
        landmarks[mcp] = Landmark(mcp_x, 0.55, 0.0)
        landmarks[pip] = Landmark((mcp_x + tip_x) / 2, 0.45, 0.0)
        landmarks[dip] = Landmark((mcp_x + tip_x) / 2, 0.35, 0.0)
        landmarks[tip] = Landmark(tip_x, 0.25, 0.0)
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


def _ambiguous_double_pinch_hand(hand: NormalizedHand) -> NormalizedHand:
    landmarks = list(hand.landmarks.landmarks)
    landmarks[4] = Landmark(0.46, 0.30, 0.0)
    landmarks[8] = Landmark(0.46, 0.30, 0.0)
    landmarks[12] = Landmark(0.46, 0.30, 0.0)
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


def _weak_middle_pinch_hand(hand: NormalizedHand) -> NormalizedHand:
    landmarks = list(hand.landmarks.landmarks)
    landmarks[4] = Landmark(0.42, 0.30, 0.0)
    landmarks[12] = Landmark(0.47, 0.30, 0.0)
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


def _relaxed_curled_hand(hand: NormalizedHand) -> NormalizedHand:
    landmarks = list(hand.landmarks.landmarks)
    for index in (8, 12, 16, 20):
        point = landmarks[index]
        landmarks[index] = Landmark(point.x, 0.61, point.z, point.visibility, point.presence)
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


def _partial_curled_hand(hand: NormalizedHand) -> NormalizedHand:
    landmarks = list(hand.landmarks.landmarks)
    for index in (8, 12):
        point = landmarks[index]
        landmarks[index] = Landmark(point.x, 0.66, point.z, point.visibility, point.presence)
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


def _sideways_closed_fist(hand: NormalizedHand) -> NormalizedHand:
    landmarks = list(hand.landmarks.landmarks)
    clustered = {
        (5, 6, 7, 8): (0.39, 0.46),
        (9, 10, 11, 12): (0.47, 0.48),
        (13, 14, 15, 16): (0.55, 0.50),
        (17, 18, 19, 20): (0.63, 0.52),
    }
    for (mcp, pip, dip, tip), (mcp_x, tip_x) in clustered.items():
        landmarks[mcp] = Landmark(mcp_x, 0.55, 0.0)
        landmarks[pip] = Landmark((mcp_x + tip_x) / 2, 0.53, 0.0)
        landmarks[dip] = Landmark(tip_x, 0.54, 0.0)
        landmarks[tip] = Landmark(tip_x, 0.56, 0.0)
    landmarks[4] = Landmark(0.48, 0.55, 0.0)
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


def _pinch_like_fist_artifact(hand: NormalizedHand) -> NormalizedHand:
    landmarks = list(hand.landmarks.landmarks)
    spread_tip_xs = {8: 0.34, 12: 0.45, 16: 0.58, 20: 0.70}
    for index, x in spread_tip_xs.items():
        point = landmarks[index]
        landmarks[index] = Landmark(x, 0.66, point.z, point.visibility, point.presence)
    landmarks[4] = Landmark(0.40, 0.66, 0.0)
    landmarks[8] = Landmark(0.40, 0.66, 0.0)
    landmarks[12] = Landmark(0.40, 0.66, 0.0)
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


def _forming_fist_pinch_artifact(hand: NormalizedHand) -> NormalizedHand:
    landmarks = list(_sideways_closed_fist(hand).landmarks.landmarks)
    landmarks[16] = Landmark(0.60, 0.25, 0.0)
    landmarks[20] = Landmark(0.72, 0.25, 0.0)
    landmarks[4] = Landmark(0.47, 0.55, 0.0)
    landmarks[8] = Landmark(0.47, 0.55, 0.0)
    landmarks[12] = Landmark(0.47, 0.55, 0.0)
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
