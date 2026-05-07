from __future__ import annotations

from airdesk.gestures.decoder import DecoderFrame, EventDecoder, EventDecoderConfig
from airdesk.state.types import GestureCandidate


def test_event_decoder_commits_peak_after_recovery() -> None:
    decoder = EventDecoder(
        EventDecoderConfig(
            activation_threshold=0.6,
            release_threshold=0.4,
            min_peak_confidence=0.7,
            recovery_seconds=0.1,
            min_event_separation_seconds=0.5,
        )
    )

    events = decoder.decode(
        [
            DecoderFrame(timestamp=1.0, scores={"background": 0.8, "stroke_left": 0.2}),
            DecoderFrame(timestamp=1.1, scores={"background": 0.3, "stroke_left": 0.65}),
            DecoderFrame(timestamp=1.2, scores={"background": 0.1, "stroke_left": 0.90}),
            DecoderFrame(timestamp=1.4, scores={"background": 0.8, "stroke_left": 0.2}),
        ]
    )

    assert len(events) == 1
    assert events[0].name == "swipe_left"
    assert events[0].timestamp == 1.2
    assert events[0].confidence == 0.9


def test_event_decoder_suppresses_repeated_fire_within_separation() -> None:
    decoder = EventDecoder(
        EventDecoderConfig(
            activation_threshold=0.6,
            release_threshold=0.4,
            min_peak_confidence=0.6,
            recovery_seconds=0.0,
            min_event_separation_seconds=0.5,
        )
    )

    events = decoder.decode(
        [
            DecoderFrame(timestamp=1.0, scores={"stroke_right": 0.8}),
            DecoderFrame(timestamp=1.1, scores={"background": 0.9}),
            DecoderFrame(timestamp=1.2, scores={"stroke_right": 0.9}),
            DecoderFrame(timestamp=1.3, scores={"background": 0.9}),
            DecoderFrame(timestamp=1.7, scores={"stroke_right": 0.85}),
            DecoderFrame(timestamp=1.8, scores={"background": 0.9}),
        ]
    )

    assert [event.timestamp for event in events] == [1.0, 1.7]
    assert [event.name for event in events] == ["swipe_right", "swipe_right"]


def test_event_decoder_accepts_candidate_score_frames() -> None:
    decoder = EventDecoder(EventDecoderConfig(recovery_seconds=0.0))
    candidate = GestureCandidate(name="swipe_left", confidence=0.8, timestamp=2.0)

    events = decoder.decode(
        [DecoderFrame(timestamp=candidate.timestamp, scores={candidate.name: candidate.confidence})]
    )

    assert len(events) == 1
    assert events[0].name == "swipe_left"
