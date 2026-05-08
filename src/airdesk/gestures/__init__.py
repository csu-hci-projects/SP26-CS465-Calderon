"""Gesture recognizers."""

from airdesk.gestures.base import CompositeGestureRecognizer, GestureRecognizer
from airdesk.gestures.decoder import (
    DecoderFrame,
    EventDecoder,
    EventDecoderConfig,
    frames_from_candidates,
)
from airdesk.gestures.motion import (
    MotionEventConfig,
    MotionEventRecognizer,
    diagnose_motion_rows,
)
from airdesk.gestures.phrases import (
    GesturePhase,
    IntentGatedSwipeRecognizer,
    PhraseRecognizerConfig,
)
from airdesk.gestures.primitives import StaticHandPoseRecognizer

__all__ = [
    "CompositeGestureRecognizer",
    "DecoderFrame",
    "EventDecoder",
    "EventDecoderConfig",
    "GesturePhase",
    "GestureRecognizer",
    "IntentGatedSwipeRecognizer",
    "MotionEventConfig",
    "MotionEventRecognizer",
    "PhraseRecognizerConfig",
    "StaticHandPoseRecognizer",
    "diagnose_motion_rows",
    "frames_from_candidates",
]
