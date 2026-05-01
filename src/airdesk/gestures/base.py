"""Gesture recognizer interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from airdesk.state.types import GestureCandidate, TrackingFrame


class GestureRecognizer(Protocol):
    """Produces gesture candidates from normalized tracking frames."""

    name: str

    def recognize(self, frame: TrackingFrame) -> Sequence[GestureCandidate]:
        """Return gesture candidates for a tracking frame."""


@dataclass
class CompositeGestureRecognizer:
    """Runs multiple recognizers and returns one candidate stream."""

    recognizers: Sequence[GestureRecognizer] = field(default_factory=tuple)
    name: str = "composite"

    def recognize(self, frame: TrackingFrame) -> list[GestureCandidate]:
        candidates: list[GestureCandidate] = []
        for recognizer in self.recognizers:
            candidates.extend(recognizer.recognize(frame))
        return candidates
