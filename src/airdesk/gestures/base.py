"""Gesture recognizer interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from airdesk.state.types import GestureCandidate, TrackingFrame


class GestureRecognizer(Protocol):
    """Produces gesture candidates from normalized tracking frames."""

    name: str

    def recognize(self, frame: TrackingFrame) -> Sequence[GestureCandidate]:
        """Return gesture candidates for a tracking frame."""
