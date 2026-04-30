"""Hand tracking backend interface."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from airdesk.state.types import TrackingFrame


class HandTrackerBackend(Protocol):
    """Produces normalized tracking frames from live, mock, or recorded sources."""

    name: str

    def start(self) -> None:
        """Prepare tracking resources."""

    def stop(self) -> None:
        """Release tracking resources."""

    def frames(self) -> Iterator[TrackingFrame]:
        """Yield normalized tracking frames."""
