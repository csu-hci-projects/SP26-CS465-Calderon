"""Capture backend interfaces."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from airdesk.state.types import CapturedFrame


class CaptureBackend(Protocol):
    """Produces raw or metadata-only captured frames."""

    name: str

    def start(self) -> None:
        """Prepare capture resources."""

    def stop(self) -> None:
        """Release capture resources."""

    def frames(self) -> Iterator[CapturedFrame]:
        """Yield captured frames."""
