"""Shared helpers for hand-scoped feature row streams."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

NO_HAND_STREAM_ID = "__no_hand__"


class FeatureStreamRow(Protocol):
    """Structural row shape needed for stream grouping."""

    tracking_present: int
    hand_id: str


def is_tracked_feature_row(row: FeatureStreamRow) -> bool:
    """Return true when a feature row belongs to a visible hand stream."""
    return row.tracking_present == 1 and bool(row.hand_id)


def feature_stream_id(row: FeatureStreamRow) -> str:
    """Return the stable stream id for a tracked or no-hand feature row."""
    if is_tracked_feature_row(row):
        return row.hand_id
    return NO_HAND_STREAM_ID


def group_feature_rows_by_stream[RowT: FeatureStreamRow](
    rows: Iterable[RowT],
    *,
    include_no_hand: bool = False,
) -> list[list[RowT]]:
    """Group feature rows into sorted hand streams, optionally including no-hand rows."""
    streams: dict[str, list[RowT]] = {}
    for row in rows:
        if not include_no_hand and not is_tracked_feature_row(row):
            continue
        streams.setdefault(feature_stream_id(row), []).append(row)
    return [streams[key] for key in sorted(streams)]


def group_indexed_feature_rows_by_stream[RowT: FeatureStreamRow](
    rows: Iterable[RowT],
    *,
    include_no_hand: bool = False,
) -> list[list[tuple[int, RowT]]]:
    """Group indexed feature rows into sorted hand/no-hand streams."""
    streams: dict[str, list[tuple[int, RowT]]] = {}
    for index, row in enumerate(rows):
        if not include_no_hand and not is_tracked_feature_row(row):
            continue
        streams.setdefault(feature_stream_id(row), []).append((index, row))
    return [streams[key] for key in sorted(streams)]
