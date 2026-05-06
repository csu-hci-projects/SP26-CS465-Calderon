"""Dependency-free feature diagnostics for gesture train/test splits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from airdesk.labels import GestureEventLabel, load_label_file
from airdesk.ml.dataset import load_feature_rows_csv

if TYPE_CHECKING:
    from airdesk.labels import GestureLabelFile


DIAGNOSTIC_NUMERIC_FIELDS = (
    "row_count",
    "tracking_fraction",
    "event_duration_seconds",
    "event_row_count",
    "event_start_relative_seconds",
    "event_end_relative_seconds",
    "event_first_row_offset_seconds",
    "event_last_row_offset_seconds",
    "palm_dx",
    "palm_dy",
    "palm_abs_dx",
    "palm_abs_dy",
    "palm_dx_per_second",
    "palm_dx_per_hand_scale",
    "mean_palm_speed",
    "max_palm_speed",
    "mean_abs_palm_vx",
    "max_abs_palm_vx",
    "mean_abs_palm_ax",
    "max_abs_palm_ax",
    "direction_consistency",
    "mean_hand_scale",
    "mean_confidence",
    "min_confidence",
)


@dataclass(frozen=True)
class FeatureHoldoutSplit:
    """Filename-ordered feature split matching the DTW/TCN holdout shape."""

    train: tuple[Path, ...]
    test: tuple[Path, ...]


@dataclass(frozen=True)
class FeatureFileDiagnostic:
    """Feature/timing summary for one recording-derived CSV."""

    feature_path: str
    label_path: str
    split: str
    group: str
    gesture: str | None
    row_count: int
    source_start_time: float | None
    source_end_time: float | None
    tracking_fraction: float
    event_start_time: float | None
    event_end_time: float | None
    event_duration_seconds: float | None
    event_start_relative_seconds: float | None
    event_end_relative_seconds: float | None
    event_row_count: int
    event_first_row_time: float | None
    event_last_row_time: float | None
    event_first_row_offset_seconds: float | None
    event_last_row_offset_seconds: float | None
    rows_before_event: int
    rows_after_event: int
    palm_dx: float
    palm_dy: float
    palm_abs_dx: float
    palm_abs_dy: float
    palm_dx_per_second: float | None
    palm_dx_per_hand_scale: float | None
    mean_palm_speed: float
    max_palm_speed: float
    mean_abs_palm_vx: float
    max_abs_palm_vx: float
    mean_abs_palm_ax: float
    max_abs_palm_ax: float
    direction_consistency: float | None
    mean_hand_scale: float
    mean_confidence: float
    min_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureDiagnosticsReport:
    """Serializable feature diagnostics report."""

    schema_version: int
    features_dir: str
    labels_dir: str
    split_config: dict[str, int]
    files: tuple[FeatureFileDiagnostic, ...]
    aggregates: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "features_dir": self.features_dir,
            "labels_dir": self.labels_dir,
            "split_config": self.split_config,
            "files": [item.to_dict() for item in self.files],
            "aggregates": self.aggregates,
        }


def build_feature_diagnostics_report(
    *,
    features_dir: Path,
    labels_dir: Path,
    train_per_gesture: int = 6,
    test_per_gesture: int = 2,
    train_negatives: int = 6,
    test_negatives: int = 2,
) -> FeatureDiagnosticsReport:
    """Build deterministic feature diagnostics over the standard holdout split."""
    split = split_feature_holdout(
        features_dir=features_dir,
        labels_dir=labels_dir,
        train_per_gesture=train_per_gesture,
        test_per_gesture=test_per_gesture,
        train_negatives=train_negatives,
        test_negatives=test_negatives,
    )
    diagnostics = tuple(
        diagnose_feature_file(path, labels_dir=labels_dir, split="train")
        for path in split.train
    ) + tuple(
        diagnose_feature_file(path, labels_dir=labels_dir, split="test")
        for path in split.test
    )
    return FeatureDiagnosticsReport(
        schema_version=1,
        features_dir=str(features_dir),
        labels_dir=str(labels_dir),
        split_config={
            "train_per_gesture": train_per_gesture,
            "test_per_gesture": test_per_gesture,
            "train_negatives": train_negatives,
            "test_negatives": test_negatives,
        },
        files=diagnostics,
        aggregates=aggregate_feature_diagnostics(diagnostics),
    )


def save_feature_diagnostics_report(report: FeatureDiagnosticsReport, path: Path) -> None:
    """Write a diagnostics report as stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def split_feature_holdout(
    *,
    features_dir: Path,
    labels_dir: Path,
    train_per_gesture: int,
    test_per_gesture: int,
    train_negatives: int,
    test_negatives: int,
) -> FeatureHoldoutSplit:
    """Split positive gesture groups and negatives by filename order."""
    positives: dict[str, list[Path]] = {}
    negatives: list[Path] = []
    for feature_path in sorted(features_dir.glob("*.csv")):
        label_path = labels_dir / f"{feature_path.stem}.labels.json"
        if not label_path.exists():
            raise ValueError(f"Missing label file for features={feature_path}: {label_path}")
        labels = load_label_file(label_path)
        event = _first_gesture_event(labels)
        if event is None:
            negatives.append(feature_path)
        else:
            positives.setdefault(event.gesture, []).append(feature_path)

    train: list[Path] = []
    test: list[Path] = []
    for gesture in sorted(positives):
        items = positives[gesture]
        needed = train_per_gesture + test_per_gesture
        if len(items) < needed:
            raise ValueError(
                f"Not enough features for gesture={gesture}: need {needed}, found {len(items)}"
            )
        train.extend(items[:train_per_gesture])
        test.extend(items[train_per_gesture:needed])

    if train_negatives + test_negatives > 0:
        needed = train_negatives + test_negatives
        if len(negatives) < needed:
            raise ValueError(f"Not enough negative features: need {needed}, found {len(negatives)}")
        train.extend(negatives[:train_negatives])
        test.extend(negatives[train_negatives:needed])

    if not train:
        raise ValueError("Feature diagnostics require at least one training feature file")
    if not test:
        raise ValueError("Feature diagnostics require at least one test feature file")
    return FeatureHoldoutSplit(train=tuple(train), test=tuple(test))


def diagnose_feature_file(
    feature_path: Path,
    *,
    labels_dir: Path,
    split: str,
) -> FeatureFileDiagnostic:
    """Summarize event-aligned motion and tracking quality for one feature CSV."""
    rows = load_feature_rows_csv(feature_path)
    label_path = labels_dir / f"{feature_path.stem}.labels.json"
    labels = load_label_file(label_path)
    event = _first_gesture_event(labels)
    gesture = event.gesture if event is not None else None
    source_start = rows[0].timestamp if rows else None
    source_end = rows[-1].timestamp if rows else None
    analysis_rows = _event_rows(rows, event) if event is not None else rows
    tracked_rows = [row for row in analysis_rows if row.tracking_present]
    source_tracked_rows = [row for row in rows if row.tracking_present]
    first_event_row = analysis_rows[0] if analysis_rows else None
    last_event_row = analysis_rows[-1] if analysis_rows else None
    mean_hand_scale = _mean([row.hand_scale for row in tracked_rows])
    palm_dx = _delta([row.palm_x for row in tracked_rows])
    palm_dy = _delta([row.palm_y for row in tracked_rows])
    event_duration = (event.end_time - event.start_time) if event is not None else _duration(rows)

    return FeatureFileDiagnostic(
        feature_path=str(feature_path),
        label_path=str(label_path),
        split=split,
        group=gesture or _feature_group(feature_path),
        gesture=gesture,
        row_count=len(rows),
        source_start_time=source_start,
        source_end_time=source_end,
        tracking_fraction=(len(source_tracked_rows) / len(rows)) if rows else 0.0,
        event_start_time=event.start_time if event is not None else None,
        event_end_time=event.end_time if event is not None else None,
        event_duration_seconds=event_duration,
        event_start_relative_seconds=(
            event.start_time - source_start
            if event is not None and source_start is not None
            else None
        ),
        event_end_relative_seconds=(
            event.end_time - source_start
            if event is not None and source_start is not None
            else None
        ),
        event_row_count=len(analysis_rows),
        event_first_row_time=first_event_row.timestamp if first_event_row is not None else None,
        event_last_row_time=last_event_row.timestamp if last_event_row is not None else None,
        event_first_row_offset_seconds=(
            first_event_row.timestamp - event.start_time
            if first_event_row is not None and event is not None
            else None
        ),
        event_last_row_offset_seconds=(
            last_event_row.timestamp - event.end_time
            if last_event_row is not None and event is not None
            else None
        ),
        rows_before_event=_rows_before(rows, event),
        rows_after_event=_rows_after(rows, event),
        palm_dx=palm_dx,
        palm_dy=palm_dy,
        palm_abs_dx=abs(palm_dx),
        palm_abs_dy=abs(palm_dy),
        palm_dx_per_second=(
            (palm_dx / event_duration) if event_duration and event_duration > 0 else None
        ),
        palm_dx_per_hand_scale=(palm_dx / mean_hand_scale) if mean_hand_scale > 0 else None,
        mean_palm_speed=_mean([row.palm_speed for row in tracked_rows]),
        max_palm_speed=_max([row.palm_speed for row in tracked_rows]),
        mean_abs_palm_vx=_mean([abs(row.palm_vx) for row in tracked_rows]),
        max_abs_palm_vx=_max([abs(row.palm_vx) for row in tracked_rows]),
        mean_abs_palm_ax=_mean([abs(row.palm_ax) for row in tracked_rows]),
        max_abs_palm_ax=_max([abs(row.palm_ax) for row in tracked_rows]),
        direction_consistency=_direction_consistency(tracked_rows, palm_dx),
        mean_hand_scale=mean_hand_scale,
        mean_confidence=_mean([row.confidence for row in tracked_rows]),
        min_confidence=_min([row.confidence for row in tracked_rows]),
    )


def aggregate_feature_diagnostics(
    diagnostics: tuple[FeatureFileDiagnostic, ...],
) -> dict[str, dict[str, Any]]:
    """Aggregate diagnostics by split and gesture/group."""
    groups: dict[str, list[FeatureFileDiagnostic]] = {}
    for item in diagnostics:
        groups.setdefault(f"{item.split}:{item.group}", []).append(item)

    return {
        key: {
            "count": len(items),
            "metrics": {
                field: _summary([getattr(item, field) for item in items])
                for field in DIAGNOSTIC_NUMERIC_FIELDS
            },
        }
        for key, items in sorted(groups.items())
    }


def _first_gesture_event(labels: GestureLabelFile) -> GestureEventLabel | None:
    return next((event for event in labels.event_labels if event.label_type == "gesture"), None)


def _event_rows(rows: list[Any], event: GestureEventLabel | None) -> list[Any]:
    if event is None:
        return rows
    return [row for row in rows if event.start_time <= row.timestamp <= event.end_time]


def _feature_group(feature_path: Path) -> str:
    stem = feature_path.stem
    parts = stem.rsplit("-", maxsplit=1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def _duration(rows: list[Any]) -> float | None:
    if len(rows) < 2:
        return None
    return rows[-1].timestamp - rows[0].timestamp


def _rows_before(rows: list[Any], event: GestureEventLabel | None) -> int:
    if event is None:
        return 0
    return sum(1 for row in rows if row.timestamp < event.start_time)


def _rows_after(rows: list[Any], event: GestureEventLabel | None) -> int:
    if event is None:
        return 0
    return sum(1 for row in rows if row.timestamp > event.end_time)


def _delta(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return values[-1] - values[0]


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _max(values: list[float]) -> float:
    return max(values) if values else 0.0


def _min(values: list[float]) -> float:
    return min(values) if values else 0.0


def _direction_consistency(rows: list[Any], palm_dx: float) -> float | None:
    if len(rows) < 2 or palm_dx == 0:
        return None
    expected_sign = 1 if palm_dx > 0 else -1
    signed_steps = [
        rows[index].palm_x - rows[index - 1].palm_x
        for index in range(1, len(rows))
    ]
    moving_steps = [step for step in signed_steps if step != 0]
    if not moving_steps:
        return None
    aligned = sum(1 for step in moving_steps if (1 if step > 0 else -1) == expected_sign)
    return aligned / len(moving_steps)


def _summary(values: list[float | int | None]) -> dict[str, float | int | None]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": fmean(present),
        "min": min(present),
        "max": max(present),
    }
