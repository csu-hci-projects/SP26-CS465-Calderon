"""Dependency-free dataset manifest builder for causal TCN experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from airdesk.features import FrameFeatureRow
from airdesk.labels import GestureLabelFile, load_label_file

TCN_EVENT_TARGETS = ("background", "swipe_left", "swipe_right")
TCN_PHASE_TARGETS = ("background", "stroke_left", "stroke_right", "recovery")
TCN_PHASE_STROKE_TARGETS = ("background", "stroke_left", "stroke_right")
TCN_V2_EVIDENCE_TARGETS = (
    "intentional_motion",
    "stroke_left",
    "stroke_right",
    "start",
    "end",
)
TCN_V2_WINDOW_TARGETS = ("background",) + TCN_V2_EVIDENCE_TARGETS
TCN_TARGETS = TCN_EVENT_TARGETS
TCN_LEGACY_FEATURE_COLUMNS = (
    "dt",
    "tracking_present",
    "hand_count",
    "confidence",
    "palm_x",
    "palm_y",
    "palm_z",
    "palm_vx",
    "palm_vy",
    "palm_speed",
    "palm_ax",
    "palm_ay",
    "palm_window_dx",
    "palm_window_dx_per_hand_scale",
    "palm_window_peak_abs_vx",
    "palm_window_direction_consistency",
    "index_rel_x",
    "index_rel_y",
    "index_rel_vx",
    "index_rel_vy",
    "pinch_distance",
    "pinch_velocity",
    "hand_scale",
    "extended_fingers",
    "folded_fingers",
)
TCN_STREAM_INVARIANT_FEATURE_COLUMNS = (
    "dt",
    "tracking_present",
    "hand_count",
    "confidence",
    "palm_vx",
    "palm_vy",
    "palm_speed",
    "palm_ax",
    "palm_ay",
    "palm_window_dx",
    "palm_window_dx_per_hand_scale",
    "palm_window_peak_abs_vx",
    "palm_window_direction_consistency",
    "index_rel_x",
    "index_rel_y",
    "index_rel_vx",
    "index_rel_vy",
    "pinch_distance",
    "pinch_velocity",
    "hand_scale",
    "extended_fingers",
    "folded_fingers",
)
TCN_FEATURE_COLUMNS = TCN_LEGACY_FEATURE_COLUMNS
TCN_FEATURE_PRESETS = {
    "legacy": TCN_LEGACY_FEATURE_COLUMNS,
    "stream-invariant": TCN_STREAM_INVARIANT_FEATURE_COLUMNS,
}
TCN_TARGET_MODES = ("event", "phase", "phase-stroke", "v2-evidence")
TCN_TARGET_ASSIGNMENTS = ("label", "motion-gated")
DEFAULT_MOTION_GATE_MIN_DX_PER_HAND_SCALE = 0.35
DEFAULT_MOTION_GATE_MIN_DIRECTION_CONSISTENCY = 0.45

_INT_FIELDS = {
    "frame_index",
    "tracking_present",
    "hand_count",
    "extended_fingers",
    "folded_fingers",
}
_FLOAT_FIELDS = {
    "timestamp",
    "dt",
    "confidence",
    "palm_x",
    "palm_y",
    "palm_z",
    "palm_vx",
    "palm_vy",
    "palm_speed",
    "palm_ax",
    "palm_ay",
    "palm_window_dx",
    "palm_window_dx_per_hand_scale",
    "palm_window_peak_abs_vx",
    "palm_window_direction_consistency",
    "index_rel_x",
    "index_rel_y",
    "index_rel_vx",
    "index_rel_vy",
    "pinch_distance",
    "pinch_velocity",
    "hand_scale",
}


@dataclass(frozen=True)
class TcnFeatureSource:
    """One exported feature sequence used by a TCN dataset manifest."""

    feature_path: str
    label_path: str | None
    row_count: int
    start_time: float | None
    end_time: float | None
    duration_seconds: float
    target_frame_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TcnFeatureSource:
        return cls(
            feature_path=str(data["feature_path"]),
            label_path=str(data["label_path"]) if data.get("label_path") is not None else None,
            row_count=int(data["row_count"]),
            start_time=(
                float(data["start_time"]) if data.get("start_time") is not None else None
            ),
            end_time=float(data["end_time"]) if data.get("end_time") is not None else None,
            duration_seconds=float(data["duration_seconds"]),
            target_frame_counts={
                str(target): int(count)
                for target, count in data.get("target_frame_counts", {}).items()
            },
        )


@dataclass(frozen=True)
class TcnWindowSample:
    """One deterministic sliding-window training sample."""

    sample_id: str
    feature_path: str
    label_path: str | None
    start_row: int
    end_row: int
    start_time: float
    end_time: float
    row_count: int
    target: str
    target_index: int
    target_frame_counts: dict[str, int]
    hand_id: str = ""
    frame_targets: tuple[dict[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TcnWindowSample:
        return cls(
            sample_id=str(data["sample_id"]),
            feature_path=str(data["feature_path"]),
            label_path=str(data["label_path"]) if data.get("label_path") is not None else None,
            hand_id=str(data.get("hand_id", "")),
            start_row=int(data["start_row"]),
            end_row=int(data["end_row"]),
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            row_count=int(data["row_count"]),
            target=str(data["target"]),
            target_index=int(data["target_index"]),
            target_frame_counts={
                str(target): int(count)
                for target, count in data.get("target_frame_counts", {}).items()
            },
            frame_targets=tuple(
                {str(target): float(value) for target, value in frame.items()}
                for frame in data.get("frame_targets", ())
            ),
        )


@dataclass(frozen=True)
class TcnDatasetManifest:
    """Serializable manifest for dependency-free TCN window construction."""

    schema_version: int
    targets: tuple[str, ...]
    feature_columns: tuple[str, ...]
    window_seconds: float
    stride_seconds: float
    min_rows: int
    min_gesture_fraction: float
    sources: tuple[TcnFeatureSource, ...]
    windows: tuple[TcnWindowSample, ...]
    feature_preset: str = "legacy"
    target_mode: str = "event"
    target_assignment: str = "label"
    motion_gate_min_dx_per_hand_scale: float = DEFAULT_MOTION_GATE_MIN_DX_PER_HAND_SCALE
    motion_gate_min_direction_consistency: float = DEFAULT_MOTION_GATE_MIN_DIRECTION_CONSISTENCY
    evidence_targets: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "targets": list(self.targets),
            "feature_columns": list(self.feature_columns),
            "window_seconds": self.window_seconds,
            "stride_seconds": self.stride_seconds,
            "min_rows": self.min_rows,
            "min_gesture_fraction": self.min_gesture_fraction,
            "feature_preset": self.feature_preset,
            "target_mode": self.target_mode,
            "target_assignment": self.target_assignment,
            "evidence_targets": list(self.evidence_targets),
            "motion_gate_min_dx_per_hand_scale": self.motion_gate_min_dx_per_hand_scale,
            "motion_gate_min_direction_consistency": (
                self.motion_gate_min_direction_consistency
            ),
            "sources": [source.to_dict() for source in self.sources],
            "windows": [window.to_dict() for window in self.windows],
            "summary": summarize_tcn_manifest(self),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TcnDatasetManifest:
        return cls(
            schema_version=int(data["schema_version"]),
            targets=tuple(str(target) for target in data["targets"]),
            feature_columns=tuple(str(column) for column in data["feature_columns"]),
            window_seconds=float(data["window_seconds"]),
            stride_seconds=float(data["stride_seconds"]),
            min_rows=int(data["min_rows"]),
            min_gesture_fraction=float(data["min_gesture_fraction"]),
            sources=tuple(TcnFeatureSource.from_dict(item) for item in data.get("sources", [])),
            windows=tuple(TcnWindowSample.from_dict(item) for item in data.get("windows", [])),
            feature_preset=str(data.get("feature_preset", "legacy")),
            target_mode=str(data.get("target_mode", "event")),
            target_assignment=str(data.get("target_assignment", "label")),
            evidence_targets=tuple(str(target) for target in data.get("evidence_targets", ())),
            motion_gate_min_dx_per_hand_scale=float(
                data.get(
                    "motion_gate_min_dx_per_hand_scale",
                    DEFAULT_MOTION_GATE_MIN_DX_PER_HAND_SCALE,
                )
            ),
            motion_gate_min_direction_consistency=float(
                data.get(
                    "motion_gate_min_direction_consistency",
                    DEFAULT_MOTION_GATE_MIN_DIRECTION_CONSISTENCY,
                )
            ),
        )


def load_feature_rows_csv(path: Path) -> list[FrameFeatureRow]:
    """Load exported AirDesk CSV feature rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [_feature_row_from_csv(row) for row in reader]


def load_tcn_dataset_manifest(path: Path) -> TcnDatasetManifest:
    """Load a TCN dataset manifest from stable JSON."""
    with path.open(encoding="utf-8") as handle:
        return TcnDatasetManifest.from_dict(json.load(handle))


def build_tcn_dataset_manifest(
    feature_paths: list[Path],
    *,
    labels_dir: Path | None = None,
    window_seconds: float = 0.8,
    stride_seconds: float = 0.2,
    min_rows: int = 4,
    min_gesture_fraction: float = 0.35,
    targets: tuple[str, ...] = TCN_TARGETS,
    feature_columns: tuple[str, ...] | None = None,
    feature_preset: str = "legacy",
    target_mode: str = "event",
    target_assignment: str = "label",
    motion_gate_min_dx_per_hand_scale: float = DEFAULT_MOTION_GATE_MIN_DX_PER_HAND_SCALE,
    motion_gate_min_direction_consistency: float = DEFAULT_MOTION_GATE_MIN_DIRECTION_CONSISTENCY,
) -> TcnDatasetManifest:
    """Build deterministic sliding-window metadata over exported feature CSVs."""
    resolved_feature_columns = feature_columns or feature_columns_for_preset(feature_preset)
    resolved_targets = targets_for_mode(target_mode) if targets == TCN_TARGETS else targets
    evidence_targets = TCN_V2_EVIDENCE_TARGETS if target_mode == "v2-evidence" else ()
    _validate_manifest_config(
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        min_rows=min_rows,
        min_gesture_fraction=min_gesture_fraction,
        targets=resolved_targets,
        feature_preset=feature_preset,
        target_mode=target_mode,
        target_assignment=target_assignment,
        motion_gate_min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
        motion_gate_min_direction_consistency=motion_gate_min_direction_consistency,
    )
    sources: list[TcnFeatureSource] = []
    windows: list[TcnWindowSample] = []
    for feature_path in sorted(feature_paths):
        rows = load_feature_rows_csv(feature_path)
        label_path, labels = _matching_labels(feature_path, labels_dir)
        evidence_by_row = (
            _v2_evidence_targets_for_rows(
                rows,
                labels,
                target_assignment=target_assignment,
                motion_gate_min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
                motion_gate_min_direction_consistency=motion_gate_min_direction_consistency,
            )
            if target_mode == "v2-evidence"
            else None
        )
        target_by_row = (
            [_v2_window_target_from_evidence(evidence) for evidence in evidence_by_row]
            if evidence_by_row is not None
            else [
                _target_for_row(
                    row,
                    labels,
                    resolved_targets,
                    target_mode=target_mode,
                    target_assignment=target_assignment,
                    motion_gate_min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
                    motion_gate_min_direction_consistency=(
                        motion_gate_min_direction_consistency
                    ),
                )
                for row in rows
            ]
        )
        sources.append(
            TcnFeatureSource(
                feature_path=str(feature_path),
                label_path=str(label_path) if label_path is not None else None,
                row_count=len(rows),
                start_time=rows[0].timestamp if rows else None,
                end_time=rows[-1].timestamp if rows else None,
                duration_seconds=_duration(rows),
                target_frame_counts=_target_counts(target_by_row, resolved_targets),
            )
        )
        for stream_rows in _feature_streams(rows):
            windows.extend(
                _windows_for_source(
                    stream_rows,
                    target_by_row=target_by_row,
                    feature_path=feature_path,
                    label_path=label_path,
                    window_seconds=window_seconds,
                    stride_seconds=stride_seconds,
                    min_rows=min_rows,
                    min_gesture_fraction=min_gesture_fraction,
                    targets=resolved_targets,
                    evidence_by_row=evidence_by_row,
                    sample_offset=len(windows),
                )
            )

    return TcnDatasetManifest(
        schema_version=1,
        targets=resolved_targets,
        feature_columns=resolved_feature_columns,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        min_rows=min_rows,
        min_gesture_fraction=min_gesture_fraction,
        sources=tuple(sources),
        windows=tuple(windows),
        feature_preset=feature_preset,
        target_mode=target_mode,
        target_assignment=target_assignment,
        motion_gate_min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
        motion_gate_min_direction_consistency=motion_gate_min_direction_consistency,
        evidence_targets=evidence_targets,
    )


def feature_columns_for_preset(feature_preset: str) -> tuple[str, ...]:
    """Return numeric feature columns for a named TCN/stream-model preset."""
    try:
        return TCN_FEATURE_PRESETS[feature_preset]
    except KeyError as exc:
        options = ", ".join(sorted(TCN_FEATURE_PRESETS))
        message = f"unsupported feature_preset={feature_preset}; use one of: {options}"
        raise ValueError(message) from exc


def targets_for_mode(target_mode: str) -> tuple[str, ...]:
    """Return default targets for a dataset target mode."""
    if target_mode == "event":
        return TCN_EVENT_TARGETS
    if target_mode == "phase":
        return TCN_PHASE_TARGETS
    if target_mode == "phase-stroke":
        return TCN_PHASE_STROKE_TARGETS
    if target_mode == "v2-evidence":
        return TCN_V2_WINDOW_TARGETS
    options = ", ".join(TCN_TARGET_MODES)
    raise ValueError(f"unsupported target_mode={target_mode}; use one of: {options}")


def target_assignments() -> tuple[str, ...]:
    """Return supported weak-label assignment strategies for TCN manifests."""
    return TCN_TARGET_ASSIGNMENTS


def save_tcn_dataset_manifest(manifest: TcnDatasetManifest, path: Path) -> None:
    """Write a TCN dataset manifest as stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def feature_window_matrix(
    window: TcnWindowSample,
    *,
    feature_columns: tuple[str, ...],
) -> list[list[float]]:
    """Load numeric feature values for one manifest window."""
    rows = load_feature_rows_csv(Path(window.feature_path))[window.start_row : window.end_row]
    if window.hand_id:
        rows = [row for row in rows if row.hand_id == window.hand_id]
    return [[_numeric_feature_value(row, column) for column in feature_columns] for row in rows]


def feature_window_frame_targets(
    window: TcnWindowSample,
    *,
    evidence_targets: tuple[str, ...],
) -> list[list[float]]:
    """Return multi-label evidence targets aligned to one manifest window."""
    if not window.frame_targets:
        raise ValueError("TCN v2 window is missing frame evidence targets")
    return [
        [float(frame.get(target, 0.0)) for target in evidence_targets]
        for frame in window.frame_targets
    ]


def summarize_tcn_manifest(manifest: TcnDatasetManifest) -> dict[str, Any]:
    """Return compact count totals for display and JSON exports."""
    window_counts = {target: 0 for target in manifest.targets}
    frame_counts = {target: 0 for target in manifest.targets}
    for source in manifest.sources:
        for target, count in source.target_frame_counts.items():
            frame_counts[target] += count
    for window in manifest.windows:
        window_counts[window.target] += 1
    return {
        "source_count": len(manifest.sources),
        "window_count": len(manifest.windows),
        "window_counts": window_counts,
        "target_frame_counts": frame_counts,
    }


def _feature_row_from_csv(row: dict[str, str]) -> FrameFeatureRow:
    values: dict[str, Any] = {}
    for field in fields(FrameFeatureRow):
        raw = row.get(field.name, "")
        if field.name in _INT_FIELDS:
            values[field.name] = int(raw or 0)
        elif field.name in _FLOAT_FIELDS:
            values[field.name] = float(raw or 0.0)
        else:
            values[field.name] = raw
    return FrameFeatureRow(**values)


def _numeric_feature_value(row: FrameFeatureRow, column: str) -> float:
    value = getattr(row, column)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"TCN feature column must be numeric: {column}")


def _matching_labels(
    feature_path: Path,
    labels_dir: Path | None,
) -> tuple[Path | None, GestureLabelFile | None]:
    if labels_dir is None:
        return None, None
    label_path = labels_dir / f"{feature_path.stem}.labels.json"
    if not label_path.exists():
        raise ValueError(f"Missing label file for features={feature_path}: {label_path}")
    return label_path, load_label_file(label_path)


def _target_for_row(
    row: FrameFeatureRow,
    labels: GestureLabelFile | None,
    targets: tuple[str, ...],
    *,
    target_mode: str,
    target_assignment: str,
    motion_gate_min_dx_per_hand_scale: float,
    motion_gate_min_direction_consistency: float,
) -> str:
    if target_mode in {"phase", "phase-stroke"}:
        phase = (
            row.phase
            if row.phase and row.phase != "background"
            else _phase_at(labels, row.timestamp)
        )
        if phase in {"stroke_left", "stroke_right"} and phase in targets:
            return _apply_target_assignment(
                row,
                phase,
                target_assignment=target_assignment,
                motion_gate_min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
                motion_gate_min_direction_consistency=motion_gate_min_direction_consistency,
            )
        if phase in {"recovery", "reset", "release", "cooldown"} and "recovery" in targets:
            return "recovery"
        return "background"
    event = row.event or _event_at(labels, row.timestamp)
    if target_mode == "v2-evidence":
        evidence = _v2_evidence_for_row(
            row,
            labels,
            target_assignment=target_assignment,
            motion_gate_min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
            motion_gate_min_direction_consistency=motion_gate_min_direction_consistency,
        )
        if evidence["stroke_left"]:
            return "stroke_left"
        if evidence["stroke_right"]:
            return "stroke_right"
        if evidence["start"]:
            return "start"
        if evidence["end"]:
            return "end"
        if evidence["intentional_motion"]:
            return "intentional_motion"
        return "background"
    if event in targets and event != "background":
        return _apply_target_assignment(
            row,
            event,
            target_assignment=target_assignment,
            motion_gate_min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
            motion_gate_min_direction_consistency=motion_gate_min_direction_consistency,
        )
    return "background"


def _apply_target_assignment(
    row: FrameFeatureRow,
    target: str,
    *,
    target_assignment: str,
    motion_gate_min_dx_per_hand_scale: float,
    motion_gate_min_direction_consistency: float,
) -> str:
    if target_assignment == "label":
        return target
    if target_assignment == "motion-gated":
        if _row_motion_matches_target(
            row,
            target,
            min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
            min_direction_consistency=motion_gate_min_direction_consistency,
        ):
            return target
        return "background"
    options = ", ".join(TCN_TARGET_ASSIGNMENTS)
    raise ValueError(f"unsupported target_assignment={target_assignment}; use one of: {options}")


def _row_motion_matches_target(
    row: FrameFeatureRow,
    target: str,
    *,
    min_dx_per_hand_scale: float,
    min_direction_consistency: float,
) -> bool:
    if row.tracking_present != 1:
        return False
    if row.hand_scale <= 0:
        return False
    if row.palm_window_direction_consistency < min_direction_consistency:
        return False
    dx = abs(row.palm_window_dx_per_hand_scale)
    if target in {"swipe_left", "stroke_left", "swipe_right", "stroke_right"}:
        return dx >= min_dx_per_hand_scale
    return True


def _event_at(labels: GestureLabelFile | None, timestamp: float) -> str:
    if labels is None:
        return ""
    for event in labels.event_labels:
        if event.label_type == "gesture" and event.start_time <= timestamp <= event.end_time:
            return event.gesture
    return ""


def _v2_evidence_targets_for_rows(
    rows: list[FrameFeatureRow],
    labels: GestureLabelFile | None,
    *,
    target_assignment: str,
    motion_gate_min_dx_per_hand_scale: float,
    motion_gate_min_direction_consistency: float,
) -> list[dict[str, float]]:
    evidence = [
        _v2_evidence_for_row(
            row,
            labels,
            target_assignment=target_assignment,
            motion_gate_min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
            motion_gate_min_direction_consistency=motion_gate_min_direction_consistency,
        )
        for row in rows
    ]
    if labels is None or not rows:
        return evidence
    for event in labels.event_labels:
        if event.label_type != "gesture":
            continue
        candidate_indices = [
            index
            for index, frame_evidence in enumerate(evidence)
            if frame_evidence["intentional_motion"] > 0
        ]
        if not candidate_indices:
            continue
        start_index = min(
            candidate_indices,
            key=lambda index: abs(rows[index].timestamp - event.start_time),
        )
        end_index = min(
            candidate_indices,
            key=lambda index: abs(rows[index].timestamp - event.end_time),
        )
        evidence[start_index]["start"] = 1.0
        evidence[end_index]["end"] = 1.0
    return evidence


def _v2_evidence_for_row(
    row: FrameFeatureRow,
    labels: GestureLabelFile | None,
    *,
    target_assignment: str,
    motion_gate_min_dx_per_hand_scale: float,
    motion_gate_min_direction_consistency: float,
) -> dict[str, float]:
    phase = (
        row.phase
        if row.phase and row.phase != "background"
        else _phase_at(labels, row.timestamp)
    )
    event = row.event or _event_at(labels, row.timestamp)
    stroke_left = phase == "stroke_left" or event == "swipe_left"
    stroke_right = phase == "stroke_right" or event == "swipe_right"
    if target_assignment == "motion-gated":
        if stroke_left and not _row_motion_matches_target(
            row,
            "stroke_left",
            min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
            min_direction_consistency=motion_gate_min_direction_consistency,
        ):
            stroke_left = False
        if stroke_right and not _row_motion_matches_target(
            row,
            "stroke_right",
            min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
            min_direction_consistency=motion_gate_min_direction_consistency,
        ):
            stroke_right = False
    intentional_motion = bool(
        stroke_left
        or stroke_right
        or phase in {"recovery", "reset", "release", "cooldown"}
    )
    return {
        "intentional_motion": 1.0 if intentional_motion else 0.0,
        "stroke_left": 1.0 if stroke_left else 0.0,
        "stroke_right": 1.0 if stroke_right else 0.0,
        "start": 0.0,
        "end": 0.0,
    }


def _v2_window_target_from_evidence(evidence: dict[str, float]) -> str:
    for target in ("stroke_left", "stroke_right", "start", "end", "intentional_motion"):
        if evidence.get(target, 0.0) > 0:
            return target
    return "background"


def _phase_at(labels: GestureLabelFile | None, timestamp: float) -> str:
    if labels is None:
        return ""
    fallback = ""
    for phase in labels.phase_labels:
        if phase.start_time <= timestamp <= phase.end_time:
            if phase.phase != "background":
                return phase.phase
            fallback = phase.phase
    return fallback


def _windows_for_source(
    indexed_rows: list[tuple[int, FrameFeatureRow]],
    *,
    target_by_row: list[str],
    feature_path: Path,
    label_path: Path | None,
    window_seconds: float,
    stride_seconds: float,
    min_rows: int,
    min_gesture_fraction: float,
    targets: tuple[str, ...],
    evidence_by_row: list[dict[str, float]] | None,
    sample_offset: int,
) -> list[TcnWindowSample]:
    rows = [row for _index, row in indexed_rows]
    if not rows:
        return []
    windows: list[TcnWindowSample] = []
    start_index = 0
    while start_index < len(rows):
        start_time = rows[start_index].timestamp
        end_time = start_time + window_seconds
        if end_time > rows[-1].timestamp + 1e-9:
            break
        end_index = start_index
        while end_index < len(rows) and rows[end_index].timestamp <= end_time + 1e-9:
            end_index += 1
        if end_index - start_index >= min_rows:
            window_targets = [
                target_by_row[original_index]
                for original_index, _row in indexed_rows[start_index:end_index]
            ]
            target_counts = _target_counts(window_targets, targets)
            frame_targets = (
                tuple(
                    evidence_by_row[original_index]
                    for original_index, _row in indexed_rows[start_index:end_index]
                )
                if evidence_by_row is not None
                else ()
            )
            target, target_index = _window_target(
                target_counts,
                row_count=end_index - start_index,
                targets=targets,
                min_gesture_fraction=min_gesture_fraction,
            )
            sample_number = sample_offset + len(windows) + 1
            windows.append(
                TcnWindowSample(
                    sample_id=f"window-{sample_number:06d}",
                    feature_path=str(feature_path),
                    label_path=str(label_path) if label_path is not None else None,
                    hand_id=rows[start_index].hand_id,
                    start_row=indexed_rows[start_index][0],
                    end_row=indexed_rows[end_index - 1][0] + 1,
                    start_time=start_time,
                    end_time=rows[end_index - 1].timestamp,
                    row_count=end_index - start_index,
                    target=target,
                    target_index=target_index,
                    target_frame_counts=target_counts,
                    frame_targets=frame_targets,
                )
            )
        next_start_time = start_time + stride_seconds
        next_index = start_index + 1
        while next_index < len(rows) and rows[next_index].timestamp < next_start_time - 1e-9:
            next_index += 1
        if next_index <= start_index:
            next_index = start_index + 1
        start_index = next_index
    return windows


def _feature_streams(rows: list[FrameFeatureRow]) -> list[list[tuple[int, FrameFeatureRow]]]:
    """Split flat feature rows into independent hand/no-hand streams."""
    streams: dict[str, list[tuple[int, FrameFeatureRow]]] = {}
    for index, row in enumerate(rows):
        stream_id = row.hand_id if row.tracking_present else "__no_hand__"
        streams.setdefault(stream_id, []).append((index, row))
    return [streams[key] for key in sorted(streams)]


def _window_target(
    target_counts: dict[str, int],
    *,
    row_count: int,
    targets: tuple[str, ...],
    min_gesture_fraction: float,
) -> tuple[str, int]:
    gesture_targets = [target for target in targets if target != "background"]
    best_gesture = max(gesture_targets, key=lambda target: (target_counts[target], target))
    if target_counts[best_gesture] / row_count >= min_gesture_fraction:
        return best_gesture, targets.index(best_gesture)
    return "background", targets.index("background")


def _target_counts(target_by_row: list[str], targets: tuple[str, ...]) -> dict[str, int]:
    counts = {target: 0 for target in targets}
    for target in target_by_row:
        counts[target if target in counts else "background"] += 1
    return counts


def _duration(rows: list[FrameFeatureRow]) -> float:
    if len(rows) < 2:
        return 0.0
    return rows[-1].timestamp - rows[0].timestamp


def _validate_manifest_config(
    *,
    window_seconds: float,
    stride_seconds: float,
    min_rows: int,
    min_gesture_fraction: float,
    targets: tuple[str, ...],
    feature_preset: str,
    target_mode: str,
    target_assignment: str,
    motion_gate_min_dx_per_hand_scale: float,
    motion_gate_min_direction_consistency: float,
) -> None:
    feature_columns_for_preset(feature_preset)
    targets_for_mode(target_mode)
    if target_assignment not in TCN_TARGET_ASSIGNMENTS:
        options = ", ".join(TCN_TARGET_ASSIGNMENTS)
        message = f"unsupported target_assignment={target_assignment}; use one of: {options}"
        raise ValueError(message)
    if "background" not in targets:
        raise ValueError("TCN targets must include background")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    if min_rows <= 0:
        raise ValueError("min_rows must be positive")
    if not 0 < min_gesture_fraction <= 1:
        raise ValueError("min_gesture_fraction must be in (0, 1]")
    if motion_gate_min_dx_per_hand_scale < 0:
        raise ValueError("motion_gate_min_dx_per_hand_scale must be non-negative")
    if not 0 <= motion_gate_min_direction_consistency <= 1:
        raise ValueError("motion_gate_min_direction_consistency must be in [0, 1]")
