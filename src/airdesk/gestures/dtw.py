"""Dependency-free DTW template recognizer for personalized dynamic gestures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import dist
from pathlib import Path
from typing import Any

from airdesk.features import FrameFeatureRow, extract_feature_rows
from airdesk.labels import GestureLabelFile
from airdesk.recording.jsonl import iter_recording
from airdesk.state.types import GestureCandidate, TrackingFrame

DTW_MODEL_VERSION = 1
DTW_FEATURE_NAMES = (
    "palm_rel_x",
    "palm_rel_y",
    "palm_vx",
    "palm_vy",
    "palm_window_dx",
    "palm_window_dx_per_hand_scale",
    "palm_window_peak_abs_vx",
    "palm_window_direction_consistency",
    "index_rel_x",
    "index_rel_y",
    "pinch_distance",
    "hand_scale",
    "confidence",
)


@dataclass(frozen=True)
class DtwTemplate:
    """One calibrated gesture template."""

    template_id: str
    gesture: str
    recording: str
    label_id: str
    start_time: float
    end_time: float
    vectors: tuple[tuple[float, ...], ...]
    palm_dx: float = 0.0
    palm_dy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "vectors": [list(vector) for vector in self.vectors],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DtwTemplate:
        return cls(
            template_id=str(data["template_id"]),
            gesture=str(data["gesture"]),
            recording=str(data["recording"]),
            label_id=str(data["label_id"]),
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            vectors=tuple(tuple(float(value) for value in vector) for vector in data["vectors"]),
            palm_dx=float(data.get("palm_dx", 0.0)),
            palm_dy=float(data.get("palm_dy", 0.0)),
        )


@dataclass(frozen=True)
class DtwGestureModel:
    """Serializable DTW gesture model."""

    schema_version: int
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    templates: tuple[DtwTemplate, ...]
    thresholds: dict[str, float]
    negative_distances: dict[str, float]
    palm_dx_signs: dict[str, int] = field(default_factory=dict)
    min_palm_dx: dict[str, float] = field(default_factory=dict)
    cooldown_seconds: float = 0.5
    min_window_seconds: float = 0.25
    max_window_seconds: float = 1.25
    window_step_seconds: float = 0.1
    min_points: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "std": list(self.std),
            "templates": [template.to_dict() for template in self.templates],
            "thresholds": self.thresholds,
            "negative_distances": self.negative_distances,
            "palm_dx_signs": self.palm_dx_signs,
            "min_palm_dx": self.min_palm_dx,
            "cooldown_seconds": self.cooldown_seconds,
            "min_window_seconds": self.min_window_seconds,
            "max_window_seconds": self.max_window_seconds,
            "window_step_seconds": self.window_step_seconds,
            "min_points": self.min_points,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DtwGestureModel:
        return cls(
            schema_version=int(data["schema_version"]),
            feature_names=tuple(str(name) for name in data["feature_names"]),
            mean=tuple(float(value) for value in data["mean"]),
            std=tuple(float(value) for value in data["std"]),
            templates=tuple(DtwTemplate.from_dict(item) for item in data["templates"]),
            thresholds={str(key): float(value) for key, value in data["thresholds"].items()},
            negative_distances={
                str(key): float(value) for key, value in data.get("negative_distances", {}).items()
            },
            palm_dx_signs={
                str(key): int(value) for key, value in data.get("palm_dx_signs", {}).items()
            },
            min_palm_dx={
                str(key): float(value) for key, value in data.get("min_palm_dx", {}).items()
            },
            cooldown_seconds=float(data.get("cooldown_seconds", 0.5)),
            min_window_seconds=float(data.get("min_window_seconds", 0.25)),
            max_window_seconds=float(data.get("max_window_seconds", 1.25)),
            window_step_seconds=float(data.get("window_step_seconds", 0.1)),
            min_points=int(data.get("min_points", 4)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> DtwGestureModel:
        with path.open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass(frozen=True)
class DtwCalibrationInput:
    """Recording and labels used for DTW calibration."""

    recording: Path
    labels: GestureLabelFile
    label_path: Path


@dataclass(frozen=True)
class DtwMatch:
    """Best DTW match for one candidate window."""

    gesture: str
    template_id: str
    distance: float
    threshold: float
    confidence: float
    palm_dx: float = 0.0
    palm_dy: float = 0.0


@dataclass(frozen=True)
class DtwBestWindow:
    """Best observed DTW window for one gesture, including rejected windows."""

    gesture: str
    template_id: str
    distance: float
    threshold: float
    window_start: float
    window_end: float
    window_points: int
    palm_dx: float = 0.0
    palm_dy: float = 0.0
    min_palm_dx: float = 0.0
    expected_palm_dx_sign: int = 0

    @property
    def distance_ratio(self) -> float:
        if self.threshold <= 0:
            return float("inf")
        return self.distance / self.threshold

    @property
    def distance_accepted(self) -> bool:
        return self.distance <= self.threshold

    @property
    def displacement_accepted(self) -> bool:
        if self.min_palm_dx <= 0 or self.expected_palm_dx_sign == 0:
            return True
        return (
            self.palm_dx * self.expected_palm_dx_sign >= self.min_palm_dx
        )

    @property
    def accepted(self) -> bool:
        return self.distance_accepted and self.displacement_accepted

    def to_dict(self) -> dict[str, Any]:
        return {
            "gesture": self.gesture,
            "template_id": self.template_id,
            "distance": self.distance,
            "threshold": self.threshold,
            "distance_ratio": self.distance_ratio,
            "distance_accepted": self.distance_accepted,
            "displacement_accepted": self.displacement_accepted,
            "accepted": self.accepted,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "window_points": self.window_points,
            "palm_dx": self.palm_dx,
            "palm_dy": self.palm_dy,
            "min_palm_dx": self.min_palm_dx,
            "expected_palm_dx_sign": self.expected_palm_dx_sign,
        }


@dataclass
class DtwTemplateRecognizer:
    """Offline sliding-window DTW recognizer."""

    model: DtwGestureModel
    name: str = "dtw"

    def recognize_rows(self, rows: list[FrameFeatureRow]) -> list[GestureCandidate]:
        raw_candidates: list[GestureCandidate] = []
        for start_index, start_row in enumerate(rows):
            if not _usable_row(start_row):
                continue
            for window_rows in _candidate_windows(rows, start_index, self.model):
                match = self.best_match(window_rows)
                if match is None:
                    continue
                raw_candidates.append(
                    GestureCandidate(
                        name=match.gesture,
                        confidence=match.confidence,
                        timestamp=window_rows[-1].timestamp,
                        hand_id=window_rows[-1].hand_id or None,
                        metadata={
                            "recognizer": self.name,
                            "distance": match.distance,
                            "threshold": match.threshold,
                            "template_id": match.template_id,
                            "palm_dx": match.palm_dx,
                            "palm_dy": match.palm_dy,
                            "window_start": window_rows[0].timestamp,
                            "window_end": window_rows[-1].timestamp,
                            "window_points": len(window_rows),
                        },
                    )
                )
        return _suppress_candidates(raw_candidates, cooldown_seconds=self.model.cooldown_seconds)

    def recognize_latest_rows(self, rows: list[FrameFeatureRow]) -> list[GestureCandidate]:
        """Return DTW candidates for windows ending at the latest usable row."""
        end_index = _latest_usable_index(rows)
        if end_index is None:
            return []
        raw_candidates: list[GestureCandidate] = []
        for window_rows in _candidate_windows_ending_at(rows, end_index, self.model):
            match = self.best_match(window_rows)
            if match is None:
                continue
            raw_candidates.append(
                GestureCandidate(
                    name=match.gesture,
                    confidence=match.confidence,
                    timestamp=window_rows[-1].timestamp,
                    hand_id=window_rows[-1].hand_id or None,
                    metadata={
                        "recognizer": self.name,
                        "distance": match.distance,
                        "threshold": match.threshold,
                        "template_id": match.template_id,
                        "palm_dx": match.palm_dx,
                        "palm_dy": match.palm_dy,
                        "window_start": window_rows[0].timestamp,
                        "window_end": window_rows[-1].timestamp,
                        "window_points": len(window_rows),
                    },
                )
            )
        return _suppress_candidates(raw_candidates, cooldown_seconds=self.model.cooldown_seconds)

    def best_match(self, rows: list[FrameFeatureRow]) -> DtwMatch | None:
        sequence = _normalize_sequence(
            _raw_sequence(rows, self.model.feature_names),
            self.model.mean,
            self.model.std,
        )
        if len(sequence) < self.model.min_points:
            return None
        best: tuple[DtwTemplate, float] | None = None
        for template in self.model.templates:
            distance_value = dtw_distance(sequence, template.vectors)
            if best is None or distance_value < best[1]:
                best = (template, distance_value)
        if best is None:
            return None

        template, distance_value = best
        threshold = self.model.thresholds.get(template.gesture)
        if threshold is None or distance_value > threshold:
            return None
        palm_dx, palm_dy = _window_palm_delta(rows)
        if not _passes_palm_dx_gate(template.gesture, palm_dx, self.model):
            return None
        confidence = max(0.0, min(1.0, 1.0 - (distance_value / threshold)))
        return DtwMatch(
            gesture=template.gesture,
            template_id=template.template_id,
            distance=distance_value,
            threshold=threshold,
            confidence=confidence,
            palm_dx=palm_dx,
            palm_dy=palm_dy,
        )

    def best_windows_by_gesture(
        self,
        rows: list[FrameFeatureRow],
    ) -> dict[str, DtwBestWindow]:
        """Return the closest window per gesture, even if it misses the threshold."""
        best: dict[str, DtwBestWindow] = {}
        for start_index, start_row in enumerate(rows):
            if not _usable_row(start_row):
                continue
            for window_rows in _candidate_windows(rows, start_index, self.model):
                sequence = _normalize_sequence(
                    _raw_sequence(window_rows, self.model.feature_names),
                    self.model.mean,
                    self.model.std,
                )
                if len(sequence) < self.model.min_points:
                    continue
                for template in self.model.templates:
                    threshold = self.model.thresholds.get(template.gesture)
                    if threshold is None:
                        continue
                    distance_value = dtw_distance(sequence, template.vectors)
                    current = best.get(template.gesture)
                    if current is not None and current.distance <= distance_value:
                        continue
                    palm_dx, palm_dy = _window_palm_delta(window_rows)
                    best[template.gesture] = DtwBestWindow(
                        gesture=template.gesture,
                        template_id=template.template_id,
                        distance=distance_value,
                        threshold=threshold,
                        window_start=window_rows[0].timestamp,
                        window_end=window_rows[-1].timestamp,
                        window_points=len(window_rows),
                        palm_dx=palm_dx,
                        palm_dy=palm_dy,
                        min_palm_dx=self.model.min_palm_dx.get(template.gesture, 0.0),
                        expected_palm_dx_sign=self.model.palm_dx_signs.get(
                            template.gesture,
                            0,
                        ),
                    )
        return best


def calibrate_dtw_model(
    inputs: list[DtwCalibrationInput],
    *,
    cooldown_seconds: float = 0.5,
    min_window_seconds: float = 0.25,
    max_window_seconds: float = 1.25,
    window_step_seconds: float = 0.1,
    min_points: int = 4,
    negative_distance_margin: float = 0.85,
    min_palm_dx_fraction: float = 0.0,
) -> DtwGestureModel:
    """Build a DTW model from labeled gesture recordings."""
    raw_templates: list[tuple[DtwTemplate, tuple[tuple[float, ...], ...]]] = []
    normalization_vectors: list[tuple[float, ...]] = []
    negative_rows: list[list[FrameFeatureRow]] = []

    for item in inputs:
        frames = _frames_from_recording(item.recording)
        rows = extract_feature_rows(frames, labels=item.labels)
        gesture_events = [
            event for event in item.labels.event_labels if event.label_type == "gesture"
        ]
        if not gesture_events:
            negative_rows.append(rows)
            continue
        for event in gesture_events:
            event_rows = [
                row
                for row in rows
                if event.start_time <= row.timestamp <= event.end_time and _usable_row(row)
            ]
            if len(event_rows) < min_points:
                continue
            raw_vectors = _raw_sequence(event_rows, DTW_FEATURE_NAMES)
            normalization_vectors.extend(raw_vectors)
            raw_templates.append(
                (
                    DtwTemplate(
                        template_id=f"template-{len(raw_templates) + 1:03d}",
                        gesture=event.gesture,
                        recording=str(item.recording),
                        label_id=event.label_id,
                        start_time=event.start_time,
                        end_time=event.end_time,
                        vectors=(),
                        palm_dx=event_rows[-1].palm_x - event_rows[0].palm_x,
                        palm_dy=event_rows[-1].palm_y - event_rows[0].palm_y,
                    ),
                    raw_vectors,
                )
            )

    if not raw_templates:
        raise ValueError("DTW calibration requires at least one labeled gesture event")

    mean, std = _normalizer(normalization_vectors)
    templates = tuple(
        DtwTemplate(
            template_id=template.template_id,
            gesture=template.gesture,
            recording=template.recording,
            label_id=template.label_id,
            start_time=template.start_time,
            end_time=template.end_time,
            vectors=_normalize_sequence(raw_vectors, mean, std),
            palm_dx=template.palm_dx,
            palm_dy=template.palm_dy,
        )
        for template, raw_vectors in raw_templates
    )
    thresholds = _thresholds_for_templates(templates)
    negative_distances = _negative_distances(
        negative_rows,
        mean=mean,
        std=std,
        templates=templates,
        min_points=min_points,
        min_window_seconds=min_window_seconds,
        max_window_seconds=max_window_seconds,
        window_step_seconds=window_step_seconds,
    )
    for gesture, negative_distance in negative_distances.items():
        thresholds[gesture] = min(thresholds[gesture], negative_distance * negative_distance_margin)
    palm_dx_signs, min_palm_dx = _palm_dx_gates(
        templates,
        min_palm_dx_fraction=min_palm_dx_fraction,
    )

    return DtwGestureModel(
        schema_version=DTW_MODEL_VERSION,
        feature_names=DTW_FEATURE_NAMES,
        mean=mean,
        std=std,
        templates=templates,
        thresholds=thresholds,
        negative_distances=negative_distances,
        palm_dx_signs=palm_dx_signs,
        min_palm_dx=min_palm_dx,
        cooldown_seconds=cooldown_seconds,
        min_window_seconds=min_window_seconds,
        max_window_seconds=max_window_seconds,
        window_step_seconds=window_step_seconds,
        min_points=min_points,
    )


def dtw_distance(
    first: tuple[tuple[float, ...], ...],
    second: tuple[tuple[float, ...], ...],
) -> float:
    """Return normalized DTW distance between two vector sequences."""
    if not first or not second:
        return float("inf")
    previous = [float("inf")] * (len(second) + 1)
    previous[0] = 0.0
    for first_vector in first:
        current = [float("inf")] * (len(second) + 1)
        for j, second_vector in enumerate(second, start=1):
            cost = _vector_distance(first_vector, second_vector)
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return previous[-1] / (len(first) + len(second))


def _frames_from_recording(recording: Path) -> list[TrackingFrame]:
    return [
        record.payload
        for record in iter_recording(recording)
        if record.kind == "tracking_frame" and isinstance(record.payload, TrackingFrame)
    ]


def _raw_sequence(
    rows: list[FrameFeatureRow],
    feature_names: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    usable = [row for row in rows if _usable_row(row)]
    if not usable:
        return ()
    origin_x = usable[0].palm_x
    origin_y = usable[0].palm_y
    return tuple(
        tuple(
            _dtw_feature_value(
                row,
                feature_name,
                origin_x=origin_x,
                origin_y=origin_y,
            )
            for feature_name in feature_names
        )
        for row in usable
    )


def _normalizer(vectors: list[tuple[float, ...]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    dimensions = len(vectors[0]) if vectors else len(DTW_FEATURE_NAMES)
    mean = tuple(
        sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimensions)
    )
    variance = tuple(
        sum((vector[index] - mean[index]) ** 2 for vector in vectors) / len(vectors)
        for index in range(dimensions)
    )
    std = tuple(value**0.5 if value > 1e-8 else 1.0 for value in variance)
    return mean, std


def _dtw_feature_value(
    row: FrameFeatureRow,
    feature_name: str,
    *,
    origin_x: float,
    origin_y: float,
) -> float:
    if feature_name == "palm_rel_x":
        return row.palm_x - origin_x
    if feature_name == "palm_rel_y":
        return row.palm_y - origin_y
    value = getattr(row, feature_name)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"DTW feature must be numeric: {feature_name}")


def _normalize_sequence(
    sequence: tuple[tuple[float, ...], ...],
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple((value - mean[index]) / std[index] for index, value in enumerate(vector))
        for vector in sequence
    )


def _thresholds_for_templates(templates: tuple[DtwTemplate, ...]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    gestures = sorted({template.gesture for template in templates})
    for gesture in gestures:
        gesture_templates = [template for template in templates if template.gesture == gesture]
        distances: list[float] = []
        for left_index, left in enumerate(gesture_templates):
            for right in gesture_templates[left_index + 1 :]:
                distances.append(dtw_distance(left.vectors, right.vectors))
        if distances:
            thresholds[gesture] = max(distances) * 1.35 + 0.05
        else:
            thresholds[gesture] = 1.0
    return thresholds


def _negative_distances(
    negative_rows: list[list[FrameFeatureRow]],
    *,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    templates: tuple[DtwTemplate, ...],
    min_points: int,
    min_window_seconds: float,
    max_window_seconds: float,
    window_step_seconds: float,
) -> dict[str, float]:
    distances: dict[str, float] = {}
    scratch_model = DtwGestureModel(
        schema_version=DTW_MODEL_VERSION,
        feature_names=DTW_FEATURE_NAMES,
        mean=mean,
        std=std,
        templates=templates,
        thresholds={template.gesture: float("inf") for template in templates},
        negative_distances={},
        min_window_seconds=min_window_seconds,
        max_window_seconds=max_window_seconds,
        window_step_seconds=window_step_seconds,
        min_points=min_points,
    )
    for rows in negative_rows:
        for start_index, start_row in enumerate(rows):
            if not _usable_row(start_row):
                continue
            for window_rows in _candidate_windows(rows, start_index, scratch_model):
                sequence = _normalize_sequence(
                    _raw_sequence(window_rows, DTW_FEATURE_NAMES),
                    mean,
                    std,
                )
                if len(sequence) < min_points:
                    continue
                for template in templates:
                    value = dtw_distance(sequence, template.vectors)
                    previous = distances.get(template.gesture)
                    distances[template.gesture] = (
                        value if previous is None else min(previous, value)
                    )
    return distances


def _candidate_windows(
    rows: list[FrameFeatureRow],
    start_index: int,
    model: DtwGestureModel,
) -> list[list[FrameFeatureRow]]:
    windows: list[list[FrameFeatureRow]] = []
    start_time = rows[start_index].timestamp
    next_duration = model.min_window_seconds
    while next_duration <= model.max_window_seconds + 1e-9:
        end_time = start_time + next_duration
        window_rows = [
            row
            for row in rows[start_index:]
            if start_time <= row.timestamp <= end_time and _usable_row(row)
        ]
        if len(window_rows) >= model.min_points:
            windows.append(window_rows)
        next_duration += model.window_step_seconds
    return windows


def _candidate_windows_ending_at(
    rows: list[FrameFeatureRow],
    end_index: int,
    model: DtwGestureModel,
) -> list[list[FrameFeatureRow]]:
    windows: list[list[FrameFeatureRow]] = []
    end_time = rows[end_index].timestamp
    next_duration = model.min_window_seconds
    while next_duration <= model.max_window_seconds + 1e-9:
        start_time = end_time - next_duration
        window_rows = [
            row
            for row in rows[: end_index + 1]
            if start_time <= row.timestamp <= end_time and _usable_row(row)
        ]
        if len(window_rows) >= model.min_points:
            windows.append(window_rows)
        next_duration += model.window_step_seconds
    return windows


def _latest_usable_index(rows: list[FrameFeatureRow]) -> int | None:
    for index in range(len(rows) - 1, -1, -1):
        if _usable_row(rows[index]):
            return index
    return None


def _palm_dx_gates(
    templates: tuple[DtwTemplate, ...],
    *,
    min_palm_dx_fraction: float,
) -> tuple[dict[str, int], dict[str, float]]:
    if min_palm_dx_fraction <= 0:
        return {}, {}
    signs: dict[str, int] = {}
    minimums: dict[str, float] = {}
    for gesture in sorted({template.gesture for template in templates}):
        deltas = [template.palm_dx for template in templates if template.gesture == gesture]
        if not deltas:
            continue
        mean_delta = sum(deltas) / len(deltas)
        if abs(mean_delta) < 1e-6:
            continue
        signs[gesture] = 1 if mean_delta > 0 else -1
        minimums[gesture] = min(abs(delta) for delta in deltas) * min_palm_dx_fraction
    return signs, minimums


def _passes_palm_dx_gate(
    gesture: str,
    palm_dx: float,
    model: DtwGestureModel,
) -> bool:
    min_palm_dx = model.min_palm_dx.get(gesture, 0.0)
    sign = model.palm_dx_signs.get(gesture, 0)
    if min_palm_dx <= 0 or sign == 0:
        return True
    return palm_dx * sign >= min_palm_dx


def _window_palm_delta(rows: list[FrameFeatureRow]) -> tuple[float, float]:
    usable = [row for row in rows if _usable_row(row)]
    if not usable:
        return 0.0, 0.0
    return usable[-1].palm_x - usable[0].palm_x, usable[-1].palm_y - usable[0].palm_y


def _suppress_candidates(
    candidates: list[GestureCandidate],
    *,
    cooldown_seconds: float,
) -> list[GestureCandidate]:
    selected: list[GestureCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        window_start = float(candidate.metadata.get("window_start", candidate.timestamp))
        window_end = float(candidate.metadata.get("window_end", candidate.timestamp))
        overlaps = False
        for existing in selected:
            existing_start = float(existing.metadata.get("window_start", existing.timestamp))
            existing_end = float(existing.metadata.get("window_end", existing.timestamp))
            if candidate.name != existing.name:
                continue
            if window_start <= existing_end + cooldown_seconds and existing_start <= window_end:
                overlaps = True
                break
        if not overlaps:
            selected.append(candidate)
    return sorted(selected, key=lambda item: item.timestamp)


def _usable_row(row: FrameFeatureRow) -> bool:
    return row.tracking_present == 1 and bool(row.hand_id)


def _vector_distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return dist(first, second)
