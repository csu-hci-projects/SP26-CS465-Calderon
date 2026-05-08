"""Deterministic motion-event spotting over per-hand feature streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

from airdesk.feature_streams import group_feature_rows_by_stream, is_tracked_feature_row
from airdesk.state.types import GestureCandidate

if TYPE_CHECKING:
    from airdesk.features import FrameFeatureRow

SwipeGesture = Literal["swipe_left", "swipe_right"]


@dataclass(frozen=True)
class MotionEventConfig:
    """Thresholds for the first replay-safe motion-event baseline."""

    min_dx_per_hand_scale: float = 0.65
    min_peak_velocity: float = 0.45
    min_direction_consistency: float = 0.60
    release_velocity: float = 0.20
    recovery_seconds: float = 0.10
    min_duration_seconds: float = 0.08
    max_duration_seconds: float = 1.25
    min_event_separation_seconds: float = 0.20
    positive_dx_gesture: SwipeGesture = "swipe_right"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MotionEvidence:
    """One row's motion evidence after thresholding and direction mapping."""

    name: SwipeGesture
    score: float
    raw_dx: float
    dx_per_hand_scale: float
    peak_velocity: float
    direction_consistency: float
    palm_x: float
    palm_y: float
    palm_vx: float
    palm_vy: float
    phase: str
    event: str


@dataclass(frozen=True)
class MotionRowDiagnostic:
    """A compact per-row diagnostic for replay motion failures."""

    hand_id: str
    frame_index: int
    timestamp: float
    mapped_gesture: SwipeGesture
    would_emit: bool
    rejection_reasons: tuple[str, ...]
    raw_dx: float
    dx_per_hand_scale: float
    peak_velocity: float
    direction_consistency: float
    palm_x: float
    palm_y: float
    palm_vx: float
    palm_vy: float
    phase: str
    event: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _ActiveMotion:
    name: SwipeGesture
    start_time: float
    start_frame_index: int
    peak_time: float
    peak_frame_index: int
    peak_score: float
    peak_evidence: MotionEvidence
    last_motion_time: float


class MotionEventRecognizer:
    """Spot swipe-like motion events from existing AirDesk feature rows."""

    name = "motion"

    def __init__(self, config: MotionEventConfig | None = None) -> None:
        self.config = config or MotionEventConfig()
        _validate_config(self.config)

    def recognize_rows(self, rows: list[FrameFeatureRow]) -> list[GestureCandidate]:
        """Return motion events from hand-scoped feature rows."""
        candidates: list[GestureCandidate] = []
        for hand_rows in _hand_streams(rows):
            candidates.extend(self._recognize_hand_stream(hand_rows))
        return sorted(candidates, key=lambda candidate: candidate.timestamp)

    def _recognize_hand_stream(self, rows: list[FrameFeatureRow]) -> list[GestureCandidate]:
        events: list[GestureCandidate] = []
        active: _ActiveMotion | None = None
        last_commit_by_name: dict[str, float] = {}

        for row in rows:
            evidence = _motion_evidence(row, self.config)
            if active is None:
                if evidence is None:
                    continue
                if not _separation_ok(
                    evidence.name,
                    row.timestamp,
                    last_commit_by_name,
                    self.config,
                ):
                    continue
                active = _start_motion(row, evidence)
                continue

            if _release_row(row, active, self.config):
                candidate = _candidate_from_motion(active, hand_id=row.hand_id, config=self.config)
                if candidate is not None and _separation_ok(
                    candidate.name,
                    candidate.timestamp,
                    last_commit_by_name,
                    self.config,
                ):
                    events.append(candidate)
                    last_commit_by_name[candidate.name] = candidate.timestamp
                active = None
                continue

            if evidence is not None and evidence.name == active.name:
                active.last_motion_time = row.timestamp
                if evidence.score >= active.peak_score:
                    active.peak_time = row.timestamp
                    active.peak_frame_index = row.frame_index
                    active.peak_score = evidence.score
                    active.peak_evidence = evidence
                continue

            should_commit = _should_commit(active, row, self.config)
            if should_commit:
                candidate = _candidate_from_motion(active, hand_id=row.hand_id, config=self.config)
                if candidate is not None and _separation_ok(
                    candidate.name,
                    candidate.timestamp,
                    last_commit_by_name,
                    self.config,
                ):
                    events.append(candidate)
                    last_commit_by_name[candidate.name] = candidate.timestamp
                active = None

            if active is not None:
                continue
            if evidence is None:
                continue
            if not _separation_ok(evidence.name, row.timestamp, last_commit_by_name, self.config):
                continue
            active = _start_motion(row, evidence)

        if active is not None:
            candidate = _candidate_from_motion(active, hand_id=rows[-1].hand_id, config=self.config)
            if candidate is not None and _separation_ok(
                candidate.name,
                candidate.timestamp,
                last_commit_by_name,
                self.config,
            ):
                events.append(candidate)
        return events


def _motion_evidence(
    row: FrameFeatureRow,
    config: MotionEventConfig,
) -> MotionEvidence | None:
    if not _usable_row(row):
        return None
    if abs(row.palm_vx) <= config.release_velocity:
        return None
    dx_per_scale = row.palm_window_dx_per_hand_scale
    if abs(dx_per_scale) < config.min_dx_per_hand_scale:
        return None
    if row.palm_window_peak_abs_vx < config.min_peak_velocity:
        return None
    if row.palm_window_direction_consistency < config.min_direction_consistency:
        return None
    name = _gesture_for_dx(row.palm_window_dx, config)
    score = _motion_score(
        dx_per_hand_scale=dx_per_scale,
        peak_velocity=row.palm_window_peak_abs_vx,
        direction_consistency=row.palm_window_direction_consistency,
        config=config,
    )
    return MotionEvidence(
        name=name,
        score=score,
        raw_dx=row.palm_window_dx,
        dx_per_hand_scale=dx_per_scale,
        peak_velocity=row.palm_window_peak_abs_vx,
        direction_consistency=row.palm_window_direction_consistency,
        palm_x=row.palm_x,
        palm_y=row.palm_y,
        palm_vx=row.palm_vx,
        palm_vy=row.palm_vy,
        phase=row.phase,
        event=row.event,
    )


def _motion_score(
    *,
    dx_per_hand_scale: float,
    peak_velocity: float,
    direction_consistency: float,
    config: MotionEventConfig,
) -> float:
    dx_ratio = min(1.0, abs(dx_per_hand_scale) / config.min_dx_per_hand_scale)
    velocity_ratio = min(1.0, peak_velocity / config.min_peak_velocity)
    consistency = max(0.0, min(1.0, direction_consistency))
    return (0.40 * dx_ratio) + (0.35 * velocity_ratio) + (0.25 * consistency)


def _start_motion(row: FrameFeatureRow, evidence: MotionEvidence) -> _ActiveMotion:
    start_time = _estimated_start_time(row, evidence)
    return _ActiveMotion(
        name=evidence.name,
        start_time=start_time,
        start_frame_index=row.frame_index,
        peak_time=row.timestamp,
        peak_frame_index=row.frame_index,
        peak_score=evidence.score,
        peak_evidence=evidence,
        last_motion_time=row.timestamp,
    )


def _estimated_start_time(row: FrameFeatureRow, evidence: MotionEvidence) -> float:
    if evidence.peak_velocity <= 0:
        return row.timestamp
    estimated_duration = abs(evidence.raw_dx) / evidence.peak_velocity
    return row.timestamp - max(0.0, estimated_duration)


def _should_commit(
    active: _ActiveMotion,
    row: FrameFeatureRow,
    config: MotionEventConfig,
) -> bool:
    if row.timestamp - active.start_time > config.max_duration_seconds:
        return True
    low_velocity = abs(row.palm_vx) <= config.release_velocity
    recovered = row.timestamp - active.last_motion_time >= config.recovery_seconds
    return low_velocity or recovered or not _usable_row(row)


def _release_row(
    row: FrameFeatureRow,
    active: _ActiveMotion,
    config: MotionEventConfig,
) -> bool:
    if row.timestamp <= active.start_time:
        return False
    return abs(row.palm_vx) <= config.release_velocity


def _candidate_from_motion(
    active: _ActiveMotion,
    *,
    hand_id: str,
    config: MotionEventConfig,
) -> GestureCandidate | None:
    duration = active.peak_time - active.start_time
    if duration < config.min_duration_seconds:
        return None
    if duration > config.max_duration_seconds:
        return None
    evidence = active.peak_evidence
    evidence_id = (
        f"motion:{hand_id}:{active.name}:"
        f"{active.start_frame_index}:{active.peak_frame_index}"
    )
    return GestureCandidate(
        name=active.name,
        confidence=active.peak_score,
        timestamp=active.peak_time,
        hand_id=hand_id or None,
        metadata={
            "recognizer": MotionEventRecognizer.name,
            "evidence_id": evidence_id,
            "window_start": active.start_time,
            "window_end": active.peak_time,
            "peak_time": active.peak_time,
            "start_frame_index": active.start_frame_index,
            "peak_frame_index": active.peak_frame_index,
            "duration_seconds": duration,
            "raw_dx": evidence.raw_dx,
            "dx_per_hand_scale": evidence.dx_per_hand_scale,
            "peak_velocity": evidence.peak_velocity,
            "direction_consistency": evidence.direction_consistency,
            "peak_palm_x": evidence.palm_x,
            "peak_palm_y": evidence.palm_y,
            "peak_palm_vx": evidence.palm_vx,
            "peak_palm_vy": evidence.palm_vy,
            "peak_phase": evidence.phase,
            "peak_event": evidence.event,
            "positive_dx_gesture": config.positive_dx_gesture,
        },
    )


def _hand_streams(rows: list[FrameFeatureRow]) -> list[list[FrameFeatureRow]]:
    return group_feature_rows_by_stream(rows)


def _usable_row(row: FrameFeatureRow) -> bool:
    return is_tracked_feature_row(row)


def _gesture_for_dx(raw_dx: float, config: MotionEventConfig) -> SwipeGesture:
    if raw_dx >= 0:
        return config.positive_dx_gesture
    return "swipe_left" if config.positive_dx_gesture == "swipe_right" else "swipe_right"


def diagnose_motion_rows(
    rows: list[FrameFeatureRow],
    config: MotionEventConfig | None = None,
    *,
    limit_per_hand: int = 8,
) -> list[MotionRowDiagnostic]:
    """Return strongest row-level motion diagnostics for replay inspection.

    The motion baseline deliberately emits only accepted candidates. This helper keeps
    near-misses visible without turning the recognizer into a threshold sweep.
    """
    active_config = config or MotionEventConfig()
    _validate_config(active_config)
    if limit_per_hand < 0:
        raise ValueError("limit_per_hand must be non-negative")
    diagnostics: list[MotionRowDiagnostic] = []
    for hand_rows in _diagnostic_hand_streams(rows):
        ranked = sorted(
            hand_rows,
            key=lambda row: (
                abs(row.palm_window_dx_per_hand_scale),
                row.palm_window_peak_abs_vx,
                row.palm_window_direction_consistency,
            ),
            reverse=True,
        )
        for row in ranked[:limit_per_hand]:
            reasons = _motion_rejection_reasons(row, active_config)
            diagnostics.append(
                MotionRowDiagnostic(
                    hand_id=row.hand_id,
                    frame_index=row.frame_index,
                    timestamp=row.timestamp,
                    mapped_gesture=_gesture_for_dx(row.palm_window_dx, active_config),
                    would_emit=not reasons,
                    rejection_reasons=tuple(reasons),
                    raw_dx=row.palm_window_dx,
                    dx_per_hand_scale=row.palm_window_dx_per_hand_scale,
                    peak_velocity=row.palm_window_peak_abs_vx,
                    direction_consistency=row.palm_window_direction_consistency,
                    palm_x=row.palm_x,
                    palm_y=row.palm_y,
                    palm_vx=row.palm_vx,
                    palm_vy=row.palm_vy,
                    phase=row.phase,
                    event=row.event,
                )
            )
    return sorted(
        diagnostics,
        key=lambda item: (
            item.hand_id,
            -abs(item.dx_per_hand_scale),
            -item.peak_velocity,
            item.frame_index,
        ),
    )


def _diagnostic_hand_streams(rows: list[FrameFeatureRow]) -> list[list[FrameFeatureRow]]:
    return group_feature_rows_by_stream(rows)


def _motion_rejection_reasons(
    row: FrameFeatureRow,
    config: MotionEventConfig,
) -> list[str]:
    reasons: list[str] = []
    if not _usable_row(row):
        reasons.append("tracking_missing")
        return reasons
    if abs(row.palm_vx) <= config.release_velocity:
        reasons.append("immediate_velocity_below_release")
    if abs(row.palm_window_dx_per_hand_scale) < config.min_dx_per_hand_scale:
        reasons.append("dx_per_hand_scale_below_min")
    if row.palm_window_peak_abs_vx < config.min_peak_velocity:
        reasons.append("peak_velocity_below_min")
    if row.palm_window_direction_consistency < config.min_direction_consistency:
        reasons.append("direction_consistency_below_min")
    return reasons


def _separation_ok(
    name: str,
    timestamp: float,
    last_commit_by_name: dict[str, float],
    config: MotionEventConfig,
) -> bool:
    last_commit = last_commit_by_name.get(name)
    if last_commit is None:
        return True
    return timestamp - last_commit >= config.min_event_separation_seconds


def _validate_config(config: MotionEventConfig) -> None:
    if config.min_dx_per_hand_scale < 0:
        raise ValueError("min_dx_per_hand_scale must be non-negative")
    if config.min_peak_velocity < 0:
        raise ValueError("min_peak_velocity must be non-negative")
    if not 0 <= config.min_direction_consistency <= 1:
        raise ValueError("min_direction_consistency must be in [0, 1]")
    if config.release_velocity < 0:
        raise ValueError("release_velocity must be non-negative")
    if config.recovery_seconds < 0:
        raise ValueError("recovery_seconds must be non-negative")
    if config.min_duration_seconds < 0:
        raise ValueError("min_duration_seconds must be non-negative")
    if config.max_duration_seconds < config.min_duration_seconds:
        raise ValueError("max_duration_seconds must be >= min_duration_seconds")
    if config.min_event_separation_seconds < 0:
        raise ValueError("min_event_separation_seconds must be non-negative")
    if config.positive_dx_gesture not in {"swipe_left", "swipe_right"}:
        raise ValueError("positive_dx_gesture must be swipe_left or swipe_right")
