"""Motion-based weak-label refinement for chart-collected gesture data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from airdesk.features import FrameFeatureRow
from airdesk.labels import GestureEventLabel, GestureLabelFile, GesturePhaseLabel, load_label_file
from airdesk.ml.dataset import load_feature_rows_csv


@dataclass(frozen=True)
class RefinedGestureLabel:
    """One original chart event aligned to a motion peak."""

    label_id: str
    gesture: str
    original_start: float
    original_end: float
    refined_start: float
    refined_end: float
    commit_time: float
    hand_id: str | None
    peak_time: float | None
    motion_score: float
    changed: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RefinedLabelFileResult:
    """Refined label file plus per-event diagnostics."""

    label_file: GestureLabelFile
    source_label_path: str
    source_feature_path: str
    refined_events: tuple[RefinedGestureLabel, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_label_path": self.source_label_path,
            "source_feature_path": self.source_feature_path,
            "refined_events": [event.to_dict() for event in self.refined_events],
        }


def refine_motion_aligned_label_file(
    *,
    feature_path: Path,
    label_path: Path,
    search_padding_seconds: float = 1.5,
    min_motion_score: float = 0.35,
    min_direction_consistency: float = 0.35,
    stroke_seconds: float | None = None,
    recovery_seconds: float = 0.35,
) -> RefinedLabelFileResult:
    """Align prompt-time gesture labels to nearby per-hand motion peaks."""
    if search_padding_seconds < 0:
        raise ValueError("search_padding_seconds must be non-negative")
    if min_motion_score < 0:
        raise ValueError("min_motion_score must be non-negative")
    if not 0 <= min_direction_consistency <= 1:
        raise ValueError("min_direction_consistency must be in [0, 1]")
    if stroke_seconds is not None and stroke_seconds <= 0:
        raise ValueError("stroke_seconds must be positive")
    if recovery_seconds < 0:
        raise ValueError("recovery_seconds must be non-negative")

    rows = load_feature_rows_csv(feature_path)
    label_file = load_label_file(label_path)
    refined_events: list[GestureEventLabel] = []
    refined_phases = [
        phase for phase in label_file.phase_labels if phase.phase == "background"
    ]
    diagnostics: list[RefinedGestureLabel] = []

    gesture_events = [
        event
        for event in label_file.event_labels
        if event.label_type == "gesture" and event.gesture in {"swipe_left", "swipe_right"}
    ]
    event_bounds = _event_search_bounds(
        gesture_events,
        label_file=label_file,
        search_padding_seconds=search_padding_seconds,
    )

    for event in label_file.event_labels:
        if event.label_type != "gesture" or event.gesture not in {"swipe_left", "swipe_right"}:
            refined_events.append(event)
            continue
        search_start, search_end = event_bounds[event.label_id]
        refined_event, refined_phase, diagnostic = _refine_event(
            event,
            rows=rows,
            label_file=label_file,
            search_start=search_start,
            search_end=search_end,
            min_motion_score=min_motion_score,
            min_direction_consistency=min_direction_consistency,
            stroke_seconds=stroke_seconds,
        )
        refined_events.append(refined_event)
        refined_phases.append(refined_phase)
        if recovery_seconds > 0:
            refined_phases.append(
                GesturePhaseLabel(
                    label_id=f"phase-refined-recovery-{len(diagnostics) + 1:03d}",
                    phase="recovery",
                    start_time=refined_event.end_time,
                    end_time=_clamp_time(
                        refined_event.end_time + recovery_seconds,
                        label_file=label_file,
                    ),
                    gesture=event.gesture,
                    notes="Motion-refined recovery label generated from chart prompt timing.",
                )
            )
        diagnostics.append(diagnostic)

    refined_label_file = GestureLabelFile(
        schema_version=label_file.schema_version,
        created_at=label_file.created_at,
        session=replace(
            label_file.session,
            notes=_append_note(
                label_file.session.notes,
                (
                    "Motion-refined label copy; original prompt-time labels preserved "
                    "elsewhere."
                ),
            ),
        ),
        event_labels=tuple(refined_events),
        phase_labels=tuple(refined_phases),
    )
    return RefinedLabelFileResult(
        label_file=refined_label_file,
        source_label_path=str(label_path),
        source_feature_path=str(feature_path),
        refined_events=tuple(diagnostics),
    )


def _refine_event(
    event: GestureEventLabel,
    *,
    rows: list[FrameFeatureRow],
    label_file: GestureLabelFile,
    search_start: float,
    search_end: float,
    min_motion_score: float,
    min_direction_consistency: float,
    stroke_seconds: float | None,
) -> tuple[GestureEventLabel, GesturePhaseLabel, RefinedGestureLabel]:
    peak = _best_motion_peak(
        rows,
        start_time=search_start,
        end_time=search_end,
        min_direction_consistency=min_direction_consistency,
    )
    original_duration = max(0.001, event.end_time - event.start_time)
    duration = stroke_seconds or original_duration
    if peak is None or peak[1] < min_motion_score:
        phase = _phase_for_event(event)
        note = _append_note(event.notes, "Motion refinement kept original timing; weak peak.")
        refined_event = replace(event, notes=note)
        refined_phase = GesturePhaseLabel(
            label_id=f"phase-refined-{event.label_id}",
            phase=phase,
            start_time=event.start_time,
            end_time=event.end_time,
            gesture=event.gesture,
            notes="Motion refinement kept original timing; no strong nearby motion peak.",
        )
        diagnostic = RefinedGestureLabel(
            label_id=event.label_id,
            gesture=event.gesture,
            original_start=event.start_time,
            original_end=event.end_time,
            refined_start=event.start_time,
            refined_end=event.end_time,
            commit_time=event.commit_time or event.end_time,
            hand_id=None,
            peak_time=None,
            motion_score=0.0 if peak is None else peak[1],
            changed=False,
            reason="no_strong_motion_peak",
        )
        return refined_event, refined_phase, diagnostic

    peak_row, motion_score = peak
    refined_end = _clamp_time(peak_row.timestamp, label_file=label_file)
    refined_start = _clamp_time(refined_end - duration, label_file=label_file)
    if refined_end <= refined_start:
        refined_start = event.start_time
        refined_end = event.end_time
    phase = _phase_for_event(event)
    note = _append_note(
        event.notes,
        (
            "Motion-refined from chart prompt timing "
            f"using hand_id={peak_row.hand_id} peak_time={peak_row.timestamp:.3f}."
        ),
    )
    refined_event = replace(
        event,
        start_time=refined_start,
        end_time=refined_end,
        commit_time=refined_end,
        notes=note,
    )
    refined_phase = GesturePhaseLabel(
        label_id=f"phase-refined-{event.label_id}",
        phase=phase,
        start_time=refined_start,
        end_time=refined_end,
        gesture=event.gesture,
        notes=(
            "Motion-refined stroke label generated from chart prompt timing "
            f"using hand_id={peak_row.hand_id}."
        ),
    )
    diagnostic = RefinedGestureLabel(
        label_id=event.label_id,
        gesture=event.gesture,
        original_start=event.start_time,
        original_end=event.end_time,
        refined_start=refined_start,
        refined_end=refined_end,
        commit_time=refined_end,
        hand_id=peak_row.hand_id,
        peak_time=peak_row.timestamp,
        motion_score=motion_score,
        changed=abs(refined_start - event.start_time) > 1e-6
        or abs(refined_end - event.end_time) > 1e-6,
        reason="motion_peak",
    )
    return refined_event, refined_phase, diagnostic


def _best_motion_peak(
    rows: list[FrameFeatureRow],
    *,
    start_time: float,
    end_time: float,
    min_direction_consistency: float,
) -> tuple[FrameFeatureRow, float] | None:
    best: tuple[FrameFeatureRow, float] | None = None
    for row in rows:
        if row.tracking_present != 1:
            continue
        if row.timestamp < start_time or row.timestamp > end_time:
            continue
        if row.hand_scale <= 0:
            continue
        if row.palm_window_direction_consistency < min_direction_consistency:
            continue
        score = abs(row.palm_window_dx_per_hand_scale)
        if best is None or score > best[1]:
            best = (row, score)
    return best


def _event_search_bounds(
    events: list[GestureEventLabel],
    *,
    label_file: GestureLabelFile,
    search_padding_seconds: float,
) -> dict[str, tuple[float, float]]:
    sorted_events = sorted(events, key=lambda item: (item.start_time, item.end_time))
    bounds: dict[str, tuple[float, float]] = {}
    for index, event in enumerate(sorted_events):
        start = event.start_time - search_padding_seconds
        end = event.end_time + search_padding_seconds
        center = (event.start_time + event.end_time) / 2
        if index > 0:
            previous = sorted_events[index - 1]
            previous_center = (previous.start_time + previous.end_time) / 2
            start = max(start, (previous_center + center) / 2)
        if index + 1 < len(sorted_events):
            following = sorted_events[index + 1]
            following_center = (following.start_time + following.end_time) / 2
            end = min(end, (center + following_center) / 2)
        bounds[event.label_id] = (
            _clamp_time(start, label_file=label_file),
            _clamp_time(end, label_file=label_file),
        )
    return bounds


def _phase_for_event(event: GestureEventLabel) -> str:
    if event.gesture == "swipe_left":
        return "stroke_left"
    if event.gesture == "swipe_right":
        return "stroke_right"
    return event.gesture


def _clamp_time(timestamp: float, *, label_file: GestureLabelFile) -> float:
    start = label_file.session.start_timestamp
    end = label_file.session.end_timestamp
    if start is not None and timestamp < start:
        return start
    if end is not None and timestamp > end:
        return end
    return timestamp


def _append_note(existing: str, note: str) -> str:
    if not existing:
        return note
    return f"{existing} {note}"
