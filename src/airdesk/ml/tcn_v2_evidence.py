"""TCN v2 frame-evidence target construction."""

from __future__ import annotations

from airdesk.features import FrameFeatureRow, is_tracked_feature_row
from airdesk.labels import GestureLabelFile

TCN_V2_EVIDENCE_TARGETS = (
    "intentional_motion",
    "stroke_left",
    "stroke_right",
    "start",
    "end",
)
TCN_V2_WINDOW_TARGETS = ("background",) + TCN_V2_EVIDENCE_TARGETS
TCN_V2_CONTROL_TARGETS = {"intentional_motion", "start", "end"}


def tcn_v2_frame_evidence_targets(
    rows: list[FrameFeatureRow],
    labels: GestureLabelFile | None,
    *,
    target_assignment: str,
    motion_gate_min_dx_per_hand_scale: float,
    motion_gate_min_direction_consistency: float,
    evidence_targets: tuple[str, ...] = TCN_V2_EVIDENCE_TARGETS,
) -> list[dict[str, float]]:
    """Return decoder-facing evidence targets aligned to feature rows."""
    evidence = [
        tcn_v2_evidence_for_row(
            row,
            labels,
            target_assignment=target_assignment,
            motion_gate_min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
            motion_gate_min_direction_consistency=motion_gate_min_direction_consistency,
            evidence_targets=evidence_targets,
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
            for index, (row, frame_evidence) in enumerate(zip(rows, evidence, strict=True))
            if event.start_time <= row.timestamp <= event.end_time
            and is_tracked_feature_row(row)
            and frame_evidence["intentional_motion"] > 0
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


def tcn_v2_evidence_for_row(
    row: FrameFeatureRow,
    labels: GestureLabelFile | None,
    *,
    target_assignment: str,
    motion_gate_min_dx_per_hand_scale: float,
    motion_gate_min_direction_consistency: float,
    evidence_targets: tuple[str, ...] = TCN_V2_EVIDENCE_TARGETS,
) -> dict[str, float]:
    """Return multi-label TCN v2 evidence for one feature row."""
    frame_evidence = _empty_tcn_v2_evidence(evidence_targets)
    if not is_tracked_feature_row(row):
        return frame_evidence
    phase = (
        row.phase
        if row.phase and row.phase != "background"
        else _phase_at(labels, row.timestamp)
    )
    event = row.event or _event_at(labels, row.timestamp)
    active_targets = {
        target
        for target in evidence_targets
        if target not in TCN_V2_CONTROL_TARGETS and _target_matches_label(target, phase, event)
    }
    if target_assignment == "motion-gated":
        active_targets = {
            target
            for target in active_targets
            if row_motion_matches_tcn_target(
                row,
                target,
                min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
                min_direction_consistency=motion_gate_min_direction_consistency,
            )
        }
    intentional_motion = bool(
        active_targets
        or phase in {"recovery", "reset", "release", "cooldown"}
    )
    for target in active_targets:
        frame_evidence[target] = 1.0
    if "intentional_motion" in frame_evidence:
        frame_evidence["intentional_motion"] = 1.0 if intentional_motion else 0.0
    return frame_evidence


def _target_matches_label(target: str, phase: str, event: str) -> bool:
    if target in (phase, event):
        return True
    if target == "stroke_left":
        return phase == "stroke_left" or event == "swipe_left"
    if target == "stroke_right":
        return phase == "stroke_right" or event == "swipe_right"
    return False


def row_motion_matches_tcn_target(
    row: FrameFeatureRow,
    target: str,
    *,
    min_dx_per_hand_scale: float,
    min_direction_consistency: float,
) -> bool:
    """Return whether a row has enough recent motion for a weak label target."""
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


def tcn_v2_window_target_from_evidence(evidence: dict[str, float]) -> str:
    """Return the collapsed display/window target for one frame's v2 evidence."""
    for target, value in evidence.items():
        if target not in TCN_V2_CONTROL_TARGETS and value > 0:
            return target
    for target in ("start", "end", "intentional_motion"):
        if evidence.get(target, 0.0) > 0:
            return target
    return "background"


def tcn_v2_evidence_counts(
    evidence_by_row: list[dict[str, float]],
    evidence_targets: tuple[str, ...],
) -> dict[str, int]:
    """Count positive frame-evidence heads for one feature source."""
    counts = {target: 0 for target in evidence_targets}
    for frame_evidence in evidence_by_row:
        for target in evidence_targets:
            if frame_evidence.get(target, 0.0) > 0:
                counts[target] += 1
    return counts


def _empty_tcn_v2_evidence(evidence_targets: tuple[str, ...]) -> dict[str, float]:
    return {target: 0.0 for target in evidence_targets}


def _event_at(labels: GestureLabelFile | None, timestamp: float) -> str:
    if labels is None:
        return ""
    for event in labels.event_labels:
        if event.label_type == "gesture" and event.start_time <= timestamp <= event.end_time:
            return event.gesture
    return ""


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
