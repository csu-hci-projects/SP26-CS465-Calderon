"""Primitive landmark facts for deterministic live control."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import dist

from airdesk.gestures.primitives import FINGER_MCPS, FINGER_TIPS, INDEX_TIP, THUMB_TIP
from airdesk.state.types import NormalizedHand, TrackingFrame

MIDDLE_TIP = 12
FINGER_PIPS = (6, 10, 14, 18)
FINGER_DIPS = (7, 11, 15, 19)
PINCH_POSES = frozenset({"index_pinch", "middle_pinch"})


@dataclass(frozen=True)
class ControlPoseFeatures:
    """Observable per-hand facts used by the control grammar."""

    hand_id: str
    timestamp: float
    palm_x: float
    palm_y: float
    palm_zone: str
    palm_vertical_zone: str
    extended_fingers: int
    folded_fingers: int
    finger_spread: float
    index_pinch_distance: float
    middle_pinch_distance: float
    poses: frozenset[str]
    suppressed_poses: frozenset[str] = frozenset()
    pose_scores: dict[str, float] = field(default_factory=dict)
    pose_evidence: dict[str, object] = field(default_factory=dict)
    ambiguity: str | None = None
    suppression_reason: str | None = None
    hand_confidence: float | None = None
    handedness: str | None = None

    def sees(self) -> str:
        """Return a compact user-facing summary of the current pose facts."""
        active = "relaxed" if not self.poses else ",".join(sorted(self.poses))
        ambiguity = f" ambiguous={self.ambiguity}" if self.ambiguity else ""
        suppressed = (
            f" suppressed={','.join(sorted(self.suppressed_poses))}"
            if self.suppressed_poses
            else ""
        )
        return (
            f"{self.hand_id}: {active} {self.palm_zone}/{self.palm_vertical_zone}"
            f"{ambiguity}{suppressed}"
        )

    def to_log_dict(self) -> dict[str, object]:
        """Return JSON-safe diagnostics for this hand's control pose facts."""
        return {
            "hand_id": self.hand_id,
            "palm": {
                "x": round(self.palm_x, 4),
                "y": round(self.palm_y, 4),
                "zone": self.palm_zone,
                "vertical_zone": self.palm_vertical_zone,
            },
            "poses": sorted(self.poses),
            "suppressed_poses": sorted(self.suppressed_poses),
            "pose_scores": {
                pose: round(score, 4) for pose, score in sorted(self.pose_scores.items())
            },
            "pose_evidence": self.pose_evidence,
            "ambiguity": self.ambiguity,
            "suppression_reason": self.suppression_reason,
            "extended_fingers": self.extended_fingers,
            "folded_fingers": self.folded_fingers,
            "finger_spread": round(self.finger_spread, 4),
            "index_pinch_distance": round(self.index_pinch_distance, 4),
            "middle_pinch_distance": round(self.middle_pinch_distance, 4),
            "hand_confidence": self.hand_confidence,
            "handedness": self.handedness,
        }


@dataclass(frozen=True)
class ControlPoseRecognizer:
    """Classify direct MediaPipe landmarks into deterministic control facts."""

    extended_threshold: float = 0.08
    fist_fold_threshold: float = 0.09
    open_spread_threshold: float = 0.16
    index_pinch_threshold: float = 0.06
    middle_pinch_threshold: float = 0.065
    min_fist_folded_fingers: int = 4
    fist_tip_spread_max: float = 0.30
    fist_tip_cluster_max: float = 0.34
    fist_thumb_cluster_max: float = 0.22
    fist_confidence_threshold: float = 0.72
    clean_pinch_confidence_threshold: float = 0.55
    conflict_margin: float = 0.18
    left_zone_max: float = 0.30
    right_zone_min: float = 0.70
    top_zone_max: float = 0.30
    bottom_zone_min: float = 0.70

    def features_for_frame(self, frame: TrackingFrame) -> list[ControlPoseFeatures]:
        """Return control features for each visible hand."""
        features: list[ControlPoseFeatures] = []
        for hand in frame.hands:
            hand_features = self.features_for_hand(hand, timestamp=frame.timestamp)
            if hand_features is not None:
                features.append(hand_features)
        return features

    def features_for_hand(
        self, hand: NormalizedHand, *, timestamp: float
    ) -> ControlPoseFeatures | None:
        """Return control features for one hand, or None for incomplete landmarks."""
        landmarks = hand.landmarks.landmarks
        if len(landmarks) < 21:
            return None

        palm_x, palm_y, _palm_z = hand.palm_center
        palm_zone = self._palm_zone(palm_x)
        palm_vertical_zone = self._palm_vertical_zone(palm_y)
        extended = self._extended_fingers(hand)
        fist_score, fist_evidence = self._fist_evidence(hand)
        folded = int(fist_evidence["strong_folded_fingers"])
        finger_spread = self._finger_spread(hand)
        index_pinch = self._distance(hand, THUMB_TIP, INDEX_TIP)
        middle_pinch = self._distance(hand, THUMB_TIP, MIDDLE_TIP)
        open_score = self._open_palm_score(
            extended_fingers=extended,
            finger_spread=finger_spread,
        )
        index_pinch_score = self._pinch_score(index_pinch, self.index_pinch_threshold)
        middle_pinch_score = self._pinch_score(middle_pinch, self.middle_pinch_threshold)
        pose_scores = {
            "fist": fist_score,
            "open_palm": open_score,
            "index_pinch": index_pinch_score,
            "middle_pinch": middle_pinch_score,
        }
        pose_evidence: dict[str, object] = {
            "fist": fist_evidence,
            "open_palm": {
                "score": round(open_score, 4),
                "extended_fingers": extended,
                "finger_spread": round(finger_spread, 4),
                "spread_threshold": self.open_spread_threshold,
            },
            "index_pinch": {
                "score": round(index_pinch_score, 4),
                "distance": round(index_pinch, 4),
                "threshold": self.index_pinch_threshold,
            },
            "middle_pinch": {
                "score": round(middle_pinch_score, 4),
                "distance": round(middle_pinch, 4),
                "threshold": self.middle_pinch_threshold,
            },
        }

        raw_poses: set[str] = set()
        if self._is_fist_candidate(fist_score=fist_score, evidence=fist_evidence):
            raw_poses.add("fist")
        if extended >= 4 and finger_spread >= self.open_spread_threshold:
            raw_poses.add("open_palm")
            if palm_zone in {"left", "right"}:
                raw_poses.add(f"sideways_open_palm_{palm_zone}")
        if index_pinch_score > 0.0:
            raw_poses.add("index_pinch")
        if middle_pinch_score > 0.0:
            raw_poses.add("middle_pinch")

        poses, ambiguity, suppression_reason = self._resolve_control_poses(
            raw_poses,
            pose_scores=pose_scores,
        )
        suppressed_poses = raw_poses - poses

        return ControlPoseFeatures(
            hand_id=hand.hand_id,
            timestamp=timestamp,
            palm_x=palm_x,
            palm_y=palm_y,
            palm_zone=palm_zone,
            palm_vertical_zone=palm_vertical_zone,
            extended_fingers=extended,
            folded_fingers=folded,
            finger_spread=finger_spread,
            index_pinch_distance=index_pinch,
            middle_pinch_distance=middle_pinch,
            poses=frozenset(poses),
            suppressed_poses=frozenset(suppressed_poses),
            pose_scores=pose_scores,
            pose_evidence=pose_evidence,
            ambiguity=ambiguity,
            suppression_reason=suppression_reason,
            hand_confidence=hand.confidence,
            handedness=hand.handedness,
        )

    def _resolve_control_poses(
        self,
        raw_poses: set[str],
        *,
        pose_scores: dict[str, float],
    ) -> tuple[set[str], str | None, str | None]:
        """Apply control-lane priority so noisy landmark facts do not all fire."""
        if "fist" in raw_poses:
            competing_scores = [
                pose_scores[pose]
                for pose in raw_poses
                if pose != "fist" and pose in pose_scores
            ]
            strongest_competitor = max(competing_scores, default=0.0)
            if strongest_competitor > 0.0 and (
                pose_scores["fist"] - strongest_competitor < self.conflict_margin
            ):
                return set(), "fist_pose_conflict", "fist evidence is not dominant"
            reason = (
                "fist dominated overlapping pose evidence"
                if strongest_competitor > 0.0
                else None
            )
            return {"fist"}, None, reason

        if len(raw_poses & PINCH_POSES) > 1:
            strongest = max(raw_poses & PINCH_POSES, key=lambda pose: pose_scores[pose])
            weakest = min(raw_poses & PINCH_POSES, key=lambda pose: pose_scores[pose])
            if pose_scores[strongest] - pose_scores[weakest] < self.conflict_margin:
                return set(), "index_middle_pinch_conflict", "pinch evidence is not dominant"
            raw_poses = (raw_poses - PINCH_POSES) | {strongest}

        side_poses = {"sideways_open_palm_left", "sideways_open_palm_right"}
        sideways_poses = {
            pose for pose in raw_poses if pose in side_poses
        }
        if sideways_poses:
            strongest_pinch = max(
                (pose_scores[pose] for pose in raw_poses & PINCH_POSES),
                default=0.0,
            )
            if strongest_pinch > 0.0 and (
                pose_scores["open_palm"] - strongest_pinch < self.conflict_margin
            ):
                return set(), "sideways_palm_pinch_conflict", "sideways palm is not dominant"
            reason = "sideways palm suppressed pinch evidence" if strongest_pinch > 0.0 else None
            return {"open_palm", *sideways_poses}, None, reason

        pinch_poses = {
            pose for pose in raw_poses if pose in {"index_pinch", "middle_pinch"}
        }
        if pinch_poses:
            strongest_pinch_score = max(pose_scores[pose] for pose in pinch_poses)
            if "open_palm" in raw_poses and (
                strongest_pinch_score < self.clean_pinch_confidence_threshold
            ):
                return set(), "open_palm_pinch_conflict", "pinch evidence is not clean"
            reason = "clean pinch suppressed open palm" if "open_palm" in raw_poses else None
            return pinch_poses, None, reason

        if "open_palm" in raw_poses:
            return {"open_palm"}, None, None
        return set(), None, None

    def _extended_fingers(self, hand: NormalizedHand) -> int:
        landmarks = hand.landmarks.landmarks
        count = 0
        for tip_index, mcp_index in zip(FINGER_TIPS, FINGER_MCPS, strict=True):
            if landmarks[tip_index].y < landmarks[mcp_index].y - self.extended_threshold:
                count += 1
        return count

    def _strongly_folded_fingers(self, hand: NormalizedHand) -> int:
        landmarks = hand.landmarks.landmarks
        count = 0
        for tip_index, mcp_index in zip(FINGER_TIPS, FINGER_MCPS, strict=True):
            if landmarks[tip_index].y > landmarks[mcp_index].y + self.fist_fold_threshold:
                count += 1
        return count

    def _fist_evidence(self, hand: NormalizedHand) -> tuple[float, dict[str, object]]:
        landmarks = hand.landmarks.landmarks
        fold_depths = [
            landmarks[tip_index].y - landmarks[mcp_index].y
            for tip_index, mcp_index in zip(FINGER_TIPS, FINGER_MCPS, strict=True)
        ]
        pip_depths = [
            landmarks[tip_index].y - landmarks[pip_index].y
            for tip_index, pip_index in zip(FINGER_TIPS, FINGER_PIPS, strict=True)
        ]
        dip_depths = [
            landmarks[tip_index].y - landmarks[dip_index].y
            for tip_index, dip_index in zip(FINGER_TIPS, FINGER_DIPS, strict=True)
        ]
        strong_folded = sum(depth >= self.fist_fold_threshold for depth in fold_depths)
        fold_score = sum(
            _clamp(depth / self.fist_fold_threshold) for depth in fold_depths
        ) / len(fold_depths)
        tip_points = [landmarks[index] for index in FINGER_TIPS]
        tip_cluster = max(
            dist((a.x, a.y, a.z), (b.x, b.y, b.z))
            for offset, a in enumerate(tip_points)
            for b in tip_points[offset + 1 :]
        )
        tip_spread = self._finger_spread(hand)
        thumb = landmarks[THUMB_TIP]
        thumb_to_tip = min(
            dist((thumb.x, thumb.y, thumb.z), (tip.x, tip.y, tip.z))
            for tip in tip_points
        )
        thumb_to_cluster = dist(
            (thumb.x, thumb.y, thumb.z),
            (
                sum(tip.x for tip in tip_points) / len(tip_points),
                sum(tip.y for tip in tip_points) / len(tip_points),
                sum(tip.z for tip in tip_points) / len(tip_points),
            ),
        )
        tip_spread_ok = tip_spread <= self.fist_tip_spread_max
        tip_cluster_ok = tip_cluster <= self.fist_tip_cluster_max
        thumb_ok = (
            thumb_to_cluster <= self.fist_thumb_cluster_max
            or thumb_to_tip <= self.fist_thumb_cluster_max
        )
        score = (
            fold_score * 0.55
            + (1.0 if tip_cluster_ok else 0.0) * 0.20
            + (1.0 if tip_spread_ok else 0.0) * 0.15
            + (1.0 if thumb_ok else 0.0) * 0.10
        )
        evidence: dict[str, object] = {
            "score": round(score, 4),
            "strong_folded_fingers": strong_folded,
            "required_folded_fingers": self.min_fist_folded_fingers,
            "fold_depths": [round(depth, 4) for depth in fold_depths],
            "tip_minus_pip_depths": [round(depth, 4) for depth in pip_depths],
            "tip_minus_dip_depths": [round(depth, 4) for depth in dip_depths],
            "tip_spread": round(tip_spread, 4),
            "tip_spread_max": self.fist_tip_spread_max,
            "tip_cluster": round(tip_cluster, 4),
            "tip_cluster_max": self.fist_tip_cluster_max,
            "thumb_to_cluster": round(thumb_to_cluster, 4),
            "thumb_to_tip": round(thumb_to_tip, 4),
            "thumb_cluster_max": self.fist_thumb_cluster_max,
            "tip_spread_ok": tip_spread_ok,
            "tip_cluster_ok": tip_cluster_ok,
            "thumb_ok": thumb_ok,
        }
        return score, evidence

    def _is_fist_candidate(self, *, fist_score: float, evidence: dict[str, object]) -> bool:
        return (
            int(evidence["strong_folded_fingers"]) >= self.min_fist_folded_fingers
            and bool(evidence["tip_spread_ok"])
            and bool(evidence["tip_cluster_ok"])
            and bool(evidence["thumb_ok"])
            and fist_score >= self.fist_confidence_threshold
        )

    def _open_palm_score(self, *, extended_fingers: int, finger_spread: float) -> float:
        return (
            _clamp(extended_fingers / 4.0) * 0.65
            + _clamp(finger_spread / self.open_spread_threshold) * 0.35
        )

    @staticmethod
    def _pinch_score(distance_value: float, threshold: float) -> float:
        if distance_value > threshold:
            return 0.0
        return _clamp(1.0 - (distance_value / threshold))

    @staticmethod
    def _finger_spread(hand: NormalizedHand) -> float:
        landmarks = hand.landmarks.landmarks
        xs = [landmarks[index].x for index in FINGER_TIPS]
        return max(xs) - min(xs)

    @staticmethod
    def _distance(hand: NormalizedHand, first: int, second: int) -> float:
        landmarks = hand.landmarks.landmarks
        a = landmarks[first]
        b = landmarks[second]
        return dist((a.x, a.y, a.z), (b.x, b.y, b.z))

    def _palm_zone(self, palm_x: float) -> str:
        if palm_x <= self.left_zone_max:
            return "left"
        if palm_x >= self.right_zone_min:
            return "right"
        return "center"

    def _palm_vertical_zone(self, palm_y: float) -> str:
        if palm_y <= self.top_zone_max:
            return "top"
        if palm_y >= self.bottom_zone_min:
            return "bottom"
        return "middle"


def _clamp(value: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))
