"""Primitive landmark facts for deterministic live control."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import dist

from airdesk.gestures.primitives import FINGER_MCPS, FINGER_TIPS, INDEX_TIP, THUMB_TIP
from airdesk.state.types import NormalizedHand, TrackingFrame

WRIST = 0
MIDDLE_TIP = 12
FINGER_JOINTS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
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
    min_fist_closed_fingers: int = 3
    fist_closed_finger_threshold: float = 0.58
    fist_tip_spread_max: float = 0.30
    fist_tip_cluster_max: float = 0.34
    fist_tip_cluster_scale_max: float = 0.95
    fist_thumb_cluster_max: float = 0.22
    fist_thumb_cluster_scale_max: float = 0.95
    fist_open_score_max: float = 0.42
    fist_confidence_threshold: float = 0.68
    forming_fist_confidence_threshold: float = 0.45
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
            pose_evidence=pose_evidence,
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
        pose_evidence: dict[str, object],
    ) -> tuple[set[str], str | None, str | None]:
        """Apply control-lane priority so noisy landmark facts do not all fire."""
        fist_evidence = pose_evidence.get("fist")
        forming_fist = (
            isinstance(fist_evidence, dict)
            and bool(fist_evidence.get("forming_fist"))
            and bool(raw_poses & PINCH_POSES)
        )
        if forming_fist and "fist" not in raw_poses:
            return set(), "forming_fist_pinch_conflict", "closed hand is not a clean pinch"

        if "fist" in raw_poses:
            competing_scores = [
                pose_scores[pose]
                for pose in raw_poses
                if pose != "fist" and pose in pose_scores
            ]
            strongest_competitor = max(competing_scores, default=0.0)
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
        palm_scale = self._palm_scale(hand)
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
        legacy_fold_score = sum(
            _clamp(depth / self.fist_fold_threshold) for depth in fold_depths
        ) / len(fold_depths)
        curl_facts = [
            self._finger_curl_fact(
                hand,
                mcp_index=mcp_index,
                pip_index=pip_index,
                dip_index=dip_index,
                tip_index=tip_index,
                palm_scale=palm_scale,
            )
            for mcp_index, pip_index, dip_index, tip_index in FINGER_JOINTS
        ]
        finger_curl_scores = [fact["score"] for fact in curl_facts]
        closed_fingers = sum(
            score >= self.fist_closed_finger_threshold for score in finger_curl_scores
        )
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
        tip_cluster_ratio = tip_cluster / palm_scale
        thumb_cluster_ratio = min(thumb_to_cluster, thumb_to_tip) / palm_scale
        tip_spread_ok = tip_spread <= self.fist_tip_spread_max
        tip_cluster_ok = tip_cluster_ratio <= self.fist_tip_cluster_scale_max
        thumb_ok = (
            thumb_to_cluster <= self.fist_thumb_cluster_max
            or thumb_to_tip <= self.fist_thumb_cluster_max
            or thumb_cluster_ratio <= self.fist_thumb_cluster_scale_max
        )
        open_score = self._open_palm_score(
            extended_fingers=self._extended_fingers(hand),
            finger_spread=tip_spread,
        )
        closed_shape_score = sum(finger_curl_scores) / len(finger_curl_scores)
        cluster_score = _inverse_ratio_score(
            tip_cluster_ratio,
            closed_at=0.42,
            open_at=self.fist_tip_cluster_scale_max,
        )
        thumb_support_score = _inverse_ratio_score(
            thumb_cluster_ratio,
            closed_at=0.35,
            open_at=self.fist_thumb_cluster_scale_max,
        )
        low_open_score = 1.0 - _clamp(open_score / self.fist_open_score_max)
        score = (
            closed_shape_score * 0.52
            + cluster_score * 0.18
            + thumb_support_score * 0.14
            + low_open_score * 0.10
            + legacy_fold_score * 0.06
        )
        forming_fist = (
            score >= self.forming_fist_confidence_threshold
            and closed_fingers >= 2
            and thumb_support_score > 0.35
            and open_score <= 0.75
        )
        evidence: dict[str, object] = {
            "score": round(score, 4),
            "closed_fingers": closed_fingers,
            "required_closed_fingers": self.min_fist_closed_fingers,
            "closed_finger_threshold": self.fist_closed_finger_threshold,
            "finger_curl_scores": [round(score, 4) for score in finger_curl_scores],
            "finger_curl_facts": [
                {
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in fact.items()
                }
                for fact in curl_facts
            ],
            "strong_folded_fingers": strong_folded,
            "required_folded_fingers": self.min_fist_folded_fingers,
            "fold_depths": [round(depth, 4) for depth in fold_depths],
            "legacy_fold_score": round(legacy_fold_score, 4),
            "tip_minus_pip_depths": [round(depth, 4) for depth in pip_depths],
            "tip_minus_dip_depths": [round(depth, 4) for depth in dip_depths],
            "palm_scale": round(palm_scale, 4),
            "tip_spread": round(tip_spread, 4),
            "tip_spread_max": self.fist_tip_spread_max,
            "tip_cluster": round(tip_cluster, 4),
            "tip_cluster_max": self.fist_tip_cluster_max,
            "tip_cluster_ratio": round(tip_cluster_ratio, 4),
            "tip_cluster_scale_max": self.fist_tip_cluster_scale_max,
            "cluster_score": round(cluster_score, 4),
            "thumb_to_cluster": round(thumb_to_cluster, 4),
            "thumb_to_tip": round(thumb_to_tip, 4),
            "thumb_cluster_max": self.fist_thumb_cluster_max,
            "thumb_cluster_ratio": round(thumb_cluster_ratio, 4),
            "thumb_cluster_scale_max": self.fist_thumb_cluster_scale_max,
            "thumb_support_score": round(thumb_support_score, 4),
            "open_score": round(open_score, 4),
            "open_score_max": self.fist_open_score_max,
            "tip_spread_ok": tip_spread_ok,
            "tip_cluster_ok": tip_cluster_ok,
            "thumb_ok": thumb_ok,
            "forming_fist": forming_fist,
        }
        return score, evidence

    def _is_fist_candidate(self, *, fist_score: float, evidence: dict[str, object]) -> bool:
        return (
            int(evidence["closed_fingers"]) >= self.min_fist_closed_fingers
            and bool(evidence["tip_spread_ok"])
            and bool(evidence["tip_cluster_ok"])
            and bool(evidence["thumb_ok"])
            and float(evidence["cluster_score"]) >= 0.10
            and float(evidence["open_score"]) <= self.fist_open_score_max
            and fist_score >= self.fist_confidence_threshold
        )

    def _finger_curl_fact(
        self,
        hand: NormalizedHand,
        *,
        mcp_index: int,
        pip_index: int,
        dip_index: int,
        tip_index: int,
        palm_scale: float,
    ) -> dict[str, float]:
        landmarks = hand.landmarks.landmarks
        mcp = landmarks[mcp_index]
        pip = landmarks[pip_index]
        dip = landmarks[dip_index]
        tip = landmarks[tip_index]
        mcp_to_tip = dist((mcp.x, mcp.y, mcp.z), (tip.x, tip.y, tip.z))
        mcp_to_dip = dist((mcp.x, mcp.y, mcp.z), (dip.x, dip.y, dip.z))
        chain_length = (
            dist((mcp.x, mcp.y, mcp.z), (pip.x, pip.y, pip.z))
            + dist((pip.x, pip.y, pip.z), (dip.x, dip.y, dip.z))
            + dist((dip.x, dip.y, dip.z), (tip.x, tip.y, tip.z))
        )
        straightness = 1.0 if chain_length <= 0 else mcp_to_tip / chain_length
        tip_to_mcp_ratio = mcp_to_tip / palm_scale
        dip_to_mcp_ratio = mcp_to_dip / palm_scale
        tip_close_score = _inverse_ratio_score(
            tip_to_mcp_ratio,
            closed_at=0.55,
            open_at=1.28,
        )
        joint_close_score = _inverse_ratio_score(
            dip_to_mcp_ratio,
            closed_at=0.42,
            open_at=0.88,
        )
        bend_score = _inverse_ratio_score(
            straightness,
            closed_at=0.58,
            open_at=0.88,
        )
        vertical_score = _clamp((tip.y - mcp.y) / self.fist_fold_threshold)
        score = (
            tip_close_score * 0.38
            + joint_close_score * 0.30
            + bend_score * 0.22
            + vertical_score * 0.10
        )
        return {
            "finger_tip": float(tip_index),
            "score": score,
            "tip_to_mcp_ratio": tip_to_mcp_ratio,
            "dip_to_mcp_ratio": dip_to_mcp_ratio,
            "straightness": straightness,
            "tip_close_score": tip_close_score,
            "joint_close_score": joint_close_score,
            "bend_score": bend_score,
            "vertical_score": vertical_score,
        }

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

    @staticmethod
    def _palm_scale(hand: NormalizedHand) -> float:
        landmarks = hand.landmarks.landmarks
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        pinky_mcp = landmarks[17]
        wrist = landmarks[WRIST]
        palm_width = dist(
            (index_mcp.x, index_mcp.y, index_mcp.z),
            (pinky_mcp.x, pinky_mcp.y, pinky_mcp.z),
        )
        palm_length = dist(
            (wrist.x, wrist.y, wrist.z),
            (middle_mcp.x, middle_mcp.y, middle_mcp.z),
        )
        bbox_width = max(0.0, hand.bbox[2] - hand.bbox[0])
        bbox_height = max(0.0, hand.bbox[3] - hand.bbox[1])
        return max(palm_width, palm_length, (bbox_width**2 + bbox_height**2) ** 0.5 * 0.35, 0.05)

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


def _inverse_ratio_score(value: float, *, closed_at: float, open_at: float) -> float:
    if open_at <= closed_at:
        return 0.0
    return _clamp((open_at - value) / (open_at - closed_at))
