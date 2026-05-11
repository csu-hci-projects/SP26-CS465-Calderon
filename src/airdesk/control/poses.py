"""Primitive landmark facts for deterministic live control."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

from airdesk.gestures.primitives import FINGER_MCPS, FINGER_TIPS, INDEX_TIP, THUMB_TIP
from airdesk.state.types import NormalizedHand, TrackingFrame

MIDDLE_TIP = 12


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
    hand_confidence: float | None = None
    handedness: str | None = None

    def sees(self) -> str:
        """Return a compact user-facing summary of the current pose facts."""
        active = "relaxed" if not self.poses else ",".join(sorted(self.poses))
        suppressed = (
            f" suppressed={','.join(sorted(self.suppressed_poses))}"
            if self.suppressed_poses
            else ""
        )
        return f"{self.hand_id}: {active} {self.palm_zone}/{self.palm_vertical_zone}{suppressed}"


@dataclass(frozen=True)
class ControlPoseRecognizer:
    """Classify direct MediaPipe landmarks into deterministic control facts."""

    extended_threshold: float = 0.08
    open_spread_threshold: float = 0.16
    index_pinch_threshold: float = 0.06
    middle_pinch_threshold: float = 0.065
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
        folded = 4 - extended
        finger_spread = self._finger_spread(hand)
        index_pinch = self._distance(hand, THUMB_TIP, INDEX_TIP)
        middle_pinch = self._distance(hand, THUMB_TIP, MIDDLE_TIP)

        raw_poses: set[str] = set()
        if folded >= 4:
            raw_poses.add("fist")
        if extended >= 4 and finger_spread >= self.open_spread_threshold:
            raw_poses.add("open_palm")
            if palm_zone in {"left", "right"}:
                raw_poses.add(f"sideways_open_palm_{palm_zone}")
        if index_pinch <= self.index_pinch_threshold:
            raw_poses.add("index_pinch")
        if middle_pinch <= self.middle_pinch_threshold:
            raw_poses.add("middle_pinch")

        poses = self._resolve_control_poses(raw_poses)
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
            hand_confidence=hand.confidence,
            handedness=hand.handedness,
        )

    @staticmethod
    def _resolve_control_poses(raw_poses: set[str]) -> set[str]:
        """Apply control-lane priority so noisy landmark facts do not all fire."""
        if "fist" in raw_poses:
            return {"fist"}

        side_poses = {"sideways_open_palm_left", "sideways_open_palm_right"}
        sideways_poses = {
            pose for pose in raw_poses if pose in side_poses
        }
        if sideways_poses:
            return {"open_palm", *sideways_poses}

        pinch_poses = {
            pose for pose in raw_poses if pose in {"index_pinch", "middle_pinch"}
        }
        if pinch_poses:
            return pinch_poses

        if "open_palm" in raw_poses:
            return {"open_palm"}
        return set()

    def _extended_fingers(self, hand: NormalizedHand) -> int:
        landmarks = hand.landmarks.landmarks
        count = 0
        for tip_index, mcp_index in zip(FINGER_TIPS, FINGER_MCPS, strict=True):
            if landmarks[tip_index].y < landmarks[mcp_index].y - self.extended_threshold:
                count += 1
        return count

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
