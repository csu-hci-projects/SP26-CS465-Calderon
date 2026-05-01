"""Rule-based primitive static gesture recognizers."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

from airdesk.state.types import GestureCandidate, NormalizedHand, TrackingFrame

FINGER_TIPS = (8, 12, 16, 20)
FINGER_MCPS = (5, 9, 13, 17)
INDEX_TIP = 8
INDEX_MCP = 5
THUMB_TIP = 4


@dataclass(frozen=True)
class HandPoseFeatures:
    """Primitive features used for static hand-pose tuning."""

    hand_id: str
    extended_fingers: int
    folded_fingers: int
    finger_spread: float
    pinch_distance: float
    hand_confidence: float | None
    handedness: str | None
    index_direction: str | None = None

    def to_flat_dict(self) -> dict[str, float | int | str]:
        return {
            "hand_id": self.hand_id,
            "extended": self.extended_fingers,
            "folded": self.folded_fingers,
            "spread": round(self.finger_spread, 4),
            "pinch": round(self.pinch_distance, 4),
            "confidence": round(self.hand_confidence, 4)
            if self.hand_confidence is not None
            else "unknown",
            "handedness": self.handedness or "unknown",
            "index_direction": self.index_direction or "unknown",
        }


@dataclass(frozen=True)
class StaticHandPoseRecognizer:
    """Recognizes the first Sprint 0 synthetic hand-pose primitives."""

    extended_threshold: float = 0.08
    pinch_threshold: float = 0.06
    name: str = "static-hand-pose"

    def recognize(self, frame: TrackingFrame) -> list[GestureCandidate]:
        candidates: list[GestureCandidate] = []
        for hand in frame.hands:
            candidates.extend(self._recognize_hand(frame.timestamp, hand))
        return candidates

    def _recognize_hand(self, timestamp: float, hand: NormalizedHand) -> list[GestureCandidate]:
        results: list[GestureCandidate] = []
        features = self.features_for_hand(hand)
        if features is None:
            return results

        if features.extended_fingers >= 4 and features.finger_spread >= 0.16:
            confidence = min(1.0, 0.75 + features.finger_spread)
            results.append(
                GestureCandidate(
                    name="open_palm",
                    confidence=confidence,
                    timestamp=timestamp,
                    hand_id=hand.hand_id,
                    metadata={
                        "extended_fingers": features.extended_fingers,
                        "spread": features.finger_spread,
                    },
                )
            )

        if features.folded_fingers >= 4:
            results.append(
                GestureCandidate(
                    name="fist",
                    confidence=1.0,
                    timestamp=timestamp,
                    hand_id=hand.hand_id,
                    metadata={"folded_fingers": features.folded_fingers},
                )
            )

        if features.pinch_distance <= self.pinch_threshold:
            confidence = max(0.0, 1.0 - (features.pinch_distance / self.pinch_threshold))
            results.append(
                GestureCandidate(
                    name="pinch",
                    confidence=confidence,
                    timestamp=timestamp,
                    hand_id=hand.hand_id,
                    metadata={"pinch_distance": features.pinch_distance},
                )
            )

        if features.index_direction in {"left", "right"} and features.folded_fingers >= 3:
            results.append(
                GestureCandidate(
                    name=f"point_{features.index_direction}",
                    confidence=0.85,
                    timestamp=timestamp,
                    hand_id=hand.hand_id,
                    metadata={
                        "index_direction": features.index_direction,
                        "folded_fingers": features.folded_fingers,
                    },
                )
            )

        return results

    def features_for_frame(self, frame: TrackingFrame) -> list[HandPoseFeatures]:
        """Return primitive tuning features for each hand in a frame."""
        features: list[HandPoseFeatures] = []
        for hand in frame.hands:
            hand_features = self.features_for_hand(hand)
            if hand_features is not None:
                features.append(hand_features)
        return features

    def features_for_hand(self, hand: NormalizedHand) -> HandPoseFeatures | None:
        """Return primitive tuning features for one hand."""
        if len(hand.landmarks.landmarks) < 21:
            return None
        extended = self._extended_fingers(hand)
        return HandPoseFeatures(
            hand_id=hand.hand_id,
            extended_fingers=extended,
            folded_fingers=4 - extended,
            finger_spread=self._finger_spread(hand),
            pinch_distance=self._pinch_distance(hand),
            hand_confidence=hand.confidence,
            handedness=hand.handedness,
            index_direction=self._index_direction(hand),
        )

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
    def _pinch_distance(hand: NormalizedHand) -> float:
        landmarks = hand.landmarks.landmarks
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        return dist((thumb.x, thumb.y, thumb.z), (index.x, index.y, index.z))

    @staticmethod
    def _index_direction(hand: NormalizedHand) -> str | None:
        landmarks = hand.landmarks.landmarks
        index_tip = landmarks[INDEX_TIP]
        index_mcp = landmarks[INDEX_MCP]
        dx = index_tip.x - index_mcp.x
        dy = index_tip.y - index_mcp.y
        if abs(dx) < 0.12 or abs(dx) < abs(dy) * 1.25:
            return None
        return "right" if dx > 0 else "left"
