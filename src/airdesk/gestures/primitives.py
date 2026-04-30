"""Rule-based primitive static gesture recognizers."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

from airdesk.state.types import GestureCandidate, NormalizedHand, TrackingFrame

FINGER_TIPS = (8, 12, 16, 20)
FINGER_MCPS = (5, 9, 13, 17)
THUMB_TIP = 4
INDEX_TIP = 8


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
        if len(hand.landmarks.landmarks) < 21:
            return results

        extended = self._extended_fingers(hand)
        folded = 4 - extended
        spread = self._finger_spread(hand)

        if extended >= 4 and spread >= 0.16:
            confidence = min(1.0, 0.75 + spread)
            results.append(
                GestureCandidate(
                    name="open_palm",
                    confidence=confidence,
                    timestamp=timestamp,
                    hand_id=hand.hand_id,
                    metadata={"extended_fingers": extended, "spread": spread},
                )
            )

        if folded >= 4:
            results.append(
                GestureCandidate(
                    name="fist",
                    confidence=1.0,
                    timestamp=timestamp,
                    hand_id=hand.hand_id,
                    metadata={"folded_fingers": folded},
                )
            )

        pinch_distance = self._pinch_distance(hand)
        if pinch_distance <= self.pinch_threshold:
            confidence = max(0.0, 1.0 - (pinch_distance / self.pinch_threshold))
            results.append(
                GestureCandidate(
                    name="pinch",
                    confidence=confidence,
                    timestamp=timestamp,
                    hand_id=hand.hand_id,
                    metadata={"pinch_distance": pinch_distance},
                )
            )

        return results

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
