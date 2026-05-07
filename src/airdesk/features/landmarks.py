"""Landmark-derived frame features for dynamic gesture recognition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import dist

from airdesk.gestures.primitives import INDEX_TIP, THUMB_TIP, StaticHandPoseRecognizer
from airdesk.labels import GestureLabelFile
from airdesk.state.types import NormalizedHand, TrackingFrame


@dataclass(frozen=True)
class FrameFeatureRow:
    """One deterministic feature row for a tracking frame."""

    frame_index: int
    timestamp: float
    dt: float
    tracking_present: int
    hand_count: int
    hand_id: str
    confidence: float
    palm_x: float
    palm_y: float
    palm_z: float
    palm_vx: float
    palm_vy: float
    palm_speed: float
    palm_ax: float
    palm_ay: float
    palm_window_dx: float
    palm_window_dx_per_hand_scale: float
    palm_window_peak_abs_vx: float
    palm_window_direction_consistency: float
    index_rel_x: float
    index_rel_y: float
    index_rel_vx: float
    index_rel_vy: float
    pinch_distance: float
    pinch_velocity: float
    hand_scale: float
    extended_fingers: int
    folded_fingers: int
    phase: str
    event: str

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass
class _HandHistory:
    timestamp: float | None = None
    palm_x: float = 0.0
    palm_y: float = 0.0
    palm_vx: float = 0.0
    palm_vy: float = 0.0
    index_rel_x: float = 0.0
    index_rel_y: float = 0.0
    pinch_distance: float = 0.0
    tracked_rows: list[tuple[float, float, float, float]] | None = None


@dataclass
class FeatureRowStream:
    """Stateful frame-to-feature converter for live/replay rolling windows."""

    labels: GestureLabelFile | None = None

    def __post_init__(self) -> None:
        self._history_by_hand_id: dict[str, _HandHistory] = {}
        self._no_hand_history = _HandHistory()
        self._pose_recognizer = StaticHandPoseRecognizer()
        self._frame_index = 0

    def append(self, frame: TrackingFrame) -> FrameFeatureRow:
        """Convert one tracking frame into the first feature row.

        This keeps the older single-row live callers working. New recognizer paths should
        call :meth:`append_rows` so both visible hands are represented.
        """
        return self.append_rows(frame)[0]

    def append_rows(self, frame: TrackingFrame) -> list[FrameFeatureRow]:
        """Convert one tracking frame into one no-hand row or one row per visible hand."""
        frame_index = self._frame_index
        self._frame_index += 1
        if not frame.hands:
            self._history_by_hand_id.clear()
            return [
                _row_for_frame(
                    frame_index=frame_index,
                    frame=frame,
                    hand=None,
                    history=self._no_hand_history,
                    pose_recognizer=self._pose_recognizer,
                    labels=self.labels,
                )
            ]

        visible_hand_ids = {hand.hand_id for hand in frame.hands}
        for missing_hand_id in set(self._history_by_hand_id) - visible_hand_ids:
            del self._history_by_hand_id[missing_hand_id]
        self._no_hand_history = _HandHistory()
        return [
            _row_for_frame(
                frame_index=frame_index,
                frame=frame,
                hand=hand,
                history=self._history_by_hand_id.setdefault(hand.hand_id, _HandHistory()),
                pose_recognizer=self._pose_recognizer,
                labels=self.labels,
            )
            for hand in frame.hands
        ]


def extract_feature_rows(
    frames: list[TrackingFrame],
    *,
    labels: GestureLabelFile | None = None,
) -> list[FrameFeatureRow]:
    """Extract deterministic per-frame features from tracking frames."""
    stream = FeatureRowStream(labels=labels)
    rows: list[FrameFeatureRow] = []
    for frame in frames:
        rows.extend(stream.append_rows(frame))
    return rows


def _row_for_frame(
    *,
    frame_index: int,
    frame: TrackingFrame,
    hand: NormalizedHand | None,
    history: _HandHistory,
    pose_recognizer: StaticHandPoseRecognizer,
    labels: GestureLabelFile | None,
) -> FrameFeatureRow:
    phase = _phase_at(labels, frame.timestamp)
    event = _event_at(labels, frame.timestamp)
    if hand is None:
        dt = _dt(history, frame.timestamp)
        history.timestamp = frame.timestamp
        history.tracked_rows = []
        return FrameFeatureRow(
            frame_index=frame_index,
            timestamp=frame.timestamp,
            dt=dt,
            tracking_present=0,
            hand_count=0,
            hand_id="",
            confidence=0.0,
            palm_x=0.0,
            palm_y=0.0,
            palm_z=0.0,
            palm_vx=0.0,
            palm_vy=0.0,
            palm_speed=0.0,
            palm_ax=0.0,
            palm_ay=0.0,
            palm_window_dx=0.0,
            palm_window_dx_per_hand_scale=0.0,
            palm_window_peak_abs_vx=0.0,
            palm_window_direction_consistency=0.0,
            index_rel_x=0.0,
            index_rel_y=0.0,
            index_rel_vx=0.0,
            index_rel_vy=0.0,
            pinch_distance=0.0,
            pinch_velocity=0.0,
            hand_scale=0.0,
            extended_fingers=0,
            folded_fingers=0,
            phase=phase,
            event=event,
        )

    landmarks = hand.landmarks.landmarks
    palm_x, palm_y, palm_z = hand.palm_center
    dt = _dt(history, frame.timestamp)
    palm_vx = (palm_x - history.palm_x) / dt if dt > 0 else 0.0
    palm_vy = (palm_y - history.palm_y) / dt if dt > 0 else 0.0
    palm_ax = (palm_vx - history.palm_vx) / dt if dt > 0 else 0.0
    palm_ay = (palm_vy - history.palm_vy) / dt if dt > 0 else 0.0
    index_rel_x, index_rel_y = _index_relative(landmarks, palm_x, palm_y)
    index_rel_vx = (index_rel_x - history.index_rel_x) / dt if dt > 0 else 0.0
    index_rel_vy = (index_rel_y - history.index_rel_y) / dt if dt > 0 else 0.0
    pinch_distance = _pinch_distance(hand)
    pinch_velocity = (pinch_distance - history.pinch_distance) / dt if dt > 0 else 0.0
    hand_scale = max(0.0, hand.bbox[2] - hand.bbox[0])
    window_motion = _update_window_motion(
        history,
        timestamp=frame.timestamp,
        palm_x=palm_x,
        palm_vx=palm_vx,
        hand_scale=hand_scale,
    )
    pose_features = pose_recognizer.features_for_hand(hand)
    extended = pose_features.extended_fingers if pose_features else 0
    folded = pose_features.folded_fingers if pose_features else 0

    history.timestamp = frame.timestamp
    history.palm_x = palm_x
    history.palm_y = palm_y
    history.palm_vx = palm_vx
    history.palm_vy = palm_vy
    history.index_rel_x = index_rel_x
    history.index_rel_y = index_rel_y
    history.pinch_distance = pinch_distance

    return FrameFeatureRow(
        frame_index=frame_index,
        timestamp=frame.timestamp,
        dt=dt,
        tracking_present=1,
        hand_count=len(frame.hands),
        hand_id=hand.hand_id,
        confidence=hand.confidence or 0.0,
        palm_x=palm_x,
        palm_y=palm_y,
        palm_z=palm_z,
        palm_vx=palm_vx,
        palm_vy=palm_vy,
        palm_speed=(palm_vx**2 + palm_vy**2) ** 0.5,
        palm_ax=palm_ax,
        palm_ay=palm_ay,
        palm_window_dx=window_motion[0],
        palm_window_dx_per_hand_scale=window_motion[1],
        palm_window_peak_abs_vx=window_motion[2],
        palm_window_direction_consistency=window_motion[3],
        index_rel_x=index_rel_x,
        index_rel_y=index_rel_y,
        index_rel_vx=index_rel_vx,
        index_rel_vy=index_rel_vy,
        pinch_distance=pinch_distance,
        pinch_velocity=pinch_velocity,
        hand_scale=hand_scale,
        extended_fingers=extended,
        folded_fingers=folded,
        phase=phase,
        event=event,
    )


def _dt(history: _HandHistory, timestamp: float) -> float:
    if history.timestamp is None or timestamp <= history.timestamp:
        return 0.0
    return timestamp - history.timestamp


def _update_window_motion(
    history: _HandHistory,
    *,
    timestamp: float,
    palm_x: float,
    palm_vx: float,
    hand_scale: float,
    window_seconds: float = 0.8,
) -> tuple[float, float, float, float]:
    tracked_rows = history.tracked_rows or []
    tracked_rows.append((timestamp, palm_x, palm_vx, hand_scale))
    cutoff = timestamp - window_seconds
    tracked_rows = [row for row in tracked_rows if row[0] >= cutoff]
    history.tracked_rows = tracked_rows
    if len(tracked_rows) < 2:
        return 0.0, 0.0, abs(palm_vx), 0.0
    palm_dx = tracked_rows[-1][1] - tracked_rows[0][1]
    mean_scale = sum(row[3] for row in tracked_rows) / len(tracked_rows)
    peak_abs_vx = max(abs(row[2]) for row in tracked_rows)
    return (
        palm_dx,
        palm_dx / mean_scale if mean_scale > 0 else 0.0,
        peak_abs_vx,
        _direction_consistency(tracked_rows, palm_dx),
    )


def _direction_consistency(
    tracked_rows: list[tuple[float, float, float, float]],
    palm_dx: float,
) -> float:
    if len(tracked_rows) < 2 or palm_dx == 0:
        return 0.0
    expected_sign = 1 if palm_dx > 0 else -1
    steps = [
        tracked_rows[index][1] - tracked_rows[index - 1][1]
        for index in range(1, len(tracked_rows))
    ]
    moving_steps = [step for step in steps if step != 0]
    if not moving_steps:
        return 0.0
    aligned = sum(1 for step in moving_steps if (1 if step > 0 else -1) == expected_sign)
    return aligned / len(moving_steps)


def _index_relative(landmarks: object, palm_x: float, palm_y: float) -> tuple[float, float]:
    if len(landmarks) <= INDEX_TIP:
        return 0.0, 0.0
    index_tip = landmarks[INDEX_TIP]
    return index_tip.x - palm_x, index_tip.y - palm_y


def _pinch_distance(hand: NormalizedHand) -> float:
    landmarks = hand.landmarks.landmarks
    if len(landmarks) <= max(THUMB_TIP, INDEX_TIP):
        return 0.0
    thumb = landmarks[THUMB_TIP]
    index = landmarks[INDEX_TIP]
    return dist((thumb.x, thumb.y, thumb.z), (index.x, index.y, index.z))


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


def _event_at(labels: GestureLabelFile | None, timestamp: float) -> str:
    if labels is None:
        return ""
    for event in labels.event_labels:
        if event.start_time <= timestamp <= event.end_time:
            return event.gesture
    return ""
