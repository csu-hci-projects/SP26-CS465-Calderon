"""Gesture label suggestion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from airdesk.recording.jsonl import iter_recording
from airdesk.state.types import TrackingFrame


@dataclass(frozen=True)
class StrokeLabelSuggestion:
    """Suggested stroke interval for one dynamic gesture recording."""

    gesture: str
    phase: str
    start_time: float
    end_time: float
    start_seconds: float
    end_seconds: float
    direction: str
    confidence: float
    displacement_x: float
    displacement_y: float


def suggest_stroke_label(
    recording: Path,
    *,
    gesture: str | None = None,
    min_duration: float = 0.25,
    max_duration: float = 1.25,
    pad_seconds: float = 0.08,
) -> StrokeLabelSuggestion:
    """Suggest the strongest palm-motion stroke window in a recording."""
    frames = _tracking_frames(recording)
    if len(frames) < 2:
        raise ValueError(f"{recording} does not contain enough tracking frames")

    inferred_gesture = gesture or _infer_gesture(recording, frames)
    if inferred_gesture is None:
        raise ValueError("gesture could not be inferred; pass --gesture swipe_left or swipe_right")

    hand_frames = [frame for frame in frames if frame.hands]
    if len(hand_frames) < 2:
        raise ValueError(f"{recording} does not contain enough hand-present frames")

    best: tuple[float, TrackingFrame, TrackingFrame] | None = None
    for start in hand_frames:
        for end in hand_frames:
            duration = end.timestamp - start.timestamp
            if duration < min_duration:
                continue
            if duration > max_duration:
                break
            start_hand = start.hands[0]
            end_hand = end.hands[0]
            dx = end_hand.palm_center[0] - start_hand.palm_center[0]
            dy = end_hand.palm_center[1] - start_hand.palm_center[1]
            score = (abs(dx) + abs(dy) * 0.35) / duration
            if best is None or score > best[0]:
                best = (score, start, end)

    if best is None:
        raise ValueError(
            f"no hand-motion window found between {min_duration:.2f}s and {max_duration:.2f}s"
        )

    _score, start_frame, end_frame = best
    start_hand = start_frame.hands[0]
    end_hand = end_frame.hands[0]
    dx = end_hand.palm_center[0] - start_hand.palm_center[0]
    dy = end_hand.palm_center[1] - start_hand.palm_center[1]
    session_start = frames[0].timestamp
    session_end = frames[-1].timestamp
    start_time = max(session_start, start_frame.timestamp - pad_seconds)
    end_time = min(session_end, end_frame.timestamp + pad_seconds)

    direction = "right" if dx > 0 else "left"
    return StrokeLabelSuggestion(
        gesture=inferred_gesture,
        phase=_phase_for_gesture(inferred_gesture),
        start_time=start_time,
        end_time=end_time,
        start_seconds=start_time - session_start,
        end_seconds=end_time - session_start,
        direction=direction,
        confidence=min(1.0, (abs(dx) + abs(dy) * 0.35) / 0.25),
        displacement_x=dx,
        displacement_y=dy,
    )


def _tracking_frames(recording: Path) -> list[TrackingFrame]:
    frames: list[TrackingFrame] = []
    for record in iter_recording(recording):
        if record.kind == "tracking_frame":
            assert isinstance(record.payload, TrackingFrame)
            frames.append(record.payload)
    return frames


def _infer_gesture(recording: Path, frames: list[TrackingFrame]) -> str | None:
    name = recording.stem.replace("-", "_")
    if "swipe_left" in name:
        return "swipe_left"
    if "swipe_right" in name:
        return "swipe_right"
    for frame in frames:
        if not frame.hands:
            continue
        # Fall back to observed palm displacement if the file name does not carry a label.
        first = frame.hands[0].palm_center[0]
        last = next(
            (later.hands[0].palm_center[0] for later in reversed(frames) if later.hands),
            first,
        )
        return "swipe_right" if last > first else "swipe_left"
    return None


def _phase_for_gesture(gesture: str) -> str:
    if gesture == "swipe_left":
        return "stroke_left"
    if gesture == "swipe_right":
        return "stroke_right"
    raise ValueError(f"unsupported suggested gesture={gesture}")
