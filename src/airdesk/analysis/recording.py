"""Analysis utilities for replayed landmark recordings."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from pathlib import Path
from statistics import fmean, pstdev

from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.recording.jsonl import iter_recording
from airdesk.state.types import TrackingFrame

PRIMITIVES = ("open_palm", "fist", "pinch")
JITTER_LANDMARKS = (0, 9)


@dataclass(frozen=True)
class RecordingAnalysis:
    """Summary metrics for a replayable tracking recording."""

    frames: int = 0
    events: int = 0
    hand_frames: int = 0
    average_fps: float | None = None
    candidate_counts: dict[str, int] = field(default_factory=dict)
    longest_runs: dict[str, int] = field(default_factory=dict)
    landmark_jitter: dict[str, float] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, str | int | float]:
        data: dict[str, str | int | float] = {
            "frames": self.frames,
            "events": self.events,
            "hand_frames": self.hand_frames,
            "average_fps": (
                round(self.average_fps, 2) if self.average_fps is not None else "unknown"
            ),
        }
        for name in PRIMITIVES:
            data[f"{name}_count"] = self.candidate_counts.get(name, 0)
            data[f"{name}_longest_run"] = self.longest_runs.get(name, 0)
        for name, value in self.landmark_jitter.items():
            data[f"jitter_{name}"] = round(value, 5)
        return data


def analyze_recording(path: Path) -> RecordingAnalysis:
    """Analyze frame timing, hand presence, primitive candidates, and simple jitter."""
    recognizer = StaticHandPoseRecognizer()
    frames: list[TrackingFrame] = []
    event_count = 0
    candidate_counts = {name: 0 for name in PRIMITIVES}
    longest_runs = {name: 0 for name in PRIMITIVES}
    current_runs = {name: 0 for name in PRIMITIVES}
    positions: dict[int, list[tuple[float, float]]] = {index: [] for index in JITTER_LANDMARKS}

    for record in iter_recording(path):
        if record.kind == "event":
            event_count += 1
            continue
        assert isinstance(record.payload, TrackingFrame)
        frame = record.payload
        frames.append(frame)

        frame_candidates = {candidate.name for candidate in recognizer.recognize(frame)}
        for name in PRIMITIVES:
            if name in frame_candidates:
                candidate_counts[name] += 1
                current_runs[name] += 1
                longest_runs[name] = max(longest_runs[name], current_runs[name])
            else:
                current_runs[name] = 0

        if frame.hands:
            landmarks = frame.hands[0].landmarks.landmarks
            for index in JITTER_LANDMARKS:
                if index < len(landmarks):
                    positions[index].append((landmarks[index].x, landmarks[index].y))

    average_fps = _average_fps(frames)
    jitter = _landmark_jitter(positions)
    return RecordingAnalysis(
        frames=len(frames),
        events=event_count,
        hand_frames=sum(1 for frame in frames if frame.hands),
        average_fps=average_fps,
        candidate_counts=candidate_counts,
        longest_runs=longest_runs,
        landmark_jitter=jitter,
    )


def format_analysis(analysis: RecordingAnalysis) -> str:
    """Format analysis as one CLI-friendly line."""
    return " ".join(f"{key}={value}" for key, value in analysis.to_flat_dict().items())


def _average_fps(frames: list[TrackingFrame]) -> float | None:
    if len(frames) < 2:
        return None
    timestamps = [frame.timestamp for frame in frames]
    intervals = []
    for earlier, later in zip(timestamps, timestamps[1:], strict=False):
        if later > earlier:
            intervals.append(later - earlier)
    if not intervals:
        return None
    return 1.0 / fmean(intervals)


def _landmark_jitter(positions: dict[int, list[tuple[float, float]]]) -> dict[str, float]:
    jitter: dict[str, float] = {}
    for index, points in positions.items():
        if len(points) < 2:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        jitter[f"landmark_{index}"] = hypot(pstdev(xs), pstdev(ys))
    return jitter
