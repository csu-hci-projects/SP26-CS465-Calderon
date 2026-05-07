"""Replayable probability/candidate event decoding for continuous gesture spotting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from airdesk.state.types import GestureCandidate

BACKGROUND_TARGETS = frozenset({"background", "recovery", "reset", "cooldown", ""})
DEFAULT_EVENT_MAP = {
    "stroke_left": "swipe_left",
    "stroke_right": "swipe_right",
}


@dataclass(frozen=True)
class EventDecoderConfig:
    """Thresholds and timing controls for probability-to-event decoding."""

    activation_threshold: float = 0.55
    release_threshold: float = 0.35
    min_peak_confidence: float = 0.60
    min_event_separation_seconds: float = 0.50
    recovery_seconds: float = 0.25
    cooldown_seconds: float = 0.50
    repeated_fire_window_seconds: float = 0.80

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecoderFrame:
    """One timestamped target-score snapshot for the event decoder."""

    timestamp: float
    scores: dict[str, float]
    source_id: str = ""
    window_start: float | None = None
    window_end: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _ActivePeak:
    name: str
    start_time: float
    peak_time: float
    peak_confidence: float
    peak_frame: DecoderFrame
    last_active_time: float


class EventDecoder:
    """Decode target probabilities into one-shot gesture events."""

    def __init__(
        self,
        config: EventDecoderConfig | None = None,
        *,
        event_map: dict[str, str] | None = None,
    ) -> None:
        self.config = config or EventDecoderConfig()
        self.event_map = event_map or DEFAULT_EVENT_MAP

    def decode(self, frames: list[DecoderFrame]) -> list[GestureCandidate]:
        """Return committed gesture events from timestamp-ordered score frames."""
        self._validate_config()
        events: list[GestureCandidate] = []
        active: _ActivePeak | None = None
        last_commit_by_name: dict[str, float] = {}

        for frame in sorted(frames, key=lambda item: item.timestamp):
            target, confidence = _best_non_background(frame.scores)
            event_name = self.event_map.get(target, target)
            if active is None:
                if confidence < self.config.activation_threshold:
                    continue
                if not self._separation_ok(event_name, frame.timestamp, last_commit_by_name):
                    continue
                active = _ActivePeak(
                    name=event_name,
                    start_time=frame.window_start or frame.timestamp,
                    peak_time=frame.timestamp,
                    peak_confidence=confidence,
                    peak_frame=frame,
                    last_active_time=frame.timestamp,
                )
                continue

            if event_name == active.name and confidence >= self.config.release_threshold:
                active.last_active_time = frame.timestamp
                if confidence >= active.peak_confidence:
                    active.peak_time = frame.timestamp
                    active.peak_confidence = confidence
                    active.peak_frame = frame
                continue

            should_commit = (
                active.peak_confidence >= self.config.min_peak_confidence
                and frame.timestamp - active.last_active_time >= self.config.recovery_seconds
            )
            if not should_commit:
                continue
            if should_commit and self._separation_ok(
                active.name,
                active.peak_time,
                last_commit_by_name,
            ):
                events.append(self._candidate_from_peak(active))
                last_commit_by_name[active.name] = active.peak_time
            active = None

            if confidence >= self.config.activation_threshold and self._separation_ok(
                event_name,
                frame.timestamp,
                last_commit_by_name,
            ):
                active = _ActivePeak(
                    name=event_name,
                    start_time=frame.window_start or frame.timestamp,
                    peak_time=frame.timestamp,
                    peak_confidence=confidence,
                    peak_frame=frame,
                    last_active_time=frame.timestamp,
                )

        if (
            active is not None
            and active.peak_confidence >= self.config.min_peak_confidence
            and self._separation_ok(active.name, active.peak_time, last_commit_by_name)
        ):
            events.append(self._candidate_from_peak(active))
        return events

    def _candidate_from_peak(self, active: _ActivePeak) -> GestureCandidate:
        peak = active.peak_frame
        return GestureCandidate(
            name=active.name,
            confidence=active.peak_confidence,
            timestamp=active.peak_time,
            hand_id=None,
            metadata={
                "recognizer": "event_decoder",
                "source_id": peak.source_id,
                "window_start": active.start_time,
                "window_end": peak.window_end or active.peak_time,
                "peak_time": active.peak_time,
                "scores": peak.scores,
                **peak.metadata,
            },
        )

    def _separation_ok(
        self,
        event_name: str,
        timestamp: float,
        last_commit_by_name: dict[str, float],
    ) -> bool:
        last = last_commit_by_name.get(event_name)
        if last is None:
            return True
        return timestamp - last >= max(
            self.config.min_event_separation_seconds,
            self.config.cooldown_seconds,
        )

    def _validate_config(self) -> None:
        if not 0 <= self.config.release_threshold <= self.config.activation_threshold <= 1:
            raise ValueError("decoder thresholds must satisfy 0 <= release <= activation <= 1")
        if not 0 <= self.config.min_peak_confidence <= 1:
            raise ValueError("min_peak_confidence must be in [0, 1]")
        if self.config.min_event_separation_seconds < 0:
            raise ValueError("min_event_separation_seconds must be non-negative")
        if self.config.recovery_seconds < 0 or self.config.cooldown_seconds < 0:
            raise ValueError("recovery_seconds and cooldown_seconds must be non-negative")


def frames_from_candidates(candidates: list[GestureCandidate]) -> list[DecoderFrame]:
    """Convert scored recognizer candidates into decoder frames."""
    frames: list[DecoderFrame] = []
    for candidate in candidates:
        window_start = _float_or_none(candidate.metadata.get("window_start"))
        window_end = _float_or_none(candidate.metadata.get("window_end"))
        frames.append(
            DecoderFrame(
                timestamp=candidate.timestamp,
                scores={candidate.name: candidate.confidence},
                source_id=str(candidate.metadata.get("recognizer", "")),
                window_start=window_start,
                window_end=window_end,
                metadata={"candidate": candidate.to_dict()},
            )
        )
    return frames


def _best_non_background(scores: dict[str, float]) -> tuple[str, float]:
    best_target = ""
    best_score = 0.0
    for target, score in scores.items():
        if target in BACKGROUND_TARGETS:
            continue
        if score > best_score:
            best_target = target
            best_score = score
    return best_target, best_score


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
