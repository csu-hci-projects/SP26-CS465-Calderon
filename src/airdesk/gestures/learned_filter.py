"""Mode-aware filters for learned/custom gesture evidence heads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TCN_V2_CUSTOM_DISPLAY_NAMES: dict[str, str] = {
    "ipn_b0a": "Point one finger",
    "ipn_b0b": "Point two fingers",
    "ipn_g01": "Click one finger",
    "ipn_g02": "Click two fingers",
    "ipn_g03": "Throw up",
    "ipn_g04": "Throw down",
    "ipn_g05": "Throw left",
    "ipn_g06": "Throw right",
    "ipn_g07": "Open twice",
    "ipn_g08": "Double click one finger",
    "ipn_g09": "Double click two fingers",
    "ipn_g10": "Zoom in",
    "ipn_g11": "Zoom out",
}

TCN_V2_META_HEADS = frozenset(
    {"intentional_motion", "start", "end", "stroke_left", "stroke_right"}
)
TCN_V2_IPN_HEADS = tuple(TCN_V2_CUSTOM_DISPLAY_NAMES)
TCN_V2_RECOGNITION_MODES: dict[str, frozenset[str]] = {
    "all-ipn": frozenset(TCN_V2_IPN_HEADS),
    "command": frozenset({"ipn_g05", "ipn_g06"}),
    "cursor": frozenset(
        {"ipn_b0a", "ipn_b0b", "ipn_g01", "ipn_g02", "ipn_g08", "ipn_g09"}
    ),
    "zoom-media": frozenset({"ipn_g10", "ipn_g11"}),
}


@dataclass(frozen=True)
class LearnedRecognition:
    """A filtered custom-head recognition safe enough to show as confident."""

    target: str
    name: str
    score: float
    margin: float
    mode: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "name": self.name,
            "score": self.score,
            "margin": self.margin,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class LearnedRecognitionFrame:
    """Per-frame filter explanation for preview and replay diagnostics."""

    mode: str
    enabled_heads: tuple[str, ...]
    top: tuple[str, float] | None
    top_enabled: tuple[str, float] | None
    runner_up_enabled: tuple[str, float] | None
    threshold: float
    margin: float
    persistence_count: int
    required_persistence: int
    suppressed_reason: str | None
    recognition: LearnedRecognition | None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "enabled_heads": list(self.enabled_heads),
            "top": _pair_to_dict(self.top),
            "top_enabled": _pair_to_dict(self.top_enabled),
            "runner_up_enabled": _pair_to_dict(self.runner_up_enabled),
            "threshold": self.threshold,
            "margin": self.margin,
            "persistence_count": self.persistence_count,
            "required_persistence": self.required_persistence,
            "suppressed_reason": self.suppressed_reason,
            "recognition": (
                self.recognition.to_dict() if self.recognition is not None else None
            ),
        }


@dataclass(frozen=True)
class LearnedRecognitionFilterConfig:
    """Static policy for filtering custom TCN v2 heads."""

    mode: str = "command"
    score_threshold: float = 0.80
    head_thresholds: dict[str, float] = field(default_factory=dict)
    margin: float = 0.15
    persistence_frames: int = 3
    cooldown_seconds: float = 1.5
    debug_all_heads: bool = False

    def __post_init__(self) -> None:
        if self.mode not in TCN_V2_RECOGNITION_MODES:
            modes = ", ".join(sorted(TCN_V2_RECOGNITION_MODES))
            raise ValueError(f"unknown recognition mode {self.mode!r}; expected one of {modes}")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score threshold must be in [0, 1]")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if self.persistence_frames <= 0:
            raise ValueError("persistence-frames must be positive")
        if self.cooldown_seconds < 0:
            raise ValueError("recognition-cooldown-seconds must be non-negative")
        for head, threshold in self.head_thresholds.items():
            if head not in TCN_V2_CUSTOM_DISPLAY_NAMES:
                raise ValueError(f"unknown TCN v2 custom head {head!r}")
            if not 0 <= threshold <= 1:
                raise ValueError(f"threshold for {head} must be in [0, 1]")

    @property
    def enabled_heads(self) -> frozenset[str]:
        if self.debug_all_heads:
            return TCN_V2_RECOGNITION_MODES["all-ipn"]
        return TCN_V2_RECOGNITION_MODES[self.mode]

    def threshold_for(self, target: str) -> float:
        return self.head_thresholds.get(target, self.score_threshold)


@dataclass
class _HandRecognitionState:
    target: str = ""
    count: int = 0
    last_emitted_target: str = ""
    last_emitted_at: float = -1_000_000.0


class LearnedRecognitionFilter:
    """Stateful persistence/cooldown filter for custom learned heads."""

    def __init__(self, config: LearnedRecognitionFilterConfig) -> None:
        self.config = config
        self._states: dict[str, _HandRecognitionState] = {}

    def update(
        self,
        *,
        hand_id: str,
        evidence: dict[str, float],
        timestamp: float,
    ) -> LearnedRecognitionFrame:
        enabled = self.config.enabled_heads
        top = top_custom_evidence(evidence, limit=1)
        top_pair = top[0] if top else None
        enabled_scores = top_custom_evidence(evidence, enabled_heads=enabled, limit=2)
        top_enabled = enabled_scores[0] if enabled_scores else None
        runner_up = enabled_scores[1] if len(enabled_scores) > 1 else None
        threshold = (
            self.config.threshold_for(top_enabled[0])
            if top_enabled is not None
            else self.config.score_threshold
        )
        margin = (
            top_enabled[1] - runner_up[1]
            if top_enabled is not None and runner_up is not None
            else top_enabled[1]
            if top_enabled is not None
            else 0.0
        )
        state = self._states.setdefault(hand_id, _HandRecognitionState())
        if top_enabled is not None and top_enabled[0] == state.target:
            state.count += 1
        else:
            state.target = top_enabled[0] if top_enabled is not None else ""
            state.count = 1 if top_enabled is not None else 0

        reason = _suppression_reason(
            top_pair=top_pair,
            top_enabled=top_enabled,
            threshold=threshold,
            margin=margin,
            required_margin=self.config.margin,
            count=state.count,
            required_count=self.config.persistence_frames,
            enabled_heads=enabled,
            last_target=state.last_emitted_target,
            last_at=state.last_emitted_at,
            now=timestamp,
            cooldown_seconds=self.config.cooldown_seconds,
        )
        recognition: LearnedRecognition | None = None
        if reason is None and top_enabled is not None:
            target, score = top_enabled
            recognition = LearnedRecognition(
                target=target,
                name=tcn_v2_evidence_display_name(target),
                score=score,
                margin=margin,
                mode=self.config.mode,
            )
            state.last_emitted_target = target
            state.last_emitted_at = timestamp
        return LearnedRecognitionFrame(
            mode=self.config.mode,
            enabled_heads=tuple(sorted(enabled)),
            top=top_pair,
            top_enabled=top_enabled,
            runner_up_enabled=runner_up,
            threshold=threshold,
            margin=margin,
            persistence_count=state.count,
            required_persistence=self.config.persistence_frames,
            suppressed_reason=reason,
            recognition=recognition,
        )


def top_custom_evidence(
    evidence: dict[str, float],
    *,
    enabled_heads: frozenset[str] | None = None,
    limit: int = 4,
) -> list[tuple[str, float]]:
    scored = [
        (target, float(score))
        for target, score in evidence.items()
        if target not in TCN_V2_META_HEADS
        and (enabled_heads is None or target in enabled_heads)
    ]
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]


def tcn_v2_evidence_display_name(target: str) -> str:
    if target in TCN_V2_CUSTOM_DISPLAY_NAMES:
        return TCN_V2_CUSTOM_DISPLAY_NAMES[target]
    cleaned = target.removeprefix("ipn_").replace("_", " ").strip()
    return cleaned.title() if cleaned else target


def parse_head_thresholds(value: str | None) -> dict[str, float]:
    if value is None or not value.strip():
        return {}
    thresholds: dict[str, float] = {}
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("head thresholds must use head=value entries")
        head, raw_threshold = (item.strip() for item in part.split("=", 1))
        if not head:
            raise ValueError("head threshold entry is missing a head name")
        thresholds[head] = float(raw_threshold)
    return thresholds


def evaluate_tcn_v2_live_jsonl(
    path: Any,
    *,
    config: LearnedRecognitionFilterConfig,
) -> dict[str, object]:
    import json

    recognizer = LearnedRecognitionFilter(config)
    predictions = 0
    recognitions: list[dict[str, object]] = []
    raw_top_above_threshold: dict[str, int] = {}
    suppressed: dict[str, int] = {}
    enabled_top_counts: dict[str, int] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            event = record.get("event", {})
            if event.get("event_type") != "tcn_v2_live_prediction":
                continue
            payload = event.get("payload", {})
            prediction = payload.get("prediction", {})
            evidence = prediction.get("evidence", {})
            if not isinstance(evidence, dict):
                continue
            predictions += 1
            hand_id = str(prediction.get("hand_id") or "hand")
            timestamp = float(prediction.get("end_time", payload.get("relative_time_seconds", 0.0)))
            frame = recognizer.update(
                hand_id=hand_id,
                evidence={str(key): float(value) for key, value in evidence.items()},
                timestamp=timestamp,
            )
            if frame.top is not None and frame.top[1] >= config.score_threshold:
                raw_top_above_threshold[frame.top[0]] = (
                    raw_top_above_threshold.get(frame.top[0], 0) + 1
                )
            if frame.top_enabled is not None:
                enabled_top_counts[frame.top_enabled[0]] = (
                    enabled_top_counts.get(frame.top_enabled[0], 0) + 1
                )
            if frame.suppressed_reason is not None:
                suppressed[frame.suppressed_reason] = suppressed.get(frame.suppressed_reason, 0) + 1
            if frame.recognition is not None:
                recognitions.append(
                    {
                        **frame.recognition.to_dict(),
                        "hand_id": hand_id,
                        "timestamp": timestamp,
                    }
                )
    recognition_counts: dict[str, int] = {}
    for item in recognitions:
        target = str(item["target"])
        recognition_counts[target] = recognition_counts.get(target, 0) + 1
    return {
        "mode": config.mode,
        "debug_all_heads": config.debug_all_heads,
        "enabled_heads": sorted(config.enabled_heads),
        "predictions": predictions,
        "recognitions": len(recognitions),
        "recognition_counts": recognition_counts,
        "raw_top_above_threshold": raw_top_above_threshold,
        "enabled_top_counts": enabled_top_counts,
        "suppressed": suppressed,
        "events": recognitions,
    }


def _suppression_reason(
    *,
    top_pair: tuple[str, float] | None,
    top_enabled: tuple[str, float] | None,
    threshold: float,
    margin: float,
    required_margin: float,
    count: int,
    required_count: int,
    enabled_heads: frozenset[str],
    last_target: str,
    last_at: float,
    now: float,
    cooldown_seconds: float,
) -> str | None:
    if top_pair is None:
        return "no_custom_head"
    if top_enabled is None:
        return "top_head_not_enabled" if top_pair[0] not in enabled_heads else "no_enabled_head"
    if top_pair[0] != top_enabled[0]:
        return "top_head_suppressed_by_mode"
    if top_enabled[1] < threshold:
        return "below_threshold"
    if margin < required_margin:
        return "margin_too_small"
    if count < required_count:
        return "persistence_pending"
    if top_enabled[0] == last_target and now - last_at < cooldown_seconds:
        return "cooldown"
    return None


def _pair_to_dict(pair: tuple[str, float] | None) -> dict[str, object] | None:
    if pair is None:
        return None
    return {
        "target": pair[0],
        "name": tcn_v2_evidence_display_name(pair[0]),
        "score": pair[1],
    }
