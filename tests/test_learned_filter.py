from __future__ import annotations

import json
from pathlib import Path

from airdesk.gestures.learned_filter import (
    TCN_V2_RECOGNITION_MODES,
    LearnedRecognitionFilter,
    LearnedRecognitionFilterConfig,
    evaluate_tcn_v2_live_jsonl,
    parse_head_thresholds,
)


def test_learned_filter_respects_mode_membership() -> None:
    filter_ = LearnedRecognitionFilter(
        LearnedRecognitionFilterConfig(
            mode="command",
            score_threshold=0.8,
            margin=0.1,
            persistence_frames=1,
        )
    )

    frame = filter_.update(
        hand_id="hand-0",
        evidence={"ipn_g03": 0.95, "ipn_g05": 0.70},
        timestamp=1.0,
    )

    assert frame.recognition is None
    assert frame.suppressed_reason == "top_head_suppressed_by_mode"
    assert frame.top == ("ipn_g03", 0.95)
    assert frame.top_enabled == ("ipn_g05", 0.70)


def test_learned_filter_requires_threshold_margin_and_persistence() -> None:
    filter_ = LearnedRecognitionFilter(
        LearnedRecognitionFilterConfig(
            mode="command",
            score_threshold=0.8,
            head_thresholds={"ipn_g05": 0.85},
            margin=0.2,
            persistence_frames=2,
            cooldown_seconds=1.0,
        )
    )

    first = filter_.update(
        hand_id="hand-0",
        evidence={"ipn_g05": 0.90, "ipn_g06": 0.78},
        timestamp=1.0,
    )
    second = filter_.update(
        hand_id="hand-0",
        evidence={"ipn_g05": 0.91, "ipn_g06": 0.60},
        timestamp=1.2,
    )
    third = filter_.update(
        hand_id="hand-0",
        evidence={"ipn_g05": 0.92, "ipn_g06": 0.50},
        timestamp=1.4,
    )
    fourth = filter_.update(
        hand_id="hand-0",
        evidence={"ipn_g05": 0.93, "ipn_g06": 0.50},
        timestamp=2.6,
    )

    assert first.suppressed_reason == "margin_too_small"
    assert second.recognition is not None
    assert second.recognition.target == "ipn_g05"
    assert third.suppressed_reason == "cooldown"
    assert fourth.recognition is not None


def test_parse_head_thresholds() -> None:
    assert parse_head_thresholds("ipn_g05=0.85, ipn_g06=0.9") == {
        "ipn_g05": 0.85,
        "ipn_g06": 0.9,
    }


def test_cursor_mode_excludes_point_heads_but_keeps_clicks_and_zooms() -> None:
    assert "ipn_b0a" not in TCN_V2_RECOGNITION_MODES["cursor"]
    assert "ipn_b0b" not in TCN_V2_RECOGNITION_MODES["cursor"]
    assert {"ipn_g01", "ipn_g02", "ipn_g10", "ipn_g11"}.issubset(
        TCN_V2_RECOGNITION_MODES["cursor"]
    )


def test_point_heads_are_suppressed_from_custom_top_evidence() -> None:
    filter_ = LearnedRecognitionFilter(
        LearnedRecognitionFilterConfig(
            mode="cursor",
            score_threshold=0.5,
            margin=0.1,
            persistence_frames=1,
        )
    )

    frame = filter_.update(
        hand_id="hand-0",
        evidence={"ipn_b0a": 0.99, "ipn_g10": 0.72, "ipn_g01": 0.40},
        timestamp=1.0,
    )

    assert frame.top == ("ipn_g10", 0.72)
    assert frame.recognition is not None
    assert frame.recognition.target == "ipn_g10"


def test_tcn_v2_live_jsonl_replay_scores_filtered_recognitions(tmp_path: Path) -> None:
    log = tmp_path / "live.jsonl"
    records = [
        _prediction_record(1.0, {"ipn_g05": 0.90, "ipn_g06": 0.40}),
        _prediction_record(1.2, {"ipn_g05": 0.91, "ipn_g06": 0.41}),
        _prediction_record(1.4, {"ipn_g03": 0.96, "ipn_g05": 0.88}),
    ]
    log.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    summary = evaluate_tcn_v2_live_jsonl(
        log,
        config=LearnedRecognitionFilterConfig(
            mode="command",
            score_threshold=0.8,
            margin=0.2,
            persistence_frames=2,
            cooldown_seconds=1.0,
        ),
    )

    assert summary["predictions"] == 3
    assert summary["recognition_counts"] == {"ipn_g05": 1}
    assert summary["raw_top_above_threshold"] == {"ipn_g03": 1, "ipn_g05": 2}
    assert summary["raw_top_motion"] == {
        "ipn_g03": {"dx_flat": 1},
        "ipn_g05": {"dx_pos": 2},
    }
    assert summary["suppressed"]["persistence_pending"] == 1
    assert summary["suppressed"]["top_head_suppressed_by_mode"] == 1


def _prediction_record(timestamp: float, evidence: dict[str, float]) -> dict[str, object]:
    full_evidence = {
        "intentional_motion": 0.9,
        "start": 0.2,
        "end": 0.1,
        **evidence,
    }
    return {
        "kind": "event",
        "version": 1,
        "event": {
            "event_type": "tcn_v2_live_prediction",
            "timestamp": timestamp,
            "session_id": "test",
            "payload": {
                "features": {"dx_scale": 0.5 if evidence.get("ipn_g05", 0.0) > 0.89 else 0.0},
                "prediction": {
                    "hand_id": "hand-0",
                    "start_time": timestamp - 0.2,
                    "end_time": timestamp,
                    "evidence": full_evidence,
                }
            },
        },
    }
