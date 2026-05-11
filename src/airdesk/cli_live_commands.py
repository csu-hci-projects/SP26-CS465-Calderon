"""Live tracking and recognizer-preview CLI commands."""

from __future__ import annotations

from pathlib import Path
from statistics import fmean
from time import monotonic
from typing import Annotated
from uuid import uuid4

import typer

from airdesk.analysis.tcn_v2 import tcn_v2_decoder_scores
from airdesk.capture.opencv import CameraSettings
from airdesk.cli_live import (
    _format_live_dtw_candidate,
    _format_live_tcn_prediction,
    _format_live_tcn_preview_predictions,
    _format_live_tcn_v2_candidate,
    _format_live_tcn_v2_prediction,
    _format_live_tcn_v2_preview_predictions,
    _format_tracker_timing,
    _is_live_tcn_gesture_target,
    _live_dtw_preview_status,
    _live_feature_streams,
    _live_tcn_preview_status,
    _live_tcn_v2_dashboard_snapshot,
    _live_tcn_v2_preview_status,
    _live_tcn_v2_row_motion_features,
    _max_tcn_v2_visible_evidence,
    _recognized_tcn_v2_custom_evidence,
    _show_live_tcn_prediction,
)
from airdesk.cli_tracking import _make_tracker
from airdesk.features import FeatureRowStream, FrameFeatureRow
from airdesk.gestures.decoder import DecoderFrame, EventDecoder, EventDecoderConfig
from airdesk.gestures.dtw import DtwGestureModel, DtwTemplateRecognizer
from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.ml import (
    CausalTcnLivePrediction,
    CausalTcnLivePredictor,
    CausalTcnV2LivePrediction,
    CausalTcnV2LivePredictor,
    MissingMlDependencyError,
)
from airdesk.recording.jsonl import JsonlRecordingWriter
from airdesk.state.types import EventLogEntry, GestureCandidate, TrackingFrame, utc_timestamp
from airdesk.tracking.mediapipe import (
    DEFAULT_HAND_LANDMARKER_DELEGATE,
    DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
    DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    DEFAULT_HAND_LANDMARKER_MODEL,
)


def register_live_tracking_commands(app: typer.Typer, gesture_app: typer.Typer) -> None:
    """Register live diagnostic commands on the existing public CLI apps."""

    @gesture_app.command("watch-tcn")
    def gesture_watch_tcn(
        tcn_model: Annotated[
            Path,
            typer.Option("--model", exists=True, readable=True, help="TCN checkpoint path."),
        ],
        backend: Annotated[str, typer.Option(help="Tracking backend to watch.")] = "mediapipe",
        device: Annotated[
            str,
            typer.Option(help="Camera path, numeric index, or replay path."),
        ] = "/dev/video0",
        width: Annotated[int | None, typer.Option(help="Requested capture width.")] = 640,
        height: Annotated[int | None, typer.Option(help="Requested capture height.")] = 480,
        fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = 30,
        fourcc: Annotated[
            str | None,
            typer.Option(help="Requested camera FOURCC, e.g. MJPG."),
        ] = "MJPG",
        hand_model_path: Annotated[
            Path,
            typer.Option(
                "--hand-model-path", help="MediaPipe Hand Landmarker .task model path."
            ),
        ] = DEFAULT_HAND_LANDMARKER_MODEL,
        max_num_hands: Annotated[
            int,
            typer.Option(help="Maximum number of hands for MediaPipe to track."),
        ] = 2,
        min_detection_confidence: Annotated[
            float,
            typer.Option(help="Minimum palm detection confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_presence_confidence: Annotated[
            float,
            typer.Option(help="Minimum hand landmark presence confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_tracking_confidence: Annotated[
            float,
            typer.Option(help="Minimum tracking confidence / box IoU threshold."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        hand_delegate: Annotated[
            str,
            typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
        ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
        auto_download_model: Annotated[
            bool,
            typer.Option(help="Download the MediaPipe model to --hand-model-path if missing."),
        ] = True,
        max_frames: Annotated[
            int | None, typer.Option(help="Stop after this many frames.")
        ] = None,
        show: Annotated[bool, typer.Option(help="Show an OpenCV live preview.")] = True,
        mirror: Annotated[bool, typer.Option(help="Mirror the preview window.")] = True,
        confidence_threshold: Annotated[
            float,
            typer.Option(help="Minimum confidence before printing a prediction line."),
        ] = 0.0,
        min_rows: Annotated[
            int,
            typer.Option(help="Minimum feature rows before the first TCN prediction."),
        ] = 4,
        include_background: Annotated[
            bool,
            typer.Option(help="Print background predictions as well as gestures."),
        ] = False,
        include_recovery: Annotated[
            bool,
            typer.Option(help="Print recovery/reset phase predictions."),
        ] = False,
        show_motion: Annotated[
            bool,
            typer.Option(help="Show hand-normalized horizontal motion in the HUD."),
        ] = True,
        profile_timing: Annotated[
            bool,
            typer.Option(help="Print per-prediction TCN timing diagnostics."),
        ] = False,
    ) -> None:
        """Watch live/replay TCN classification without triggering desktop actions."""
        if not 0 <= confidence_threshold <= 1:
            typer.echo("confidence-threshold must be in [0, 1]", err=True)
            raise typer.Exit(code=1)
        try:
            predictor = CausalTcnLivePredictor.load(tcn_model)
        except (MissingMlDependencyError, ValueError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

        tracker = _make_tracker(
            backend=backend,
            device=device,
            max_frames=max_frames,
            show=show,
            camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
            model_path=hand_model_path,
            auto_download_model=auto_download_model,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            delegate=hand_delegate,
            preview_mirror=mirror,
            preview_gestures=False,
        )
        stream = FeatureRowStream()
        rows: list[FrameFeatureRow] = []
        latest_predictions: dict[str, CausalTcnLivePrediction] = {}
        latest_rows_by_hand: dict[str, object] = {}
        state: dict[str, object] = {
            "status": "warming up",
            "alert": "",
            "alert_until": 0.0,
            "predictions": latest_predictions,
            "rows_by_hand": latest_rows_by_hand,
            "show_motion": show_motion,
            "stream_count": 0,
            "row_count": 0,
        }
        first_timestamp: float | None = None
        next_prediction_time_by_hand: dict[str, float] = {}

        if hasattr(tracker, "preview_status_provider"):
            tracker.preview_status_provider = lambda: _live_tcn_preview_status(state)  # type: ignore[attr-defined]

        typer.echo(
            "watching tcn "
            f"model={tcn_model} backend={backend} window={predictor.window_seconds:.2f}s "
            f"stride={predictor.stride_seconds:.2f}s targets={','.join(predictor.targets)}"
        )
        try:
            tracker.start()
            for frame in tracker.frames():
                first_timestamp = frame.timestamp if first_timestamp is None else first_timestamp
                rows.extend(stream.append_rows(frame))
                cutoff = frame.timestamp - predictor.window_seconds
                rows = [item for item in rows if item.timestamp >= cutoff]
                hand_streams = _live_feature_streams(rows)
                if not hand_streams:
                    state["status"] = "warming rows=0"
                    state["stream_count"] = 0
                    state["row_count"] = 0
                    continue
                state["stream_count"] = len(hand_streams)
                state["row_count"] = len(rows)
                state["status"] = _format_live_tcn_preview_predictions(state)
                for hand_id, hand_rows in hand_streams.items():
                    latest_rows_by_hand[hand_id] = hand_rows[-1]
                    if len(hand_rows) < min_rows:
                        continue
                    next_prediction_time = next_prediction_time_by_hand.get(hand_id)
                    if (
                        next_prediction_time is not None
                        and hand_rows[-1].timestamp < next_prediction_time
                    ):
                        continue
                    prediction_started_at = monotonic()
                    prediction = predictor.predict_rows(hand_rows)
                    prediction_ms = (monotonic() - prediction_started_at) * 1000
                    latest_predictions[hand_id] = prediction
                    state["status"] = _format_live_tcn_preview_predictions(state)
                    visible_prediction = _show_live_tcn_prediction(
                        prediction,
                        include_background=include_background,
                        include_recovery=include_recovery,
                        confidence_threshold=confidence_threshold,
                    )
                    if profile_timing and visible_prediction:
                        typer.echo(
                            f"tcn_predict_ms={prediction_ms:.2f} hand={hand_id} "
                            f"rows={len(hand_rows)} target={prediction.target} "
                            f"confidence={prediction.confidence:.3f}"
                        )
                    if _is_live_tcn_gesture_target(prediction.target) and visible_prediction:
                        state["alert"] = (
                            f"{hand_id} {prediction.target} {prediction.confidence:.2f}"
                        )
                        state["alert_until"] = monotonic() + 1.25
                    next_prediction_time_by_hand[hand_id] = (
                        hand_rows[-1].timestamp + predictor.stride_seconds
                    )
                    if visible_prediction:
                        typer.echo(
                            _format_live_tcn_prediction(
                                prediction,
                                first_timestamp=first_timestamp,
                            )
                        )
        except KeyboardInterrupt:
            typer.echo("interrupted")
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        finally:
            tracker.stop()

    @gesture_app.command("watch-tcn-v2")
    def gesture_watch_tcn_v2(
        tcn_model: Annotated[
            Path,
            typer.Option("--model", exists=True, readable=True, help="TCN v2 checkpoint path."),
        ],
        backend: Annotated[str, typer.Option(help="Tracking backend to watch.")] = "mediapipe",
        device: Annotated[
            str,
            typer.Option(help="Camera path, numeric index, or replay path."),
        ] = "/dev/video0",
        width: Annotated[int | None, typer.Option(help="Requested capture width.")] = 640,
        height: Annotated[int | None, typer.Option(help="Requested capture height.")] = 480,
        fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = 30,
        fourcc: Annotated[
            str | None,
            typer.Option(help="Requested camera FOURCC, e.g. MJPG."),
        ] = "MJPG",
        hand_model_path: Annotated[
            Path,
            typer.Option(
                "--hand-model-path", help="MediaPipe Hand Landmarker .task model path."
            ),
        ] = DEFAULT_HAND_LANDMARKER_MODEL,
        max_num_hands: Annotated[
            int,
            typer.Option(help="Maximum number of hands for MediaPipe to track."),
        ] = 2,
        min_detection_confidence: Annotated[
            float,
            typer.Option(help="Minimum palm detection confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_presence_confidence: Annotated[
            float,
            typer.Option(help="Minimum hand landmark presence confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_tracking_confidence: Annotated[
            float,
            typer.Option(help="Minimum tracking confidence / box IoU threshold."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        hand_delegate: Annotated[
            str,
            typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
        ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
        auto_download_model: Annotated[
            bool,
            typer.Option(help="Download the MediaPipe model to --hand-model-path if missing."),
        ] = True,
        max_frames: Annotated[
            int | None, typer.Option(help="Stop after this many frames.")
        ] = None,
        show: Annotated[bool, typer.Option(help="Show an OpenCV live preview.")] = True,
        mirror: Annotated[bool, typer.Option(help="Mirror the preview window.")] = True,
        preview_layout: Annotated[
            str,
            typer.Option(help="Preview layout: dashboard or camera."),
        ] = "dashboard",
        preview_width: Annotated[
            int,
            typer.Option(help="Dashboard preview canvas width in pixels."),
        ] = 1180,
        preview_height: Annotated[
            int,
            typer.Option(help="Dashboard preview canvas height in pixels."),
        ] = 720,
        camera_buffer_size: Annotated[
            int,
            typer.Option(help="Requested OpenCV camera buffer size; 0 leaves backend default."),
        ] = 1,
        min_rows: Annotated[
            int,
            typer.Option(help="Minimum feature rows before the first TCN v2 prediction."),
        ] = 4,
        evidence_threshold: Annotated[
            float,
            typer.Option(help="Minimum evidence score before printing a prediction line."),
        ] = 0.35,
        include_background: Annotated[
            bool,
            typer.Option(help="Print low-confidence/background evidence lines as well."),
        ] = False,
        activation_threshold: Annotated[
            float,
            typer.Option(help="Live decoder activation threshold over v2 stroke evidence."),
        ] = 0.35,
        release_threshold: Annotated[
            float,
            typer.Option(help="Live decoder release threshold over v2 stroke evidence."),
        ] = 0.20,
        min_peak_confidence: Annotated[
            float,
            typer.Option(help="Live decoder minimum peak stroke confidence."),
        ] = 0.35,
        cooldown_seconds: Annotated[
            float,
            typer.Option(help="Live decoder same-gesture separation/cooldown in seconds."),
        ] = 0.5,
        events_out: Annotated[
            Path | None,
            typer.Option(help="Write live TCN v2 predictions/candidates as JSONL events."),
        ] = None,
        show_motion: Annotated[
            bool,
            typer.Option(help="Show hand-normalized horizontal motion in the HUD."),
        ] = True,
        profile_timing: Annotated[
            bool,
            typer.Option(help="Print per-prediction TCN v2 timing diagnostics."),
        ] = False,
    ) -> None:
        """Watch live/replay TCN v2 evidence without triggering desktop actions."""
        if not 0 <= evidence_threshold <= 1:
            typer.echo("evidence-threshold must be in [0, 1]", err=True)
            raise typer.Exit(code=1)
        if min_rows <= 0:
            typer.echo("min-rows must be positive", err=True)
            raise typer.Exit(code=1)
        if not 0 <= release_threshold <= activation_threshold <= 1:
            typer.echo("decoder thresholds must satisfy 0 <= release <= activation <= 1", err=True)
            raise typer.Exit(code=1)
        if not 0 <= min_peak_confidence <= 1:
            typer.echo("min-peak-confidence must be in [0, 1]", err=True)
            raise typer.Exit(code=1)
        if cooldown_seconds < 0:
            typer.echo("cooldown-seconds must be non-negative", err=True)
            raise typer.Exit(code=1)
        if preview_layout not in {"dashboard", "camera"}:
            typer.echo("preview-layout must be 'dashboard' or 'camera'", err=True)
            raise typer.Exit(code=1)
        if preview_width <= 0 or preview_height <= 0:
            typer.echo("preview dimensions must be positive", err=True)
            raise typer.Exit(code=1)
        if camera_buffer_size < 0:
            typer.echo("camera-buffer-size must be non-negative", err=True)
            raise typer.Exit(code=1)
        decoder_config = EventDecoderConfig(
            activation_threshold=activation_threshold,
            release_threshold=release_threshold,
            min_peak_confidence=min_peak_confidence,
            min_event_separation_seconds=cooldown_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        try:
            predictor = CausalTcnV2LivePredictor.load(tcn_model)
        except (MissingMlDependencyError, ValueError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

        tracker = _make_tracker(
            backend=backend,
            device=device,
            max_frames=max_frames,
            show=show,
            camera_settings=CameraSettings(
                width=width,
                height=height,
                fps=fps,
                fourcc=fourcc,
                buffer_size=camera_buffer_size or None,
            ),
            model_path=hand_model_path,
            auto_download_model=auto_download_model,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            delegate=hand_delegate,
            preview_mirror=mirror,
            preview_layout=preview_layout,
            preview_canvas_width=preview_width,
            preview_canvas_height=preview_height,
            preview_gestures=False,
        )
        stream = FeatureRowStream()
        rows: list[FrameFeatureRow] = []
        latest_predictions: dict[str, CausalTcnV2LivePrediction] = {}
        latest_rows_by_hand: dict[str, object] = {}
        decoder_frames: list[DecoderFrame] = []
        emitted_candidates: set[tuple[str | None, str, int]] = set()
        state: dict[str, object] = {
            "status": "warming up",
            "alert": "",
            "alert_until": 0.0,
            "predictions": latest_predictions,
            "rows_by_hand": latest_rows_by_hand,
            "show_motion": show_motion,
            "evidence_threshold": evidence_threshold,
            "stream_count": 0,
            "row_count": 0,
            "prediction_count": 0,
            "candidate_count": 0,
            "latest_relative_time": None,
            "recent_candidates": [],
            "recent_recognitions": [],
        }
        session_id = str(uuid4())
        event_writer = JsonlRecordingWriter(events_out) if events_out is not None else None
        first_timestamp: float | None = None
        next_prediction_time_by_hand: dict[str, float] = {}
        prediction_count = 0
        candidate_count = 0
        last_custom_alert_by_hand: dict[str, tuple[str, float]] = {}
        decoder_history_seconds = (
            predictor.window_seconds
            + decoder_config.recovery_seconds
            + decoder_config.cooldown_seconds
            + 1.0
        )

        if hasattr(tracker, "preview_status_provider"):
            tracker.preview_status_provider = lambda: _live_tcn_v2_preview_status(state)  # type: ignore[attr-defined]
        if hasattr(tracker, "preview_dashboard_provider"):
            tracker.preview_dashboard_provider = lambda: _live_tcn_v2_dashboard_snapshot(  # type: ignore[attr-defined]
                state,
                first_timestamp=first_timestamp,
                timing_samples=getattr(tracker, "timing_samples", []),
            )

        _write_tcn_v2_live_event(
            event_writer,
            event_type="tcn_v2_live_session_start",
            timestamp=utc_timestamp(),
            session_id=session_id,
            payload={
                "model": str(tcn_model),
                "backend": backend,
                "device": device,
                "schema_version": predictor.schema_version,
                "evidence_targets": list(predictor.evidence_targets),
                "window_seconds": predictor.window_seconds,
                "stride_seconds": predictor.stride_seconds,
                "event_decoder": decoder_config.to_dict(),
                "preview_layout": preview_layout,
                "camera_buffer_size": camera_buffer_size or None,
                "dry_run_only": True,
            },
        )
        typer.echo(
            "watching tcn_v2 "
            f"model={tcn_model} backend={backend} window={predictor.window_seconds:.2f}s "
            f"stride={predictor.stride_seconds:.2f}s heads={','.join(predictor.evidence_targets)} "
            "actions=disabled"
        )
        if (
            "stroke_left" not in predictor.evidence_targets
            and "stroke_right" not in predictor.evidence_targets
        ):
            typer.echo(
                "custom evidence heads detected; displaying top heads only, "
                "AirDesk swipe decoder/candidates disabled"
            )
        try:
            tracker.start()
            for frame in tracker.frames():
                first_timestamp = frame.timestamp if first_timestamp is None else first_timestamp
                state["latest_relative_time"] = frame.timestamp - first_timestamp
                rows.extend(stream.append_rows(frame))
                cutoff = frame.timestamp - predictor.window_seconds
                rows = [item for item in rows if item.timestamp >= cutoff]
                hand_streams = _live_feature_streams(rows)
                if not hand_streams:
                    state["status"] = "TCN v2 warming rows=0"
                    state["stream_count"] = 0
                    state["row_count"] = 0
                    continue
                state["stream_count"] = len(hand_streams)
                state["row_count"] = len(rows)
                state["status"] = _format_live_tcn_v2_preview_predictions(state)
                for hand_id, hand_rows in hand_streams.items():
                    latest_rows_by_hand[hand_id] = hand_rows[-1]
                    if len(hand_rows) < min_rows:
                        continue
                    next_prediction_time = next_prediction_time_by_hand.get(hand_id)
                    if (
                        next_prediction_time is not None
                        and hand_rows[-1].timestamp < next_prediction_time
                    ):
                        continue
                    prediction_started_at = monotonic()
                    prediction = predictor.predict_rows(hand_rows)
                    prediction_ms = (monotonic() - prediction_started_at) * 1000
                    decoder_scores = tcn_v2_decoder_scores(prediction.evidence)
                    prediction_count += 1
                    state["prediction_count"] = prediction_count
                    latest_predictions[hand_id] = prediction
                    state["status"] = _format_live_tcn_v2_preview_predictions(state)
                    decoder_frames.append(
                        _tcn_v2_live_decoder_frame(
                            prediction,
                            source_id=frame.source_id,
                            decoder_scores=decoder_scores,
                        )
                    )
                    decoder_frames = [
                        item
                        for item in decoder_frames
                        if item.timestamp >= prediction.end_time - decoder_history_seconds
                    ]
                    _write_tcn_v2_live_event(
                        event_writer,
                        event_type="tcn_v2_live_prediction",
                        timestamp=utc_timestamp(),
                        session_id=session_id,
                        payload={
                            "prediction": prediction.to_dict(),
                            "decoder_scores": decoder_scores,
                            "features": _live_tcn_v2_row_motion_features(
                                latest_rows_by_hand.get(hand_id)
                            ),
                            "relative_time_seconds": (
                                prediction.end_time - first_timestamp
                                if first_timestamp is not None
                                else prediction.end_time
                            ),
                        },
                    )
                    visible_evidence = (
                        _max_tcn_v2_visible_evidence(prediction.evidence)
                        >= evidence_threshold
                    )
                    custom_recognition = _recognized_tcn_v2_custom_evidence(
                        prediction.evidence,
                        threshold=evidence_threshold,
                    )
                    if custom_recognition is not None:
                        now = monotonic()
                        target = str(custom_recognition["target"])
                        name = str(custom_recognition["name"])
                        score = float(custom_recognition["score"])
                        last_target, last_at = last_custom_alert_by_hand.get(
                            hand_id,
                            ("", -999.0),
                        )
                        if target != last_target or now - last_at >= 1.5:
                            state["alert"] = f"{hand_id} {name} {score:.2f}"
                            state["alert_until"] = now + 1.5
                            last_custom_alert_by_hand[hand_id] = (target, now)
                            recent_recognitions = list(state.get("recent_recognitions", []))
                            recent_recognitions.append(
                                {
                                    "name": name,
                                    "target": target,
                                    "hand_id": hand_id,
                                    "confidence": score,
                                    "emitted": (
                                        prediction.end_time - first_timestamp
                                        if first_timestamp is not None
                                        else prediction.end_time
                                    ),
                                }
                            )
                            state["recent_recognitions"] = recent_recognitions[-8:]
                    if profile_timing and visible_evidence:
                        typer.echo(
                            f"tcn_v2_predict_ms={prediction_ms:.2f} hand={hand_id} "
                            f"rows={len(hand_rows)}"
                        )
                    if include_background or visible_evidence:
                        typer.echo(
                            _format_live_tcn_v2_prediction(
                                prediction,
                                first_timestamp=first_timestamp,
                                decoder_scores=decoder_scores,
                            )
                        )
                    candidates = EventDecoder(decoder_config).decode(
                        decoder_frames,
                        flush_open_events=False,
                    )
                    for candidate in candidates:
                        key = _live_candidate_key(candidate)
                        if key in emitted_candidates:
                            continue
                        emitted_candidates.add(key)
                        candidate_count += 1
                        state["candidate_count"] = candidate_count
                        hand_label = candidate.hand_id or "hand"
                        state["alert"] = f"{hand_label} {candidate.name} {candidate.confidence:.2f}"
                        state["alert_until"] = monotonic() + 1.8
                        recent_candidates = list(state.get("recent_candidates", []))
                        recent_candidates.append(
                            {
                                "name": candidate.name,
                                "hand_id": candidate.hand_id or "hand",
                                "confidence": candidate.confidence,
                                "peak": (
                                    candidate.timestamp - first_timestamp
                                    if first_timestamp is not None
                                    else candidate.timestamp
                                ),
                                "emitted": (
                                    prediction.end_time - first_timestamp
                                    if first_timestamp is not None
                                    else prediction.end_time
                                ),
                                "delay": max(0.0, prediction.end_time - candidate.timestamp),
                            }
                        )
                        state["recent_candidates"] = recent_candidates[-8:]
                        _write_tcn_v2_live_event(
                            event_writer,
                            event_type="tcn_v2_live_candidate",
                            timestamp=utc_timestamp(),
                            session_id=session_id,
                            payload={
                                "candidate": candidate.to_dict(),
                                "name": candidate.name,
                                "hand_id": candidate.hand_id,
                                "confidence": candidate.confidence,
                                "peak_at": candidate.timestamp,
                                "emitted_at": prediction.end_time,
                                "delay": max(0.0, prediction.end_time - candidate.timestamp),
                                "relative_peak_seconds": (
                                    candidate.timestamp - first_timestamp
                                    if first_timestamp is not None
                                    else candidate.timestamp
                                ),
                                "relative_emitted_at_seconds": (
                                    prediction.end_time - first_timestamp
                                    if first_timestamp is not None
                                    else prediction.end_time
                                ),
                            },
                        )
                        typer.echo(
                            _format_live_tcn_v2_candidate(
                                candidate,
                                first_timestamp=first_timestamp,
                                emitted_at=prediction.end_time,
                            )
                        )
                    next_prediction_time_by_hand[hand_id] = (
                        hand_rows[-1].timestamp + predictor.stride_seconds
                    )
        except KeyboardInterrupt:
            typer.echo("interrupted")
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        finally:
            _write_tcn_v2_live_event(
                event_writer,
                event_type="tcn_v2_live_session_finish",
                timestamp=utc_timestamp(),
                session_id=session_id,
                payload={"predictions": prediction_count, "candidates": candidate_count},
            )
            if event_writer is not None:
                event_writer.close()
            if profile_timing:
                timing_text = _format_tracker_timing(tracker)
                if timing_text:
                    typer.echo(timing_text)
            tracker.stop()

    @gesture_app.command("watch-dtw")
    def gesture_watch_dtw(
        model: Annotated[
            Path,
            typer.Option("--model", exists=True, readable=True, help="DTW model JSON path."),
        ],
        backend: Annotated[str, typer.Option(help="Tracking backend to watch.")] = "mediapipe",
        device: Annotated[
            str,
            typer.Option(help="Camera path, numeric index, or replay path."),
        ] = "/dev/video0",
        width: Annotated[int | None, typer.Option(help="Requested capture width.")] = 640,
        height: Annotated[int | None, typer.Option(help="Requested capture height.")] = 480,
        fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = 30,
        fourcc: Annotated[
            str | None,
            typer.Option(help="Requested camera FOURCC, e.g. MJPG."),
        ] = "MJPG",
        hand_model_path: Annotated[
            Path,
            typer.Option(
                "--hand-model-path", help="MediaPipe Hand Landmarker .task model path."
            ),
        ] = DEFAULT_HAND_LANDMARKER_MODEL,
        max_num_hands: Annotated[
            int,
            typer.Option(help="Maximum number of hands for MediaPipe to track."),
        ] = DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
        min_detection_confidence: Annotated[
            float,
            typer.Option(help="Minimum palm detection confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_presence_confidence: Annotated[
            float,
            typer.Option(help="Minimum hand landmark presence confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_tracking_confidence: Annotated[
            float,
            typer.Option(help="Minimum tracking confidence / box IoU threshold."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        hand_delegate: Annotated[
            str,
            typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
        ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
        auto_download_model: Annotated[
            bool,
            typer.Option(help="Download the MediaPipe model to --hand-model-path if missing."),
        ] = True,
        max_frames: Annotated[
            int | None, typer.Option(help="Stop after this many frames.")
        ] = None,
        show: Annotated[bool, typer.Option(help="Show an OpenCV live preview.")] = True,
        mirror: Annotated[bool, typer.Option(help="Mirror the preview window.")] = True,
        confidence_threshold: Annotated[
            float,
            typer.Option(help="Minimum DTW confidence before printing a candidate."),
        ] = 0.0,
        watch_stride_seconds: Annotated[
            float,
            typer.Option(help="Minimum seconds between DTW scans over the rolling buffer."),
        ] = 0.08,
        profile_timing: Annotated[
            bool,
            typer.Option(help="Print per-scan DTW timing diagnostics."),
        ] = False,
    ) -> None:
        """Watch live/replay DTW candidate spotting without triggering desktop actions."""
        if not 0 <= confidence_threshold <= 1:
            typer.echo("confidence-threshold must be in [0, 1]", err=True)
            raise typer.Exit(code=1)
        if watch_stride_seconds <= 0:
            typer.echo("watch-stride-seconds must be positive", err=True)
            raise typer.Exit(code=1)
        try:
            dtw_model = DtwGestureModel.load(model)
        except (OSError, ValueError, KeyError) as exc:
            typer.echo(f"Failed to load DTW model: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        tracker = _make_tracker(
            backend=backend,
            device=device,
            max_frames=max_frames,
            show=show,
            camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
            model_path=hand_model_path,
            auto_download_model=auto_download_model,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            delegate=hand_delegate,
            preview_mirror=mirror,
        )
        recognizer = DtwTemplateRecognizer(dtw_model)
        stream = FeatureRowStream()
        rows: list[FrameFeatureRow] = []
        state = {"status": "warming up", "alert": "", "alert_until": 0.0}
        first_timestamp: float | None = None
        next_scan_time: float | None = None
        last_reported_timestamp: float | None = None

        if hasattr(tracker, "preview_status_provider"):
            tracker.preview_status_provider = lambda: _live_dtw_preview_status(state)  # type: ignore[attr-defined]

        typer.echo(
            "watching dtw "
            f"model={model} backend={backend} "
            f"window={dtw_model.min_window_seconds:.2f}-{dtw_model.max_window_seconds:.2f}s "
            f"step={dtw_model.window_step_seconds:.2f}s"
        )
        try:
            tracker.start()
            for frame in tracker.frames():
                first_timestamp = frame.timestamp if first_timestamp is None else first_timestamp
                rows.extend(stream.append_rows(frame))
                cutoff = frame.timestamp - dtw_model.max_window_seconds - watch_stride_seconds
                rows = [item for item in rows if item.timestamp >= cutoff]
                state["status"] = f"DTW rows={len(rows)} hands={len(frame.hands)}"
                if next_scan_time is not None and frame.timestamp < next_scan_time:
                    continue
                scan_started_at = monotonic()
                candidates = [
                    candidate
                    for candidate in recognizer.recognize_latest_rows(rows)
                    if candidate.confidence >= confidence_threshold
                ]
                scan_ms = (monotonic() - scan_started_at) * 1000
                if profile_timing:
                    typer.echo(
                        f"dtw_scan_ms={scan_ms:.2f} rows={len(rows)} "
                        f"candidates={len(candidates)} hands={len(frame.hands)}"
                    )
                fresh = [
                    candidate
                    for candidate in candidates
                    if last_reported_timestamp is None
                    or candidate.timestamp > last_reported_timestamp + 1e-6
                ]
                if fresh:
                    for candidate in fresh:
                        typer.echo(
                            _format_live_dtw_candidate(
                                candidate,
                                first_timestamp=first_timestamp,
                            )
                        )
                    best = max(fresh, key=lambda item: item.confidence)
                    state["alert"] = f"{best.name} {best.confidence:.2f}"
                    state["alert_until"] = monotonic() + 1.25
                    last_reported_timestamp = max(candidate.timestamp for candidate in fresh)
                next_scan_time = frame.timestamp + watch_stride_seconds
        except KeyboardInterrupt:
            typer.echo("interrupted")
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        finally:
            tracker.stop()

    @app.command()
    def track(
        backend: Annotated[str, typer.Option(help="Tracking backend to run.")] = "mediapipe",
        device: Annotated[
            str, typer.Option(help="Camera path or numeric index.")
        ] = "/dev/video0",
        width: Annotated[int | None, typer.Option(help="Requested capture width.")] = None,
        height: Annotated[int | None, typer.Option(help="Requested capture height.")] = None,
        fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = None,
        fourcc: Annotated[
            str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")
        ] = None,
        model_path: Annotated[
            Path,
            typer.Option(help="MediaPipe Hand Landmarker .task model path."),
        ] = DEFAULT_HAND_LANDMARKER_MODEL,
        max_num_hands: Annotated[
            int,
            typer.Option(help="Maximum number of hands for MediaPipe to track."),
        ] = DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
        min_detection_confidence: Annotated[
            float,
            typer.Option(help="Minimum palm detection confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_presence_confidence: Annotated[
            float,
            typer.Option(help="Minimum hand landmark presence confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_tracking_confidence: Annotated[
            float,
            typer.Option(help="Minimum tracking confidence / box IoU threshold."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        hand_delegate: Annotated[
            str,
            typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
        ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
        auto_download_model: Annotated[
            bool,
            typer.Option(help="Download the MediaPipe model to --model-path if missing."),
        ] = True,
        max_frames: Annotated[
            int | None, typer.Option(help="Stop after this many frames.")
        ] = None,
        show: Annotated[
            bool, typer.Option(help="Show an OpenCV landmark debug window.")
        ] = False,
        mirror: Annotated[bool, typer.Option(help="Mirror the preview window.")] = True,
    ) -> None:
        """Run live tracking and print compact frame summaries without recording or actions."""
        tracker = _make_tracker(
            backend=backend,
            device=device,
            max_frames=max_frames,
            show=show,
            camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
            model_path=model_path,
            auto_download_model=auto_download_model,
            preview_mirror=mirror,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            delegate=hand_delegate,
        )
        recognizer = StaticHandPoseRecognizer()
        try:
            tracker.start()
            for frame in tracker.frames():
                candidates = recognizer.recognize(frame)
                typer.echo(_format_frame_summary(frame, candidates))
        except KeyboardInterrupt:
            typer.echo("interrupted")
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        finally:
            tracker.stop()

    @app.command()
    def tune(
        backend: Annotated[str, typer.Option(help="Tracking backend to tune.")] = "mediapipe",
        device: Annotated[
            str, typer.Option(help="Camera path or numeric index.")
        ] = "/dev/video0",
        width: Annotated[int | None, typer.Option(help="Requested capture width.")] = 640,
        height: Annotated[int | None, typer.Option(help="Requested capture height.")] = 480,
        fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = 30,
        fourcc: Annotated[
            str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")
        ] = "MJPG",
        extended_threshold: Annotated[
            float,
            typer.Option(help="Finger tip-vs-MCP y-distance threshold."),
        ] = 0.08,
        pinch_threshold: Annotated[
            float,
            typer.Option(help="Thumb/index distance threshold for pinch."),
        ] = 0.06,
        model_path: Annotated[
            Path,
            typer.Option(help="MediaPipe Hand Landmarker .task model path."),
        ] = DEFAULT_HAND_LANDMARKER_MODEL,
        max_num_hands: Annotated[
            int,
            typer.Option(help="Maximum number of hands for MediaPipe to track."),
        ] = DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
        min_detection_confidence: Annotated[
            float,
            typer.Option(help="Minimum palm detection confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_presence_confidence: Annotated[
            float,
            typer.Option(help="Minimum hand landmark presence confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_tracking_confidence: Annotated[
            float,
            typer.Option(help="Minimum tracking confidence / box IoU threshold."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        hand_delegate: Annotated[
            str,
            typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
        ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
        auto_download_model: Annotated[
            bool,
            typer.Option(help="Download the MediaPipe model to --model-path if missing."),
        ] = True,
        max_frames: Annotated[
            int | None, typer.Option(help="Stop after this many frames.")
        ] = None,
        show: Annotated[
            bool, typer.Option(help="Show an OpenCV landmark debug window.")
        ] = False,
        mirror: Annotated[bool, typer.Option(help="Mirror the preview window.")] = True,
    ) -> None:
        """Run a live primitive-tuning session with per-frame landmark features."""
        tracker = _make_tracker(
            backend=backend,
            device=device,
            max_frames=max_frames,
            show=show,
            camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
            model_path=model_path,
            auto_download_model=auto_download_model,
            preview_mirror=mirror,
            preview_extended_threshold=extended_threshold,
            preview_pinch_threshold=pinch_threshold,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            delegate=hand_delegate,
        )
        recognizer = StaticHandPoseRecognizer(
            extended_threshold=extended_threshold,
            pinch_threshold=pinch_threshold,
        )
        previous_timestamp: float | None = None
        typer.echo(
            "target: open_palm extended=4 spread>=0.16 | fist folded=4 | "
            f"pinch distance<={pinch_threshold:.3f}"
        )
        try:
            tracker.start()
            for frame in tracker.frames():
                candidates = recognizer.recognize(frame)
                features = recognizer.features_for_frame(frame)
                frame_fps = _instant_fps(previous_timestamp, frame.timestamp)
                previous_timestamp = frame.timestamp
                typer.echo(_format_tune_summary(frame, candidates, features, frame_fps))
        except KeyboardInterrupt:
            typer.echo("interrupted")
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        finally:
            tracker.stop()

    @app.command()
    def view(
        backend: Annotated[str, typer.Option(help="Tracking backend to view.")] = "mediapipe",
        device: Annotated[
            str, typer.Option(help="Camera path or numeric index.")
        ] = "/dev/video0",
        width: Annotated[int | None, typer.Option(help="Requested capture width.")] = 640,
        height: Annotated[int | None, typer.Option(help="Requested capture height.")] = 480,
        fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = 30,
        fourcc: Annotated[
            str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")
        ] = "MJPG",
        model_path: Annotated[
            Path,
            typer.Option(help="MediaPipe Hand Landmarker .task model path."),
        ] = DEFAULT_HAND_LANDMARKER_MODEL,
        max_num_hands: Annotated[
            int,
            typer.Option(help="Maximum number of hands for MediaPipe to track."),
        ] = DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
        min_detection_confidence: Annotated[
            float,
            typer.Option(help="Minimum palm detection confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_presence_confidence: Annotated[
            float,
            typer.Option(help="Minimum hand landmark presence confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_tracking_confidence: Annotated[
            float,
            typer.Option(help="Minimum tracking confidence / box IoU threshold."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        hand_delegate: Annotated[
            str,
            typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
        ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
        auto_download_model: Annotated[
            bool,
            typer.Option(help="Download the MediaPipe model to --model-path if missing."),
        ] = True,
        max_frames: Annotated[
            int | None, typer.Option(help="Stop after this many frames.")
        ] = None,
        mirror: Annotated[bool, typer.Option(help="Mirror the preview window.")] = True,
    ) -> None:
        """Open a live webcam preview with MediaPipe hand overlays."""
        tracker = _make_tracker(
            backend=backend,
            device=device,
            max_frames=max_frames,
            show=True,
            camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
            model_path=model_path,
            auto_download_model=auto_download_model,
            preview_mirror=mirror,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            delegate=hand_delegate,
        )
        typer.echo("Opening AirDesk live view. Press q or esc in the preview window to quit.")
        try:
            tracker.start()
            for _frame in tracker.frames():
                pass
        except KeyboardInterrupt:
            typer.echo("interrupted")
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        finally:
            tracker.stop()

    @app.command()
    def benchmark(
        backend: Annotated[
            str, typer.Option(help="Tracking backend to benchmark.")
        ] = "mediapipe",
        device: Annotated[
            str, typer.Option(help="Camera path or numeric index.")
        ] = "/dev/video0",
        width: Annotated[int | None, typer.Option(help="Requested capture width.")] = 640,
        height: Annotated[int | None, typer.Option(help="Requested capture height.")] = 480,
        fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = 30,
        fourcc: Annotated[
            str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")
        ] = "MJPG",
        model_path: Annotated[
            Path,
            typer.Option(help="MediaPipe Hand Landmarker .task model path."),
        ] = DEFAULT_HAND_LANDMARKER_MODEL,
        max_num_hands: Annotated[
            int,
            typer.Option(help="Maximum number of hands for MediaPipe to track."),
        ] = DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
        min_detection_confidence: Annotated[
            float,
            typer.Option(help="Minimum palm detection confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_presence_confidence: Annotated[
            float,
            typer.Option(help="Minimum hand landmark presence confidence."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        min_tracking_confidence: Annotated[
            float,
            typer.Option(help="Minimum tracking confidence / box IoU threshold."),
        ] = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
        hand_delegate: Annotated[
            str,
            typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
        ] = DEFAULT_HAND_LANDMARKER_DELEGATE,
        auto_download_model: Annotated[
            bool,
            typer.Option(help="Download the MediaPipe model to --model-path if missing."),
        ] = True,
        max_frames: Annotated[int, typer.Option(help="Frames to benchmark.")] = 120,
    ) -> None:
        """Benchmark live tracking FPS and hand-present frames for a configuration."""
        tracker = _make_tracker(
            backend=backend,
            device=device,
            max_frames=max_frames,
            show=False,
            camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
            model_path=model_path,
            auto_download_model=auto_download_model,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            delegate=hand_delegate,
        )
        timestamps: list[float] = []
        hand_frames = 0
        hand_confidences: list[float] = []
        current_missing_streak = 0
        longest_missing_streak = 0
        try:
            tracker.start()
            for frame in tracker.frames():
                timestamps.append(frame.timestamp)
                if frame.hands:
                    hand_frames += 1
                    current_missing_streak = 0
                    hand_confidences.extend(
                        hand.confidence for hand in frame.hands if hand.confidence is not None
                    )
                else:
                    current_missing_streak += 1
                    longest_missing_streak = max(
                        longest_missing_streak, current_missing_streak
                    )
        except KeyboardInterrupt:
            typer.echo("interrupted")
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        finally:
            tracker.stop()

        average_fps = _average_fps_from_timestamps(timestamps)
        fps_text = f"{average_fps:.2f}" if average_fps is not None else "unknown"
        hand_ratio = hand_frames / len(timestamps) if timestamps else 0.0
        mean_confidence = f"{fmean(hand_confidences):.3f}" if hand_confidences else "unknown"
        typer.echo(
            f"frames={len(timestamps)} hand_frames={hand_frames} average_fps={fps_text} "
            f"hand_present_ratio={hand_ratio:.3f} "
            f"longest_no_hand_streak={longest_missing_streak} "
            f"mean_hand_confidence={mean_confidence} "
            f"model_path={model_path} max_num_hands={max_num_hands} "
            f"delegate={hand_delegate} "
            f"min_detection={min_detection_confidence:.2f} "
            f"min_presence={min_presence_confidence:.2f} "
            f"min_tracking={min_tracking_confidence:.2f}"
        )
        timing_text = _format_tracker_timing(tracker)
        if timing_text:
            typer.echo(timing_text)


def _format_frame_summary(frame: TrackingFrame, candidates: object) -> str:
    names = ",".join(candidate.name for candidate in candidates) or "none"
    return (
        f"frame={frame.frame.sequence} hands={len(frame.hands)} "
        f"size={frame.frame.width}x{frame.frame.height} candidates={names}"
    )


def _format_tune_summary(
    frame: TrackingFrame,
    candidates: object,
    features: object,
    frame_fps: float | None,
) -> str:
    names = ",".join(candidate.name for candidate in candidates) or "none"
    fps = f"{frame_fps:.1f}" if frame_fps is not None else "unknown"
    if not features:
        return (
            f"frame={frame.frame.sequence} fps={fps} hands=0 "
            f"size={frame.frame.width}x{frame.frame.height} candidates={names}"
        )
    feature_parts = []
    for feature in features:
        values = feature.to_flat_dict()
        feature_parts.append(
            "hand={hand_id} side={handedness} conf={confidence} extended={extended} "
            "folded={folded} spread={spread} pinch={pinch}".format(**values)
        )
    return (
        f"frame={frame.frame.sequence} fps={fps} hands={len(feature_parts)} "
        f"size={frame.frame.width}x{frame.frame.height} candidates={names} | "
        + " | ".join(feature_parts)
    )


def _instant_fps(previous_timestamp: float | None, timestamp: float) -> float | None:
    if previous_timestamp is None:
        return None
    elapsed = timestamp - previous_timestamp
    if elapsed <= 0:
        return None
    return 1.0 / elapsed


def _average_fps_from_timestamps(timestamps: list[float]) -> float | None:
    if len(timestamps) < 2:
        return None
    intervals = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:], strict=False)
        if later > earlier
    ]
    if not intervals:
        return None
    return 1.0 / fmean(intervals)


def _tcn_v2_live_decoder_frame(
    prediction: CausalTcnV2LivePrediction,
    *,
    source_id: str,
    decoder_scores: dict[str, float],
) -> DecoderFrame:
    return DecoderFrame(
        timestamp=prediction.end_time,
        scores=decoder_scores,
        source_id=source_id,
        hand_id=prediction.hand_id,
        window_start=prediction.start_time,
        window_end=prediction.end_time,
        metadata={
            "recognizer": "tcn_v2_live",
            "intentional_motion": prediction.evidence.get("intentional_motion", 0.0),
            "start": prediction.evidence.get("start", 0.0),
            "end": prediction.evidence.get("end", 0.0),
            "raw_evidence": prediction.evidence,
            "decoder_scores": decoder_scores,
        },
    )


def _live_candidate_key(candidate: GestureCandidate) -> tuple[str | None, str, int]:
    return (candidate.hand_id, candidate.name, int(round(candidate.timestamp * 1000)))


def _write_tcn_v2_live_event(
    writer: JsonlRecordingWriter | None,
    *,
    event_type: str,
    timestamp: float,
    session_id: str,
    payload: dict[str, object],
) -> None:
    if writer is None:
        return
    writer.write_event(
        EventLogEntry(
            event_type=event_type,
            timestamp=timestamp,
            session_id=session_id,
            payload=payload,
        )
    )
