"""Live tracking and recognizer-preview CLI commands."""

from __future__ import annotations

from pathlib import Path
from statistics import fmean
from time import monotonic
from typing import Annotated

import typer

from airdesk.capture.opencv import CameraSettings
from airdesk.cli_live import (
    _format_live_dtw_candidate,
    _format_live_tcn_prediction,
    _format_live_tcn_preview_predictions,
    _format_tracker_timing,
    _is_live_tcn_gesture_target,
    _live_dtw_preview_status,
    _live_feature_streams,
    _live_tcn_preview_status,
    _show_live_tcn_prediction,
)
from airdesk.cli_tracking import _make_tracker
from airdesk.features import FeatureRowStream, FrameFeatureRow
from airdesk.gestures.dtw import DtwGestureModel, DtwTemplateRecognizer
from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.ml import (
    CausalTcnLivePrediction,
    CausalTcnLivePredictor,
    MissingMlDependencyError,
)
from airdesk.state.types import TrackingFrame
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
