"""Recording, prompted collection, and chart-label CLI surfaces."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Annotated

import typer

from airdesk.analysis import analyze_recording
from airdesk.capture.opencv import CameraSettings
from airdesk.cli_support import _save_valid_label_file
from airdesk.cli_tracking import _make_tracker
from airdesk.gestures.base import CompositeGestureRecognizer
from airdesk.gestures.phrases import IntentGatedSwipeRecognizer
from airdesk.gestures.primitives import StaticHandPoseRecognizer
from airdesk.labels import add_event_label, add_phase_label, init_label_file
from airdesk.recording.jsonl import JsonlRecordingWriter, iter_recording
from airdesk.state.types import EventLogEntry, TrackingFrame, utc_timestamp
from airdesk.tracking.interfaces import HandTrackerBackend
from airdesk.tracking.mediapipe import (
    DEFAULT_HAND_LANDMARKER_DELEGATE,
    DEFAULT_HAND_LANDMARKER_MAX_NUM_HANDS,
    DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    DEFAULT_HAND_LANDMARKER_MODEL,
    MediaPipeHandTrackerBackend,
)

DEFAULT_COLLECTION_LABELS = (
    "open-palm-hold",
    "fist-hold",
    "pinch-hold",
    "swipe-left-positive",
    "swipe-right-positive",
    "normal-desk-motion-negative",
)


@dataclass(frozen=True)
class CollectionTakeResult:
    """Result of one prompted collection take."""

    frames: int
    decision: str


@dataclass(frozen=True)
class RecordPromptSegment:
    """One preview prompt segment for a structured recording."""

    start: float
    end: float
    text: str
    kind: str = "prompt"
    gesture: str | None = None
    block_index: int | None = None
    gesture_index: int | None = None


@dataclass(frozen=True)
class ChartGestureWindow:
    """One generated gesture window from a compact recording chart."""

    start: float
    stroke_start: float
    stroke_end: float
    recovery_end: float
    token: str
    gesture: str
    phase: str
    block_index: int
    gesture_index: int


@dataclass(frozen=True)
class RecordChartPlan:
    """Expanded recording chart with preview prompts and label windows."""

    chart: str
    duration: float
    segments: tuple[RecordPromptSegment, ...]
    gestures: tuple[ChartGestureWindow, ...]


def replay(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    recognize: Annotated[
        bool,
        typer.Option(help="Run the static primitive recognizer over replayed frames."),
    ] = True,
) -> None:
    """Read a JSONL recording and report replayable frame/event counts."""
    summary = _summarize_records(path, recognize=recognize)
    typer.echo(_format_summary(summary))


def collection_summary(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    pattern: Annotated[
        str,
        typer.Option(help="Recording filename pattern for directories."),
    ] = "*.jsonl",
) -> None:
    """Summarize candidate counts across a collection directory or recording files."""
    paths = _collection_paths(path, pattern=pattern)
    if not paths:
        typer.echo(f"No recordings matched {path}/{pattern}", err=True)
        raise typer.Exit(code=1)

    rows = [_collection_summary_row(recording_path) for recording_path in paths]
    for row in rows:
        typer.echo(_format_collection_row(row))
    typer.echo(_format_collection_totals(rows))


def gesture_chart_record(
    out: Annotated[Path, typer.Option(help="Output JSONL recording path.")],
    chart: Annotated[
        str,
        typer.Option(
            help="Compact chart, e.g. 'RR | rest | RL | rest | RRR'. R/L are swipe cues.",
        ),
    ],
    labels_out: Annotated[
        Path | None,
        typer.Option(help="Output label JSON path. Defaults to recording .labels.json."),
    ] = None,
    write_labels: Annotated[
        bool,
        typer.Option(help="Write coarse stroke/recovery/event labels from the chart."),
    ] = True,
    backend: Annotated[str, typer.Option(help="Tracking backend to record.")] = "mediapipe",
    device: Annotated[str, typer.Option(help="Camera path or numeric index.")] = "/dev/video0",
    width: Annotated[int | None, typer.Option(help="Requested capture width.")] = None,
    height: Annotated[int | None, typer.Option(help="Requested capture height.")] = None,
    fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = None,
    fourcc: Annotated[str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")] = None,
    label: Annotated[str | None, typer.Option(help="Short label for this recording.")] = None,
    countdown: Annotated[float, typer.Option(help="Countdown seconds after space/start.")] = 3.0,
    lead_in_seconds: Annotated[
        float,
        typer.Option(help="Seconds of chart preview before the first gesture block."),
    ] = 3.0,
    cue_seconds: Annotated[
        float,
        typer.Option(help="Seconds of get-ready cue before each gesture."),
    ] = 1.5,
    gesture_seconds: Annotated[
        float,
        typer.Option(help="Seconds allocated to each gesture stroke."),
    ] = 0.75,
    recovery_seconds: Annotated[
        float,
        typer.Option(help="Seconds allocated to reset/recovery after each gesture."),
    ] = 0.75,
    rest_seconds: Annotated[
        float,
        typer.Option(help="Seconds allocated to each rest/background block."),
    ] = 10.0,
    wait_for_space: Annotated[
        bool,
        typer.Option(help="When previewing, wait for space before starting countdown."),
    ] = True,
    model_path: Annotated[
        Path,
        typer.Option(help="MediaPipe Hand Landmarker .task model path."),
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
        typer.Option(help="Download the MediaPipe model to --model-path if missing."),
    ] = True,
    max_frames: Annotated[int | None, typer.Option(help="Stop after this many frames.")] = None,
    show: Annotated[bool, typer.Option(help="Show an OpenCV prompt/landmark window.")] = True,
) -> None:
    """Record a structured swipe chart with two-hand-capable timing prompts."""
    if countdown < 0:
        typer.echo("--countdown cannot be negative.", err=True)
        raise typer.Exit(code=1)
    try:
        plan = _parse_record_chart(
            chart=chart,
            lead_in_seconds=lead_in_seconds,
            cue_seconds=cue_seconds,
            gesture_seconds=gesture_seconds,
            recovery_seconds=recovery_seconds,
            rest_seconds=rest_seconds,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    recording_label = label or out.stem
    frame_count = _record_tracking(
        out=out,
        backend=backend,
        device=device,
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
        label=recording_label,
        duration=plan.duration,
        countdown=countdown,
        prompt_segments=plan.segments,
        wait_for_space=wait_for_space,
        model_path=model_path,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        hand_delegate=hand_delegate,
        auto_download_model=auto_download_model,
        max_frames=max_frames,
        show=show,
        extra_started_payload={
            "chart": _record_chart_payload(plan),
        },
        chart_plan=plan,
    )
    typer.echo(
        f"chart recorded frames={frame_count} gestures={len(plan.gestures)} "
        f"duration={plan.duration:.2f}s out={out}"
    )
    if not write_labels:
        return
    output_labels = labels_out or out.with_suffix(".labels.json")
    try:
        _write_chart_label_file(
            recording=out,
            out=output_labels,
            plan=plan,
            participant="caden",
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote chart_labels={output_labels}")


def gesture_chart_label(
    recording: Annotated[Path, typer.Argument(exists=True, readable=True)],
    chart: Annotated[
        str,
        typer.Option(help="Compact chart used for the recording, e.g. 'RR | rest | RL'."),
    ],
    out: Annotated[Path | None, typer.Option(help="Output label JSON path.")] = None,
    cue_seconds: Annotated[
        float,
        typer.Option(help="Seconds of get-ready cue before each gesture."),
    ] = 1.5,
    lead_in_seconds: Annotated[
        float,
        typer.Option(help="Seconds of chart preview before the first gesture block."),
    ] = 3.0,
    gesture_seconds: Annotated[
        float,
        typer.Option(help="Seconds allocated to each gesture stroke."),
    ] = 0.75,
    recovery_seconds: Annotated[
        float,
        typer.Option(help="Seconds allocated to reset/recovery after each gesture."),
    ] = 0.75,
    rest_seconds: Annotated[
        float,
        typer.Option(help="Seconds allocated to each rest/background block."),
    ] = 10.0,
    participant: Annotated[str, typer.Option(help="Participant/user identifier.")] = "caden",
) -> None:
    """Create coarse labels from a compact chart for an existing recording."""
    try:
        plan = _parse_record_chart(
            chart=chart,
            lead_in_seconds=lead_in_seconds,
            cue_seconds=cue_seconds,
            gesture_seconds=gesture_seconds,
            recovery_seconds=recovery_seconds,
            rest_seconds=rest_seconds,
        )
        output = out or recording.with_suffix(".labels.json")
        _write_chart_label_file(recording=recording, out=output, plan=plan, participant=participant)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote chart_labels={output} gestures={len(plan.gestures)}")


def record(
    out: Annotated[Path, typer.Option(help="Output JSONL recording path.")],
    backend: Annotated[str, typer.Option(help="Tracking backend to record.")] = "mediapipe",
    device: Annotated[str, typer.Option(help="Camera path or numeric index.")] = "/dev/video0",
    width: Annotated[int | None, typer.Option(help="Requested capture width.")] = None,
    height: Annotated[int | None, typer.Option(help="Requested capture height.")] = None,
    fps: Annotated[float | None, typer.Option(help="Requested capture FPS.")] = None,
    fourcc: Annotated[str | None, typer.Option(help="Requested camera FOURCC, e.g. MJPG.")] = None,
    label: Annotated[str | None, typer.Option(help="Short label for this recording.")] = None,
    duration: Annotated[
        float | None,
        typer.Option(help="Stop after this many seconds based on frame timestamps."),
    ] = None,
    countdown: Annotated[
        float,
        typer.Option(help="Preview countdown seconds before recording frames."),
    ] = 0.0,
    segment: Annotated[
        list[str] | None,
        typer.Option(
            "--segment",
            help="Preview prompt segment as start:end:text, e.g. 0:10:R R. Repeatable.",
        ),
    ] = None,
    wait_for_space: Annotated[
        bool,
        typer.Option(help="When previewing, wait for space before starting countdown."),
    ] = False,
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
    max_frames: Annotated[int | None, typer.Option(help="Stop after this many frames.")] = None,
    show: Annotated[bool, typer.Option(help="Show an OpenCV landmark debug window.")] = False,
) -> None:
    """Record normalized tracking frames and runtime events as JSONL."""
    if countdown < 0:
        typer.echo("--countdown cannot be negative.", err=True)
        raise typer.Exit(code=1)
    try:
        prompt_segments = _parse_record_prompt_segments(segment or [])
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    frame_count = _record_tracking(
        out=out,
        backend=backend,
        device=device,
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
        label=label,
        duration=duration,
        countdown=countdown,
        prompt_segments=prompt_segments,
        wait_for_space=wait_for_space,
        model_path=model_path,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        hand_delegate=hand_delegate,
        auto_download_model=auto_download_model,
        max_frames=max_frames,
        show=show,
        extra_started_payload={},
        chart_plan=None,
    )
    typer.echo(f"recorded frames={frame_count} out={out}")


def collect(
    out_dir: Annotated[Path, typer.Option(help="Directory for collected JSONL takes.")] = Path(
        "data/recordings/collection"
    ),
    label: Annotated[
        list[str] | None,
        typer.Option("--label", "-l", help="Gesture/session label to collect. Repeatable."),
    ] = None,
    reps: Annotated[int, typer.Option(help="Kept repetitions per label.")] = 3,
    duration: Annotated[float, typer.Option(help="Recording duration per take in seconds.")] = 5.0,
    countdown: Annotated[float, typer.Option(help="Countdown duration before recording.")] = 3.0,
    backend: Annotated[str, typer.Option(help="Tracking backend to collect from.")] = "mediapipe",
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
    show: Annotated[bool, typer.Option(help="Show webcam preview with collection status.")] = True,
    auto_keep: Annotated[
        bool,
        typer.Option(help="Keep every take without prompting."),
    ] = False,
) -> None:
    """Collect prompted gesture recordings with countdown and keep/redo flow."""
    labels = tuple(label) if label else DEFAULT_COLLECTION_LABELS
    if reps < 1:
        typer.echo("--reps must be at least 1.", err=True)
        raise typer.Exit(code=1)
    if duration <= 0:
        typer.echo("--duration must be positive.", err=True)
        raise typer.Exit(code=1)
    if countdown < 0:
        typer.echo("--countdown cannot be negative.", err=True)
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    kept_paths: list[Path] = []
    for current_label in labels:
        kept = 0
        while kept < reps:
            repetition = kept + 1
            output = _next_collection_path(out_dir, current_label, repetition)
            typer.echo(f"\n{current_label} rep {repetition}/{reps}")
            preview_driven = show and not auto_keep and backend == "mediapipe"
            if preview_driven:
                typer.echo("Use the preview: space=start, k=keep, r=redo, s=skip, q=quit.")
            elif not auto_keep:
                typer.echo("Press Enter to start, or type s to skip this rep, q to quit.")
                response = input("> ").strip().lower()
                if response == "q":
                    raise typer.Exit()
                if response == "s":
                    kept += 1
                    continue

            take = _record_collection_take(
                output=output,
                label=current_label,
                repetition=repetition,
                backend=backend,
                device=device,
                width=width,
                height=height,
                fps=fps,
                fourcc=fourcc,
                duration=duration,
                countdown=countdown,
                model_path=model_path,
                auto_download_model=auto_download_model,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
                hand_delegate=hand_delegate,
                show=show,
                preview_driven=preview_driven,
            )

            if auto_keep:
                keep = take.decision == "keep"
            else:
                if preview_driven:
                    if take.decision == "quit":
                        raise typer.Exit()
                    if take.decision == "redo":
                        output.unlink(missing_ok=True)
                        continue
                    if take.decision == "skip":
                        output.unlink(missing_ok=True)
                        kept += 1
                        continue
                    keep = take.decision == "keep"
                    if keep:
                        typer.echo(f"Kept frames={take.frames} out={output}")
                else:
                    typer.echo(f"Recorded frames={take.frames} out={output}")
                    response = (
                        input("Keep, redo, skip, or quit? [k/r/s/q] ").strip().lower() or "k"
                    )
                    if response == "q":
                        raise typer.Exit()
                    keep = response == "k"
                    if response == "r":
                        output.unlink(missing_ok=True)
                        continue
                    if response == "s":
                        output.unlink(missing_ok=True)
                        kept += 1
                        continue
            if keep:
                kept_paths.append(output)
                kept += 1

    typer.echo(f"collection complete kept={len(kept_paths)} out_dir={out_dir}")


def _record_tracking(
    *,
    out: Path,
    backend: str,
    device: str,
    width: int | None,
    height: int | None,
    fps: float | None,
    fourcc: str | None,
    label: str | None,
    duration: float | None,
    countdown: float,
    prompt_segments: tuple[RecordPromptSegment, ...],
    wait_for_space: bool,
    model_path: Path,
    max_num_hands: int,
    min_detection_confidence: float,
    min_presence_confidence: float,
    min_tracking_confidence: float,
    hand_delegate: str,
    auto_download_model: bool,
    max_frames: int | None,
    show: bool,
    extra_started_payload: dict[str, object],
    chart_plan: RecordChartPlan | None,
) -> int:
    """Record normalized tracking frames with shared prompt/countdown behavior."""
    tracker = _make_tracker(
        backend=backend,
        device=device,
        max_frames=max_frames,
        show=show,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=model_path,
        auto_download_model=auto_download_model,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        delegate=hand_delegate,
    )
    status = {"text": f"ready: {label or out.stem}"}
    preview_state = {
        "phase": "waiting" if wait_for_space and show else "countdown",
        "quit": False,
        "elapsed": 0.0,
        "duration": duration,
        "countdown_remaining": countdown,
    }
    if isinstance(tracker, MediaPipeHandTrackerBackend):
        tracker.preview_status_provider = lambda: status["text"]
        if chart_plan is not None:
            tracker.preview_chart_provider = lambda: _record_chart_preview_payload(
                label=label or out.stem,
                state=preview_state,
                plan=chart_plan,
            )
        tracker.preview_key_handler = lambda key: _handle_record_preview_key(
            key,
            preview_state,
        )
    frame_count = 0
    interrupted = False
    countdown_started_at: float | None = None
    recording_started_at: float | None = None
    try:
        tracker.start()
        with JsonlRecordingWriter(out) as writer:
            writer.write_event(
                EventLogEntry(
                    event_type="recording_started",
                    timestamp=utc_timestamp(),
                    payload={
                        "backend": backend,
                        "device": device,
                        "max_frames": max_frames,
                        "duration": duration,
                        "countdown": countdown,
                        "wait_for_space": wait_for_space,
                        "label": label,
                        "model_path": str(model_path),
                        "prompt_segments": [asdict(item) for item in prompt_segments],
                        "mediapipe": {
                            "max_num_hands": max_num_hands,
                            "min_detection_confidence": min_detection_confidence,
                            "min_presence_confidence": min_presence_confidence,
                            "min_tracking_confidence": min_tracking_confidence,
                            "delegate": hand_delegate,
                        },
                        "camera_settings": CameraSettings(
                            width=width,
                            height=height,
                            fps=fps,
                            fourcc=fourcc,
                        ).to_dict(),
                        **extra_started_payload,
                    },
                )
            )
            try:
                for frame in tracker.frames():
                    if preview_state["quit"]:
                        interrupted = True
                        break
                    if preview_state["phase"] == "waiting":
                        preview_state["elapsed"] = 0.0
                        status["text"] = (
                            f"ready: {label or out.stem} | space=start q/esc=quit"
                        )
                        continue
                    if recording_started_at is None:
                        if countdown_started_at is None:
                            countdown_started_at = frame.timestamp
                        countdown_elapsed = frame.timestamp - countdown_started_at
                        if countdown_elapsed < countdown:
                            remaining = countdown - countdown_elapsed
                            preview_state["countdown_remaining"] = remaining
                            status["text"] = (
                                f"countdown {remaining:.1f}s | {label or out.stem}"
                            )
                            continue
                        recording_started_at = frame.timestamp
                        preview_state["phase"] = "recording"
                        writer.write_event(
                            EventLogEntry(
                                event_type="recording_frames_started",
                                timestamp=utc_timestamp(),
                                payload={"label": label, "countdown": countdown},
                            )
                        )
                    recorded_elapsed = frame.timestamp - recording_started_at
                    if duration is not None and recorded_elapsed >= duration:
                        status["text"] = f"done frames={frame_count} | q/esc quits"
                        break
                    preview_state["elapsed"] = recorded_elapsed
                    status["text"] = _record_preview_status(
                        label=label or out.stem,
                        elapsed=recorded_elapsed,
                        duration=duration,
                        segments=prompt_segments,
                    )
                    writer.write_tracking_frame(frame)
                    frame_count += 1
            except KeyboardInterrupt:
                interrupted = True
                writer.write_event(
                    EventLogEntry(
                        event_type="recording_interrupted",
                        timestamp=utc_timestamp(),
                        payload={"frames": frame_count},
                    )
                )
            writer.write_event(
                EventLogEntry(
                    event_type="recording_finished",
                    timestamp=utc_timestamp(),
                    payload={
                        "frames": frame_count,
                        "interrupted": interrupted,
                        "duration": duration,
                        "countdown": countdown,
                        "wait_for_space": wait_for_space,
                        "label": label,
                    },
                )
            )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        tracker.stop()
    return frame_count


def _parse_record_prompt_segments(values: list[str]) -> tuple[RecordPromptSegment, ...]:
    segments: list[RecordPromptSegment] = []
    for value in values:
        pieces = value.split(":", maxsplit=2)
        if len(pieces) != 3:
            raise ValueError(f"segment must use start:end:text format, got {value!r}")
        start_text, end_text, text = pieces
        try:
            start = float(start_text)
            end = float(end_text)
        except ValueError as exc:
            raise ValueError(f"segment start/end must be numeric, got {value!r}") from exc
        if start < 0 or end <= start:
            raise ValueError(f"segment end must be greater than start, got {value!r}")
        label = text.strip()
        if not label:
            raise ValueError(f"segment text cannot be empty, got {value!r}")
        segments.append(RecordPromptSegment(start=start, end=end, text=label))
    return tuple(sorted(segments, key=lambda item: (item.start, item.end)))


def _parse_record_chart(
    *,
    chart: str,
    lead_in_seconds: float,
    cue_seconds: float,
    gesture_seconds: float,
    recovery_seconds: float,
    rest_seconds: float,
) -> RecordChartPlan:
    if lead_in_seconds < 0:
        raise ValueError("--lead-in-seconds cannot be negative")
    if cue_seconds < 0:
        raise ValueError("--cue-seconds cannot be negative")
    if gesture_seconds <= 0:
        raise ValueError("--gesture-seconds must be positive")
    if recovery_seconds < 0:
        raise ValueError("--recovery-seconds cannot be negative")
    if rest_seconds <= 0:
        raise ValueError("--rest-seconds must be positive")

    blocks = [block.strip() for block in chart.split("|")]
    blocks = [block for block in blocks if block]
    if not blocks:
        raise ValueError("chart must contain at least one gesture or rest block")

    elapsed = lead_in_seconds
    segments: list[RecordPromptSegment] = []
    if lead_in_seconds > 0:
        segments.append(
            RecordPromptSegment(
                start=0.0,
                end=lead_in_seconds,
                text="get ready",
                kind="lead_in",
            )
        )
    gestures: list[ChartGestureWindow] = []
    gesture_index = 0
    for block_index, block in enumerate(blocks, start=1):
        tokens = _record_chart_block_tokens(block)
        if len(tokens) == 1 and _is_record_chart_rest(tokens[0]):
            segments.append(
                RecordPromptSegment(
                    start=elapsed,
                    end=elapsed + rest_seconds,
                    text="rest",
                    kind="rest",
                    block_index=block_index,
                )
            )
            elapsed += rest_seconds
            continue
        block_gestures = [_record_chart_token_to_gesture(token) for token in tokens]
        block_display = " ".join(display for _gesture, _phase, display in block_gestures)
        cue_start = elapsed
        stroke_start = cue_start + cue_seconds
        stroke_duration = gesture_seconds * len(block_gestures)
        stroke_end = stroke_start + stroke_duration
        recovery_end = stroke_end + recovery_seconds
        if cue_seconds > 0:
            segments.append(
                RecordPromptSegment(
                    start=cue_start,
                    end=stroke_start,
                    text=f"{block_display} ready",
                    kind="cue",
                    gesture=block_display,
                    block_index=block_index,
                )
            )
        segments.append(
            RecordPromptSegment(
                start=stroke_start,
                end=stroke_end,
                text=f"SWIPE {block_display}",
                kind="stroke",
                gesture=block_display,
                block_index=block_index,
            )
        )
        if recovery_seconds > 0:
            segments.append(
                RecordPromptSegment(
                    start=stroke_end,
                    end=recovery_end,
                    text="reset",
                    kind="recovery",
                    gesture=block_display,
                    block_index=block_index,
                )
            )
        slot_seconds = stroke_duration / len(block_gestures)
        for block_offset, (_token_gesture, _token_phase, _display) in enumerate(block_gestures):
            gesture, phase, display = _token_gesture, _token_phase, _display
            gesture_index += 1
            token_start = stroke_start + block_offset * slot_seconds
            token_end = stroke_start + (block_offset + 1) * slot_seconds
            gestures.append(
                ChartGestureWindow(
                    start=cue_start,
                    stroke_start=token_start,
                    stroke_end=token_end,
                    recovery_end=recovery_end,
                    token=display,
                    gesture=gesture,
                    phase=phase,
                    block_index=block_index,
                    gesture_index=gesture_index,
                )
            )
        elapsed = recovery_end
    if not gestures:
        raise ValueError("chart must contain at least one R or L gesture")
    return RecordChartPlan(
        chart=chart,
        duration=elapsed,
        segments=tuple(segments),
        gestures=tuple(gestures),
    )


def _record_chart_block_tokens(block: str) -> list[str]:
    normalized = block.strip()
    if re.fullmatch(r"[RrLl]+", normalized):
        return list(normalized.upper())
    return [token for token in re.split(r"[\s,]+", normalized) if token]


def _is_record_chart_rest(token: str) -> bool:
    return token.strip().lower() in {"rest", "background", "bg", "_", "-"}


def _record_chart_token_to_gesture(token: str) -> tuple[str, str, str]:
    normalized = token.strip().lower()
    if normalized in {"r", "right", "swipe_right", "stroke_right"}:
        return "swipe_right", "stroke_right", "R"
    if normalized in {"l", "left", "swipe_left", "stroke_left"}:
        return "swipe_left", "stroke_left", "L"
    raise ValueError(f"unsupported chart token={token!r}; use R, L, or rest")


def _record_chart_payload(plan: RecordChartPlan) -> dict[str, object]:
    return {
        "source": plan.chart,
        "duration": plan.duration,
        "segments": [asdict(segment) for segment in plan.segments],
        "gestures": [asdict(gesture) for gesture in plan.gestures],
    }


def _record_chart_preview_payload(
    *,
    label: str,
    state: dict[str, object],
    plan: RecordChartPlan,
) -> dict[str, object]:
    elapsed = float(state.get("elapsed", 0.0))
    current = _record_prompt_segment_at(elapsed, plan.segments)
    return {
        "label": label,
        "phase": state.get("phase", "recording"),
        "elapsed": elapsed,
        "duration": plan.duration,
        "countdown_remaining": state.get("countdown_remaining", 0.0),
        "current_text": current.text if current is not None else label,
        "current_kind": current.kind if current is not None else "prompt",
        "segments": [asdict(segment) for segment in plan.segments],
    }


def _write_chart_label_file(
    *,
    recording: Path,
    out: Path,
    plan: RecordChartPlan,
    participant: str,
) -> None:
    label_file = init_label_file(
        recording,
        participant_id=participant,
        notes=(
            "Initialized from an AirDesk chart. Coarse stroke/recovery labels are "
            "timing prompts, not hand-refined ground truth."
        ),
    )
    if label_file.session.start_timestamp is None:
        raise ValueError("recording has no tracking frames; cannot create chart labels")
    recording_start = label_file.session.start_timestamp
    recording_end = label_file.session.end_timestamp
    required_duration = max(gesture.recovery_end for gesture in plan.gestures)
    if recording_end is not None and recording_end - recording_start + 1e-6 < required_duration:
        raise ValueError(
            "recording ended before the final gesture/recovery label window; rerun the take "
            "or pass the same chart timing used during recording"
        )

    updated = label_file
    notes = "Generated from chart timing; refine before final training."
    last_gesture_index_by_block = {
        gesture.block_index: gesture.gesture_index for gesture in plan.gestures
    }
    for gesture in plan.gestures:
        stroke_start = recording_start + gesture.stroke_start
        stroke_end = recording_start + gesture.stroke_end
        updated = add_phase_label(
            updated,
            phase=gesture.phase,
            start_time=stroke_start,
            end_time=stroke_end,
            gesture=gesture.gesture,
            notes=notes,
        )
        if (
            gesture.gesture_index == last_gesture_index_by_block[gesture.block_index]
            and gesture.recovery_end > gesture.stroke_end
        ):
            updated = add_phase_label(
                updated,
                phase="recovery",
                start_time=stroke_end,
                end_time=recording_start + gesture.recovery_end,
                gesture=gesture.gesture,
                notes=notes,
            )
        updated = add_event_label(
            updated,
            gesture=gesture.gesture,
            start_time=stroke_start,
            end_time=stroke_end,
            commit_time=stroke_end,
            notes=notes,
        )
    _save_valid_label_file(updated, out)


def _record_preview_status(
    *,
    label: str,
    elapsed: float,
    duration: float | None,
    segments: tuple[RecordPromptSegment, ...],
) -> str:
    segment = _record_prompt_segment_at(elapsed, segments)
    total = f"{elapsed:.1f}s" if duration is None else f"{elapsed:.1f}/{duration:.1f}s"
    if segment is None:
        return f"recording {total} | {label}"
    remaining = max(0.0, segment.end - elapsed)
    next_segment = _next_record_prompt_segment(segment, segments)
    next_text = f" | NEXT {next_segment.text}" if next_segment is not None else ""
    if segment.kind == "cue" and segment.gesture is not None:
        return f"recording {total} | GET READY {segment.gesture} in {ceil(remaining)}{next_text}"
    if segment.kind == "stroke":
        return f"recording {total} | {segment.text} | {remaining:.1f}s{next_text}"
    return f"recording {total} | NOW {segment.text} | {remaining:.1f}s left{next_text}"


def _record_prompt_segment_at(
    elapsed: float,
    segments: tuple[RecordPromptSegment, ...],
) -> RecordPromptSegment | None:
    for segment in segments:
        if segment.start <= elapsed < segment.end:
            return segment
    return None


def _next_record_prompt_segment(
    current: RecordPromptSegment,
    segments: tuple[RecordPromptSegment, ...],
) -> RecordPromptSegment | None:
    for segment in segments:
        if segment.start >= current.end:
            return segment
    return None


def _handle_record_preview_key(key: int, state: dict[str, object]) -> bool:
    if state.get("phase") == "waiting" and key == ord(" "):
        state["phase"] = "countdown"
        return True
    if key in (27, ord("q")):
        state["quit"] = True
        return True
    return False


def _record_collection_take(
    *,
    output: Path,
    label: str,
    repetition: int,
    backend: str,
    device: str,
    width: int | None,
    height: int | None,
    fps: float | None,
    fourcc: str | None,
    duration: float,
    countdown: float,
    model_path: Path,
    auto_download_model: bool,
    max_num_hands: int,
    min_detection_confidence: float,
    min_presence_confidence: float,
    min_tracking_confidence: float,
    hand_delegate: str,
    show: bool,
    preview_driven: bool,
) -> CollectionTakeResult:
    status = {"text": f"ready: {label} rep {repetition}"}
    state: dict[str, str | None] = {
        "phase": "waiting" if preview_driven else "countdown",
        "decision": None,
    }
    tracker = _make_tracker(
        backend=backend,
        device=device,
        max_frames=None,
        show=show,
        camera_settings=CameraSettings(width=width, height=height, fps=fps, fourcc=fourcc),
        model_path=model_path,
        auto_download_model=auto_download_model,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        delegate=hand_delegate,
    )
    _attach_collection_preview_controls(
        tracker,
        status_provider=lambda: status["text"],
        key_handler=lambda key: _handle_collection_preview_key(key, state),
    )
    frame_count = 0
    interrupted = False
    first_frame_timestamp: float | None = None
    recording_started_at: float | None = None

    try:
        tracker.start()
        with JsonlRecordingWriter(output) as writer:
            writer.write_event(
                EventLogEntry(
                    event_type="collection_take_started",
                    timestamp=utc_timestamp(),
                    payload={
                        "backend": backend,
                        "device": device,
                        "duration": duration,
                        "countdown": countdown,
                        "label": label,
                        "repetition": repetition,
                        "model_path": str(model_path),
                        "mediapipe": {
                            "max_num_hands": max_num_hands,
                            "min_detection_confidence": min_detection_confidence,
                            "min_presence_confidence": min_presence_confidence,
                            "min_tracking_confidence": min_tracking_confidence,
                            "delegate": hand_delegate,
                        },
                        "camera_settings": CameraSettings(
                            width=width,
                            height=height,
                            fps=fps,
                            fourcc=fourcc,
                        ).to_dict(),
                    },
                )
            )
            try:
                for frame in tracker.frames():
                    if state["phase"] == "done":
                        break
                    if state["phase"] == "waiting":
                        status["text"] = (
                            f"{label} rep {repetition} | position hand | "
                            "space=start s=skip q=quit"
                        )
                        continue
                    if state["phase"] == "review":
                        if state["decision"] is not None:
                            break
                        status["text"] = (
                            f"done frames={frame_count} | k=keep r=redo s=skip q=quit"
                        )
                        continue
                    if state["phase"] == "countdown":
                        if first_frame_timestamp is None:
                            first_frame_timestamp = frame.timestamp
                        elapsed = frame.timestamp - first_frame_timestamp
                        if elapsed < countdown:
                            remaining = countdown - elapsed
                            status["text"] = (
                                f"countdown {remaining:.1f}s | {label} rep {repetition}"
                            )
                            continue
                        state["phase"] = "recording"
                    if recording_started_at is None:
                        recording_started_at = frame.timestamp
                        writer.write_event(
                            EventLogEntry(
                                event_type="collection_recording_started",
                                timestamp=utc_timestamp(),
                                payload={"label": label, "repetition": repetition},
                            )
                        )
                    recorded_elapsed = frame.timestamp - recording_started_at
                    if recorded_elapsed >= duration:
                        state["phase"] = "review"
                        if not preview_driven:
                            state["decision"] = "keep"
                            break
                        status["text"] = (
                            f"done frames={frame_count} | k=keep r=redo s=skip q=quit"
                        )
                        continue
                    status["text"] = (
                        f"recording {recorded_elapsed:.1f}/{duration:.1f}s | "
                        f"{label} rep {repetition}"
                    )
                    writer.write_tracking_frame(frame)
                    frame_count += 1
            except KeyboardInterrupt:
                interrupted = True
                writer.write_event(
                    EventLogEntry(
                        event_type="collection_take_interrupted",
                        timestamp=utc_timestamp(),
                        payload={"frames": frame_count, "label": label, "repetition": repetition},
                    )
                )
            status["text"] = f"done: {label} rep {repetition} frames={frame_count}"
            writer.write_event(
                EventLogEntry(
                    event_type="collection_take_finished",
                    timestamp=utc_timestamp(),
                    payload={
                        "frames": frame_count,
                        "interrupted": interrupted,
                        "duration": duration,
                        "countdown": countdown,
                        "label": label,
                        "repetition": repetition,
                    },
                )
            )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        tracker.stop()
    return CollectionTakeResult(frames=frame_count, decision=state["decision"] or "keep")


def _attach_collection_preview_controls(
    tracker: HandTrackerBackend,
    status_provider: Callable[[], str],
    key_handler: Callable[[int], bool] | None = None,
) -> None:
    if isinstance(tracker, MediaPipeHandTrackerBackend):
        tracker.preview_status_provider = status_provider
        tracker.preview_key_handler = key_handler


def _handle_collection_preview_key(key: int, state: dict[str, str | None]) -> bool:
    phase = state["phase"]
    if phase == "waiting":
        if key == ord(" "):
            state["phase"] = "countdown"
            return True
        if key == ord("s"):
            state["decision"] = "skip"
            state["phase"] = "done"
            return True
        if key == ord("q"):
            state["decision"] = "quit"
            state["phase"] = "done"
            return True
    if phase == "review":
        decisions = {
            ord("k"): "keep",
            ord("r"): "redo",
            ord("s"): "skip",
            ord("q"): "quit",
        }
        if key in decisions:
            state["decision"] = decisions[key]
            state["phase"] = "done"
            return True
    return False


def _next_collection_path(out_dir: Path, label: str, repetition: int) -> Path:
    slug = _slugify_label(label)
    candidate = out_dir / f"{slug}-{repetition:03d}.jsonl"
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = out_dir / f"{slug}-{repetition:03d}-{suffix}.jsonl"
        if not candidate.exists():
            return candidate
        suffix += 1


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "take"


def _summarize_records(path: Path, *, recognize: bool) -> dict[str, int]:
    summary = {
        "frames": 0,
        "events": 0,
        "hands": 0,
        "open_palm": 0,
        "fist": 0,
        "pinch": 0,
        "swipe_left": 0,
        "swipe_right": 0,
        "point_left": 0,
        "point_right": 0,
    }
    static_recognizer = StaticHandPoseRecognizer()
    recognizer = CompositeGestureRecognizer(
        recognizers=(
            static_recognizer,
            IntentGatedSwipeRecognizer(pose_recognizer=static_recognizer),
        )
    )
    for record_item in iter_recording(path):
        if record_item.kind == "tracking_frame":
            assert isinstance(record_item.payload, TrackingFrame)
            summary["frames"] += 1
            summary["hands"] += len(record_item.payload.hands)
            if recognize:
                for candidate in recognizer.recognize(record_item.payload):
                    if candidate.name in summary:
                        summary[candidate.name] += 1
        elif record_item.kind == "event":
            summary["events"] += 1
    return summary


def _format_summary(summary: dict[str, int]) -> str:
    return " ".join(f"{key}={value}" for key, value in summary.items())


def _collection_paths(path: Path, *, pattern: str) -> list[Path]:
    if path.is_dir():
        return sorted(recording for recording in path.glob(pattern) if recording.is_file())
    return [path]


def _collection_summary_row(path: Path) -> dict[str, str | int | float]:
    analysis = analyze_recording(path)
    return {
        "file": path.name,
        "label": _recording_label(path),
        **analysis.to_flat_dict(),
    }


def _recording_label(path: Path) -> str:
    for record_item in iter_recording(path):
        if record_item.kind != "event":
            continue
        assert isinstance(record_item.payload, EventLogEntry)
        label = record_item.payload.payload.get("label")
        if isinstance(label, str) and label:
            return label
    return re.sub(r"-\d{3}(?:-\d+)?$", "", path.stem)


def _format_collection_row(row: dict[str, str | int | float]) -> str:
    keys = (
        "file",
        "label",
        "frames",
        "hand_frames",
        "average_fps",
        "open_palm_count",
        "swipe_left_count",
        "swipe_right_count",
        "point_left_count",
        "point_right_count",
        "pinch_count",
        "fist_count",
    )
    return " ".join(f"{key}={row.get(key, 0)}" for key in keys)


def _format_collection_totals(rows: list[dict[str, str | int | float]]) -> str:
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        label = str(row["label"])
        label_totals = totals.setdefault(
            label,
            {
                "files": 0,
                "frames": 0,
                "hand_frames": 0,
                "swipe_left_count": 0,
                "swipe_right_count": 0,
                "point_left_count": 0,
                "point_right_count": 0,
                "pinch_count": 0,
                "fist_count": 0,
            },
        )
        label_totals["files"] += 1
        for key in tuple(label_totals):
            if key == "files":
                continue
            label_totals[key] += int(row.get(key, 0))

    parts = []
    for label, label_totals in sorted(totals.items()):
        values = " ".join(f"{key}={value}" for key, value in label_totals.items())
        parts.append(f"label={label} {values}")
    return "totals | " + " | ".join(parts)


def register_recording_commands(app: typer.Typer, gesture_app: typer.Typer) -> None:
    """Attach recording-related commands to the public CLI app."""
    app.command()(replay)
    app.command("collection-summary")(collection_summary)
    app.command()(record)
    app.command()(collect)
    gesture_app.command("chart-record")(gesture_chart_record)
    gesture_app.command("chart-label")(gesture_chart_label)
