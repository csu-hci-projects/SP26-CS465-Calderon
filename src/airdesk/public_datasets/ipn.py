"""IPN Hand dataset conversion into AirDesk replay/feature artifacts."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from airdesk.features import export_features_csv
from airdesk.labels import (
    GestureEventLabel,
    GestureLabelFile,
    GesturePhaseLabel,
    init_label_file,
    save_label_file,
    validate_label_file,
)
from airdesk.ml import build_tcn_dataset_manifest, save_tcn_dataset_manifest
from airdesk.recording.jsonl import JsonlRecordingWriter
from airdesk.state.types import FrameMetadata, TrackingFrame
from airdesk.tracking.mediapipe import (
    DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    DEFAULT_HAND_LANDMARKER_MODEL,
    ensure_hand_landmarker_model,
    normalized_hands_from_mediapipe_results,
)

IPN_CLASS_NAMES = {
    "D0X": "Non-gesture",
    "B0A": "Pointing with one finger",
    "B0B": "Pointing with two fingers",
    "G01": "Click with one finger",
    "G02": "Click with two fingers",
    "G03": "Throw up",
    "G04": "Throw down",
    "G05": "Throw left",
    "G06": "Throw right",
    "G07": "Open twice",
    "G08": "Double click with one finger",
    "G09": "Double click with two fingers",
    "G10": "Zoom in",
    "G11": "Zoom out",
}

IPN_AIRDESK_ATOMIC_MAP = {
    "G05": ("swipe_left", "stroke_left"),
    "G06": ("swipe_right", "stroke_right"),
}

_IPN_SPLIT_FILES = {
    "train": "trainlistall.txt",
    "training": "trainlistall.txt",
    "val": "vallistall.txt",
    "valid": "vallistall.txt",
    "validation": "vallistall.txt",
}

_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


@dataclass(frozen=True)
class IpnSegment:
    """One IPN Hand annotation interval."""

    video_id: str
    class_index: int
    label: str
    start_frame: int
    end_frame: int
    subset: str

    @property
    def maps_to_airdesk(self) -> bool:
        return self.label in IPN_AIRDESK_ATOMIC_MAP


@dataclass(frozen=True)
class IpnConversionResult:
    """Paths and counts emitted by one IPN conversion run."""

    converted_videos: int
    converted_segments: int
    mapped_segments: int
    recording_paths: tuple[Path, ...]
    label_paths: tuple[Path, ...]
    feature_paths: tuple[Path, ...]
    manifest_path: Path | None = None


def load_ipn_class_index(path: Path) -> dict[int, str]:
    """Load an IPN ``classInd*.txt`` file as ``class_index -> class_code``."""
    class_index: dict[int, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: expected '<index> <label>'")
        class_index[int(parts[0])] = parts[1]
    if not class_index:
        raise ValueError(f"{path}: no IPN classes found")
    return class_index


def load_ipn_split_segments(
    path: Path,
    *,
    class_index: dict[int, str] | None = None,
    subset: str | None = None,
) -> list[IpnSegment]:
    """Load an IPN train/validation list into typed segments."""
    resolved_class_index = class_index or {}
    segments: list[IpnSegment] = []
    resolved_subset = subset or _subset_from_split_path(path)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 4:
            raise ValueError(
                f"{path}:{line_number}: expected '<video_or_frames_path> <class> <start> <end>'"
            )
        video_id = _video_id_from_ipn_path(parts[0])
        class_number = int(parts[1])
        label = resolved_class_index.get(class_number, str(class_number))
        start_frame = int(parts[2])
        end_frame = int(parts[3])
        if start_frame <= 0 or end_frame < start_frame:
            raise ValueError(f"{path}:{line_number}: invalid frame interval")
        segments.append(
            IpnSegment(
                video_id=video_id,
                class_index=class_number,
                label=label,
                start_frame=start_frame,
                end_frame=end_frame,
                subset=resolved_subset,
            )
        )
    return segments


def split_file_for_ipn_annotations(annotations_dir: Path, split: str) -> Path:
    """Return the standard IPN annotation-list path for a split name."""
    try:
        filename = _IPN_SPLIT_FILES[split.strip().lower()]
    except KeyError as exc:
        options = ", ".join(sorted(_IPN_SPLIT_FILES))
        raise ValueError(f"unsupported IPN split={split!r}; use one of: {options}") from exc
    path = annotations_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"missing IPN annotation split file: {path}")
    return path


def find_ipn_video_path(videos_dir: Path, video_id: str) -> Path:
    """Find an IPN video by id under a downloaded video directory."""
    candidates = [videos_dir / video_id]
    candidates.extend(videos_dir / f"{video_id}{suffix}" for suffix in _VIDEO_SUFFIXES)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches: list[Path] = []
    for suffix in _VIDEO_SUFFIXES:
        matches.extend(videos_dir.rglob(f"{video_id}{suffix}"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        formatted = ", ".join(str(path) for path in sorted(matches)[:5])
        raise ValueError(f"multiple IPN video matches for {video_id}: {formatted}")
    raise FileNotFoundError(f"could not find IPN video {video_id} under {videos_dir}")


def convert_ipn_videos(
    *,
    videos_dir: Path,
    annotations_dir: Path,
    out_dir: Path,
    split: str = "train",
    video_ids: tuple[str, ...] = (),
    limit: int | None = 1,
    model_path: Path = DEFAULT_HAND_LANDMARKER_MODEL,
    auto_download_model: bool = True,
    max_num_hands: int = 2,
    min_detection_confidence: float = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    min_hand_presence_confidence: float = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    min_tracking_confidence: float = DEFAULT_HAND_LANDMARKER_MIN_CONFIDENCE,
    delegate: str = "cpu",
    manifest_out: Path | None = None,
    frame_limit: int | None = None,
) -> IpnConversionResult:
    """Convert selected IPN videos into AirDesk recordings, labels, and features."""
    class_index = load_ipn_class_index(annotations_dir / "classIndAll.txt")
    split_path = split_file_for_ipn_annotations(annotations_dir, split)
    segments = load_ipn_split_segments(split_path, class_index=class_index, subset=split)
    segments_by_video = _group_segments_by_video(segments)
    selected_video_ids = _selected_video_ids(
        tuple(segments_by_video),
        requested=video_ids,
        limit=limit,
    )

    recording_dir = out_dir / "recordings"
    labels_dir = out_dir / "labels"
    features_dir = out_dir / "features"
    recording_paths: list[Path] = []
    label_paths: list[Path] = []
    feature_paths: list[Path] = []
    converted_segments = 0
    mapped_segments = 0

    for video_id in selected_video_ids:
        video_segments = segments_by_video[video_id]
        video_path = find_ipn_video_path(videos_dir, video_id)
        recording_path = recording_dir / f"{video_id}.jsonl"
        label_path = labels_dir / f"{video_id}.labels.json"
        feature_path = features_dir / f"{video_id}.csv"

        frame_count, fps = _track_video_to_recording(
            video_path=video_path,
            recording_path=recording_path,
            model_path=model_path,
            auto_download_model=auto_download_model,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            delegate=delegate,
            frame_limit=frame_limit,
        )
        label_file = _ipn_label_file_for_recording(
            recording_path=recording_path,
            segments=video_segments,
            fps=fps,
            observed_frame_count=frame_count,
        )
        validation = validate_label_file(label_file)
        if not validation.ok:
            details = "; ".join(validation.errors)
            raise ValueError(f"invalid generated IPN labels for {video_id}: {details}")
        save_label_file(label_file, label_path)
        export_features_csv(recording_path, feature_path, labels=label_file)

        recording_paths.append(recording_path)
        label_paths.append(label_path)
        feature_paths.append(feature_path)
        converted_segments += len(video_segments)
        mapped_segments += sum(1 for segment in video_segments if segment.maps_to_airdesk)

    if manifest_out is not None:
        manifest = build_tcn_dataset_manifest(
            feature_paths,
            labels_dir=labels_dir,
            feature_preset="stream-invariant-v2",
            target_mode="v2-evidence",
            target_assignment="label",
        )
        save_tcn_dataset_manifest(manifest, manifest_out)

    return IpnConversionResult(
        converted_videos=len(recording_paths),
        converted_segments=converted_segments,
        mapped_segments=mapped_segments,
        recording_paths=tuple(recording_paths),
        label_paths=tuple(label_paths),
        feature_paths=tuple(feature_paths),
        manifest_path=manifest_out,
    )


def _track_video_to_recording(
    *,
    video_path: Path,
    recording_path: Path,
    model_path: Path,
    auto_download_model: bool,
    max_num_hands: int,
    min_detection_confidence: float,
    min_hand_presence_confidence: float,
    min_tracking_confidence: float,
    delegate: str,
    frame_limit: int | None,
) -> tuple[int, float]:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )
    except ImportError as exc:
        raise RuntimeError(
            "IPN conversion requires MediaPipe/OpenCV live dependencies. Run "
            "`uv sync --dev --extra live` before converting videos."
        ) from exc

    resolved_model = ensure_hand_landmarker_model(model_path, download=auto_download_model)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open IPN video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(resolved_model),
            delegate=_mediapipe_delegate(BaseOptions, delegate),
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    landmarker = HandLandmarker.create_from_options(options)
    sequence = 0
    try:
        with JsonlRecordingWriter(recording_path) as writer:
            while frame_limit is None or sequence < frame_limit:
                ok, image = capture.read()
                if not ok:
                    break
                sequence += 1
                if width <= 0 or height <= 0:
                    height, width = image.shape[:2]
                timestamp = (sequence - 1) / fps
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect_for_video(media_image, int(timestamp * 1000))
                hands = normalized_hands_from_mediapipe_results(results)
                metadata = FrameMetadata(
                    timestamp=timestamp,
                    source_id=video_path.stem,
                    width=width,
                    height=height,
                    sequence=sequence,
                    color_format="bgr",
                )
                writer.write_tracking_frame(
                    TrackingFrame(
                        timestamp=timestamp,
                        source_id=video_path.stem,
                        frame=metadata,
                        hands=hands,
                    )
                )
    finally:
        landmarker.close()
        capture.release()
    return sequence, fps


def _ipn_label_file_for_recording(
    *,
    recording_path: Path,
    segments: list[IpnSegment],
    fps: float,
    observed_frame_count: int,
) -> GestureLabelFile:
    label_file = init_label_file(
        recording_path,
        participant_id="ipn-public",
        notes=(
            "Generated from IPN Hand annotations. Only AirDesk atomic left/right "
            "swipe mappings are positive labels; other IPN gestures remain background "
            "for the first v2-evidence training pass."
        ),
    )
    event_labels: list[GestureEventLabel] = []
    phase_labels: list[GesturePhaseLabel] = list(label_file.phase_labels)
    max_frame = max(1, observed_frame_count)
    for segment in segments:
        mapped = IPN_AIRDESK_ATOMIC_MAP.get(segment.label)
        if mapped is None:
            continue
        gesture, phase = mapped
        start_time = _frame_to_time(min(segment.start_frame, max_frame), fps)
        end_time = _frame_to_time(min(segment.end_frame, max_frame), fps)
        if end_time < start_time:
            continue
        event_labels.append(
            GestureEventLabel(
                label_id=f"event-{len(event_labels) + 1:03d}",
                label_type="gesture",
                gesture=gesture,
                start_time=start_time,
                end_time=end_time,
                commit_time=end_time,
                notes=(
                    f"IPN {segment.label} {IPN_CLASS_NAMES.get(segment.label, '')}; "
                    "mapped to AirDesk atomic swipe evidence."
                ),
            )
        )
        phase_labels.append(
            GesturePhaseLabel(
                label_id=f"phase-{len(phase_labels) + 1:03d}",
                phase=phase,
                start_time=start_time,
                end_time=end_time,
                gesture=gesture,
                notes=f"IPN {segment.label} mapped to {phase}.",
            )
        )
    return GestureLabelFile(
        schema_version=label_file.schema_version,
        created_at=label_file.created_at,
        session=label_file.session,
        event_labels=tuple(event_labels),
        phase_labels=tuple(phase_labels),
    )


def write_ipn_mapping_csv(path: Path) -> None:
    """Write the current IPN-to-AirDesk mapping as a reviewable CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("ipn_label", "ipn_name", "airdesk_gesture", "airdesk_phase"),
        )
        writer.writeheader()
        for label, name in IPN_CLASS_NAMES.items():
            gesture, phase = IPN_AIRDESK_ATOMIC_MAP.get(label, ("", ""))
            writer.writerow(
                {
                    "ipn_label": label,
                    "ipn_name": name,
                    "airdesk_gesture": gesture,
                    "airdesk_phase": phase,
                }
            )


def _selected_video_ids(
    available_video_ids: tuple[str, ...],
    *,
    requested: tuple[str, ...],
    limit: int | None,
) -> tuple[str, ...]:
    if requested:
        missing = [video_id for video_id in requested if video_id not in available_video_ids]
        if missing:
            raise ValueError(f"requested IPN videos are not in the split: {', '.join(missing)}")
        selected = requested
    else:
        selected = available_video_ids
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive when provided")
        selected = selected[:limit]
    return selected


def _group_segments_by_video(segments: list[IpnSegment]) -> dict[str, list[IpnSegment]]:
    grouped: dict[str, list[IpnSegment]] = defaultdict(list)
    for segment in segments:
        grouped[segment.video_id].append(segment)
    return dict(grouped)


def _frame_to_time(frame_number: int, fps: float) -> float:
    return (frame_number - 1) / fps


def _video_id_from_ipn_path(raw_path: str) -> str:
    normalized = raw_path.strip().replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", maxsplit=1)[-1]


def _subset_from_split_path(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("train"):
        return "train"
    if name.startswith("val"):
        return "validation"
    return path.stem


def _mediapipe_delegate(base_options: Any, delegate: str) -> Any:
    normalized = delegate.strip().lower()
    if normalized == "cpu":
        return base_options.Delegate.CPU
    if normalized == "gpu":
        return base_options.Delegate.GPU
    raise RuntimeError("MediaPipe delegate must be 'cpu' or 'gpu'")
