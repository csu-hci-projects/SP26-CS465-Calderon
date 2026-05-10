"""Public dataset import CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from airdesk.public_datasets.ipn import convert_ipn_videos, write_ipn_mapping_csv
from airdesk.tracking.mediapipe import DEFAULT_HAND_LANDMARKER_MODEL


def public_data_ipn_convert(
    videos_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="IPN MP4 video root."),
    ],
    annotations_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            readable=True,
            help="IPN annotation directory containing classIndAll.txt and split lists.",
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(help="Output root for AirDesk recordings, labels, and features."),
    ],
    split: Annotated[str, typer.Option(help="IPN split to convert: train or val.")] = "train",
    video_id: Annotated[
        list[str] | None,
        typer.Option(help="Specific IPN video id to convert. Repeat for multiple videos."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Maximum videos to convert when --video-id is omitted."),
    ] = 1,
    manifest_out: Annotated[
        Path | None,
        typer.Option(help="Optional stream-invariant-v2 v2-evidence manifest path."),
    ] = None,
    mapping_out: Annotated[
        Path | None,
        typer.Option(help="Optional CSV documenting the current IPN-to-AirDesk mapping."),
    ] = None,
    model_path: Annotated[
        Path,
        typer.Option(help="MediaPipe Hand Landmarker .task model path."),
    ] = DEFAULT_HAND_LANDMARKER_MODEL,
    max_num_hands: Annotated[
        int,
        typer.Option(help="Maximum hands for MediaPipe to track."),
    ] = 2,
    hand_delegate: Annotated[
        str,
        typer.Option("--hand-delegate", help="MediaPipe delegate: cpu or gpu."),
    ] = "cpu",
    download_model: Annotated[
        bool,
        typer.Option(help="Download the MediaPipe model to --model-path if missing."),
    ] = True,
    frame_limit: Annotated[
        int | None,
        typer.Option(help="Debug limit on frames per video."),
    ] = None,
) -> None:
    """Convert IPN Hand videos into AirDesk replay/features/labels artifacts."""
    try:
        result = convert_ipn_videos(
            videos_dir=videos_dir,
            annotations_dir=annotations_dir,
            out_dir=out_dir,
            split=split,
            video_ids=tuple(video_id or ()),
            limit=limit,
            model_path=model_path,
            auto_download_model=download_model,
            max_num_hands=max_num_hands,
            delegate=hand_delegate,
            manifest_out=manifest_out,
            frame_limit=frame_limit,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if mapping_out is not None:
        write_ipn_mapping_csv(mapping_out)

    manifest_suffix = f" manifest={result.manifest_path}" if result.manifest_path else ""
    typer.echo(
        f"converted ipn_videos={result.converted_videos} "
        f"segments={result.converted_segments} mapped_atomic={result.mapped_segments} "
        f"recordings={len(result.recording_paths)} features={len(result.feature_paths)}"
        f"{manifest_suffix}"
    )


def register_public_data_commands(public_data_app: typer.Typer) -> None:
    """Register public dataset import commands."""
    public_data_app.command("ipn-convert")(public_data_ipn_convert)
