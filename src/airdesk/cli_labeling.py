"""Label and feature-export CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from airdesk.cli_support import (
    _relative_label_time,
    _save_valid_label_file,
)
from airdesk.features import export_features_csv
from airdesk.labels import (
    add_event_label,
    add_ordered_sequence_labels,
    add_phase_label,
    init_label_file,
    load_label_file,
    save_label_file,
    suggest_stroke_label,
    validate_label_file,
)


def label_init(
    recording: Annotated[Path, typer.Argument(exists=True, readable=True)],
    out: Annotated[Path | None, typer.Option(help="Output label JSON path.")] = None,
    participant: Annotated[str, typer.Option(help="Participant/user identifier.")] = "caden",
    notes: Annotated[str, typer.Option(help="Starter notes for this label file.")] = "",
    overwrite: Annotated[bool, typer.Option(help="Overwrite an existing label file.")] = False,
) -> None:
    """Create a starter label file for a continuous recording."""
    output = out or recording.with_suffix(".labels.json")
    if output.exists() and not overwrite:
        typer.echo(f"Label file already exists: {output}. Use --overwrite to replace it.", err=True)
        raise typer.Exit(code=1)
    label_file = init_label_file(recording, participant_id=participant, notes=notes)
    save_label_file(label_file, output)
    typer.echo(
        f"wrote labels={output} frames={label_file.session.frame_count} "
        f"hand_frames={label_file.session.hand_frame_count}"
    )


def label_validate(path: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Validate a gesture label file."""
    result = validate_label_file(load_label_file(path))
    if not result.ok:
        for error in result.errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"valid labels={path}")


def label_add_phase(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    phase: Annotated[str, typer.Option(help="Phase label, e.g. stroke_left.")],
    start: Annotated[float, typer.Option(help="Start seconds relative to recording start.")],
    end: Annotated[float, typer.Option(help="End seconds relative to recording start.")],
    gesture: Annotated[str | None, typer.Option(help="Optional gesture name.")] = None,
    notes: Annotated[str, typer.Option(help="Optional notes.")] = "",
) -> None:
    """Append one phase label using relative seconds from the recording start."""
    label_file = load_label_file(path)
    updated = add_phase_label(
        label_file,
        phase=phase,
        start_time=_relative_label_time(label_file.session.start_timestamp, start),
        end_time=_relative_label_time(label_file.session.start_timestamp, end),
        gesture=gesture,
        notes=notes,
    )
    _save_valid_label_file(updated, path)
    typer.echo(f"added phase={phase} labels={path}")


def label_add_event(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    gesture: Annotated[str, typer.Option(help="Gesture name, e.g. swipe_left.")],
    start: Annotated[float, typer.Option(help="Start seconds relative to recording start.")],
    end: Annotated[float, typer.Option(help="End seconds relative to recording start.")],
    label_type: Annotated[str, typer.Option(help="Event label type.")] = "gesture",
    commit: Annotated[
        float | None,
        typer.Option(help="Optional commit seconds relative to recording start."),
    ] = None,
    intended_command: Annotated[str | None, typer.Option(help="Optional intended command.")] = None,
    success: Annotated[bool | None, typer.Option(help="Optional success flag.")] = None,
    notes: Annotated[str, typer.Option(help="Optional notes.")] = "",
) -> None:
    """Append one event label using relative seconds from the recording start."""
    label_file = load_label_file(path)
    updated = add_event_label(
        label_file,
        gesture=gesture,
        start_time=_relative_label_time(label_file.session.start_timestamp, start),
        end_time=_relative_label_time(label_file.session.start_timestamp, end),
        label_type=label_type,
        commit_time=(
            _relative_label_time(label_file.session.start_timestamp, commit)
            if commit is not None
            else None
        ),
        intended_command=intended_command,
        success=success,
        notes=notes,
    )
    _save_valid_label_file(updated, path)
    typer.echo(f"added event={gesture} labels={path}")


def label_add_sequence(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    sequence: Annotated[str, typer.Option(help="Ordered tokens, e.g. 'R L R R L L'.")],
    start: Annotated[
        float,
        typer.Option(help="Active-window start seconds relative to recording."),
    ],
    end: Annotated[float, typer.Option(help="Active-window end seconds relative to recording.")],
    gesture_fraction: Annotated[
        float,
        typer.Option(help="Fraction of each coarse slot assigned to stroke/event."),
    ] = 0.65,
    recovery_fraction: Annotated[
        float,
        typer.Option(help="Fraction of each coarse slot assigned to recovery/reset."),
    ] = 0.35,
    notes: Annotated[str, typer.Option(help="Optional notes for generated labels.")] = (
        "Coarse ordered-sequence label; refine timestamps before final training."
    ),
) -> None:
    """Append coarse labels for a remembered chained L/R sequence."""
    label_file = load_label_file(path)
    try:
        updated = add_ordered_sequence_labels(
            label_file,
            sequence=sequence.split(),
            start_time=_relative_label_time(label_file.session.start_timestamp, start),
            end_time=_relative_label_time(label_file.session.start_timestamp, end),
            gesture_fraction=gesture_fraction,
            recovery_fraction=recovery_fraction,
            notes=notes,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _save_valid_label_file(updated, path)
    typer.echo(f"added sequence_count={len(sequence.split())} labels={path}")


def label_suggest(
    recording: Annotated[Path, typer.Argument(exists=True, readable=True)],
    gesture: Annotated[
        str | None,
        typer.Option(help="Gesture name to label, e.g. swipe_left. Inferred when possible."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Label JSON path to create or update when --apply is used."),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(help="Apply the suggested phase and event labels to --out."),
    ] = False,
    participant: Annotated[str, typer.Option(help="Participant/user identifier.")] = "caden",
    min_duration: Annotated[
        float,
        typer.Option(help="Minimum stroke-window duration in seconds."),
    ] = 0.25,
    max_duration: Annotated[
        float,
        typer.Option(help="Maximum stroke-window duration in seconds."),
    ] = 1.25,
    pad_seconds: Annotated[
        float,
        typer.Option(help="Context padding added before/after the detected stroke."),
    ] = 0.08,
) -> None:
    """Suggest a stroke label from the strongest palm-motion window."""
    suggestion = suggest_stroke_label(
        recording,
        gesture=gesture,
        min_duration=min_duration,
        max_duration=max_duration,
        pad_seconds=pad_seconds,
    )
    typer.echo(
        "suggestion "
        f"gesture={suggestion.gesture} phase={suggestion.phase} "
        f"start={suggestion.start_seconds:.3f} end={suggestion.end_seconds:.3f} "
        f"direction={suggestion.direction} confidence={suggestion.confidence:.2f}"
    )

    output = out or recording.with_suffix(".labels.json")
    if not apply:
        typer.echo(
            "apply with: "
            f"uv run airdesk label suggest {recording} --gesture {suggestion.gesture} "
            f"--out {output} --apply"
        )
        return

    label_file = load_label_file(output) if output.exists() else init_label_file(
        recording,
        participant_id=participant,
        notes="Initialized from label suggest.",
    )
    label_file = add_phase_label(
        label_file,
        phase=suggestion.phase,
        start_time=suggestion.start_time,
        end_time=suggestion.end_time,
        gesture=suggestion.gesture,
        notes=(
            "Auto-suggested from strongest palm-motion window; "
            "review before training/evaluation."
        ),
    )
    label_file = add_event_label(
        label_file,
        gesture=suggestion.gesture,
        start_time=suggestion.start_time,
        end_time=suggestion.end_time,
        label_type="gesture",
        notes=(
            "Auto-suggested from strongest palm-motion window; "
            "review before training/evaluation."
        ),
    )
    _save_valid_label_file(label_file, output)
    typer.echo(f"applied suggestion labels={output}")


def features_export(
    recording: Annotated[Path, typer.Argument(exists=True, readable=True)],
    out: Annotated[Path, typer.Option(help="Output CSV feature path.")],
    labels: Annotated[
        Path | None,
        typer.Option(help="Optional gesture labels JSON path."),
    ] = None,
) -> None:
    """Export deterministic landmark-derived features as CSV."""
    label_file = load_label_file(labels) if labels is not None else None
    rows = export_features_csv(recording, out, labels=label_file)
    typer.echo(f"exported features={out} rows={len(rows)}")




def register_label_commands(label_app: typer.Typer) -> None:
    """Register continuous-label commands."""
    label_app.command("init")(label_init)
    label_app.command("validate")(label_validate)
    label_app.command("add-phase")(label_add_phase)
    label_app.command("add-event")(label_add_event)
    label_app.command("add-sequence")(label_add_sequence)
    label_app.command("suggest")(label_suggest)


def register_feature_commands(features_app: typer.Typer) -> None:
    """Register feature export commands."""
    features_app.command("export")(features_export)
