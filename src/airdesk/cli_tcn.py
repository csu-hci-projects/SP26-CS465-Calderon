"""Offline TCN and feature-diagnostic CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from airdesk.analysis import (
    diagnose_tcn_manifest_events,
    diagnose_tcn_v2_manifest_events,
    evaluate_tcn_manifest,
    evaluate_tcn_v2_manifest,
    holdout_totals,
)
from airdesk.gestures.decoder import EventDecoderConfig
from airdesk.labels import load_label_file
from airdesk.ml import (
    CausalTcnTrainingConfig,
    CausalTcnV2TrainingConfig,
    FeatureDiagnosticsReport,
    MissingMlDependencyError,
    build_feature_diagnostics_report,
    build_tcn_dataset_manifest,
    save_feature_diagnostics_report,
    save_tcn_dataset_manifest,
    train_causal_tcn,
    train_causal_tcn_v2,
)


def gesture_build_tcn_dataset(
    features_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Feature CSV directory."),
    ],
    out: Annotated[Path, typer.Option(help="Output dataset manifest JSON path.")],
    labels_dir: Annotated[
        Path | None,
        typer.Option(file_okay=False, readable=True, help="Optional matching label directory."),
    ] = None,
    pattern: Annotated[str, typer.Option(help="Feature filename glob pattern.")] = "*.csv",
    window_seconds: Annotated[
        float,
        typer.Option(help="Sliding window duration in seconds."),
    ] = 0.8,
    stride_seconds: Annotated[
        float,
        typer.Option(help="Sliding window stride in seconds."),
    ] = 0.2,
    min_rows: Annotated[
        int,
        typer.Option(help="Minimum feature rows required per window."),
    ] = 4,
    min_gesture_fraction: Annotated[
        float,
        typer.Option(help="Minimum in-window gesture-frame fraction for a swipe target."),
    ] = 0.35,
    feature_preset: Annotated[
        str,
        typer.Option(help="Feature preset: legacy or stream-invariant."),
    ] = "legacy",
    target_mode: Annotated[
        str,
        typer.Option(help="Target mode: event, phase, phase-stroke, or v2-evidence."),
    ] = "event",
    target_assignment: Annotated[
        str,
        typer.Option(help="Target assignment: label or motion-gated."),
    ] = "label",
    motion_gate_min_dx_per_hand_scale: Annotated[
        float,
        typer.Option(
            help=(
                "For motion-gated targets, minimum absolute trailing palm dx normalized by "
                "hand scale."
            ),
        ),
    ] = 0.35,
    motion_gate_min_direction_consistency: Annotated[
        float,
        typer.Option(
            help="For motion-gated targets, minimum same-direction motion consistency.",
        ),
    ] = 0.45,
) -> None:
    """Build a dependency-free manifest for TCN context windows."""
    feature_paths = sorted(features_dir.glob(pattern))
    if not feature_paths:
        typer.echo(f"No feature CSV files matched {features_dir}/{pattern}", err=True)
        raise typer.Exit(code=1)
    try:
        manifest = build_tcn_dataset_manifest(
            feature_paths,
            labels_dir=labels_dir,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            min_rows=min_rows,
            min_gesture_fraction=min_gesture_fraction,
            feature_preset=feature_preset,
            target_mode=target_mode,
            target_assignment=target_assignment,
            motion_gate_min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
            motion_gate_min_direction_consistency=motion_gate_min_direction_consistency,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    save_tcn_dataset_manifest(manifest, out)
    summary = manifest.to_dict()["summary"]
    window_counts = " ".join(
        f"{target}={count}" for target, count in summary["window_counts"].items()
    )
    evidence_counts = summary.get("evidence_frame_counts")
    evidence_suffix = ""
    if evidence_counts:
        formatted_evidence = " ".join(
            f"{target}={count}" for target, count in evidence_counts.items()
        )
        evidence_suffix = f" evidence_frames=({formatted_evidence})"
    typer.echo(
        f"wrote tcn_manifest={out} sources={summary['source_count']} "
        f"windows={summary['window_count']} preset={manifest.feature_preset} "
        f"target_mode={manifest.target_mode} target_assignment={manifest.target_assignment} "
        f"{window_counts}{evidence_suffix}"
    )


def gesture_train_tcn(
    manifest: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="TCN dataset manifest JSON path."),
    ],
    out: Annotated[Path, typer.Option(help="Output Torch checkpoint path.")],
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 25,
    learning_rate: Annotated[float, typer.Option(help="Adam learning rate.")] = 0.001,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 16,
    hidden_channels: Annotated[int, typer.Option(help="TCN hidden channels.")] = 24,
    levels: Annotated[int, typer.Option(help="Dilated causal convolution levels.")] = 2,
    kernel_size: Annotated[int, typer.Option(help="Causal convolution kernel size.")] = 3,
    dropout: Annotated[float, typer.Option(help="Dropout probability.")] = 0.0,
    validation_fraction: Annotated[
        float,
        typer.Option(help="Deterministic validation split fraction."),
    ] = 0.2,
    seed: Annotated[int, typer.Option(help="Deterministic training seed.")] = 7,
) -> None:
    """Train the first optional offline causal TCN classifier."""
    config = CausalTcnTrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        levels=levels,
        kernel_size=kernel_size,
        dropout=dropout,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    try:
        result = train_causal_tcn(manifest_path=manifest, out_path=out, config=config)
    except (MissingMlDependencyError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    validation = (
        f"{result.validation_accuracy:.3f}"
        if result.validation_accuracy is not None
        else "none"
    )
    typer.echo(
        f"wrote tcn_model={out} samples={result.samples} train={result.train_samples} "
        f"validation={result.validation_samples} final_loss={result.final_train_loss:.4f} "
        f"train_accuracy={result.train_accuracy:.3f} validation_accuracy={validation}"
    )


def gesture_train_tcn_v2(
    manifest: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="TCN v2 evidence manifest JSON path."),
    ],
    out: Annotated[Path, typer.Option(help="Output Torch checkpoint path.")],
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 25,
    learning_rate: Annotated[float, typer.Option(help="Adam learning rate.")] = 0.001,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 16,
    hidden_channels: Annotated[int, typer.Option(help="TCN hidden channels.")] = 32,
    levels: Annotated[int, typer.Option(help="Dilated causal convolution levels.")] = 3,
    kernel_size: Annotated[int, typer.Option(help="Causal convolution kernel size.")] = 3,
    dropout: Annotated[float, typer.Option(help="Dropout probability.")] = 0.10,
    validation_fraction: Annotated[
        float,
        typer.Option(help="Deterministic validation split fraction."),
    ] = 0.2,
    seed: Annotated[int, typer.Option(help="Deterministic training seed.")] = 7,
    positive_weight_cap: Annotated[
        float,
        typer.Option(help="Maximum BCE positive-class weight per evidence head."),
    ] = 30.0,
    boundary_positive_weight_multiplier: Annotated[
        float,
        typer.Option(help="Extra positive-weight multiplier for sparse start/end heads."),
    ] = 2.0,
    focal_gamma: Annotated[
        float,
        typer.Option(help="Focal loss gamma for hard evidence frames; 0 disables focal scaling."),
    ] = 1.0,
) -> None:
    """Train a TCN v2 sequence model for decoder-facing evidence heads."""
    config = CausalTcnV2TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        levels=levels,
        kernel_size=kernel_size,
        dropout=dropout,
        validation_fraction=validation_fraction,
        seed=seed,
        positive_weight_cap=positive_weight_cap,
        boundary_positive_weight_multiplier=boundary_positive_weight_multiplier,
        focal_gamma=focal_gamma,
    )
    try:
        result = train_causal_tcn_v2(manifest_path=manifest, out_path=out, config=config)
    except (MissingMlDependencyError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    validation = (
        f"{result.validation_accuracy:.3f}"
        if result.validation_accuracy is not None
        else "none"
    )
    typer.echo(
        f"wrote tcn_v2_model={out} samples={result.samples} train={result.train_samples} "
        f"validation={result.validation_samples} final_loss={result.final_train_loss:.4f} "
        f"train_frame_accuracy={result.train_accuracy:.3f} "
        f"validation_frame_accuracy={validation} heads={','.join(result.targets)}"
    )


def gesture_evaluate_tcn(
    manifest: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="TCN dataset manifest JSON path."),
    ],
    model: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="TCN checkpoint path."),
    ],
    out: Annotated[Path | None, typer.Option(help="Optional JSON summary output path.")] = None,
    confidence_threshold: Annotated[
        float,
        typer.Option(help="Minimum softmax confidence for a non-background candidate."),
    ] = 0.5,
    cooldown_seconds: Annotated[
        float,
        typer.Option(help="Suppress repeated same-gesture windows within this many seconds."),
    ] = 0.5,
    match_tolerance_seconds: Annotated[
        float,
        typer.Option(help="Tolerance after an event interval for window-end matching."),
    ] = 0.5,
    event_decoder: Annotated[
        bool,
        typer.Option(help="Decode probability streams with hysteresis/peaks/cooldown."),
    ] = False,
    activation_threshold: Annotated[
        float,
        typer.Option(help="Event decoder activation threshold."),
    ] = 0.55,
    release_threshold: Annotated[
        float,
        typer.Option(help="Event decoder release threshold."),
    ] = 0.35,
    min_peak_confidence: Annotated[
        float,
        typer.Option(help="Event decoder minimum peak confidence."),
    ] = 0.60,
) -> None:
    """Evaluate a trained TCN checkpoint over a labeled feature manifest."""
    try:
        decoder_config = (
            EventDecoderConfig(
                activation_threshold=activation_threshold,
                release_threshold=release_threshold,
                min_peak_confidence=min_peak_confidence,
                min_event_separation_seconds=cooldown_seconds,
                cooldown_seconds=cooldown_seconds,
            )
            if event_decoder
            else None
        )
        evaluations = evaluate_tcn_manifest(
            manifest_path=manifest,
            model_path=model,
            confidence_threshold=confidence_threshold,
            cooldown_seconds=cooldown_seconds,
            match_tolerance_seconds=match_tolerance_seconds,
            event_decoder_config=decoder_config,
        )
    except (MissingMlDependencyError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    summary = holdout_totals(evaluations)
    payload = {
        "recognizer": "tcn_event_decoder" if event_decoder else "tcn",
        "manifest": str(manifest),
        "model": str(model),
        "confidence_threshold": confidence_threshold,
        "cooldown_seconds": cooldown_seconds,
        "match_tolerance_seconds": match_tolerance_seconds,
        "event_decoder": decoder_config.to_dict() if decoder_config is not None else None,
        "summary": summary,
        "evaluations": [evaluation.to_dict() for evaluation in evaluations],
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    latency = summary["mean_latency_seconds"]
    formatted_latency = round(latency, 4) if isinstance(latency, float) else "unknown"
    typer.echo(
        f"recognizer={'tcn_event_decoder' if event_decoder else 'tcn'} "
        f"recordings={summary['recordings']} "
        f"intended={summary['intended_events']} matched={summary['matched_events']} "
        f"missed={summary['missed_events']} candidates={summary['candidate_count']} "
        f"false_activations={summary['false_activations']} "
        f"repeated_fires={summary['repeated_fires']} mean_latency={formatted_latency}"
    )
    if out is not None:
        typer.echo(f"wrote evaluation={out}")


def gesture_evaluate_tcn_v2(
    manifest: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="TCN v2 evidence manifest JSON path."),
    ],
    model: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="TCN v2 checkpoint path."),
    ],
    out: Annotated[Path | None, typer.Option(help="Optional JSON summary output path.")] = None,
    match_tolerance_seconds: Annotated[
        float,
        typer.Option(help="Tolerance after an event interval for event matching."),
    ] = 0.5,
    early_match_tolerance_seconds: Annotated[
        float,
        typer.Option(
            help="Tolerance before an event interval for causal early detections.",
        ),
    ] = 0.0,
    activation_threshold: Annotated[
        float,
        typer.Option(help="Event decoder activation threshold over stroke evidence."),
    ] = 0.55,
    release_threshold: Annotated[
        float,
        typer.Option(help="Event decoder release threshold over stroke evidence."),
    ] = 0.35,
    min_peak_confidence: Annotated[
        float,
        typer.Option(help="Event decoder minimum peak stroke confidence."),
    ] = 0.60,
    cooldown_seconds: Annotated[
        float,
        typer.Option(help="Decoder same-gesture separation/cooldown in seconds."),
    ] = 0.5,
) -> None:
    """Evaluate TCN v2 evidence heads through the replay event decoder."""
    decoder_config = EventDecoderConfig(
        activation_threshold=activation_threshold,
        release_threshold=release_threshold,
        min_peak_confidence=min_peak_confidence,
        min_event_separation_seconds=cooldown_seconds,
        cooldown_seconds=cooldown_seconds,
    )
    try:
        evaluations = evaluate_tcn_v2_manifest(
            manifest_path=manifest,
            model_path=model,
            match_tolerance_seconds=match_tolerance_seconds,
            early_match_tolerance_seconds=early_match_tolerance_seconds,
            event_decoder_config=decoder_config,
        )
    except (MissingMlDependencyError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    summary = holdout_totals(evaluations)
    payload = {
        "recognizer": "tcn_v2_event_decoder",
        "manifest": str(manifest),
        "model": str(model),
        "match_tolerance_seconds": match_tolerance_seconds,
        "early_match_tolerance_seconds": early_match_tolerance_seconds,
        "event_decoder": decoder_config.to_dict(),
        "summary": summary,
        "evaluations": [evaluation.to_dict() for evaluation in evaluations],
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    latency = summary["mean_latency_seconds"]
    formatted_latency = round(latency, 4) if isinstance(latency, float) else "unknown"
    typer.echo(
        f"recognizer=tcn_v2_event_decoder recordings={summary['recordings']} "
        f"intended={summary['intended_events']} matched={summary['matched_events']} "
        f"missed={summary['missed_events']} candidates={summary['candidate_count']} "
        f"false_activations={summary['false_activations']} "
        f"repeated_fires={summary['repeated_fires']} mean_latency={formatted_latency}"
    )
    if out is not None:
        typer.echo(f"wrote evaluation={out}")


def gesture_diagnose_tcn_events(
    manifest: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="TCN dataset manifest JSON path."),
    ],
    model: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="TCN checkpoint path."),
    ],
    out: Annotated[Path, typer.Option(help="Output detailed diagnostics JSON path.")],
    confidence_threshold: Annotated[
        float,
        typer.Option(help="Minimum softmax confidence for decoded probability frames."),
    ] = 0.35,
    cooldown_seconds: Annotated[
        float,
        typer.Option(help="Decoder same-gesture separation/cooldown in seconds."),
    ] = 0.5,
    match_tolerance_seconds: Annotated[
        float,
        typer.Option(help="Tolerance after an event interval for event matching."),
    ] = 0.5,
    activation_threshold: Annotated[
        float,
        typer.Option(help="Event decoder activation threshold."),
    ] = 0.35,
    release_threshold: Annotated[
        float,
        typer.Option(help="Event decoder release threshold."),
    ] = 0.2,
    min_peak_confidence: Annotated[
        float,
        typer.Option(help="Event decoder minimum peak confidence."),
    ] = 0.35,
) -> None:
    """Write per-event decoded TCN diagnostics for misses and false activations."""
    decoder_config = EventDecoderConfig(
        activation_threshold=activation_threshold,
        release_threshold=release_threshold,
        min_peak_confidence=min_peak_confidence,
        min_event_separation_seconds=cooldown_seconds,
        cooldown_seconds=cooldown_seconds,
    )
    try:
        payload = diagnose_tcn_manifest_events(
            manifest_path=manifest,
            model_path=model,
            confidence_threshold=confidence_threshold,
            cooldown_seconds=cooldown_seconds,
            match_tolerance_seconds=match_tolerance_seconds,
            event_decoder_config=decoder_config,
        )
    except (MissingMlDependencyError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary = payload["summary"]
    typer.echo(
        f"tcn_event_diagnostics recordings={summary['recordings']} "
        f"intended={summary['intended_events']} matched={summary['matched_events']} "
        f"missed={summary['missed_events']} candidates={summary['candidate_count']} "
        f"false_activations={summary['false_activations']} "
        f"repeated_fires={summary['repeated_fires']}"
    )
    typer.echo(f"wrote diagnostics={out}")


def gesture_diagnose_tcn_v2_events(
    manifest: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="TCN v2 evidence manifest JSON path."),
    ],
    model: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="TCN v2 checkpoint path."),
    ],
    out: Annotated[Path, typer.Option(help="Output detailed diagnostics JSON path.")],
    match_tolerance_seconds: Annotated[
        float,
        typer.Option(help="Tolerance after an event interval for event matching."),
    ] = 0.5,
    early_match_tolerance_seconds: Annotated[
        float,
        typer.Option(
            help="Tolerance before an event interval for causal early detections.",
        ),
    ] = 0.0,
    activation_threshold: Annotated[
        float,
        typer.Option(help="Event decoder activation threshold over stroke evidence."),
    ] = 0.35,
    release_threshold: Annotated[
        float,
        typer.Option(help="Event decoder release threshold over stroke evidence."),
    ] = 0.2,
    min_peak_confidence: Annotated[
        float,
        typer.Option(help="Event decoder minimum peak stroke confidence."),
    ] = 0.35,
    cooldown_seconds: Annotated[
        float,
        typer.Option(help="Decoder same-gesture separation/cooldown in seconds."),
    ] = 0.5,
) -> None:
    """Write per-event decoded TCN v2 evidence diagnostics."""
    decoder_config = EventDecoderConfig(
        activation_threshold=activation_threshold,
        release_threshold=release_threshold,
        min_peak_confidence=min_peak_confidence,
        min_event_separation_seconds=cooldown_seconds,
        cooldown_seconds=cooldown_seconds,
    )
    try:
        payload = diagnose_tcn_v2_manifest_events(
            manifest_path=manifest,
            model_path=model,
            match_tolerance_seconds=match_tolerance_seconds,
            early_match_tolerance_seconds=early_match_tolerance_seconds,
            event_decoder_config=decoder_config,
        )
    except (MissingMlDependencyError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary = payload["summary"]
    typer.echo(
        f"tcn_v2_event_diagnostics recordings={summary['recordings']} "
        f"intended={summary['intended_events']} matched={summary['matched_events']} "
        f"missed={summary['missed_events']} candidates={summary['candidate_count']} "
        f"false_activations={summary['false_activations']} "
        f"repeated_fires={summary['repeated_fires']}"
    )
    typer.echo(f"wrote diagnostics={out}")



def gesture_holdout_tcn(
    features_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Feature CSV directory."),
    ],
    labels_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Label directory."),
    ],
    out: Annotated[Path, typer.Option(help="Output JSON summary path.")],
    model_out: Annotated[Path, typer.Option(help="Output TCN checkpoint path.")],
    train_manifest_out: Annotated[
        Path | None,
        typer.Option(help="Optional output path for the train manifest."),
    ] = None,
    test_manifest_out: Annotated[
        Path | None,
        typer.Option(help="Optional output path for the test manifest."),
    ] = None,
    train_per_gesture: Annotated[int, typer.Option(help="Training files per gesture.")] = 6,
    test_per_gesture: Annotated[int, typer.Option(help="Held-out test files per gesture.")] = 2,
    train_negatives: Annotated[int, typer.Option(help="Training negative/background files.")] = 6,
    test_negatives: Annotated[int, typer.Option(help="Held-out negative/background files.")] = 2,
    window_seconds: Annotated[float, typer.Option(help="Sliding window duration.")] = 0.8,
    stride_seconds: Annotated[float, typer.Option(help="Sliding window stride.")] = 0.2,
    min_rows: Annotated[int, typer.Option(help="Minimum rows per window.")] = 4,
    min_gesture_fraction: Annotated[
        float,
        typer.Option(help="Minimum in-window gesture-frame fraction for a swipe target."),
    ] = 0.35,
    feature_preset: Annotated[
        str,
        typer.Option(help="Feature preset: legacy or stream-invariant."),
    ] = "legacy",
    target_mode: Annotated[
        str,
        typer.Option(help="Target mode: event, phase, or phase-stroke."),
    ] = "event",
    target_assignment: Annotated[
        str,
        typer.Option(help="Target assignment: label or motion-gated."),
    ] = "label",
    motion_gate_min_dx_per_hand_scale: Annotated[
        float,
        typer.Option(
            help=(
                "For motion-gated targets, minimum absolute trailing palm dx normalized by "
                "hand scale."
            ),
        ),
    ] = 0.35,
    motion_gate_min_direction_consistency: Annotated[
        float,
        typer.Option(
            help="For motion-gated targets, minimum same-direction motion consistency.",
        ),
    ] = 0.45,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 25,
    learning_rate: Annotated[float, typer.Option(help="Adam learning rate.")] = 0.001,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 16,
    hidden_channels: Annotated[int, typer.Option(help="TCN hidden channels.")] = 24,
    levels: Annotated[int, typer.Option(help="Dilated causal convolution levels.")] = 2,
    kernel_size: Annotated[int, typer.Option(help="Causal convolution kernel size.")] = 3,
    dropout: Annotated[float, typer.Option(help="Dropout probability.")] = 0.0,
    validation_fraction: Annotated[
        float,
        typer.Option(help="Deterministic validation split fraction within train windows."),
    ] = 0.2,
    seed: Annotated[int, typer.Option(help="Deterministic training seed.")] = 7,
    confidence_threshold: Annotated[
        float,
        typer.Option(help="Minimum softmax confidence for a non-background candidate."),
    ] = 0.5,
    cooldown_seconds: Annotated[
        float,
        typer.Option(help="Suppress repeated same-gesture windows within this many seconds."),
    ] = 0.5,
    match_tolerance_seconds: Annotated[
        float,
        typer.Option(help="Tolerance after an event interval for window-end matching."),
    ] = 0.5,
    event_decoder: Annotated[
        bool,
        typer.Option(help="Decode probability streams with hysteresis/peaks/cooldown."),
    ] = False,
    release_threshold: Annotated[
        float,
        typer.Option(help="Event decoder release threshold."),
    ] = 0.35,
    min_peak_confidence: Annotated[
        float,
        typer.Option(help="Event decoder minimum peak confidence."),
    ] = 0.55,
) -> None:
    """Train/evaluate TCN on a deterministic filename-ordered holdout split."""
    train_features, test_features = _split_tcn_feature_holdout(
        features_dir=features_dir,
        labels_dir=labels_dir,
        train_per_gesture=train_per_gesture,
        test_per_gesture=test_per_gesture,
        train_negatives=train_negatives,
        test_negatives=test_negatives,
    )
    train_manifest_path = train_manifest_out or out.with_name(f"{out.stem}-train-manifest.json")
    test_manifest_path = test_manifest_out or out.with_name(f"{out.stem}-test-manifest.json")
    train_manifest = build_tcn_dataset_manifest(
        train_features,
        labels_dir=labels_dir,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        min_rows=min_rows,
        min_gesture_fraction=min_gesture_fraction,
        feature_preset=feature_preset,
        target_mode=target_mode,
        target_assignment=target_assignment,
        motion_gate_min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
        motion_gate_min_direction_consistency=motion_gate_min_direction_consistency,
    )
    test_manifest = build_tcn_dataset_manifest(
        test_features,
        labels_dir=labels_dir,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        min_rows=min_rows,
        min_gesture_fraction=min_gesture_fraction,
        feature_preset=feature_preset,
        target_mode=target_mode,
        target_assignment=target_assignment,
        motion_gate_min_dx_per_hand_scale=motion_gate_min_dx_per_hand_scale,
        motion_gate_min_direction_consistency=motion_gate_min_direction_consistency,
    )
    save_tcn_dataset_manifest(train_manifest, train_manifest_path)
    save_tcn_dataset_manifest(test_manifest, test_manifest_path)
    config = CausalTcnTrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        levels=levels,
        kernel_size=kernel_size,
        dropout=dropout,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    try:
        decoder_config = (
            EventDecoderConfig(
                activation_threshold=confidence_threshold,
                release_threshold=release_threshold,
                min_peak_confidence=min_peak_confidence,
                min_event_separation_seconds=cooldown_seconds,
                cooldown_seconds=cooldown_seconds,
            )
            if event_decoder
            else None
        )
        train_result = train_causal_tcn(
            manifest_path=train_manifest_path,
            out_path=model_out,
            config=config,
        )
        evaluations = evaluate_tcn_manifest(
            manifest_path=test_manifest_path,
            model_path=model_out,
            confidence_threshold=confidence_threshold,
            cooldown_seconds=cooldown_seconds,
            match_tolerance_seconds=match_tolerance_seconds,
            event_decoder_config=decoder_config,
        )
    except (MissingMlDependencyError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    summary = holdout_totals(evaluations)
    payload = {
        "recognizer": "tcn_event_decoder" if event_decoder else "tcn",
        "model_path": str(model_out),
        "split": {
            "train_manifest": str(train_manifest_path),
            "test_manifest": str(test_manifest_path),
            "train_features": [str(path) for path in train_features],
            "test_features": [str(path) for path in test_features],
        },
        "training": train_result.to_dict(),
        "event_decoder": decoder_config.to_dict() if decoder_config is not None else None,
        "summary": summary,
        "evaluations": [evaluation.to_dict() for evaluation in evaluations],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    latency = summary["mean_latency_seconds"]
    formatted_latency = round(latency, 4) if isinstance(latency, float) else "unknown"
    typer.echo(
        f"recognizer={'tcn_event_decoder' if event_decoder else 'tcn'} "
        f"train={len(train_features)} test={len(test_features)} "
        f"intended={summary['intended_events']} matched={summary['matched_events']} "
        f"missed={summary['missed_events']} candidates={summary['candidate_count']} "
        f"false_activations={summary['false_activations']} "
        f"repeated_fires={summary['repeated_fires']} mean_latency={formatted_latency}"
    )
    typer.echo(f"wrote holdout={out}")


def gesture_diagnose_features(
    features_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Feature CSV directory."),
    ],
    labels_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True, help="Label directory."),
    ],
    out: Annotated[Path | None, typer.Option(help="Optional JSON diagnostics path.")] = None,
    train_per_gesture: Annotated[int, typer.Option(help="Training files per gesture.")] = 6,
    test_per_gesture: Annotated[int, typer.Option(help="Held-out test files per gesture.")] = 2,
    train_negatives: Annotated[int, typer.Option(help="Training negative/background files.")] = 6,
    test_negatives: Annotated[int, typer.Option(help="Held-out negative/background files.")] = 2,
) -> None:
    """Diagnose feature separation and label/window timing on a holdout split."""
    try:
        report = build_feature_diagnostics_report(
            features_dir=features_dir,
            labels_dir=labels_dir,
            train_per_gesture=train_per_gesture,
            test_per_gesture=test_per_gesture,
            train_negatives=train_negatives,
            test_negatives=test_negatives,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if out is not None:
        save_feature_diagnostics_report(report, out)
    typer.echo(_format_feature_diagnostics_summary(report))
    if out is not None:
        typer.echo(f"wrote diagnostics={out}")




def _split_tcn_feature_holdout(
    *,
    features_dir: Path,
    labels_dir: Path,
    train_per_gesture: int,
    test_per_gesture: int,
    train_negatives: int,
    test_negatives: int,
) -> tuple[list[Path], list[Path]]:
    positives: dict[str, list[Path]] = {}
    negatives: list[Path] = []
    for feature_path in sorted(features_dir.glob("*.csv")):
        label_path = labels_dir / f"{feature_path.stem}.labels.json"
        if not label_path.exists():
            raise typer.BadParameter(
                f"missing label file for features={feature_path}: {label_path}"
            )
        labels = load_label_file(label_path)
        events = [event for event in labels.event_labels if event.label_type == "gesture"]
        if not events:
            negatives.append(feature_path)
            continue
        positives.setdefault(events[0].gesture, []).append(feature_path)

    train: list[Path] = []
    test: list[Path] = []
    for gesture in sorted(positives):
        items = positives[gesture]
        if len(items) < train_per_gesture + test_per_gesture:
            raise typer.BadParameter(
                f"not enough features for gesture={gesture}: "
                f"need {train_per_gesture + test_per_gesture}, found {len(items)}"
            )
        train.extend(items[:train_per_gesture])
        test.extend(items[train_per_gesture : train_per_gesture + test_per_gesture])

    if train_negatives + test_negatives > 0:
        if len(negatives) < train_negatives + test_negatives:
            raise typer.BadParameter(
                "not enough negative features: "
                f"need {train_negatives + test_negatives}, found {len(negatives)}"
            )
        train.extend(negatives[:train_negatives])
        test.extend(negatives[train_negatives : train_negatives + test_negatives])

    if not train:
        raise typer.BadParameter("TCN holdout requires at least one training feature file")
    if not test:
        raise typer.BadParameter("TCN holdout requires at least one test feature file")
    return train, test


def _format_feature_diagnostics_summary(report: FeatureDiagnosticsReport) -> str:
    aggregates = report.aggregates
    lines = [f"feature_diagnostics files={len(report.files)}"]
    for key, aggregate in aggregates.items():
        metrics = aggregate["metrics"]
        palm_dx = metrics["palm_dx"]["mean"]
        speed = metrics["max_palm_speed"]["mean"]
        scale_dx = metrics["palm_dx_per_hand_scale"]["mean"]
        consistency = metrics["direction_consistency"]["mean"]
        palm_dx_text = f"{palm_dx:.3f}" if isinstance(palm_dx, float) else "unknown"
        speed_text = f"{speed:.3f}" if isinstance(speed, float) else "unknown"
        scale_dx_text = f"{scale_dx:.3f}" if isinstance(scale_dx, float) else "unknown"
        consistency_text = (
            f"{consistency:.3f}" if isinstance(consistency, float) else "unknown"
        )
        lines.append(
            f"{key} count={aggregate['count']} mean_palm_dx={palm_dx_text} "
            f"mean_max_speed={speed_text} mean_dx_per_scale={scale_dx_text} "
            f"mean_direction_consistency={consistency_text}"
        )
    return "\n".join(lines)




def register_tcn_commands(gesture_app: typer.Typer) -> None:
    """Register offline TCN commands on the main gesture command group."""
    gesture_app.command("build-tcn-dataset")(gesture_build_tcn_dataset)
    gesture_app.command("train-tcn")(gesture_train_tcn)
    gesture_app.command("train-tcn-v2")(gesture_train_tcn_v2)
    gesture_app.command("evaluate-tcn")(gesture_evaluate_tcn)
    gesture_app.command("evaluate-tcn-v2")(gesture_evaluate_tcn_v2)
    gesture_app.command("diagnose-tcn-events")(gesture_diagnose_tcn_events)
    gesture_app.command("diagnose-tcn-v2-events")(gesture_diagnose_tcn_v2_events)
    gesture_app.command("holdout-tcn")(gesture_holdout_tcn)
    gesture_app.command("diagnose-features")(gesture_diagnose_features)
