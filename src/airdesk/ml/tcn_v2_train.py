"""TCN v2 sequence-evidence training and prediction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from airdesk.ml.dataset import (
    TcnDatasetManifest,
    feature_window_frame_targets,
    feature_window_matrix,
    load_tcn_dataset_manifest,
)
from airdesk.ml.train import (
    CausalTcnTrainingConfig,
    CausalTcnTrainingResult,
    _batches,
    _make_causal_tcn_sequence_model,
    _normalization_stats,
    _normalized_window_tensor,
    _require_torch,
    _split_indices,
    _validate_training_config,
    _window_rows,
)


@dataclass(frozen=True)
class TcnEvidenceTrainingArrays:
    """Normalized TCN v2 samples with per-frame multi-label evidence targets."""

    samples: tuple[tuple[tuple[float, ...], ...], ...]
    frame_targets: tuple[tuple[tuple[float, ...], ...], ...]
    lengths: tuple[int, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class CausalTcnEvidencePrediction:
    """One decoder-facing TCN v2 evidence frame emitted from a causal context."""

    sample_id: str
    feature_path: str
    label_path: str | None
    hand_id: str
    timestamp: float
    window_start: float
    window_end: float
    evidence: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_tcn_v2_training_arrays(manifest: TcnDatasetManifest) -> TcnEvidenceTrainingArrays:
    """Load, normalize, and align TCN v2 frame evidence targets from a manifest."""
    if manifest.target_mode != "v2-evidence":
        raise ValueError("TCN v2 training requires target_mode='v2-evidence'")
    if not manifest.evidence_targets:
        raise ValueError("TCN v2 manifest is missing evidence_targets")
    if not manifest.windows:
        raise ValueError("TCN v2 training requires at least one manifest window")
    raw_samples = tuple(
        tuple(
            tuple(value for value in row)
            for row in feature_window_matrix(
                window,
                feature_columns=manifest.feature_columns,
            )
        )
        for window in manifest.windows
    )
    raw_targets = tuple(
        tuple(
            tuple(value for value in row)
            for row in feature_window_frame_targets(
                window,
                evidence_targets=manifest.evidence_targets,
            )
        )
        for window in manifest.windows
    )
    for sample, targets in zip(raw_samples, raw_targets, strict=True):
        if len(sample) != len(targets):
            raise ValueError("TCN v2 feature rows and frame targets are misaligned")
    feature_mean, feature_std = _normalization_stats(raw_samples, len(manifest.feature_columns))
    normalized = tuple(
        tuple(
            tuple(
                (value - feature_mean[index]) / feature_std[index]
                for index, value in enumerate(row)
            )
            for row in sample
        )
        for sample in raw_samples
    )
    return TcnEvidenceTrainingArrays(
        samples=normalized,
        frame_targets=raw_targets,
        lengths=tuple(len(sample) for sample in normalized),
        feature_mean=feature_mean,
        feature_std=feature_std,
        sample_ids=tuple(window.sample_id for window in manifest.windows),
    )


def train_causal_tcn_v2(
    *,
    manifest_path: Path,
    out_path: Path,
    config: CausalTcnTrainingConfig | None = None,
) -> CausalTcnTrainingResult:
    """Train a causal TCN v2 sequence-evidence model and save a Torch checkpoint."""
    torch, nn, functional = _require_torch()
    config = config or CausalTcnTrainingConfig()
    _validate_training_config(config)
    manifest = load_tcn_dataset_manifest(manifest_path)
    arrays = prepare_tcn_v2_training_arrays(manifest)
    sample_tensor, _length_tensor, target_tensor, mask_tensor = _evidence_tensors_from_arrays(
        torch,
        arrays,
    )
    train_indices, validation_indices = _split_indices(
        torch,
        sample_count=len(arrays.samples),
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    model = _make_causal_tcn_sequence_model(
        torch=torch,
        nn=nn,
        functional=functional,
        input_features=len(manifest.feature_columns),
        targets=len(manifest.evidence_targets),
        hidden_channels=config.hidden_channels,
        levels=config.levels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    final_train_loss = 0.0
    for _epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch_indices in _batches(train_indices, config.batch_size):
            optimizer.zero_grad()
            logits = model(sample_tensor[batch_indices])
            loss = _masked_bce_loss(
                loss_fn(logits, target_tensor[batch_indices]),
                mask_tensor[batch_indices],
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_indices)
            seen += len(batch_indices)
        final_train_loss = epoch_loss / seen if seen else 0.0

    train_accuracy = _evidence_frame_accuracy(
        model,
        sample_tensor,
        target_tensor,
        mask_tensor,
        train_indices,
        torch,
    )
    validation_accuracy = (
        _evidence_frame_accuracy(
            model,
            sample_tensor,
            target_tensor,
            mask_tensor,
            validation_indices,
            torch,
        )
        if len(validation_indices) > 0
        else None
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "metadata": {
                "schema_version": 1,
                "model_type": "causal_tcn_v2_evidence",
                "manifest_path": str(manifest_path),
                "targets": list(manifest.targets),
                "evidence_targets": list(manifest.evidence_targets),
                "feature_columns": list(manifest.feature_columns),
                "feature_mean": list(arrays.feature_mean),
                "feature_std": list(arrays.feature_std),
                "window_seconds": manifest.window_seconds,
                "stride_seconds": manifest.stride_seconds,
                "training_config": config.to_dict(),
                "train_frame_accuracy": train_accuracy,
                "validation_frame_accuracy": validation_accuracy,
            },
            "model_config": {
                "input_features": len(manifest.feature_columns),
                "targets": len(manifest.evidence_targets),
                "hidden_channels": config.hidden_channels,
                "levels": config.levels,
                "kernel_size": config.kernel_size,
                "dropout": config.dropout,
            },
        },
        out_path,
    )
    return CausalTcnTrainingResult(
        model_path=str(out_path),
        samples=len(arrays.samples),
        train_samples=len(train_indices),
        validation_samples=len(validation_indices),
        epochs=config.epochs,
        final_train_loss=final_train_loss,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        targets=manifest.evidence_targets,
        feature_columns=manifest.feature_columns,
    )


def predict_causal_tcn_v2_manifest(
    *,
    model_path: Path,
    manifest_path: Path,
    emit_all_rows: bool = False,
) -> list[CausalTcnEvidencePrediction]:
    """Run a TCN v2 evidence checkpoint over a manifest."""
    torch, nn, functional = _require_torch()
    manifest = load_tcn_dataset_manifest(manifest_path)
    if manifest.target_mode != "v2-evidence":
        raise ValueError("TCN v2 prediction requires target_mode='v2-evidence'")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    if str(metadata.get("model_type")) != "causal_tcn_v2_evidence":
        raise ValueError("TCN v2 prediction requires a causal_tcn_v2_evidence checkpoint")
    model_config = checkpoint["model_config"]
    evidence_targets = tuple(str(target) for target in metadata["evidence_targets"])
    feature_columns = tuple(str(column) for column in metadata["feature_columns"])
    if tuple(manifest.feature_columns) != feature_columns:
        raise ValueError("Manifest feature columns do not match TCN v2 checkpoint metadata")
    if tuple(manifest.evidence_targets) != evidence_targets:
        raise ValueError("Manifest evidence targets do not match TCN v2 checkpoint metadata")
    model = _make_causal_tcn_sequence_model(
        torch=torch,
        nn=nn,
        functional=functional,
        input_features=int(model_config["input_features"]),
        targets=int(model_config["targets"]),
        hidden_channels=int(model_config["hidden_channels"]),
        levels=int(model_config["levels"]),
        kernel_size=int(model_config["kernel_size"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    feature_mean = tuple(float(value) for value in metadata["feature_mean"])
    feature_std = tuple(float(value) for value in metadata["feature_std"])
    predictions: list[CausalTcnEvidencePrediction] = []
    with torch.no_grad():
        for window in manifest.windows:
            sample = _normalized_window_tensor(
                torch,
                window,
                feature_columns=feature_columns,
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            probabilities = torch.sigmoid(model(sample))[0]
            rows = _window_rows(window)
            row_indices = range(len(rows)) if emit_all_rows else (len(rows) - 1,)
            for row_index in row_indices:
                row = rows[row_index]
                predictions.append(
                    CausalTcnEvidencePrediction(
                        sample_id=window.sample_id,
                        feature_path=window.feature_path,
                        label_path=window.label_path,
                        hand_id=window.hand_id,
                        timestamp=float(row.timestamp),
                        window_start=window.start_time,
                        window_end=window.end_time,
                        evidence={
                            evidence_targets[index]: float(value.item())
                            for index, value in enumerate(probabilities[row_index])
                        },
                    )
                )
    return sorted(
        predictions,
        key=lambda item: (item.feature_path, item.timestamp, item.hand_id, item.sample_id),
    )


def _evidence_tensors_from_arrays(
    torch: Any,
    arrays: TcnEvidenceTrainingArrays,
) -> tuple[Any, Any, Any, Any]:
    max_length = max(arrays.lengths)
    feature_count = len(arrays.samples[0][0])
    target_count = len(arrays.frame_targets[0][0])
    samples = torch.zeros((len(arrays.samples), max_length, feature_count), dtype=torch.float32)
    targets = torch.zeros((len(arrays.samples), max_length, target_count), dtype=torch.float32)
    mask = torch.zeros((len(arrays.samples), max_length), dtype=torch.float32)
    for sample_index, sample in enumerate(arrays.samples):
        for row_index, row in enumerate(sample):
            samples[sample_index, row_index] = torch.tensor(row, dtype=torch.float32)
            targets[sample_index, row_index] = torch.tensor(
                arrays.frame_targets[sample_index][row_index],
                dtype=torch.float32,
            )
            mask[sample_index, row_index] = 1.0
    lengths = torch.tensor(arrays.lengths, dtype=torch.long)
    return samples, lengths, targets, mask


def _masked_bce_loss(loss: Any, mask: Any) -> Any:
    masked = loss * mask.unsqueeze(-1)
    denominator = mask.sum() * loss.shape[-1]
    return masked.sum() / denominator.clamp(min=1.0)


def _evidence_frame_accuracy(
    model: Any,
    samples: Any,
    targets: Any,
    mask: Any,
    indices: Any,
    torch: Any,
) -> float:
    if len(indices) == 0:
        return 0.0
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(samples[indices]))
        predicted = probabilities >= 0.5
        expected = targets[indices] >= 0.5
        valid = mask[indices].unsqueeze(-1).bool()
        correct = (predicted == expected) & valid
        denominator = valid.sum() * targets.shape[-1]
        if int(denominator.item()) == 0:
            return 0.0
        return float(correct.sum().float().div(denominator).item())
