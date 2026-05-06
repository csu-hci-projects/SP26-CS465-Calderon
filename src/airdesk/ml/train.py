"""Optional causal TCN training over AirDesk feature-window manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from airdesk.ml.dataset import (
    TcnDatasetManifest,
    feature_window_matrix,
    load_tcn_dataset_manifest,
)


class MissingMlDependencyError(RuntimeError):
    """Raised when optional ML dependencies are not installed."""


@dataclass(frozen=True)
class CausalTcnTrainingConfig:
    """Small, deterministic training configuration for the first TCN scaffold."""

    epochs: int = 25
    learning_rate: float = 0.001
    batch_size: int = 16
    hidden_channels: int = 24
    levels: int = 2
    kernel_size: int = 3
    dropout: float = 0.0
    validation_fraction: float = 0.2
    seed: int = 7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TcnTrainingArrays:
    """Dependency-free numeric arrays prepared from a TCN manifest."""

    samples: tuple[tuple[tuple[float, ...], ...], ...]
    labels: tuple[int, ...]
    lengths: tuple[int, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class CausalTcnTrainingResult:
    """Summary of one offline TCN training run."""

    model_path: str
    samples: int
    train_samples: int
    validation_samples: int
    epochs: int
    final_train_loss: float
    train_accuracy: float
    validation_accuracy: float | None
    targets: tuple[str, ...]
    feature_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_tcn_training_arrays(manifest: TcnDatasetManifest) -> TcnTrainingArrays:
    """Load, normalize, and pad-free window arrays from a manifest."""
    if not manifest.windows:
        raise ValueError("TCN training requires at least one manifest window")
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
    return TcnTrainingArrays(
        samples=normalized,
        labels=tuple(window.target_index for window in manifest.windows),
        lengths=tuple(len(sample) for sample in normalized),
        feature_mean=feature_mean,
        feature_std=feature_std,
        sample_ids=tuple(window.sample_id for window in manifest.windows),
    )


def train_causal_tcn(
    *,
    manifest_path: Path,
    out_path: Path,
    config: CausalTcnTrainingConfig | None = None,
) -> CausalTcnTrainingResult:
    """Train a small offline causal TCN classifier and save a Torch checkpoint."""
    torch, nn, functional = _require_torch()
    config = config or CausalTcnTrainingConfig()
    _validate_training_config(config)
    manifest = load_tcn_dataset_manifest(manifest_path)
    arrays = prepare_tcn_training_arrays(manifest)
    sample_tensor, length_tensor, label_tensor = _tensors_from_arrays(torch, arrays)
    train_indices, validation_indices = _split_indices(
        torch,
        sample_count=len(arrays.samples),
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    model = _make_causal_tcn_model(
        torch=torch,
        nn=nn,
        functional=functional,
        input_features=len(manifest.feature_columns),
        targets=len(manifest.targets),
        hidden_channels=config.hidden_channels,
        levels=config.levels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    final_train_loss = 0.0
    for _epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch_indices in _batches(train_indices, config.batch_size):
            optimizer.zero_grad()
            logits = model(sample_tensor[batch_indices], length_tensor[batch_indices])
            loss = loss_fn(logits, label_tensor[batch_indices])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_indices)
            seen += len(batch_indices)
        final_train_loss = epoch_loss / seen if seen else 0.0

    train_accuracy = _accuracy(
        model,
        sample_tensor,
        length_tensor,
        label_tensor,
        train_indices,
        torch,
    )
    validation_accuracy = (
        _accuracy(model, sample_tensor, length_tensor, label_tensor, validation_indices, torch)
        if len(validation_indices) > 0
        else None
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "metadata": {
                "schema_version": 1,
                "model_type": "causal_tcn",
                "manifest_path": str(manifest_path),
                "targets": list(manifest.targets),
                "feature_columns": list(manifest.feature_columns),
                "feature_mean": list(arrays.feature_mean),
                "feature_std": list(arrays.feature_std),
                "window_seconds": manifest.window_seconds,
                "stride_seconds": manifest.stride_seconds,
                "training_config": config.to_dict(),
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            },
            "model_config": {
                "input_features": len(manifest.feature_columns),
                "targets": len(manifest.targets),
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
        targets=manifest.targets,
        feature_columns=manifest.feature_columns,
    )


def _normalization_stats(
    samples: tuple[tuple[tuple[float, ...], ...], ...],
    feature_count: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values_by_feature: list[list[float]] = [[] for _ in range(feature_count)]
    for sample in samples:
        for row in sample:
            for index, value in enumerate(row):
                values_by_feature[index].append(value)
    means = tuple(sum(values) / len(values) if values else 0.0 for values in values_by_feature)
    stds = []
    for index, values in enumerate(values_by_feature):
        if not values:
            stds.append(1.0)
            continue
        variance = sum((value - means[index]) ** 2 for value in values) / len(values)
        stds.append(variance**0.5 if variance > 1e-8 else 1.0)
    return means, tuple(stds)


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise MissingMlDependencyError(
            "PyTorch is required for TCN training. Install with `uv sync --dev --extra ml`."
        ) from exc
    return torch, nn, functional


def _validate_training_config(config: CausalTcnTrainingConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.hidden_channels <= 0:
        raise ValueError("hidden_channels must be positive")
    if config.levels <= 0:
        raise ValueError("levels must be positive")
    if config.kernel_size <= 1:
        raise ValueError("kernel_size must be greater than 1")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not 0 <= config.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")


def _tensors_from_arrays(
    torch: Any,
    arrays: TcnTrainingArrays,
) -> tuple[Any, Any, Any]:
    max_length = max(arrays.lengths)
    feature_count = len(arrays.samples[0][0])
    padded = torch.zeros((len(arrays.samples), max_length, feature_count), dtype=torch.float32)
    for sample_index, sample in enumerate(arrays.samples):
        for row_index, row in enumerate(sample):
            padded[sample_index, row_index] = torch.tensor(row, dtype=torch.float32)
    lengths = torch.tensor(arrays.lengths, dtype=torch.long)
    labels = torch.tensor(arrays.labels, dtype=torch.long)
    return padded, lengths, labels


def _split_indices(
    torch: Any,
    *,
    sample_count: int,
    validation_fraction: float,
    seed: int,
) -> tuple[Any, Any]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(sample_count, generator=generator)
    validation_count = int(sample_count * validation_fraction)
    if sample_count > 1 and validation_fraction > 0 and validation_count == 0:
        validation_count = 1
    train_count = sample_count - validation_count
    if train_count <= 0:
        return indices, torch.tensor([], dtype=torch.long)
    return indices[:train_count], indices[train_count:]


def _batches(indices: Any, batch_size: int) -> list[Any]:
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def _accuracy(
    model: Any,
    samples: Any,
    lengths: Any,
    labels: Any,
    indices: Any,
    torch: Any,
) -> float:
    if len(indices) == 0:
        return 0.0
    model.eval()
    with torch.no_grad():
        logits = model(samples[indices], lengths[indices])
        predictions = logits.argmax(dim=1)
        return float((predictions == labels[indices]).float().mean().item())


def _make_causal_tcn_model(
    *,
    torch: Any,
    nn: Any,
    functional: Any,
    input_features: int,
    targets: int,
    hidden_channels: int,
    levels: int,
    kernel_size: int,
    dropout: float,
) -> Any:
    class CausalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            self.left_padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, value: Any) -> Any:
            value = functional.pad(value, (self.left_padding, 0))
            return self.dropout(torch.relu(self.conv(value)))

    class CausalTcnClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            in_channels = input_features
            for level in range(levels):
                layers.append(CausalBlock(in_channels, hidden_channels, dilation=2**level))
                in_channels = hidden_channels
            self.layers = nn.ModuleList(layers)
            self.classifier = nn.Linear(hidden_channels, targets)

        def forward(self, samples: Any, lengths: Any) -> Any:
            value = samples.transpose(1, 2)
            for layer in self.layers:
                value = layer(value)
            sequence = value.transpose(1, 2)
            batch_indices = torch.arange(sequence.shape[0], device=sequence.device)
            final_indices = torch.clamp(lengths - 1, min=0)
            return self.classifier(sequence[batch_indices, final_indices])

    return CausalTcnClassifier()
