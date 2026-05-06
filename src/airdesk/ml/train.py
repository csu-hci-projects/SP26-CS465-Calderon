"""Optional causal TCN training over AirDesk feature-window manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from airdesk.ml.dataset import (
    TcnDatasetManifest,
    TcnWindowSample,
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


@dataclass(frozen=True)
class CausalTcnPrediction:
    """One non-background TCN prediction over a manifest window."""

    sample_id: str
    feature_path: str
    label_path: str | None
    start_time: float
    end_time: float
    target: str
    target_index: int
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalTcnLivePrediction:
    """One live/replay TCN prediction over an in-memory feature window."""

    start_time: float
    end_time: float
    target: str
    target_index: int
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CausalTcnLivePredictor:
    """Loaded TCN checkpoint for rolling live/replay classification."""

    model: Any
    torch: Any
    targets: tuple[str, ...]
    feature_columns: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    window_seconds: float
    stride_seconds: float

    @classmethod
    def load(cls, model_path: Path) -> CausalTcnLivePredictor:
        """Load a TCN checkpoint for in-memory rolling-window predictions."""
        torch, nn, functional = _require_torch()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        model_config = checkpoint["model_config"]
        model = _make_causal_tcn_model(
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
        return cls(
            model=model,
            torch=torch,
            targets=tuple(str(target) for target in metadata["targets"]),
            feature_columns=tuple(str(column) for column in metadata["feature_columns"]),
            feature_mean=tuple(float(value) for value in metadata["feature_mean"]),
            feature_std=tuple(float(value) for value in metadata["feature_std"]),
            window_seconds=float(metadata["window_seconds"]),
            stride_seconds=float(metadata["stride_seconds"]),
        )

    def predict_rows(self, rows: list[Any]) -> CausalTcnLivePrediction:
        """Classify one in-memory feature window."""
        if not rows:
            raise ValueError("TCN live prediction requires at least one feature row")
        matrix = [
            [_numeric_row_value(row, column) for column in self.feature_columns]
            for row in rows
        ]
        normalized = [
            [
                (value - self.feature_mean[index]) / self.feature_std[index]
                for index, value in enumerate(row)
            ]
            for row in matrix
        ]
        with self.torch.no_grad():
            sample = self.torch.tensor([normalized], dtype=self.torch.float32)
            lengths = self.torch.tensor([len(rows)], dtype=self.torch.long)
            logits = self.model(sample, lengths)
            probabilities_tensor = self.torch.softmax(logits, dim=1)[0]
            target_index = int(probabilities_tensor.argmax().item())
            confidence = float(probabilities_tensor[target_index].item())
        return CausalTcnLivePrediction(
            start_time=float(rows[0].timestamp),
            end_time=float(rows[-1].timestamp),
            target=self.targets[target_index],
            target_index=target_index,
            confidence=confidence,
            probabilities={
                self.targets[index]: float(value.item())
                for index, value in enumerate(probabilities_tensor)
            },
        )


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


def predict_causal_tcn_manifest(
    *,
    model_path: Path,
    manifest_path: Path,
    confidence_threshold: float = 0.5,
    cooldown_seconds: float = 0.5,
    include_background: bool = False,
) -> list[CausalTcnPrediction]:
    """Run a trained TCN checkpoint over a manifest and return window predictions."""
    torch, nn, functional = _require_torch()
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if cooldown_seconds < 0:
        raise ValueError("cooldown_seconds must be non-negative")
    manifest = load_tcn_dataset_manifest(manifest_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    model_config = checkpoint["model_config"]
    targets = tuple(str(target) for target in metadata["targets"])
    feature_columns = tuple(str(column) for column in metadata["feature_columns"])
    if tuple(manifest.feature_columns) != feature_columns:
        raise ValueError("Manifest feature columns do not match TCN checkpoint metadata")
    if tuple(manifest.targets) != targets:
        raise ValueError("Manifest targets do not match TCN checkpoint metadata")

    model = _make_causal_tcn_model(
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

    predictions: list[CausalTcnPrediction] = []
    with torch.no_grad():
        for window in manifest.windows:
            sample = _normalized_window_tensor(
                torch,
                window,
                feature_columns=feature_columns,
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            lengths = torch.tensor([sample.shape[1]], dtype=torch.long)
            logits = model(sample, lengths)
            probabilities_tensor = torch.softmax(logits, dim=1)[0]
            target_index = int(probabilities_tensor.argmax().item())
            target = targets[target_index]
            confidence = float(probabilities_tensor[target_index].item())
            if target == "background" and not include_background:
                continue
            if confidence < confidence_threshold:
                continue
            predictions.append(
                CausalTcnPrediction(
                    sample_id=window.sample_id,
                    feature_path=window.feature_path,
                    label_path=window.label_path,
                    start_time=window.start_time,
                    end_time=window.end_time,
                    target=target,
                    target_index=target_index,
                    confidence=confidence,
                    probabilities={
                        targets[index]: float(value.item())
                        for index, value in enumerate(probabilities_tensor)
                    },
                )
            )
    if include_background:
        return sorted(predictions, key=lambda item: (item.feature_path, item.end_time))
    return _suppress_predictions(predictions, cooldown_seconds=cooldown_seconds)


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


def _numeric_row_value(row: Any, column: str) -> float:
    value = getattr(row, column)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"TCN feature column must be numeric: {column}")


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


def _normalized_window_tensor(
    torch: Any,
    window: TcnWindowSample,
    *,
    feature_columns: tuple[str, ...],
    feature_mean: tuple[float, ...],
    feature_std: tuple[float, ...],
) -> Any:
    matrix = feature_window_matrix(window, feature_columns=feature_columns)
    normalized = [
        [
            (value - feature_mean[index]) / feature_std[index]
            for index, value in enumerate(row)
        ]
        for row in matrix
    ]
    return torch.tensor([normalized], dtype=torch.float32)


def _suppress_predictions(
    predictions: list[CausalTcnPrediction],
    *,
    cooldown_seconds: float,
) -> list[CausalTcnPrediction]:
    selected: list[CausalTcnPrediction] = []
    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        overlaps = False
        for existing in selected:
            if prediction.feature_path != existing.feature_path:
                continue
            if prediction.target != existing.target:
                continue
            if (
                prediction.start_time <= existing.end_time + cooldown_seconds
                and existing.start_time <= prediction.end_time
            ):
                overlaps = True
                break
        if not overlaps:
            selected.append(prediction)
    return sorted(selected, key=lambda item: (item.feature_path, item.end_time))


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
