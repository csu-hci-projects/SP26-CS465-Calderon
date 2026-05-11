"""TCN v2 sequence-evidence training and prediction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from airdesk.ml.dataset import (
    NO_HAND_STREAM_ID,
    TcnDatasetManifest,
    feature_window_frame_targets,
    load_feature_rows_csv,
    load_tcn_dataset_manifest,
)
from airdesk.ml.train import (
    CausalTcnTrainingConfig,
    CausalTcnTrainingResult,
    _batches,
    _normalization_stats,
    _numeric_row_value,
    _require_torch,
    _split_indices,
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


@dataclass(frozen=True)
class CausalTcnV2LivePrediction:
    """One live TCN v2 evidence prediction over an in-memory feature stream."""

    hand_id: str | None
    start_time: float
    end_time: float
    evidence: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CausalTcnV2LivePredictor:
    """Loaded TCN v2 checkpoint for rolling live evidence prediction."""

    model: Any
    torch: Any
    device: Any
    evidence_targets: tuple[str, ...]
    feature_columns: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    window_seconds: float
    stride_seconds: float
    schema_version: int
    calibration_thresholds: dict[str, float]

    @classmethod
    def load(cls, model_path: Path, *, device: str = "auto") -> CausalTcnV2LivePredictor:
        """Load a TCN v2 evidence checkpoint for live rolling-window predictions."""
        torch, nn, functional = _require_torch()
        resolved_device = _resolve_tcn_device(torch, device)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        if str(metadata.get("model_type")) != "causal_tcn_v2_evidence":
            raise ValueError("TCN v2 live prediction requires a causal_tcn_v2_evidence checkpoint")
        model = _make_tcn_v2_sequence_model_for_checkpoint(
            torch=torch,
            nn=nn,
            functional=functional,
            model_config=checkpoint["model_config"],
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(resolved_device)
        model.eval()
        calibration = metadata.get("calibration", {})
        thresholds = calibration.get("evidence_thresholds", {})
        return cls(
            model=model,
            torch=torch,
            device=resolved_device,
            evidence_targets=tuple(str(target) for target in metadata["evidence_targets"]),
            feature_columns=tuple(str(column) for column in metadata["feature_columns"]),
            feature_mean=tuple(float(value) for value in metadata["feature_mean"]),
            feature_std=tuple(float(value) for value in metadata["feature_std"]),
            window_seconds=float(metadata.get("window_seconds", 0.8)),
            stride_seconds=float(metadata.get("stride_seconds", 0.2)),
            schema_version=int(metadata.get("schema_version", 1)),
            calibration_thresholds={
                str(target): float(threshold) for target, threshold in thresholds.items()
            },
        )

    def predict_rows(self, rows: list[Any]) -> CausalTcnV2LivePrediction:
        """Predict decoder-facing evidence for the latest row in one live hand stream."""
        if not rows:
            raise ValueError("TCN v2 live prediction requires at least one feature row")
        matrix = [
            [_numeric_row_value(row, column) for column in self.feature_columns] for row in rows
        ]
        normalized = [
            [
                (value - self.feature_mean[index]) / self.feature_std[index]
                for index, value in enumerate(row)
            ]
            for row in matrix
        ]
        with self.torch.no_grad():
            sample = self.torch.tensor(
                [normalized],
                dtype=self.torch.float32,
                device=self.device,
            )
            probabilities = self.torch.sigmoid(self.model(sample))[0, -1].detach().cpu()
        latest_row = rows[-1]
        return CausalTcnV2LivePrediction(
            hand_id=getattr(latest_row, "hand_id", None) or None,
            start_time=float(rows[0].timestamp),
            end_time=float(latest_row.timestamp),
            evidence={
                target: float(probabilities[index].item())
                for index, target in enumerate(self.evidence_targets)
            },
        )


@dataclass(frozen=True)
class CausalTcnV2TrainingConfig:
    """Training configuration for the boundary-aware TCN v2 evidence model."""

    epochs: int = 25
    learning_rate: float = 0.001
    batch_size: int = 16
    hidden_channels: int = 32
    levels: int = 3
    kernel_size: int = 3
    dropout: float = 0.10
    validation_fraction: float = 0.2
    seed: int = 7
    positive_weight_cap: float = 30.0
    boundary_positive_weight_multiplier: float = 2.0
    focal_gamma: float = 1.0
    device: str = "auto"

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
    row_cache: dict[str, list[Any]] = {}
    raw_samples = tuple(
        tuple(
            tuple(value for value in row)
            for row in _cached_feature_window_matrix(
                window,
                feature_columns=manifest.feature_columns,
                row_cache=row_cache,
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
    config: CausalTcnTrainingConfig | CausalTcnV2TrainingConfig | None = None,
) -> CausalTcnTrainingResult:
    """Train a causal TCN v2 sequence-evidence model and save a Torch checkpoint."""
    torch, nn, functional = _require_torch()
    config = _coerce_tcn_v2_config(config)
    _validate_tcn_v2_training_config(config)
    device = _resolve_tcn_device(torch, config.device)
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
    model = _make_causal_tcn_v2_sequence_model(
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
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    positive_weights = _evidence_positive_weights(
        torch,
        target_tensor,
        mask_tensor,
        manifest.evidence_targets,
        config,
    )
    positive_weights_for_loss = positive_weights.to(device)

    final_train_loss = 0.0
    for _epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch_indices in _batches(train_indices, config.batch_size):
            batch_samples = sample_tensor[batch_indices].to(device)
            batch_targets = target_tensor[batch_indices].to(device)
            batch_mask = mask_tensor[batch_indices].to(device)
            optimizer.zero_grad()
            logits = model(batch_samples)
            loss = _masked_weighted_bce_loss(
                functional,
                logits,
                batch_targets,
                batch_mask,
                positive_weights_for_loss,
                focal_gamma=config.focal_gamma,
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
        device=device,
        batch_size=config.batch_size,
    )
    validation_accuracy = (
        _evidence_frame_accuracy(
            model,
            sample_tensor,
            target_tensor,
            mask_tensor,
            validation_indices,
            torch,
            device=device,
            batch_size=config.batch_size,
        )
        if len(validation_indices) > 0
        else None
    )
    calibration_indices = validation_indices if len(validation_indices) > 0 else train_indices
    calibration_split = "validation" if len(validation_indices) > 0 else "train"
    evidence_thresholds = _calibrate_evidence_thresholds(
        model,
        sample_tensor,
        target_tensor,
        mask_tensor,
        calibration_indices,
        manifest.evidence_targets,
        torch,
        device=device,
        batch_size=config.batch_size,
    )
    train_head_metrics = _evidence_head_metrics(
        model,
        sample_tensor,
        target_tensor,
        mask_tensor,
        train_indices,
        manifest.evidence_targets,
        evidence_thresholds,
        torch,
        device=device,
        batch_size=config.batch_size,
    )
    validation_head_metrics = (
        _evidence_head_metrics(
            model,
            sample_tensor,
            target_tensor,
            mask_tensor,
            validation_indices,
            manifest.evidence_targets,
            evidence_thresholds,
            torch,
            device=device,
            batch_size=config.batch_size,
        )
        if len(validation_indices) > 0
        else None
    )
    receptive_field_frames = tcn_v2_receptive_field_frames(
        levels=config.levels,
        kernel_size=config.kernel_size,
    )
    estimated_frame_seconds = _estimated_frame_seconds(manifest)
    estimated_receptive_field_seconds = (
        (receptive_field_frames - 1) * estimated_frame_seconds
        if estimated_frame_seconds is not None
        else None
    )
    positive_weight_by_target = _tensor_by_target(
        positive_weights,
        manifest.evidence_targets,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "metadata": {
                "schema_version": 2,
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
                "training_device": str(device),
                "train_frame_accuracy": train_accuracy,
                "validation_frame_accuracy": validation_accuracy,
                "train_head_metrics": train_head_metrics,
                "validation_head_metrics": validation_head_metrics,
                "loss": {
                    "type": "weighted_focal_bce",
                    "positive_weights": positive_weight_by_target,
                    "positive_weight_cap": config.positive_weight_cap,
                    "boundary_positive_weight_multiplier": (
                        config.boundary_positive_weight_multiplier
                    ),
                    "focal_gamma": config.focal_gamma,
                },
                "calibration": {
                    "split": calibration_split,
                    "evidence_thresholds": evidence_thresholds,
                },
                "receptive_field_frames": receptive_field_frames,
                "estimated_frame_seconds": estimated_frame_seconds,
                "estimated_receptive_field_seconds": estimated_receptive_field_seconds,
            },
            "model_config": {
                "architecture": "residual_dilated_causal_tcn_v2",
                "input_features": len(manifest.feature_columns),
                "targets": len(manifest.evidence_targets),
                "hidden_channels": config.hidden_channels,
                "levels": config.levels,
                "kernel_size": config.kernel_size,
                "dropout": config.dropout,
                "normalization": "layer_norm",
                "convs_per_level": 2,
                "residual_blocks": True,
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
    batch_size: int = 64,
    device: str = "auto",
) -> list[CausalTcnEvidencePrediction]:
    """Run a TCN v2 evidence checkpoint over a manifest."""
    torch, nn, functional = _require_torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resolved_device = _resolve_tcn_device(torch, device)
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
    model = _make_tcn_v2_sequence_model_for_checkpoint(
        torch=torch,
        nn=nn,
        functional=functional,
        model_config=model_config,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(resolved_device)
    model.eval()
    feature_mean = tuple(float(value) for value in metadata["feature_mean"])
    feature_std = tuple(float(value) for value in metadata["feature_std"])
    sample_tensor, window_rows = _prediction_tensors_from_windows(
        torch,
        manifest.windows,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    predictions: list[CausalTcnEvidencePrediction] = []
    with torch.no_grad():
        for start_index in range(0, len(manifest.windows), batch_size):
            end_index = start_index + batch_size
            batch_samples = sample_tensor[start_index:end_index].to(resolved_device)
            probabilities = torch.sigmoid(model(batch_samples)).detach().cpu()
            for batch_offset, window in enumerate(manifest.windows[start_index:end_index]):
                rows = window_rows[start_index + batch_offset]
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
                                for index, value in enumerate(
                                    probabilities[batch_offset, row_index]
                                )
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


def _prediction_tensors_from_windows(
    torch: Any,
    windows: tuple[Any, ...],
    *,
    feature_columns: tuple[str, ...],
    feature_mean: tuple[float, ...],
    feature_std: tuple[float, ...],
) -> tuple[Any, tuple[list[Any], ...]]:
    matrices: list[list[list[float]]] = []
    window_rows: list[list[Any]] = []
    row_cache: dict[str, list[Any]] = {}
    for window in windows:
        matrix = _cached_feature_window_matrix(
            window,
            feature_columns=feature_columns,
            row_cache=row_cache,
        )
        rows = _cached_window_rows(window, row_cache)
        if len(matrix) != len(rows):
            raise ValueError("TCN v2 prediction rows and feature matrix are misaligned")
        if not matrix:
            raise ValueError("TCN v2 prediction window has no feature rows")
        matrices.append(
            [
                [
                    (value - feature_mean[index]) / feature_std[index]
                    for index, value in enumerate(row)
                ]
                for row in matrix
            ]
        )
        window_rows.append(rows)
    if not matrices:
        raise ValueError("TCN v2 prediction requires at least one manifest window")
    max_length = max(len(matrix) for matrix in matrices)
    feature_count = len(feature_columns)
    samples = torch.zeros((len(matrices), max_length, feature_count), dtype=torch.float32)
    for sample_index, matrix in enumerate(matrices):
        for row_index, row in enumerate(matrix):
            samples[sample_index, row_index] = torch.tensor(row, dtype=torch.float32)
    return samples, tuple(window_rows)


def _cached_feature_window_matrix(
    window: Any,
    *,
    feature_columns: tuple[str, ...],
    row_cache: dict[str, list[Any]],
) -> list[list[float]]:
    rows = _cached_window_rows(window, row_cache)
    return [[_numeric_row_value(row, column) for column in feature_columns] for row in rows]


def _cached_window_rows(window: Any, row_cache: dict[str, list[Any]]) -> list[Any]:
    rows = row_cache.get(window.feature_path)
    if rows is None:
        rows = load_feature_rows_csv(Path(window.feature_path))
        row_cache[window.feature_path] = rows
    window_rows = rows[window.start_row : window.end_row]
    if window.hand_id == NO_HAND_STREAM_ID:
        return [row for row in window_rows if row.tracking_present == 0]
    if window.hand_id:
        return [row for row in window_rows if row.hand_id == window.hand_id]
    return window_rows


def _masked_bce_loss(loss: Any, mask: Any) -> Any:
    masked = loss * mask.unsqueeze(-1)
    denominator = mask.sum() * loss.shape[-1]
    return masked.sum() / denominator.clamp(min=1.0)


def _masked_weighted_bce_loss(
    functional: Any,
    logits: Any,
    targets: Any,
    mask: Any,
    positive_weights: Any,
    *,
    focal_gamma: float,
) -> Any:
    loss = functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weights,
        reduction="none",
    )
    if focal_gamma > 0:
        probabilities = logits.sigmoid()
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        loss = loss * (1.0 - p_t).pow(focal_gamma)
    return _masked_bce_loss(loss, mask)


def _evidence_positive_weights(
    torch: Any,
    targets: Any,
    mask: Any,
    evidence_targets: tuple[str, ...],
    config: CausalTcnV2TrainingConfig,
) -> Any:
    valid = mask.unsqueeze(-1)
    positives = (targets * valid).sum(dim=(0, 1))
    valid_frames = mask.sum()
    weights: list[float] = []
    for index, target in enumerate(evidence_targets):
        positive_count = float(positives[index].item())
        negative_count = max(0.0, float(valid_frames.item()) - positive_count)
        weight = 1.0 if positive_count <= 0 else max(1.0, negative_count / positive_count)
        if target in {"start", "end"}:
            weight *= config.boundary_positive_weight_multiplier
        weights.append(min(config.positive_weight_cap, weight))
    return torch.tensor(weights, dtype=torch.float32)


def _evidence_probability_batches(
    model: Any,
    samples: Any,
    indices: Any,
    torch: Any,
    *,
    device: Any,
    batch_size: int,
) -> Any:
    for batch_indices in _batches(indices, batch_size):
        batch_samples = samples[batch_indices].to(device)
        probabilities = torch.sigmoid(model(batch_samples)).detach().cpu()
        yield batch_indices, probabilities


def _evidence_frame_accuracy(
    model: Any,
    samples: Any,
    targets: Any,
    mask: Any,
    indices: Any,
    torch: Any,
    *,
    device: Any,
    batch_size: int,
) -> float:
    if len(indices) == 0:
        return 0.0
    model.eval()
    correct_count = 0
    denominator_count = 0
    with torch.no_grad():
        for batch_indices, probabilities in _evidence_probability_batches(
            model,
            samples,
            indices,
            torch,
            device=device,
            batch_size=batch_size,
        ):
            predicted = probabilities >= 0.5
            expected = targets[batch_indices] >= 0.5
            valid = mask[batch_indices].unsqueeze(-1).bool()
            correct_count += int(((predicted == expected) & valid).sum().item())
            denominator_count += int(valid.sum().item()) * targets.shape[-1]
    if denominator_count == 0:
        return 0.0
    return correct_count / denominator_count


def _calibrate_evidence_thresholds(
    model: Any,
    samples: Any,
    targets: Any,
    mask: Any,
    indices: Any,
    evidence_targets: tuple[str, ...],
    torch: Any,
    *,
    device: Any,
    batch_size: int,
) -> dict[str, float]:
    if len(indices) == 0:
        return {target: 0.5 for target in evidence_targets}
    model.eval()
    threshold_values = [step / 100 for step in range(5, 100, 5)]
    positives = {target: 0 for target in evidence_targets}
    threshold_counts = {
        target: {threshold: [0, 0, 0] for threshold in threshold_values}
        for target in evidence_targets
    }
    with torch.no_grad():
        for batch_indices, probabilities in _evidence_probability_batches(
            model,
            samples,
            indices,
            torch,
            device=device,
            batch_size=batch_size,
        ):
            expected = targets[batch_indices] >= 0.5
            valid = mask[batch_indices].bool()
            for target_index, target in enumerate(evidence_targets):
                actual = expected[:, :, target_index] & valid
                positives[target] += int(actual.sum().item())
                for threshold in threshold_values:
                    predicted = (probabilities[:, :, target_index] >= threshold) & valid
                    true_positive = int((predicted & actual).sum().item())
                    false_positive = int((predicted & ~actual & valid).sum().item())
                    false_negative = int((~predicted & actual).sum().item())
                    counts = threshold_counts[target][threshold]
                    counts[0] += true_positive
                    counts[1] += false_positive
                    counts[2] += false_negative
    thresholds: dict[str, float] = {}
    for target in evidence_targets:
        if positives[target] == 0:
            thresholds[target] = 0.5
            continue
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in threshold_values:
            true_positive, false_positive, false_negative = threshold_counts[target][threshold]
            denominator = 2 * true_positive + false_positive + false_negative
            f1 = (2 * true_positive / denominator) if denominator else 0.0
            if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
                best_f1 = f1
                best_threshold = threshold
        thresholds[target] = round(best_threshold, 2)
    return thresholds


def _evidence_head_metrics(
    model: Any,
    samples: Any,
    targets: Any,
    mask: Any,
    indices: Any,
    evidence_targets: tuple[str, ...],
    thresholds: dict[str, float],
    torch: Any,
    *,
    device: Any,
    batch_size: int,
) -> dict[str, dict[str, float | int]]:
    if len(indices) == 0:
        return {}
    model.eval()
    counts = {
        target: {"positives": 0, "true_positive": 0, "false_positive": 0, "false_negative": 0}
        for target in evidence_targets
    }
    with torch.no_grad():
        for batch_indices, probabilities in _evidence_probability_batches(
            model,
            samples,
            indices,
            torch,
            device=device,
            batch_size=batch_size,
        ):
            expected = targets[batch_indices] >= 0.5
            valid = mask[batch_indices].bool()
            for target_index, target in enumerate(evidence_targets):
                threshold = thresholds.get(target, 0.5)
                predicted = (probabilities[:, :, target_index] >= threshold) & valid
                actual = expected[:, :, target_index] & valid
                counts[target]["positives"] += int(actual.sum().item())
                counts[target]["true_positive"] += int((predicted & actual).sum().item())
                counts[target]["false_positive"] += int(
                    (predicted & ~actual & valid).sum().item()
                )
                counts[target]["false_negative"] += int((~predicted & actual).sum().item())
    metrics: dict[str, dict[str, float | int]] = {}
    for target in evidence_targets:
        target_counts = counts[target]
        true_positive = target_counts["true_positive"]
        false_positive = target_counts["false_positive"]
        false_negative = target_counts["false_negative"]
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1_denominator = precision + recall
        metrics[target] = {
            "threshold": thresholds.get(target, 0.5),
            "positives": target_counts["positives"],
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / f1_denominator) if f1_denominator else 0.0,
        }
    return metrics


def _tensor_by_target(values: Any, targets: tuple[str, ...]) -> dict[str, float]:
    return {
        target: float(values[index].item())
        for index, target in enumerate(targets)
    }


def tcn_v2_receptive_field_frames(*, levels: int, kernel_size: int) -> int:
    """Return causal input frames visible to one output frame in the v2 TCN."""
    return 1 + 2 * (kernel_size - 1) * sum(2**level for level in range(levels))


def _estimated_frame_seconds(manifest: TcnDatasetManifest) -> float | None:
    intervals = sorted(
        (window.end_time - window.start_time) / (window.row_count - 1)
        for window in manifest.windows
        if window.row_count > 1 and window.end_time > window.start_time
    )
    if not intervals:
        return None
    middle = len(intervals) // 2
    if len(intervals) % 2:
        return intervals[middle]
    return (intervals[middle - 1] + intervals[middle]) / 2


def _coerce_tcn_v2_config(
    config: CausalTcnTrainingConfig | CausalTcnV2TrainingConfig | None,
) -> CausalTcnV2TrainingConfig:
    if config is None:
        return CausalTcnV2TrainingConfig()
    if isinstance(config, CausalTcnV2TrainingConfig):
        return config
    if isinstance(config, CausalTcnTrainingConfig):
        return CausalTcnV2TrainingConfig(
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            hidden_channels=config.hidden_channels,
            levels=config.levels,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )
    raise TypeError("Unsupported TCN v2 training config")


def _validate_tcn_v2_training_config(config: CausalTcnV2TrainingConfig) -> None:
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
    if config.positive_weight_cap < 1:
        raise ValueError("positive_weight_cap must be at least 1")
    if config.boundary_positive_weight_multiplier <= 0:
        raise ValueError("boundary_positive_weight_multiplier must be positive")
    if config.focal_gamma < 0:
        raise ValueError("focal_gamma must be non-negative")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


def _resolve_tcn_device(torch: Any, requested: str) -> Any:
    """Resolve a TCN v2 compute device without making CUDA a hard dependency."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    device_name = requested
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for TCN v2, but torch reports no CUDA device")
    return torch.device(device_name)


def _make_tcn_v2_sequence_model_for_checkpoint(
    *,
    torch: Any,
    nn: Any,
    functional: Any,
    model_config: dict[str, Any],
) -> Any:
    architecture = str(model_config.get("architecture", "plain_causal_conv"))
    if architecture == "residual_dilated_causal_tcn_v2":
        return _make_causal_tcn_v2_sequence_model(
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
    if architecture == "plain_causal_conv":
        return _make_legacy_causal_tcn_v2_sequence_model(
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
    raise ValueError(f"Unsupported TCN v2 checkpoint architecture: {architecture}")


def _make_causal_tcn_v2_sequence_model(
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
    class CausalResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.left_padding = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            self.norm1 = nn.LayerNorm(channels)
            self.conv2 = nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            self.norm2 = nn.LayerNorm(channels)
            self.dropout = nn.Dropout(dropout)

        def forward(self, value: Any) -> Any:
            residual = value
            value = functional.pad(value, (self.left_padding, 0))
            value = self.dropout(torch.relu(_channel_layer_norm(self.norm1, self.conv1(value))))
            value = functional.pad(value, (self.left_padding, 0))
            value = self.dropout(_channel_layer_norm(self.norm2, self.conv2(value)))
            return torch.relu(value + residual)

    class CausalTcnEvidenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Conv1d(input_features, hidden_channels, kernel_size=1)
            self.blocks = nn.ModuleList(
                CausalResidualBlock(hidden_channels, dilation=2**level)
                for level in range(levels)
            )
            self.evidence_head = nn.Linear(hidden_channels, targets)

        def forward(self, samples: Any) -> Any:
            value = samples.transpose(1, 2)
            value = torch.relu(self.input_projection(value))
            for block in self.blocks:
                value = block(value)
            sequence = value.transpose(1, 2)
            return self.evidence_head(sequence)

    return CausalTcnEvidenceModel()


def _channel_layer_norm(norm: Any, value: Any) -> Any:
    return norm(value.transpose(1, 2)).transpose(1, 2)


def _make_legacy_causal_tcn_v2_sequence_model(
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

    class CausalTcnEvidenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            in_channels = input_features
            for level in range(levels):
                layers.append(CausalBlock(in_channels, hidden_channels, dilation=2**level))
                in_channels = hidden_channels
            self.layers = nn.ModuleList(layers)
            self.evidence_head = nn.Linear(hidden_channels, targets)

        def forward(self, samples: Any) -> Any:
            value = samples.transpose(1, 2)
            for layer in self.layers:
                value = layer(value)
            sequence = value.transpose(1, 2)
            return self.evidence_head(sequence)

    return CausalTcnEvidenceModel()
