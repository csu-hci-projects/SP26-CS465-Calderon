"""Optional machine-learning scaffolding for AirDesk."""

from airdesk.ml.dataset import (
    TCN_FEATURE_COLUMNS,
    TCN_TARGETS,
    TcnDatasetManifest,
    TcnFeatureSource,
    TcnWindowSample,
    build_tcn_dataset_manifest,
    feature_window_matrix,
    load_feature_rows_csv,
    load_tcn_dataset_manifest,
    save_tcn_dataset_manifest,
)
from airdesk.ml.train import (
    CausalTcnTrainingConfig,
    CausalTcnTrainingResult,
    MissingMlDependencyError,
    TcnTrainingArrays,
    prepare_tcn_training_arrays,
    train_causal_tcn,
)

__all__ = [
    "TCN_FEATURE_COLUMNS",
    "TCN_TARGETS",
    "TcnDatasetManifest",
    "TcnFeatureSource",
    "TcnWindowSample",
    "build_tcn_dataset_manifest",
    "feature_window_matrix",
    "load_feature_rows_csv",
    "load_tcn_dataset_manifest",
    "CausalTcnTrainingConfig",
    "CausalTcnTrainingResult",
    "MissingMlDependencyError",
    "TcnTrainingArrays",
    "prepare_tcn_training_arrays",
    "save_tcn_dataset_manifest",
    "train_causal_tcn",
]
