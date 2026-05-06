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
from airdesk.ml.diagnostics import (
    FeatureDiagnosticsReport,
    FeatureFileDiagnostic,
    FeatureHoldoutSplit,
    build_feature_diagnostics_report,
    save_feature_diagnostics_report,
    split_feature_holdout,
)
from airdesk.ml.train import (
    CausalTcnPrediction,
    CausalTcnTrainingConfig,
    CausalTcnTrainingResult,
    MissingMlDependencyError,
    TcnTrainingArrays,
    predict_causal_tcn_manifest,
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
    "FeatureDiagnosticsReport",
    "FeatureFileDiagnostic",
    "FeatureHoldoutSplit",
    "load_feature_rows_csv",
    "load_tcn_dataset_manifest",
    "build_feature_diagnostics_report",
    "CausalTcnPrediction",
    "CausalTcnTrainingConfig",
    "CausalTcnTrainingResult",
    "MissingMlDependencyError",
    "TcnTrainingArrays",
    "prepare_tcn_training_arrays",
    "predict_causal_tcn_manifest",
    "save_tcn_dataset_manifest",
    "save_feature_diagnostics_report",
    "split_feature_holdout",
    "train_causal_tcn",
]
