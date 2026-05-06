"""Optional machine-learning scaffolding for AirDesk."""

from airdesk.ml.dataset import (
    TCN_FEATURE_COLUMNS,
    TCN_TARGETS,
    TcnDatasetManifest,
    TcnFeatureSource,
    TcnWindowSample,
    build_tcn_dataset_manifest,
    load_feature_rows_csv,
    save_tcn_dataset_manifest,
)

__all__ = [
    "TCN_FEATURE_COLUMNS",
    "TCN_TARGETS",
    "TcnDatasetManifest",
    "TcnFeatureSource",
    "TcnWindowSample",
    "build_tcn_dataset_manifest",
    "load_feature_rows_csv",
    "save_tcn_dataset_manifest",
]
