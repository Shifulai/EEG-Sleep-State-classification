from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_label_map(include_wake: bool = False) -> Dict[str, int]:
    classes = ["N1", "N2", "N3", "REM"]
    if include_wake:
        classes = ["W", *classes]
    return {name: index for index, name in enumerate(classes)}


@dataclass
class DataConfig:
    data_root: Path = PROJECT_ROOT / "Data"
    processed_root: Path = PROJECT_ROOT / "Data" / "processed"
    metadata_path: Path = PROJECT_ROOT / "Data" / "metadata.csv"

    split_train: str = "train"
    split_val: str = "val"
    split_test: str = "test"

    include_wake: bool = False
    context_window: int = 5
    sampling_rate: int = 100
    epoch_seconds: int = 30

    label_map: Dict[str, int] = field(default_factory=dict)


@dataclass
class ModelConfig:
    in_channels: int = 1
    embedding_dim: int = 256
    cnn_hidden_channels: int = 128

    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 9

    num_classes: int = 4


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 64
    num_workers: int = 0

    lr: float = 1e-3
    weight_decay: float = 1e-2

    early_stop_patience: int = 10
    use_weighted_sampler: bool = False
    use_class_weights: bool = True


@dataclass
class ResultConfig:
    result_root: Path = PROJECT_ROOT / "Result"
    checkpoint_dir: Path = PROJECT_ROOT / "Result" / "checkpoints"
    best_ckpt_path: Path = PROJECT_ROOT / "Result" / "checkpoints" / "best.pt"
    last_ckpt_path: Path = PROJECT_ROOT / "Result" / "checkpoints" / "last.pt"

    metrics_path: Path = PROJECT_ROOT / "Result" / "metrics.json"
    confusion_matrix_path: Path = PROJECT_ROOT / "Result" / "confusion_matrix.png"
    band_importance_path: Path = PROJECT_ROOT / "Result" / "band_importance.csv"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    result: ResultConfig = field(default_factory=ResultConfig)


def get_default_config() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.data.label_map = build_label_map(include_wake=cfg.data.include_wake)
    cfg.model.num_classes = len(cfg.data.label_map)
    return cfg


def ensure_output_dirs(cfg: ExperimentConfig) -> None:
    cfg.result.result_root.mkdir(parents=True, exist_ok=True)
    cfg.result.checkpoint_dir.mkdir(parents=True, exist_ok=True)
