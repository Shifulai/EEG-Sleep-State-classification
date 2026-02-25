from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_metadata_rows(metadata_path: Path) -> List[Dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def _normalize_eeg_array(eeg: np.ndarray) -> np.ndarray:
    if eeg.ndim == 2:
        # [epochs, samples] -> [epochs, channels=1, samples]
        eeg = eeg[:, None, :]
    if eeg.ndim != 3:
        raise ValueError(f"Expected EEG shape [epochs, channels, samples], got {eeg.shape}")
    return eeg.astype(np.float32)


def _map_labels(labels: np.ndarray, label_map: Dict[str, int]) -> np.ndarray:
    if labels.dtype.kind in {"U", "S", "O"}:
        mapped = np.array([label_map[str(label)] for label in labels], dtype=np.int64)
        return mapped
    return labels.astype(np.int64)


class SleepEpochDataset(Dataset):
    """Record-level dataset loader for preprocessed NumPy files.

    Metadata format (required columns):
    - eeg_path: path to .npy EEG array, shape [epochs, channels, samples] or [epochs, samples]
    - label_path: path to .npy labels array, shape [epochs]
    - split: train/val/test

    Optional columns:
    - subject_id
    - record_id
    """

    def __init__(
        self,
        metadata_path: Path,
        split: str,
        context_window: int,
        label_map: Optional[Dict[str, int]] = None,
        strict: bool = True,
    ) -> None:
        if context_window % 2 == 0:
            raise ValueError("context_window must be odd so a center epoch exists")

        self.metadata_path = Path(metadata_path)
        self.split = split.lower()
        self.context_window = context_window
        self.label_map = label_map or {}
        self.strict = strict

        self.records: List[Dict[str, object]] = []
        self.index_map: List[Tuple[int, int]] = []
        self.sample_labels: List[int] = []
        self.class_counts: Counter = Counter()

        self._build_index()

    def _build_index(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata file not found: {self.metadata_path}")

        rows = _load_metadata_rows(self.metadata_path)
        base_dir = self.metadata_path.parent

        for row in rows:
            row_split = row.get("split", "").strip().lower()
            if row_split != self.split:
                continue

            eeg_value = row.get("eeg_path", "").strip()
            label_value = row.get("label_path", "").strip()
            if not eeg_value or not label_value:
                if self.strict:
                    raise ValueError("metadata row missing eeg_path or label_path")
                continue

            eeg_path = _resolve_path(eeg_value, base_dir)
            label_path = _resolve_path(label_value, base_dir)
            if not eeg_path.exists() or not label_path.exists():
                if self.strict:
                    raise FileNotFoundError(f"missing npy file(s): {eeg_path}, {label_path}")
                continue

            eeg = _normalize_eeg_array(np.load(eeg_path, allow_pickle=False))
            labels = _map_labels(np.load(label_path, allow_pickle=False), self.label_map)

            if eeg.shape[0] != labels.shape[0]:
                raise ValueError(
                    "Epoch count mismatch: "
                    f"{eeg_path.name} has {eeg.shape[0]} but labels have {labels.shape[0]}"
                )

            record_idx = len(self.records)
            self.records.append(
                {
                    "subject_id": row.get("subject_id", ""),
                    "record_id": row.get("record_id", ""),
                    "eeg": eeg,
                    "labels": labels,
                }
            )

            for epoch_idx, label in enumerate(labels.tolist()):
                int_label = int(label)
                self.index_map.append((record_idx, epoch_idx))
                self.sample_labels.append(int_label)
                self.class_counts[int_label] += 1

        if not self.index_map:
            raise ValueError(f"No samples found for split={self.split} from {self.metadata_path}")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record_idx, center_epoch = self.index_map[index]
        record = self.records[record_idx]

        eeg: np.ndarray = record["eeg"]  # [epochs, channels, samples]
        labels: np.ndarray = record["labels"]  # [epochs]

        num_epochs, channels, samples = eeg.shape
        half = self.context_window // 2

        window = np.zeros((self.context_window, channels, samples), dtype=np.float32)
        for offset in range(-half, half + 1):
            src_epoch = center_epoch + offset
            dst_epoch = offset + half
            if 0 <= src_epoch < num_epochs:
                window[dst_epoch] = eeg[src_epoch]

        x = torch.from_numpy(window)  # [T, C, L]
        y = torch.tensor(int(labels[center_epoch]), dtype=torch.long)
        return x, y


def _build_weighted_sampler(dataset: SleepEpochDataset) -> WeightedRandomSampler:
    sample_weights = []
    for label in dataset.sample_labels:
        count = dataset.class_counts[int(label)]
        sample_weights.append(1.0 / max(count, 1))

    weights = torch.tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler


def make_dataloader(
    dataset: SleepEpochDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    weighted_sampling: bool,
) -> DataLoader:
    if weighted_sampling:
        sampler = _build_weighted_sampler(dataset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_dataloaders(cfg) -> Tuple[SleepEpochDataset, SleepEpochDataset, SleepEpochDataset, DataLoader, DataLoader, DataLoader]:
    train_ds = SleepEpochDataset(
        metadata_path=cfg.data.metadata_path,
        split=cfg.data.split_train,
        context_window=cfg.data.context_window,
        label_map=cfg.data.label_map,
        strict=True,
    )
    val_ds = SleepEpochDataset(
        metadata_path=cfg.data.metadata_path,
        split=cfg.data.split_val,
        context_window=cfg.data.context_window,
        label_map=cfg.data.label_map,
        strict=True,
    )
    test_ds = SleepEpochDataset(
        metadata_path=cfg.data.metadata_path,
        split=cfg.data.split_test,
        context_window=cfg.data.context_window,
        label_map=cfg.data.label_map,
        strict=True,
    )

    train_loader = make_dataloader(
        train_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=not cfg.train.use_weighted_sampler,
        weighted_sampling=cfg.train.use_weighted_sampler,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        weighted_sampling=False,
    )
    test_loader = make_dataloader(
        test_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        weighted_sampling=False,
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
