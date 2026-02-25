from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly


ANNOTATION_TO_STAGE = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}


def _as_2d_channels_first(eeg: np.ndarray) -> np.ndarray:
    if eeg.ndim == 1:
        return eeg[None, :]
    if eeg.ndim == 2:
        return eeg
    raise ValueError(f"Expected 1D or 2D EEG array, got shape {eeg.shape}")


def bandpass_filter(
    eeg: np.ndarray,
    fs: float,
    low_hz: float = 0.3,
    high_hz: float = 35.0,
    order: int = 4,
) -> np.ndarray:
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        raise ValueError("Invalid bandpass frequency range")

    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, eeg, axis=-1)


def notch_filter(eeg: np.ndarray, fs: float, notch_hz: float = 50.0, quality: float = 30.0) -> np.ndarray:
    nyq = fs / 2.0
    w0 = notch_hz / nyq
    if w0 <= 0.0 or w0 >= 1.0:
        return eeg

    b, a = iirnotch(w0=w0, Q=quality)
    return filtfilt(b, a, eeg, axis=-1)


def resample_eeg(eeg: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    if orig_fs == target_fs:
        return eeg
    return resample_poly(np.asarray(eeg), up=target_fs, down=orig_fs, axis=-1)


def zscore_normalize(eeg: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = eeg.mean(axis=-1, keepdims=True)
    std = eeg.std(axis=-1, keepdims=True)
    return (eeg - mean) / (std + eps)


def epoch_signal(eeg: np.ndarray, fs: int, epoch_seconds: int = 30) -> np.ndarray:
    samples_per_epoch = fs * epoch_seconds
    total_samples = eeg.shape[-1]
    num_epochs = total_samples // samples_per_epoch
    if num_epochs == 0:
        raise ValueError("Signal shorter than one epoch")

    trimmed = eeg[..., : num_epochs * samples_per_epoch]
    return trimmed.reshape(eeg.shape[0], num_epochs, samples_per_epoch).transpose(1, 0, 2)


def preprocess_record(
    raw_eeg: np.ndarray,
    orig_fs: int,
    target_fs: int = 100,
    epoch_seconds: int = 30,
    apply_notch_hz: Optional[float] = 50.0,
) -> np.ndarray:
    eeg = _as_2d_channels_first(np.asarray(raw_eeg, dtype=np.float32))
    eeg = resample_eeg(eeg, orig_fs=orig_fs, target_fs=target_fs)
    eeg = bandpass_filter(eeg, fs=target_fs, low_hz=0.3, high_hz=35.0)
    if apply_notch_hz is not None:
        eeg = notch_filter(eeg, fs=target_fs, notch_hz=apply_notch_hz)
    eeg = zscore_normalize(eeg)
    return epoch_signal(eeg, fs=target_fs, epoch_seconds=epoch_seconds).astype(np.float32)


def save_preprocessed_record(eeg_epochs: np.ndarray, labels: np.ndarray, out_eeg: Path, out_labels: Path) -> None:
    out_eeg.parent.mkdir(parents=True, exist_ok=True)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_eeg, eeg_epochs.astype(np.float32))
    np.save(out_labels, labels.astype(np.int64))


def split_by_subject(
    subject_ids: np.ndarray,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
):
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to 1.0")

    unique = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    n = len(unique)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_ids = set(unique[:n_train].tolist())
    val_ids = set(unique[n_train : n_train + n_val].tolist())
    test_ids = set(unique[n_train + n_val :].tolist())
    return train_ids, val_ids, test_ids


def _record_prefix(stem: str) -> str:
    s = stem.upper().replace("-PSG", "").replace("-HYPNOGRAM", "")
    m = re.search(r"[A-Z]{2}\d{4}", s)
    if m:
        return m.group(0)
    return s[:6]


def _record_id_from_psg(psg_path: Path) -> str:
    return psg_path.stem.replace("-PSG", "")


def _subject_id_from_record_id(record_id: str) -> str:
    m = re.search(r"[A-Z]{2}\d{4}", record_id.upper())
    return m.group(0) if m else record_id[:6]


def discover_sleep_edf_pairs(raw_dir: Path) -> List[Tuple[Path, Path, str, str]]:
    psg_files = sorted(raw_dir.rglob("*PSG.edf"))
    hyp_files = sorted(raw_dir.rglob("*Hypnogram.edf"))

    if not psg_files:
        raise FileNotFoundError(f"No PSG files found under {raw_dir}")
    if not hyp_files:
        raise FileNotFoundError(f"No Hypnogram files found under {raw_dir}")

    hyp_index: Dict[str, List[Path]] = {}
    for hyp in hyp_files:
        hyp_index.setdefault(_record_prefix(hyp.stem), []).append(hyp)

    pairs: List[Tuple[Path, Path, str, str]] = []
    for psg in psg_files:
        record_id = _record_id_from_psg(psg)
        subject_id = _subject_id_from_record_id(record_id)
        key = _record_prefix(psg.stem)
        candidates = hyp_index.get(key, [])
        if not candidates:
            continue
        hyp = sorted(candidates, key=lambda p: _common_prefix_len(record_id, p.stem), reverse=True)[0]
        pairs.append((psg, hyp, subject_id, record_id))

    if not pairs:
        raise ValueError("No matched PSG/Hypnogram pairs found.")
    return pairs


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _normalize_ch_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def choose_eeg_channel(raw, preferred_channel: str) -> str:
    if preferred_channel:
        wanted = _normalize_ch_name(preferred_channel)
        for name in raw.ch_names:
            if _normalize_ch_name(name) == wanted:
                return name

    eeg_channels = []
    try:
        eeg_channels = raw.copy().pick_types(eeg=True).ch_names
    except Exception:
        eeg_channels = []

    if eeg_channels:
        return eeg_channels[0]
    return raw.ch_names[0]


def build_stage_map(include_wake: bool) -> Dict[str, int]:
    names = ["N1", "N2", "N3", "REM"]
    if include_wake:
        names = ["W", *names]
    return {name: idx for idx, name in enumerate(names)}


def labels_from_annotations(
    onsets: Sequence[float],
    durations: Sequence[float],
    descriptions: Sequence[str],
    num_epochs: int,
    epoch_seconds: int,
    include_wake: bool,
) -> np.ndarray:
    stage_map = build_stage_map(include_wake=include_wake)
    labels = np.full(num_epochs, -1, dtype=np.int64)

    for onset, duration, desc in zip(onsets, durations, descriptions):
        stage_name = ANNOTATION_TO_STAGE.get(str(desc).strip())
        if stage_name is None:
            continue
        if stage_name == "W" and not include_wake:
            continue

        start = int(np.floor(onset / epoch_seconds))
        end = int(np.floor((onset + duration) / epoch_seconds + 1e-8))
        if end <= start:
            end = start + 1

        start = max(start, 0)
        end = min(end, num_epochs)
        if start >= num_epochs:
            continue
        labels[start:end] = stage_map[stage_name]

    return labels


def process_single_record(
    psg_path: Path,
    hyp_path: Path,
    out_root: Path,
    preferred_channel: str,
    target_fs: int,
    epoch_seconds: int,
    include_wake: bool,
    notch_hz: Optional[float],
) -> Tuple[Path, Path, int, Counter]:
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "mne is required for EDF processing. Install with: pip install mne"
        ) from exc

    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose="ERROR")
    ann = mne.read_annotations(str(hyp_path))

    selected_channel = choose_eeg_channel(raw, preferred_channel=preferred_channel)
    raw.pick([selected_channel])
    signal = raw.get_data().astype(np.float32)

    orig_fs = int(round(float(raw.info["sfreq"])))
    total_seconds = signal.shape[-1] / float(orig_fs)
    num_epochs_raw = int(total_seconds // epoch_seconds)
    if num_epochs_raw <= 0:
        raise ValueError(f"Record too short for epoching: {psg_path.name}")

    labels = labels_from_annotations(
        onsets=ann.onset,
        durations=ann.duration,
        descriptions=ann.description,
        num_epochs=num_epochs_raw,
        epoch_seconds=epoch_seconds,
        include_wake=include_wake,
    )

    eeg_epochs = preprocess_record(
        raw_eeg=signal,
        orig_fs=orig_fs,
        target_fs=target_fs,
        epoch_seconds=epoch_seconds,
        apply_notch_hz=notch_hz,
    )

    n = min(eeg_epochs.shape[0], labels.shape[0])
    eeg_epochs = eeg_epochs[:n]
    labels = labels[:n]

    valid = labels >= 0
    eeg_epochs = eeg_epochs[valid]
    labels = labels[valid]
    if labels.size == 0:
        raise ValueError(f"No valid labeled epochs after filtering: {psg_path.name}")

    record_id = _record_id_from_psg(psg_path)
    out_eeg = out_root / f"{record_id}_eeg.npy"
    out_label = out_root / f"{record_id}_label.npy"
    save_preprocessed_record(eeg_epochs=eeg_epochs, labels=labels, out_eeg=out_eeg, out_labels=out_label)
    return out_eeg, out_label, int(labels.size), Counter(labels.tolist())


def write_metadata(rows: List[Dict[str, str]], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subject_id", "record_id", "eeg_path", "label_path", "split"]
    with metadata_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def assign_splits(rows: List[Dict[str, str]], seed: int) -> None:
    subject_ids = np.array([row["subject_id"] for row in rows], dtype=object)
    train_set, val_set, test_set = split_by_subject(subject_ids=subject_ids, seed=seed)
    for row in rows:
        sid = row["subject_id"]
        if sid in train_set:
            row["split"] = "train"
        elif sid in val_set:
            row["split"] = "val"
        else:
            row["split"] = "test"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Preprocess Sleep-EDF into .npy epochs and metadata.csv")
    parser.add_argument("--raw-dir", type=Path, default=project_root / "Data" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=project_root / "Data" / "processed")
    parser.add_argument("--metadata-path", type=Path, default=project_root / "Data" / "metadata.csv")
    parser.add_argument("--channel", type=str, default="Fpz-Cz")
    parser.add_argument("--target-fs", type=int, default=100)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--notch-hz", type=float, default=50.0)
    parser.add_argument("--disable-notch", action="store_true")
    parser.add_argument("--include-wake", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notch_hz = None if args.disable_notch else args.notch_hz

    pairs = discover_sleep_edf_pairs(args.raw_dir)
    rows: List[Dict[str, str]] = []
    total_epochs = 0
    stage_counter: Counter = Counter()

    print(f"Found {len(pairs)} PSG/Hypnogram pairs under {args.raw_dir}")
    for idx, (psg, hyp, subject_id, record_id) in enumerate(pairs, start=1):
        try:
            out_eeg, out_label, num_epochs, label_counter = process_single_record(
                psg_path=psg,
                hyp_path=hyp,
                out_root=args.processed_dir,
                preferred_channel=args.channel,
                target_fs=args.target_fs,
                epoch_seconds=args.epoch_seconds,
                include_wake=args.include_wake,
                notch_hz=notch_hz,
            )
        except Exception as exc:
            print(f"[{idx}/{len(pairs)}] Skip {psg.name}: {exc}")
            continue

        rows.append(
            {
                "subject_id": subject_id,
                "record_id": record_id,
                "eeg_path": str(out_eeg),
                "label_path": str(out_label),
                "split": "",
            }
        )
        total_epochs += num_epochs
        stage_counter.update(label_counter)
        print(f"[{idx}/{len(pairs)}] Processed {record_id}: {num_epochs} epochs")

    if not rows:
        raise RuntimeError("No record was processed successfully.")

    assign_splits(rows, seed=args.seed)
    write_metadata(rows, args.metadata_path)

    print(f"Saved metadata: {args.metadata_path}")
    print(f"Processed records: {len(rows)}")
    print(f"Total epochs: {total_epochs}")
    print(f"Label distribution: {dict(stage_counter)}")


if __name__ == "__main__":
    main()
