from __future__ import annotations

import csv
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.signal import butter, filtfilt

try:
    from .config import get_default_config
    from .dataset import build_dataloaders
    from .evaluate import evaluate_model
    from .model import build_model
except ImportError:
    from config import get_default_config
    from dataset import build_dataloaders
    from evaluate import evaluate_model
    from model import build_model


BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),
    "beta": (13.0, 30.0),
}


def bandstop_filter_batch(x: torch.Tensor, fs: int, low_hz: float, high_hz: float, order: int = 4) -> torch.Tensor:
    """Apply band-stop filtering on shape [B, T, C, L]."""
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        return x

    b, a = butter(order, [low, high], btype="bandstop")

    x_np = x.detach().cpu().numpy()
    filtered = filtfilt(b, a, x_np, axis=-1)
    filtered = np.ascontiguousarray(filtered)
    return torch.from_numpy(filtered).to(device=x.device, dtype=x.dtype)


def _save_band_importance(rows: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["band", "low_hz", "high_hz", "macro_f1", "delta_macro_f1"]
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_band_importance(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int,
    sampling_rate: int,
    output_csv: Path,
) -> List[Dict[str, float]]:
    baseline = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
        num_classes=num_classes,
        criterion=None,
    )
    baseline_f1 = float(baseline["macro_f1"])

    rows: List[Dict[str, float]] = []
    for band_name, (low_hz, high_hz) in BANDS.items():
        transform = partial(bandstop_filter_batch, fs=sampling_rate, low_hz=low_hz, high_hz=high_hz)
        metrics = evaluate_model(
            model=model,
            data_loader=data_loader,
            device=device,
            num_classes=num_classes,
            criterion=None,
            input_transform=transform,
        )
        macro_f1 = float(metrics["macro_f1"])
        rows.append(
            {
                "band": band_name,
                "low_hz": low_hz,
                "high_hz": high_hz,
                "macro_f1": macro_f1,
                "delta_macro_f1": baseline_f1 - macro_f1,
            }
        )

    rows.sort(key=lambda x: x["delta_macro_f1"], reverse=True)
    _save_band_importance(rows, Path(output_csv))
    return rows


def main() -> None:
    cfg = get_default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, _, _, _, test_loader = build_dataloaders(cfg)

    model = build_model(cfg).to(device)
    if not cfg.result.best_ckpt_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found at {cfg.result.best_ckpt_path}. Train the model first."
        )

    ckpt = torch.load(cfg.result.best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    rows = run_band_importance(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=cfg.model.num_classes,
        sampling_rate=cfg.data.sampling_rate,
        output_csv=cfg.result.band_importance_path,
    )

    print("Band importance saved:")
    for row in rows:
        print(
            f"{row['band']}: macro_f1={row['macro_f1']:.4f}, "
            f"delta_macro_f1={row['delta_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
