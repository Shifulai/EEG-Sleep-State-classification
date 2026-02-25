from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from .config import ensure_output_dirs, get_default_config
    from .dataset import build_dataloaders
    from .evaluate import evaluate_model, save_confusion_matrix_plot
    from .model import build_model
except ImportError:
    from config import ensure_output_dirs, get_default_config
    from dataset import build_dataloaders
    from evaluate import evaluate_model, save_confusion_matrix_plot
    from model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_weights(class_counts: Dict[int, int], num_classes: int, device: torch.device) -> torch.Tensor:
    weights = np.zeros(num_classes, dtype=np.float32)
    total = float(sum(class_counts.values()))
    for c in range(num_classes):
        count = float(class_counts.get(c, 0))
        weights[c] = total / max(count, 1.0)

    weights = weights / max(weights.mean(), 1e-8)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, data_loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    losses = []

    for x, y in data_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EEG sleep stage classifier")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    cfg = get_default_config()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr

    ensure_output_dirs(cfg)
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(cfg)

    model = build_model(cfg).to(device)

    class_weights = None
    if cfg.train.use_class_weights:
        class_weights = build_class_weights(train_ds.class_counts, cfg.model.num_classes, device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.train.epochs, 1))

    best_val_f1 = -1.0
    patience_counter = 0
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=cfg.model.num_classes,
            criterion=criterion,
        )
        scheduler.step()

        val_macro_f1 = float(val_metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_macro_f1,
                "val_cohen_kappa": val_metrics["cohen_kappa"],
            }
        )

        save_checkpoint(cfg.result.last_ckpt_path, model, optimizer, epoch, best_val_f1)

        improved = val_macro_f1 > best_val_f1
        if improved:
            best_val_f1 = val_macro_f1
            patience_counter = 0
            save_checkpoint(cfg.result.best_ckpt_path, model, optimizer, epoch, best_val_f1)
        else:
            patience_counter += 1

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_f1={val_macro_f1:.4f}"
        )

        if patience_counter >= cfg.train.early_stop_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if cfg.result.best_ckpt_path.exists():
        best_ckpt = torch.load(cfg.result.best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=cfg.model.num_classes,
        criterion=criterion,
    )

    class_names = [name for name, _ in sorted(cfg.data.label_map.items(), key=lambda x: x[1])]
    save_confusion_matrix_plot(
        cm=test_metrics["confusion_matrix"],
        class_names=class_names,
        output_path=cfg.result.confusion_matrix_path,
    )

    output = {
        "config": {
            "epochs": cfg.train.epochs,
            "batch_size": cfg.train.batch_size,
            "lr": cfg.train.lr,
            "num_classes": cfg.model.num_classes,
        },
        "history": history,
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "cohen_kappa": test_metrics["cohen_kappa"],
            "per_class": test_metrics["per_class"],
            "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        },
        "dataset_size": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
    }

    cfg.result.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.result.metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    print(f"Training complete. Metrics saved to {cfg.result.metrics_path}")


if __name__ == "__main__":
    main()
