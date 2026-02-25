from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch


def build_confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _safe_div(n: float, d: float) -> float:
    return n / d if d != 0 else 0.0


def per_class_metrics(cm: np.ndarray) -> List[Dict[str, float]]:
    metrics = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        support = int(cm[c, :].sum())

        metrics.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return metrics


def cohen_kappa_from_cm(cm: np.ndarray) -> float:
    total = float(cm.sum())
    if total == 0:
        return 0.0

    po = float(np.trace(cm)) / total
    row_marginals = cm.sum(axis=1).astype(np.float64)
    col_marginals = cm.sum(axis=0).astype(np.float64)
    pe = float((row_marginals * col_marginals).sum()) / (total * total)

    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


def compute_classification_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, object]:
    cm = build_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    class_stats = per_class_metrics(cm)

    correct = float(np.trace(cm))
    total = float(cm.sum())
    accuracy = _safe_div(correct, total)
    macro_f1 = float(np.mean([row["f1"] for row in class_stats])) if class_stats else 0.0
    kappa = cohen_kappa_from_cm(cm)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "cohen_kappa": kappa,
        "per_class": class_stats,
        "confusion_matrix": cm,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int,
    criterion: Optional[torch.nn.Module] = None,
    input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, object]:
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    losses: List[float] = []

    for x, y in data_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if input_transform is not None:
            x = input_transform(x)

        logits = model(x)
        pred = logits.argmax(dim=1)

        all_true.extend(y.detach().cpu().tolist())
        all_pred.extend(pred.detach().cpu().tolist())

        if criterion is not None:
            loss = criterion(logits, y)
            losses.append(float(loss.item()))

    metrics = compute_classification_metrics(all_true, all_pred, num_classes=num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else None
    return metrics


def save_confusion_matrix_plot(cm: np.ndarray, class_names: List[str], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    try:
        from .config import get_default_config
        from .dataset import build_dataloaders
        from .model import build_model
    except ImportError:
        from config import get_default_config
        from dataset import build_dataloaders
        from model import build_model

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

    metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=cfg.model.num_classes,
        criterion=None,
    )

    class_names = [name for name, _ in sorted(cfg.data.label_map.items(), key=lambda x: x[1])]
    save_confusion_matrix_plot(metrics["confusion_matrix"], class_names, cfg.result.confusion_matrix_path)

    summary = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "cohen_kappa": metrics["cohen_kappa"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    cfg.result.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.result.metrics_path.open("w", encoding="utf-8") as fp:
        json.dump({"eval": summary}, fp, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
