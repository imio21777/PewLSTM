#!/usr/bin/env python3
"""Evaluate the pre-trained PewLSTM checkpoint (model_P1_1h.pth) on the 20% test split."""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
CKPT_PATH = SCRIPT_DIR / "model_P1_1h.pth"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = RESULTS_DIR / "pretrained_accuracy.json"
BAR_PLOT_PATH = RESULTS_DIR / "pretrained_accuracy.png"

# Reuse utilities from evaluate_methods
import sys
sys.path.append(str(PROJECT_ROOT))
from evaluate_methods import (  # type: ignore
    prepare_sequences,
    set_seed,
    compute_metrics,
    PewForecastModel,
)


def evaluate_pretrained(train_ratio: float = 0.8, seed: int = 42) -> Dict[str, float]:
    set_seed(seed)
    lot_index = 0  # P1
    horizon = 1

    (
        _train_seq,
        _train_labels_scaled,
        _train_labels_actual,
        test_seq,
        _test_labels_scaled,
        test_labels_actual,
        count_scaler,
    ) = prepare_sequences(lot_index, horizon, train_ratio)

    device = torch.device("cpu")
    test_inputs = torch.from_numpy(test_seq).to(device)

    model = PewForecastModel(hidden_dim=1, use_periodic=True, use_weather=True).to(device)
    state_dict = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        preds_scaled = model(test_inputs).cpu().numpy()

    preds_actual = count_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
    actual = test_labels_actual.reshape(-1)

    metrics = compute_metrics(actual, preds_actual)
    return dict(metrics)


def save_summary(metrics: Dict[str, float]) -> None:
    payload = [
        {
            "method": "PewLSTM (pretrained P1 checkpoint)",
            "horizon": 1,
            "metrics": metrics,
        }
    ]
    with SUMMARY_PATH.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def plot_accuracy_bar(metrics: Dict[str, float]) -> None:
    accuracy = metrics["accuracy"]
    plt.figure(figsize=(5, 5))
    plt.bar(["P1 (1h)"], [accuracy], color="#4C72B0")
    lower = min(0.0, accuracy * 1.1)
    plt.ylim(lower, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Pretrained PewLSTM Accuracy")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(BAR_PLOT_PATH)
    plt.close()


def main() -> None:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    metrics = evaluate_pretrained()
    save_summary(metrics)
    plot_accuracy_bar(metrics)

    print("Pretrained checkpoint evaluation metrics:")
    for key, value in metrics.items():
        if key == "accuracy" or key == "mape":
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value:.4f}")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved bar chart to {BAR_PLOT_PATH}")


if __name__ == "__main__":
    main()
