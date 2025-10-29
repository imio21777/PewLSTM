#!/usr/bin/env python3
"""Evaluate model_P1_1h.pth using normalization-based metric with evenly sampled test days."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import math

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = RESULTS_DIR / "pretrained_accuracy_normalized.json"
BAR_PLOT_PATH = RESULTS_DIR / "pretrained_accuracy_normalized.png"
CKPT_PATH = SCRIPT_DIR / "model_P1_1h.pth"

import sys
sys.path.append(str(SCRIPT_DIR))

from evaluate_methods_timeseries import get_full_sequences  # type: ignore
from evaluate_methods import set_seed  # type: ignore
from modifiedPSTM import pew_LSTM  # type: ignore

class CheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm1 = pew_LSTM(1, 1, 4, use_periodic=True, use_weather=True)
        self.lstm2 = pew_LSTM(1, 1, 4, use_periodic=True, use_weather=True)
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weather = inputs[:, :, :-1]
        counts = inputs[:, :, -1].unsqueeze(2)
        h1, _ = self.lstm1(counts, weather)
        h2, _ = self.lstm2(h1, weather)
        out = self.fc(h2.reshape(-1, 1)).view(-1)
        return out


def select_even_days(total_days: int, test_ratio: float) -> np.ndarray:
    test_days = max(1, int(round(total_days * test_ratio)))
    indices = np.linspace(0, total_days - 1, test_days, dtype=int)
    return np.unique(indices)


def evaluate_normalized(test_ratio: float, seed: int) -> Dict[str, float]:
    set_seed(seed)
    lot_index = 0
    horizon = 1

    sequences, labels_scaled, labels_actual, count_scaler = get_full_sequences(lot_index, horizon)
    num_days = sequences.shape[0]
    test_days = select_even_days(num_days, test_ratio)
    train_days = np.setdiff1d(np.arange(num_days), test_days)

    train_seq = sequences[train_days]
    train_labels = labels_scaled[train_days]
    test_seq = sequences[test_days]
    test_labels_scaled = labels_scaled[test_days]
    test_labels_actual = labels_actual[test_days]

    model = CheckpointModel()
    state_dict = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        preds = model(torch.from_numpy(test_seq))

    preds_np = preds.numpy().reshape(-1)
    targets_scaled_np = test_labels_scaled.reshape(-1)
    targets_actual_np = test_labels_actual.reshape(-1)

    preds_actual_np = count_scaler.inverse_transform(preds_np.reshape(-1, 1)).reshape(-1)

    if preds_actual_np.size > 1 and preds_np.size > 1:
        diffs = np.abs(targets_scaled_np[:-1] - preds_np[1:])
        mae = float(np.mean(diffs))
        p = 0.0
        k = 0
        sq_err = 0.0
        for idx in range(targets_actual_np.size - 1):
            target_val = targets_actual_np[idx]
            pred_val = preds_actual_np[idx + 1]
            k += 1
            if target_val != 0:
                p += abs(target_val - pred_val) / target_val
            else:
                p += abs(target_val - pred_val)
            sq_err += float((target_val - pred_val) ** 2)
        acc_term = p / k if k else 0.0
        accuracy = (1.0 - acc_term) * 100.0
        rmse = math.sqrt(sq_err / k) if k else 0.0
        actual_aligned = targets_actual_np[:-1]
        preds_aligned = preds_actual_np[1:]
        if actual_aligned.size:
            denom = np.where(np.abs(actual_aligned) < 1e-3, 1e-3, np.abs(actual_aligned))
            mape = float(np.mean(np.abs(preds_aligned - actual_aligned) / denom) * 100.0)
        else:
            mape = 0.0
    else:
        diffs = np.abs(targets_scaled_np - preds_np)
        mae = float(np.mean(diffs)) if diffs.size else 0.0
        if targets_actual_np.size and preds_actual_np.size:
            if targets_actual_np[0] != 0:
                accuracy = (1.0 - abs(targets_actual_np[0] - preds_actual_np[0]) / targets_actual_np[0]) * 100.0
            else:
                accuracy = (1.0 - abs(targets_actual_np[0] - preds_actual_np[0])) * 100.0
            rmse = math.sqrt(float((targets_actual_np[0] - preds_actual_np[0]) ** 2))
            denom = max(abs(targets_actual_np[0]), 1e-3)
            mape = float(abs(preds_actual_np[0] - targets_actual_np[0]) / denom * 100.0)
        else:
            accuracy = 0.0
            rmse = 0.0
            mape = 0.0
    return {"accuracy": accuracy, "mae": mae, "rmse": rmse, "mape": mape}


def plot_bar(metrics: Dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.bar(["P1 (normalized)"], [metrics["accuracy"]], color="#4C72B0")
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("P1 Pretrained (Normalized Metric)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(BAR_PLOT_PATH)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate P1 checkpoint using normalized metric.")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = evaluate_normalized(test_ratio=args.test_ratio, seed=args.seed)
    record = {
        "method": "PewLSTM (pretrained normalized)",
        "horizon": 1,
        "metrics": metrics,
        "per_lot_metrics": {"P1": metrics},
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as fp:
        json.dump([record], fp, indent=2)

    plot_bar(metrics)
    print("Normalized metrics:", metrics)
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved bar chart to {BAR_PLOT_PATH}")


if __name__ == "__main__":
    main()
