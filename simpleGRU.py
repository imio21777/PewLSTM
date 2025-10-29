#!/usr/bin/env python3
"""Simple GRU baseline using only historical parking counts."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from evaluate_methods import COUNT_TARGETS
from evaluate_methods_timeseries import (
    LOOKBACK,
    PARK_TABLE_IDS,
    compute_actual_metrics,
    load_series,
    normalized_accuracy,
    select_even_days,
    set_seed,
    train_torch_model,
)

DEVICE = torch.device("cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoint")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def flatten_counts(raw_series: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    counts = raw_series[:, -1].reshape(-1, 1)
    count_scaler = MinMaxScaler()
    counts_scaled = count_scaler.fit_transform(counts)

    features = counts_scaled[:-horizon]
    labels_scaled = counts_scaled[horizon:, 0]
    labels_actual = counts[horizon:, 0]

    usable = (features.shape[0] // LOOKBACK) * LOOKBACK
    features = features[:usable]
    labels_scaled = labels_scaled[:usable]
    labels_actual = labels_actual[:usable]

    sequences = features.reshape(-1, LOOKBACK, 1).astype("float32")
    labels_scaled_seq = labels_scaled.reshape(-1, LOOKBACK).astype("float32")
    labels_actual_seq = labels_actual.reshape(-1, LOOKBACK).astype("float32")
    return sequences, labels_scaled_seq, labels_actual_seq, count_scaler


class SimpleGRUModel(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(inputs)
        preds = self.fc(outputs.reshape(-1, outputs.size(-1))).view(-1)
        return preds


def load_checkpoint(path: Optional[str]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_checkpoint(path: Optional[str], data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
    os.replace(tmp_path, path)


def evaluate_simple_gru(
    train_ratio: float,
    test_ratio: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    checkpoint: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    checkpoint_path: Optional[str],
    target_mode: str,
) -> list[Dict[str, object]]:
    set_seed(seed)
    results: list[Dict[str, object]] = []
    method_name = f"Simple GRU ({target_mode})"
    horizons = [1, 2, 3]

    method_store = checkpoint.setdefault(method_name, {})

    for horizon in horizons:
        horizon_key = str(horizon)
        lot_store = method_store.setdefault(horizon_key, {})
        per_lot_metrics: Dict[str, Dict[str, float]] = {
            lot_id: {k: float(v) for k, v in metrics.items()}
            for lot_id, metrics in lot_store.items()
            if isinstance(metrics, dict)
        }

        for lot_index, lot_id in enumerate(PARK_TABLE_IDS):
            if lot_id in per_lot_metrics:
                continue
            raw_series = load_series(lot_index, target_mode)
            sequences, labels_scaled, labels_actual, count_scaler = flatten_counts(raw_series, horizon)

            total_days = sequences.shape[0]
            test_days = select_even_days(total_days, test_ratio)
            train_days = np.setdiff1d(np.arange(total_days), test_days)
            if train_days.size == 0 or test_days.size == 0:
                print(f"[WARN] Simple GRU {lot_id} h{horizon}: insufficient data after split")
                continue

            train_seq = torch.from_numpy(sequences[train_days]).to(DEVICE)
            train_targets = torch.from_numpy(labels_scaled[train_days].reshape(-1)).to(DEVICE)
            test_seq = torch.from_numpy(sequences[test_days]).to(DEVICE)
            test_targets = labels_scaled[test_days].reshape(-1)
            test_actual = labels_actual[test_days].reshape(-1)

            model = SimpleGRUModel().to(DEVICE)
            train_torch_model(
                model,
                train_seq,
                train_targets,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                desc=f"Simple GRU ({target_mode}) {lot_id} h{horizon}",
            )

            model.eval()
            with torch.no_grad():
                preds_norm = model(test_seq).cpu().numpy()

            accuracy = normalized_accuracy(test_targets, preds_norm)
            preds_actual = count_scaler.inverse_transform(preds_norm.reshape(-1, 1)).reshape(-1)
            metrics = compute_actual_metrics(test_actual, preds_actual)
            metrics = {"accuracy": accuracy, **metrics}
            lot_store[lot_id] = {k: float(v) for k, v in metrics.items()}
            per_lot_metrics[lot_id] = lot_store[lot_id]
            save_checkpoint(checkpoint_path, checkpoint)

        if not per_lot_metrics:
            continue

        aggregated: Dict[str, float] = {}
        keys = list(per_lot_metrics[next(iter(per_lot_metrics))].keys())
        for key in keys:
            aggregated[key] = float(np.mean([metrics[key] for metrics in per_lot_metrics.values()]))

        results.append(
            {
                "method": method_name,
                "horizon": horizon,
                "metrics": aggregated,
                "per_lot_metrics": per_lot_metrics,
            }
        )

    return results


def save_outputs(results: list[Dict[str, object]], output_dir: str) -> None:
    summary_path = os.path.join(output_dir, "accuracy_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    mean_path = os.path.join(output_dir, "accuracy_unnormalized_summary.json")
    unnorm_records = []
    for record in results:
        metrics = record["metrics"]
        per_lot = record["per_lot_metrics"]
        unnorm_records.append(
            {
                "method": record["method"],
                "horizon": record["horizon"],
                "metrics": {"accuracy_raw": metrics.get("accuracy_raw", float("nan"))},
                "per_lot_metrics": {
                    lot: {"accuracy_raw": lot_metrics.get("accuracy_raw", float("nan"))}
                    for lot, lot_metrics in per_lot.items()
                },
            }
        )
    with open(mean_path, "w", encoding="utf-8") as fp:
        json.dump(unnorm_records, fp, indent=2)

    plot_path = os.path.join(output_dir, "accuracy_comparison.png")
    horizons = sorted({record["horizon"] for record in results})
    x = np.arange(len(horizons))
    accuracies = [record["metrics"]["accuracy"] for record in results]

    plt = None
    if accuracies:
        import matplotlib.pyplot as plt  # lazy import

        plt.figure(figsize=(6, 4))
        plt.bar(x, accuracies, width=0.5, color="#4C72B0")
        plt.xticks(x, [f"{h}h" for h in horizons])
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    header = "Method\tHorizon\tAccuracy(%)\tAccuracy_raw(%)\tRMSE\tMAE\tMAPE(%)"
    lines = [header]
    for record in sorted(results, key=lambda x: (x["method"], x["horizon"])):
        metrics = record["metrics"]
        line = (
            f"{record['method']}\t{record['horizon']}\t"
            f"{metrics['accuracy']:.2f}\t{metrics.get('accuracy_raw', float('nan')):.2f}\t"
            f"{metrics['rmse']:.4f}\t{metrics['mae']:.4f}\t{metrics['mape']:.2f}"
        )
        lines.append(line)

    text_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(text_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
        fp.write("\n")

    print(f"Saved summary JSON to {summary_path}")
    print(f"Saved raw accuracy JSON to {mean_path}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved metrics table to {text_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple GRU baseline for parking prediction.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results_simple_gru")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--target",
        type=str,
        default="occupancy",
        choices=COUNT_TARGETS,
        help="Target series to forecast.",
    )
    args = parser.parse_args()

    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = args.checkpoint or os.path.join(CHECKPOINT_DIR, "simple_gru_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)

    results = evaluate_simple_gru(
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        target_mode=args.target,
    )

    save_outputs(results, output_dir)


if __name__ == "__main__":
    main()
