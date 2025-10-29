#!/usr/bin/env python3
"""Rolling time-series evaluation with reserved uniformly sampled test set."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from evaluate_methods import (  # type: ignore
    LOOKBACK,
    METHODS,
    MethodConfig,
    PARK_TABLE_IDS,
    PewForecastModel,
    SimpleLSTMModel,
    load_series,
    select_methods,
    set_seed,
    train_torch_model,
)
import run_modified_pewlstm as dataset_mod  # noqa: F401

data_root = os.path.join(SCRIPT_DIR, "data")
dataset_mod.RECORD_PATH = os.path.join(data_root, "record")
dataset_mod.WEATHER_PATH = os.path.join(data_root, "weather")

torch.set_num_threads(max(1, os.cpu_count() // 2))

CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoint")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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


def get_full_sequences(
    lot_index: int,
    horizon: int,
    target: str = "occupancy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    raw_series = load_series(lot_index, target)
    feature_scaler = MinMaxScaler()
    scaled_series = feature_scaler.fit_transform(raw_series)

    count_scaler = MinMaxScaler()
    count_scaler.fit(raw_series[:, -1].reshape(-1, 1))

    features = scaled_series[:-horizon]
    labels_scaled = scaled_series[horizon:, -1]
    labels_actual = raw_series[horizon:, -1]

    usable = (features.shape[0] // LOOKBACK) * LOOKBACK
    if usable == 0:
        raise ValueError("Not enough data for evaluation")

    features = features[:usable]
    labels_scaled = labels_scaled[:usable]
    labels_actual = labels_actual[:usable]

    sequences = features.reshape(-1, LOOKBACK, raw_series.shape[1]).astype("float32")
    labels_scaled_seq = labels_scaled.reshape(-1, LOOKBACK).astype("float32")
    labels_actual_seq = labels_actual.reshape(-1, LOOKBACK).astype("float32")
    return sequences, labels_scaled_seq, labels_actual_seq, count_scaler


def select_even_days(total_days: int, ratio: float) -> np.ndarray:
    test_days = max(1, int(round(total_days * ratio)))
    indices = np.linspace(0, total_days - 1, test_days, dtype=int)
    return np.unique(indices)


def normalized_accuracy(targets: np.ndarray, preds: np.ndarray) -> float:
    if targets.size == 0 or preds.size == 0:
        return 0.0

    targets_flat = targets.reshape(-1)
    preds_flat = preds.reshape(-1)

    if targets_flat.size > 1 and preds_flat.size > 1:
        diffs = np.abs(targets_flat[:-1] - preds_flat[1:])
        mae = float(np.mean(diffs))
        error_sum = 0.0
        count = 0
        for idx in range(targets_flat.size - 1):
            target_val = targets_flat[idx]
            pred_val = preds_flat[idx + 1]
            count += 1
            if target_val != 0:
                error_sum += abs(target_val - pred_val) / target_val
            else:
                error_sum += abs(target_val - pred_val)
        acc_term = error_sum / count if count else 0.0
    else:
        diffs = np.abs(targets_flat - preds_flat)
        mae = float(np.mean(diffs)) if diffs.size else 0.0
        if targets_flat.size and preds_flat.size:
            base_target = targets_flat[0]
            base_pred = preds_flat[0]
            if base_target != 0:
                acc_term = abs(base_target - base_pred) / base_target
            else:
                acc_term = abs(base_target - base_pred)
        else:
            acc_term = 1.0

    accuracy = (1.0 - acc_term) * 100.0
    return float(accuracy)


def compute_actual_metrics(actual: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    if preds.size > 1 and actual.size > 1:
        actual_main = actual[:-1]
        preds_main = preds[1:]
    else:
        actual_main = actual
        preds_main = preds
    diff = preds_main - actual_main
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    denom = np.where(np.abs(actual_main) < 1e-3, 1e-3, np.abs(actual_main))
    mape = float(np.mean(np.abs(diff) / denom) * 100.0)
    acc_raw = 0.0
    if actual_main.size > 0 and preds_main.size > 0:
        if actual_main.size > 0:
            error_sum = 0.0
            count = 0
            for target_val, pred_val in zip(actual_main, preds_main):
                count += 1
                if target_val != 0:
                    error_sum += abs(target_val - pred_val) / abs(target_val)
                else:
                    error_sum += abs(target_val - pred_val)
            mean_error = error_sum / count if count else 0.0
        else:
            mean_error = 0.0
        acc_raw = (1.0 - mean_error) * 100.0
    return {"rmse": rmse, "mae": mae, "mape": mape, "accuracy_raw": acc_raw}


def generate_fold_boundaries(num_days: int, train_ratio: float, folds: int, fold_size: Optional[int]) -> List[Tuple[int, int]]:
    if num_days < 2:
        return []
    base_train = max(1, int(num_days * train_ratio))
    if base_train >= num_days:
        base_train = num_days - 1
    remaining = num_days - base_train
    if remaining <= 0:
        return []

    step = fold_size if fold_size and fold_size > 0 else max(1, math.ceil(remaining / max(1, folds)))

    boundaries: List[Tuple[int, int]] = []
    start = base_train
    while start < num_days:
        end = min(num_days, start + step)
        if start >= end:
            break
        boundaries.append((start, end))
        start = end
    return boundaries


def train_model(
    method: MethodConfig,
    train_seq: np.ndarray,
    train_labels_scaled: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    desc: str,
):
    device = torch.device("cpu")
    if method.kind == "pew":
        model = PewForecastModel(
            hidden_dim=int(method.kwargs.get("hidden_dim", 8)),
            use_periodic=bool(method.kwargs.get("use_periodic", True)),
            use_weather=bool(method.kwargs.get("use_weather", True)),
        ).to(device)
        train_inputs = torch.from_numpy(train_seq).to(device)
        train_targets = torch.from_numpy(train_labels_scaled.reshape(-1)).to(device)
        train_torch_model(model, train_inputs, train_targets, epochs, lr, weight_decay, desc=desc)
        return model
    if method.kind == "simple_lstm":
        model = SimpleLSTMModel(hidden_dim=int(method.kwargs.get("hidden_dim", 16))).to(device)
        train_inputs = torch.from_numpy(train_seq[:, :, -1:].copy()).to(device)
        train_targets = torch.from_numpy(train_labels_scaled.reshape(-1)).to(device)
        train_torch_model(model, train_inputs, train_targets, epochs, lr, weight_decay, desc=desc)
        return model
    if method.kind == "regression":
        reg = RandomForestRegressor(**method.kwargs)
        X_train = train_seq.reshape(train_seq.shape[0], -1)
        y_train = train_labels_scaled.reshape(train_labels_scaled.shape[0], -1)
        reg.fit(X_train, y_train)
        return reg
    raise ValueError(f"Unsupported method kind {method.kind}")


def predict_model(method: MethodConfig, model, test_seq: np.ndarray) -> np.ndarray:
    device = torch.device("cpu")
    if method.kind == "pew":
        with torch.no_grad():
            preds = model(torch.from_numpy(test_seq).to(device)).cpu().numpy()
        return preds.reshape(-1)
    if method.kind == "simple_lstm":
        with torch.no_grad():
            preds = model(torch.from_numpy(test_seq[:, :, -1:].copy()).to(device)).cpu().numpy()
        return preds.reshape(-1)
    if method.kind == "regression":
        X_test = test_seq.reshape(test_seq.shape[0], -1)
        preds = model.predict(X_test)
        return preds.reshape(-1)
    raise ValueError(f"Unsupported method kind {method.kind}")


def evaluate_methods_timeseries(
    methods: List[MethodConfig],
    horizons: Iterable[int],
    train_ratio: float,
    test_ratio: float,
    folds: int,
    fold_size: Optional[int],
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    checkpoint: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    checkpoint_path: Optional[str],
    enable_fold_validation: bool,
) -> List[Dict[str, object]]:
    set_seed(seed)
    horizons_list = list(horizons)
    results: List[Dict[str, object]] = []

    total_runs = len(methods) * len(horizons_list)
    outer_bar = tqdm(total=total_runs, desc="Rolling evaluation", unit="run") if (tqdm and total_runs > 0) else None

    try:
        for method in methods:
            for horizon in horizons_list:
                per_lot_metrics: Dict[str, Dict[str, float]] = {}
                lot_bar = (
                    tqdm(total=len(PARK_TABLE_IDS), desc=f"{method.name} h{horizon}", leave=False, unit="lot", dynamic_ncols=True)
                    if tqdm
                    else None
                )

                method_store = checkpoint.setdefault(method.name, {})
                horizon_key = str(horizon)
                horizon_store = method_store.setdefault(horizon_key, {})
                per_lot_metrics: Dict[str, Dict[str, float]] = {
                    lot: {k: float(v) for k, v in metrics.items()}
                    for lot, metrics in horizon_store.items()
                    if isinstance(metrics, dict)
                }

                try:
                    for lot_index, lot_id in enumerate(PARK_TABLE_IDS):
                        try:
                            sequences, labels_scaled, labels_actual, count_scaler = get_full_sequences(lot_index, horizon)
                        except Exception as exc:
                            print(f"[WARN] {method.name} {lot_id} h{horizon}: {exc}")
                            continue

                        total_days = sequences.shape[0]
                        test_days = select_even_days(total_days, test_ratio)
                        train_days = np.setdiff1d(np.arange(total_days), test_days)
                        if train_days.size == 0 or test_days.size == 0:
                            print(f"[WARN] {method.name} {lot_id} h{horizon}: insufficient data after split")
                            continue

                        stored_metrics = horizon_store.get(lot_id)
                        needs_recompute = (
                            stored_metrics is None
                            or not isinstance(stored_metrics, dict)
                            or "accuracy_raw" not in stored_metrics
                            or "accuracy" not in stored_metrics
                        )
                        if not needs_recompute:
                            per_lot_metrics[lot_id] = {k: float(v) for k, v in stored_metrics.items()}
                            if lot_bar is not None:
                                lot_bar.update(1)
                            continue

                        train_seq_data = sequences[train_days]
                        train_labels_data = labels_scaled[train_days]
                        test_seq_data = sequences[test_days]
                        test_labels_scaled = labels_scaled[test_days].reshape(-1)
                        test_labels_actual = labels_actual[test_days].reshape(-1)

                        fold_boundaries: List[Tuple[int, int]] = []
                        if enable_fold_validation and folds != 0:
                            fold_boundaries = generate_fold_boundaries(
                                train_seq_data.shape[0],
                                train_ratio=train_ratio,
                                folds=folds,
                                fold_size=fold_size,
                            )
                        if fold_boundaries and tqdm:
                            fold_bar = tqdm(
                                total=len(fold_boundaries),
                                desc=f"{method.name} h{horizon} {lot_id}",
                                leave=False,
                                unit="fold",
                                dynamic_ncols=True,
                            )
                        else:
                            fold_bar = None

                        for fold_idx, (test_start, test_end) in enumerate(fold_boundaries, start=1):
                            fold_train_seq = train_seq_data[:test_start]
                            fold_train_labels = train_labels_data[:test_start]
                            fold_test_seq = train_seq_data[test_start:test_end]
                            if fold_train_seq.shape[0] == 0 or fold_test_seq.shape[0] == 0:
                                continue
                            desc = f"{method.name} h{horizon} {lot_id} fold{fold_idx}"
                            fold_model = train_model(
                                method,
                                fold_train_seq,
                                fold_train_labels,
                                epochs,
                                lr,
                                weight_decay,
                                desc,
                            )
                            predict_model(method, fold_model, fold_test_seq)
                            if fold_bar is not None:
                                fold_bar.update(1)
                        if fold_bar is not None:
                            fold_bar.close()

                        full_model = train_model(
                            method,
                            train_seq_data,
                            train_labels_data,
                            epochs,
                            lr,
                            weight_decay,
                            desc=f"{method.name} h{horizon} {lot_id} full",
                        )
                        preds_norm = predict_model(method, full_model, test_seq_data)
                        accuracy = normalized_accuracy(test_labels_scaled, preds_norm)
                        preds_actual = count_scaler.inverse_transform(preds_norm.reshape(-1, 1)).reshape(-1)
                        actual_metrics = compute_actual_metrics(test_labels_actual.reshape(-1), preds_actual)
                        lot_metrics = {"accuracy": accuracy, **actual_metrics}
                        lot_metrics_store = {k: float(v) for k, v in lot_metrics.items()}
                        horizon_store[lot_id] = lot_metrics_store
                        per_lot_metrics[lot_id] = lot_metrics_store
                        save_checkpoint(checkpoint_path, checkpoint)

                        if lot_bar is not None:
                            lot_bar.update(1)

                finally:
                    if lot_bar is not None:
                        lot_bar.close()

                if not per_lot_metrics:
                    continue

                aggregated: Dict[str, float] = {}
                keys = list(per_lot_metrics[next(iter(per_lot_metrics))].keys())
                for key in keys:
                    aggregated[key] = float(np.mean([metrics[key] for metrics in per_lot_metrics.values()]))

                results.append(
                    {
                        "method": method.name,
                        "horizon": horizon,
                        "metrics": aggregated,
                        "per_lot_metrics": per_lot_metrics,
                    }
                )

                if outer_bar is not None:
                    outer_bar.update(1)

    finally:
        if outer_bar is not None:
            outer_bar.close()

    return results


def plot_grouped_bars(results: List[Dict[str, object]], output_path: str, methods: List[MethodConfig]) -> None:
    if not results:
        return
    horizons = sorted({int(r["horizon"]) for r in results})
    method_names = [m.name for m in methods if any(r["method"] == m.name for r in results)]
    if not method_names:
        return

    acc_matrix = np.zeros((len(method_names), len(horizons)))
    for r in results:
        i = method_names.index(r["method"])
        j = horizons.index(int(r["horizon"]))
        acc_matrix[i, j] = r["metrics"]["accuracy"]

    x = np.arange(len(horizons))
    width = 0.8 / max(1, len(method_names))

    plt.figure(figsize=(10, 6))
    for idx, name in enumerate(method_names):
        offsets = x + (idx - (len(method_names) - 1) / 2) * width
        plt.bar(offsets, acc_matrix[idx], width=width, label=name)

    vals = acc_matrix.flatten()
    if vals.size:
        lower = vals.min()
        margin = max(abs(lower), abs(vals.max())) * 0.1
    else:
        lower, margin = 0, 10
    plt.ylim(lower - margin, 100)

    plt.xticks(x, [f"{h}h" for h in horizons])
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Accuracy (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling evaluation with reserved uniform test set.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results_timeseries")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--methods", nargs="*", help="Subset of methods to evaluate")
    parser.add_argument("--no-fold-validation", action="store_true", help="Disable fold-based validation")
    args = parser.parse_args()

    selected_methods = select_methods(args.methods)
    if not selected_methods:
        print("No methods selected for evaluation.")
        return

    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = args.checkpoint or os.path.join(CHECKPOINT_DIR, "timeseries_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)

    results = evaluate_methods_timeseries(
        methods=selected_methods,
        horizons=[1, 2, 3],
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        folds=0 if args.no_fold_validation else args.folds,
        fold_size=args.fold_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        enable_fold_validation=not args.no_fold_validation,
    )

    summary_path = os.path.join(output_dir, "accuracy_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    plot_path = os.path.join(output_dir, "accuracy_comparison.png")
    plot_grouped_bars(results, plot_path, selected_methods)

    unnorm_summary = []
    header = "Method\tHorizon\tAccuracy(%)\tAccuracy_raw(%)\tRMSE\tMAE\tMAPE(%)"
    print(header)
    lines = [header]
    for record in sorted(results, key=lambda x: (x["method"], x["horizon"])):
        metrics = record["metrics"]
        line = (
            f"{record['method']}\t{record['horizon']}\t"
            f"{metrics['accuracy']:.2f}\t{metrics.get('accuracy_raw', float('nan')):.2f}\t"
            f"{metrics['rmse']:.4f}\t{metrics['mae']:.4f}\t{metrics['mape']:.2f}"
        )
        print(line)
        lines.append(line)
        unnorm_entry = {
            "method": record["method"],
            "horizon": record["horizon"],
            "metrics": {"accuracy_raw": metrics.get("accuracy_raw", float("nan"))},
            "per_lot_metrics": {
                lot: {"accuracy_raw": lot_metrics.get("accuracy_raw", float("nan"))}
                for lot, lot_metrics in record["per_lot_metrics"].items()
            },
        }
        unnorm_summary.append(unnorm_entry)

    text_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(text_path, "w", encoding="utf-8") as txt:
        txt.write("\n".join(lines))
        txt.write("\n")

    unnorm_path = os.path.join(output_dir, "accuracy_unnormalized_summary.json")
    with open(unnorm_path, "w", encoding="utf-8") as fp:
        json.dump(unnorm_summary, fp, indent=2)

    print(f"Saved summary JSON to {summary_path}")
    print(f"Saved metrics table to {text_path}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved unnormalized accuracy summary to {unnorm_path}")


if __name__ == "__main__":
    main()
