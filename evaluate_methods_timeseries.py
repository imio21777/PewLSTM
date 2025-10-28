#!/usr/bin/env python3
"""Rolling and multi-fold time-series evaluation for parking prediction methods."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from evaluate_methods import (  # type: ignore
    LOOKBACK,
    METHODS,
    MethodConfig,
    PARK_TABLE_IDS,
    PewForecastModel,
    SimpleLSTMModel,
    compute_metrics,
    load_series,
    select_methods,
    set_seed,
    train_torch_model,
)

torch.set_num_threads(max(1, os.cpu_count() // 2))


def get_full_sequences(lot_index: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    raw_series = load_series(lot_index)

    feature_scaler = MinMaxScaler()
    scaled_series = feature_scaler.fit_transform(raw_series)

    count_scaler = MinMaxScaler()
    count_scaler.fit(raw_series[:, -1].reshape(-1, 1))

    features = scaled_series[:-horizon]
    labels_scaled = scaled_series[horizon:, -1]
    labels_actual = raw_series[horizon:, -1]

    usable = (features.shape[0] // LOOKBACK) * LOOKBACK
    if usable == 0:
        raise ValueError(f"Not enough data for parking lot {PARK_TABLE_IDS[lot_index]} with horizon {horizon}")

    features = features[:usable]
    labels_scaled = labels_scaled[:usable]
    labels_actual = labels_actual[:usable]

    sequences = features.reshape(-1, LOOKBACK, raw_series.shape[1]).astype("float32")
    labels_scaled_seq = labels_scaled.reshape(-1, LOOKBACK).astype("float32")
    labels_actual_seq = labels_actual.reshape(-1, LOOKBACK).astype("float32")
    return sequences, labels_scaled_seq, labels_actual_seq, count_scaler


def generate_fold_boundaries(num_days: int, train_ratio: float, folds: int, fold_size: int | None) -> List[Tuple[int, int]]:
    if num_days < 2:
        return []
    base_train_days = max(1, int(num_days * train_ratio))
    if base_train_days >= num_days:
        base_train_days = num_days - 1
    remaining = num_days - base_train_days
    if remaining <= 0:
        return []

    if fold_size and fold_size > 0:
        step = fold_size
    else:
        step = math.ceil(remaining / max(1, folds))
        step = max(1, step)

    boundaries: List[Tuple[int, int]] = []
    start = base_train_days
    while start < num_days:
        end = min(num_days, start + step)
        if start >= end:
            break
        boundaries.append((start, end))
        start = end
    return boundaries


def train_and_predict(
    method: MethodConfig,
    train_seq: np.ndarray,
    train_labels_scaled: np.ndarray,
    test_seq: np.ndarray,
    count_scaler: MinMaxScaler,
    epochs: int,
    lr: float,
    weight_decay: float,
    desc_prefix: str,
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device("cpu")

    if method.kind == "pew":
        train_inputs = torch.from_numpy(train_seq).to(device)
        train_targets = torch.from_numpy(train_labels_scaled.reshape(-1)).to(device)
        test_inputs = torch.from_numpy(test_seq).to(device)

        model = PewForecastModel(
            hidden_dim=int(method.kwargs.get("hidden_dim", 8)),
            use_periodic=bool(method.kwargs.get("use_periodic", True)),
            use_weather=bool(method.kwargs.get("use_weather", True)),
        ).to(device)
        train_torch_model(
            model,
            train_inputs,
            train_targets,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            desc=desc_prefix,
        )
        model.eval()
        with torch.no_grad():
            preds_scaled = model(test_inputs).cpu().numpy()

    elif method.kind == "simple_lstm":
        train_inputs = torch.from_numpy(train_seq[:, :, -1:].copy()).to(device)
        train_targets = torch.from_numpy(train_labels_scaled.reshape(-1)).to(device)
        test_inputs = torch.from_numpy(test_seq[:, :, -1:].copy()).to(device)

        model = SimpleLSTMModel(hidden_dim=int(method.kwargs.get("hidden_dim", 16))).to(device)
        train_torch_model(
            model,
            train_inputs,
            train_targets,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            desc=desc_prefix,
        )
        model.eval()
        with torch.no_grad():
            preds_scaled = model(test_inputs).cpu().numpy()

    elif method.kind == "regression":
        reg = RandomForestRegressor(**method.kwargs)
        X_train = train_seq.reshape(train_seq.shape[0], -1)
        y_train = train_labels_scaled.reshape(train_labels_scaled.shape[0], -1)
        reg.fit(X_train, y_train)
        X_test = test_seq.reshape(test_seq.shape[0], -1)
        preds_scaled = reg.predict(X_test).reshape(-1)

    else:
        raise ValueError(f"Unsupported method kind {method.kind}")

    preds_actual = count_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
    return preds_actual, preds_actual  # second value unused (placeholder for compatibility)


def evaluate_methods_timeseries(
    methods: List[MethodConfig],
    horizons: Iterable[int],
    train_ratio: float,
    folds: int,
    fold_size: int | None,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
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
                metric_accumulator: Dict[str, List[float]] = {}

                lot_bar = (
                    tqdm(
                        total=len(PARK_TABLE_IDS),
                        desc=f"{method.name} h{horizon}",
                        leave=False,
                        unit="lot",
                        dynamic_ncols=True,
                    )
                    if tqdm
                    else None
                )

                try:
                    for lot_index, lot_id in enumerate(PARK_TABLE_IDS):
                        try:
                            sequences, labels_scaled, labels_actual, count_scaler = get_full_sequences(lot_index, horizon)
                        except Exception as exc:
                            print(f"[WARN] {method.name} {lot_id} h{horizon}: {exc}")
                            continue

                        fold_boundaries = generate_fold_boundaries(
                            sequences.shape[0], train_ratio=train_ratio, folds=folds, fold_size=fold_size
                        )
                        if not fold_boundaries:
                            print(f"[WARN] {method.name} {lot_id} h{horizon}: insufficient data for folds")
                            continue

                        fold_preds: List[np.ndarray] = []
                        fold_actuals: List[np.ndarray] = []

                        fold_bar = (
                            tqdm(
                                total=len(fold_boundaries),
                                desc=f"{method.name} h{horizon} {lot_id}",
                                leave=False,
                                unit="fold",
                                dynamic_ncols=True,
                            )
                            if tqdm
                            else None
                        )

                        for fold_idx, (test_start, test_end) in enumerate(fold_boundaries, start=1):
                            train_seq = sequences[:test_start]
                            train_labels_scaled = labels_scaled[:test_start]
                            test_seq = sequences[test_start:test_end]
                            test_labels_actual = labels_actual[test_start:test_end]

                            if train_seq.shape[0] == 0 or test_seq.shape[0] == 0:
                                continue

                            if method.kind == "regression":
                                preds_actual, _ = train_and_predict(
                                    method,
                                    train_seq,
                                    train_labels_scaled,
                                    test_seq,
                                    count_scaler,
                                    epochs,
                                    lr,
                                    weight_decay,
                                    desc_prefix="",
                                )
                            else:
                                desc = f"{method.name} h{horizon} {lot_id} fold{fold_idx}"
                                preds_actual, _ = train_and_predict(
                                    method,
                                    train_seq,
                                    train_labels_scaled,
                                    test_seq,
                                    count_scaler,
                                    epochs,
                                    lr,
                                    weight_decay,
                                    desc_prefix=desc,
                                )

                            actual_flat = test_labels_actual.reshape(-1)
                            fold_preds.append(preds_actual.reshape(-1))
                            fold_actuals.append(actual_flat)

                            if fold_bar is not None:
                                fold_bar.update(1)

                        if fold_bar is not None:
                            fold_bar.close()

                        if not fold_preds:
                            continue

                        preds_concat = np.concatenate(fold_preds)
                        actual_concat = np.concatenate(fold_actuals)
                        lot_metrics = compute_metrics(actual_concat, preds_concat)
                        per_lot_metrics[lot_id] = lot_metrics
                        for key, value in lot_metrics.items():
                            metric_accumulator.setdefault(key, []).append(float(value))

                        if lot_bar is not None:
                            lot_bar.update(1)

                finally:
                    if lot_bar is not None:
                        lot_bar.close()

                if not metric_accumulator:
                    continue

                aggregated = {key: float(np.mean(values)) for key, values in metric_accumulator.items()}
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

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for idx, name in enumerate(method_names):
        offsets = x + (idx - (len(method_names) - 1) / 2) * width
        plt.bar(offsets, acc_matrix[idx], width=width, label=name)

    all_vals = acc_matrix.flatten()
    if all_vals.size:
        lower = all_vals.min()
        margin = max(abs(lower), abs(lower)) * 0.1
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
    parser = argparse.ArgumentParser(description="Rolling and multi-fold evaluation for parking prediction models.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--folds", type=int, default=3, help="Number of rolling folds.")
    parser.add_argument("--fold-size", type=int, default=None, help="Days per fold (optional override).")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results_timeseries")
    parser.add_argument(
        "--methods",
        nargs="*",
        help="Subset of methods to evaluate (e.g., 'PewLSTM', 'Simple LSTM'). Defaults to all.",
    )
    args = parser.parse_args()

    selected_methods = select_methods(args.methods)
    if not selected_methods:
        print("No methods selected for evaluation.")
        return

    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    results = evaluate_methods_timeseries(
        methods=selected_methods,
        horizons=[1, 2, 3],
        train_ratio=args.train_ratio,
        folds=args.folds,
        fold_size=args.fold_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    summary_path = os.path.join(output_dir, "accuracy_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    plot_path = os.path.join(output_dir, "accuracy_comparison.png")
    plot_grouped_bars(results, plot_path, selected_methods)

    sorted_records = sorted(results, key=lambda x: (x["method"], x["horizon"]))
    header = "Method\tHorizon\tAccuracy(%)\tRMSE\tMAE\tMAPE(%)"
    print(header)
    lines = [header]
    for record in sorted_records:
        metrics = record["metrics"]
        line = (
            f"{record['method']}\t{record['horizon']}\t"
            f"{metrics['accuracy']:.2f}\t{metrics['rmse']:.4f}\t"
            f"{metrics['mae']:.4f}\t{metrics['mape']:.2f}"
        )
        print(line)
        lines.append(line)

    text_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(text_path, "w", encoding="utf-8") as txt:
        txt.write("\n".join(lines))
        txt.write("\n")

    print(f"Saved summary JSON to {summary_path}")
    print(f"Saved metrics table to {text_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()

