#!/usr/bin/env python3
"""Evaluate PewLSTM variants and baselines across multiple horizons."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Ensure local imports work when executed from repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in os.sys.path:
    os.sys.path.append(SCRIPT_DIR)

import run_modified_pewlstm as dataset_mod  # type: ignore
from run_modified_pewlstm import (  # type: ignore
    PARK_TABLE_IDS,
    PARK_WEATHER_IDX,
    calc_park_cnt_from_dict,
    gen_series,
    process_weather,
    read_park_table,
    read_weather_table,
    trans_record_to_count,
)
from modifiedPSTM import pew_LSTM  # type: ignore

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

torch.set_num_threads(max(1, os.cpu_count() // 2))

DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
dataset_mod.RECORD_PATH = os.path.join(DATA_ROOT, "record")
dataset_mod.WEATHER_PATH = os.path.join(DATA_ROOT, "weather")

LOOKBACK = 24
WEATHER_DIM = 4


def load_checkpoint(path: str) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def save_checkpoint(path: str, data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> None:
    if not path:
        return
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp_path, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


COUNT_TARGETS = ("occupancy", "departure", "arrival")


def _build_count_dict(park_book, target: str) -> Dict[int, Dict[str, int]]:
    if target == "occupancy":
        return trans_record_to_count(park_book)

    p_dict: Dict[int, Dict[str, int]] = {}
    for stime, etime in zip(park_book["Lockdown Time"], park_book["Lockup Time"]):
        start_tsp = int(time.mktime(time.strptime(stime, "%Y/%m/%d %H:%M")))
        end_tsp = int(time.mktime(time.strptime(etime, "%Y/%m/%d %H:%M")))
        if end_tsp - start_tsp <= 5 * 60:
            continue
        if target == "departure":
            hour_key = int(end_tsp // (60 * 60))
        elif target == "arrival":
            hour_key = int(start_tsp // (60 * 60))
        else:
            raise ValueError(f"Unsupported target type '{target}'")
        entry = p_dict.setdefault(hour_key, {"cnt": 0})
        entry["cnt"] += 1
    return p_dict


def load_series(lot_index: int, target: str = "occupancy") -> np.ndarray:
    """Load merged weather and parking series for a given lot.

    Args:
        lot_index: Parking lot index.
        target: One of 'occupancy', 'departure', or 'arrival'.
    """
    if target not in COUNT_TARGETS:
        raise ValueError(f"Unknown target '{target}'. Expected one of {COUNT_TARGETS}")

    park_book = read_park_table(lot_index)
    weather_book = read_weather_table(PARK_WEATHER_IDX[lot_index])
    p_dict = _build_count_dict(park_book, target)
    if not p_dict:
        raise ValueError(f"No parking records for {PARK_TABLE_IDS[lot_index]}")
    start_h = min(p_dict.keys())
    end_h = max(p_dict.keys())
    park_cnt = calc_park_cnt_from_dict(p_dict)
    weather_rec = process_weather(weather_book)
    series = gen_series(park_cnt, weather_rec, start_h, end_h)

    # Fill missing values with column means, matching build_dataset logic
    for col in ["tem", "rhu", "wind_s", "pre_ih"]:
        if series[col].isnull().any():
            series[col].fillna(series[col].mean(), inplace=True)
    series["cnt"].fillna(0.0, inplace=True)
    return series.astype("float32").values


def prepare_sequences(
    lot_index: int, horizon: int, train_ratio: float, target: str = "occupancy"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    if horizon < 1:
        raise ValueError("Horizon must be >= 1")

    raw_series = load_series(lot_index, target)
    feature_scaler = MinMaxScaler()
    scaled_series = feature_scaler.fit_transform(raw_series)

    count_scaler = MinMaxScaler()
    count_scaler.fit(raw_series[:, -1].reshape(-1, 1))

    features = scaled_series[:-horizon]
    labels_scaled_flat = scaled_series[horizon:, -1]
    labels_actual_flat = raw_series[horizon:, -1]

    usable = (features.shape[0] // LOOKBACK) * LOOKBACK
    if usable == 0:
        raise ValueError(f"Not enough data for lot {PARK_TABLE_IDS[lot_index]} and horizon {horizon}")

    features = features[:usable]
    labels_scaled_flat = labels_scaled_flat[:usable]
    labels_actual_flat = labels_actual_flat[:usable]

    sequences = features.reshape(-1, LOOKBACK, raw_series.shape[1]).astype("float32")
    labels_scaled = labels_scaled_flat.reshape(-1, LOOKBACK).astype("float32")
    labels_actual = labels_actual_flat.reshape(-1, LOOKBACK).astype("float32")

    num_days = sequences.shape[0]
    train_days = max(1, int(num_days * train_ratio))
    if train_days >= num_days:
        train_days = num_days - 1

    train_seq = sequences[:train_days]
    test_seq = sequences[train_days:]
    train_labels_scaled = labels_scaled[:train_days]
    test_labels_scaled = labels_scaled[train_days:]
    train_labels_actual = labels_actual[:train_days]
    test_labels_actual = labels_actual[train_days:]

    return (
        train_seq,
        train_labels_scaled,
        train_labels_actual,
        test_seq,
        test_labels_scaled,
        test_labels_actual,
        count_scaler,
    )


class PewForecastModel(nn.Module):
    def __init__(self, hidden_dim: int, use_periodic: bool, use_weather: bool):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm1 = pew_LSTM(1, hidden_dim, WEATHER_DIM, use_periodic=use_periodic, use_weather=use_weather)
        self.lstm2 = pew_LSTM(hidden_dim, hidden_dim, WEATHER_DIM, use_periodic=use_periodic, use_weather=use_weather)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weather = inputs[:, :, :-1]
        counts = inputs[:, :, -1].unsqueeze(2)
        h1, _ = self.lstm1(counts, weather)
        h2, _ = self.lstm2(h1, weather)
        out = self.fc(h2.contiguous().view(-1, self.hidden_dim)).view(-1)
        return out


class SimpleLSTMModel(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(inputs)
        out = self.fc(out.contiguous().view(-1, self.hidden_dim)).view(-1)
        return out


def train_torch_model(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    desc: str = "Training",
) -> Tuple[float, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    last_loss = math.inf
    if tqdm:
        epoch_iterator = tqdm(
            range(epochs),
            desc=desc,
            leave=False,
            unit="epoch",
            dynamic_ncols=True,
        )
    else:
        epoch_iterator = range(epochs)

    for epoch in epoch_iterator:
        model.train()
        optimizer.zero_grad()
        outputs = model(train_inputs)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    if tqdm:
        epoch_iterator.close()  # type: ignore[attr-defined]
    return last_loss, last_loss


def predict_torch_model(model: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        outputs = model(inputs).cpu().numpy()
    return outputs


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual = actual.reshape(-1)
    predicted = predicted.reshape(-1)
    diff = predicted - actual
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    denom = np.where(np.abs(actual) < 1e-3, 1e-3, np.abs(actual))
    mape = float(np.mean(np.abs(diff) / denom) * 100.0)
    accuracy = 0.0
    if actual.size > 0 and predicted.size > 0:
        if actual.size > 1 and predicted.size > 1:
            error_sum = 0.0
            count = 0
            for target_val, pred_val in zip(actual[:-1], predicted[1:]):
                count += 1
                if target_val != 0:
                    error_sum += abs(target_val - pred_val) / abs(target_val)
                else:
                    error_sum += abs(target_val - pred_val)
            mean_error = error_sum / count if count else 0.0
        else:
            target_val = actual[0]
            pred_val = predicted[0]
            if target_val != 0:
                mean_error = abs(target_val - pred_val) / abs(target_val)
            else:
                mean_error = abs(target_val - pred_val)
        accuracy = (1.0 - mean_error) * 100.0
    return {
        "accuracy": accuracy,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


@dataclass
class MethodConfig:
    name: str
    kind: str
    kwargs: Dict[str, object]


METHODS: List[MethodConfig] = [
    MethodConfig("PewLSTM", "pew", {"use_periodic": True, "use_weather": True}),
    MethodConfig("Simple LSTM", "simple_lstm", {"hidden_dim": 16}),
    MethodConfig("Regression (Random Forest)", "regression", {"n_estimators": 200, "random_state": 42}),
    MethodConfig("PewLSTM w/o Periodic", "pew", {"use_periodic": False, "use_weather": True}),
    MethodConfig("PewLSTM w/o Weather", "pew", {"use_periodic": True, "use_weather": False}),
]

def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def select_methods(requested: Optional[List[str]]) -> List[MethodConfig]:
    if not requested:
        return METHODS

    alias_map: Dict[str, MethodConfig] = {}
    for method in METHODS:
        normalized = _normalize_name(method.name)
        alias_map.setdefault(normalized, method)

    # additional aliases for convenience
    manual_aliases = {
        "pewlstmfull": "PewLSTM",
        "fullpewlstm": "PewLSTM",
        "simple": "Simple LSTM",
        "simplelstm": "Simple LSTM",
        "regression": "Regression (Random Forest)",
        "rf": "Regression (Random Forest)",
        "randomforest": "Regression (Random Forest)",
        "pewlstmwoperiodic": "PewLSTM w/o Periodic",
        "pewlstmwithoutperiodic": "PewLSTM w/o Periodic",
        "pewlstmwoweather": "PewLSTM w/o Weather",
        "pewlstmwithoutweather": "PewLSTM w/o Weather",
    }
    for alias, target in manual_aliases.items():
        alias_key = _normalize_name(alias)
        target_key = _normalize_name(target)
        if alias_key not in alias_map and target_key in alias_map:
            alias_map[alias_key] = alias_map[target_key]

    selected: List[MethodConfig] = []
    seen = set()
    for name in requested:
        key = _normalize_name(name)
        method = alias_map.get(key)
        if method is None:
            raise ValueError(f"Unknown method '{name}'. Available options: {[m.name for m in METHODS]}")
        if method.name not in seen:
            selected.append(method)
            seen.add(method.name)

    return selected


def evaluate_method_for_lot(
    config: MethodConfig,
    lot_index: int,
    horizon: int,
    train_ratio: float,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, float]:
    (
        train_seq,
        train_labels_scaled,
        train_labels_actual,
        test_seq,
        test_labels_scaled,
        test_labels_actual,
        count_scaler,
    ) = prepare_sequences(lot_index, horizon, train_ratio)

    device = torch.device("cpu")

    if config.kind == "pew":
        train_inputs = torch.from_numpy(train_seq).to(device)
        test_inputs = torch.from_numpy(test_seq).to(device)
        train_targets = torch.from_numpy(train_labels_scaled.reshape(-1)).to(device)

        model = PewForecastModel(
            hidden_dim=int(config.kwargs.get("hidden_dim", 8)),
            use_periodic=bool(config.kwargs.get("use_periodic", True)),
            use_weather=bool(config.kwargs.get("use_weather", True)),
        ).to(device)
        train_torch_model(
            model,
            train_inputs,
            train_targets,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            desc=f"{config.name} h{horizon} {PARK_TABLE_IDS[lot_index]}",
        )
        preds_scaled = predict_torch_model(model, test_inputs)

    elif config.kind == "simple_lstm":
        train_inputs = torch.from_numpy(train_seq[:, :, -1:].copy()).to(device)
        test_inputs = torch.from_numpy(test_seq[:, :, -1:].copy()).to(device)
        train_targets = torch.from_numpy(train_labels_scaled.reshape(-1)).to(device)

        model = SimpleLSTMModel(hidden_dim=int(config.kwargs.get("hidden_dim", 16))).to(device)
        train_torch_model(
            model,
            train_inputs,
            train_targets,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            desc=f"{config.name} h{horizon} {PARK_TABLE_IDS[lot_index]}",
        )
        preds_scaled = predict_torch_model(model, test_inputs)

    elif config.kind == "regression":
        reg = RandomForestRegressor(**config.kwargs)
        X_train = train_seq.reshape(train_seq.shape[0], -1)
        y_train = train_labels_scaled
        X_test = test_seq.reshape(test_seq.shape[0], -1)
        reg.fit(X_train, y_train)
        preds_scaled = reg.predict(X_test).reshape(-1)
    else:
        raise ValueError(f"Unsupported method kind {config.kind}")

    preds_actual = count_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
    actual = test_labels_actual.reshape(-1)
    return compute_metrics(actual, preds_actual)


def evaluate_all_methods(
    methods: List[MethodConfig],
    horizons: Iterable[int],
    train_ratio: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    checkpoint: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    checkpoint_path: str,
) -> List[Dict[str, object]]:
    set_seed(seed)
    horizons_list = list(horizons)
    results: List[Dict[str, object]] = []
    progress_total = len(methods) * len(horizons_list)
    progress_bar = tqdm(total=progress_total, desc="Evaluating methods", unit="run") if (tqdm and progress_total > 0) else None
    try:
        for method in methods:
            for horizon in horizons_list:
                horizon_key = str(horizon)
                method_store = checkpoint.setdefault(method.name, {})
                lot_store = method_store.setdefault(horizon_key, {})
                lot_metrics_map: Dict[str, Dict[str, float]] = {
                    lot_id: {k: float(v) for k, v in metrics.items()}
                    for lot_id, metrics in lot_store.items()
                }
                completed_count = len(lot_metrics_map)
                lot_bar = (
                    tqdm(
                        total=len(PARK_TABLE_IDS),
                        desc=f"{method.name} h{horizon}",
                        leave=False,
                        unit="lot",
                        dynamic_ncols=True,
                        initial=completed_count,
                    )
                    if tqdm
                    else None
                )
                try:
                    for lot_index in range(len(PARK_TABLE_IDS)):
                        lot_id = PARK_TABLE_IDS[lot_index]
                        if lot_id in lot_metrics_map:
                            continue
                        try:
                            metrics = evaluate_method_for_lot(
                                method,
                                lot_index,
                                horizon,
                                train_ratio=train_ratio,
                                epochs=epochs,
                                lr=lr,
                                weight_decay=weight_decay,
                            )
                            lot_metrics_map[lot_id] = {k: float(v) for k, v in metrics.items()}
                            lot_store[lot_id] = lot_metrics_map[lot_id]
                            save_checkpoint(checkpoint_path, checkpoint)
                        except Exception as exc:
                            print(f"[WARN] {method.name} lot {PARK_TABLE_IDS[lot_index]} horizon {horizon}: {exc}")
                        finally:
                            if lot_bar is not None:
                                lot_bar.update(1)
                finally:
                    if lot_bar is not None:
                        lot_bar.close()
                if not lot_metrics_map:
                    continue
                per_metric_lists: Dict[str, List[float]] = {}
                for metrics in lot_metrics_map.values():
                    for key, value in metrics.items():
                        per_metric_lists.setdefault(key, []).append(float(value))
                aggregated = {key: float(np.mean(values)) for key, values in per_metric_lists.items()}
                results.append(
                    {
                        "method": method.name,
                        "horizon": horizon,
                        "metrics": aggregated,
                        "per_lot_metrics": lot_metrics_map,
                    }
                )
                if progress_bar is not None:
                    progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return results


def plot_results(results: List[Dict[str, object]], output_path: str, methods: List[MethodConfig]) -> None:
    if not results:
        return
    horizons = sorted({int(r["horizon"]) for r in results})
    available_methods = [m for m in methods if any(r["method"] == m.name for r in results)]
    if not horizons or not available_methods:
        return

    num_methods = len(available_methods)
    x = np.arange(len(horizons))
    width = 0.8 / max(1, num_methods)

    plt.figure(figsize=(10, 6))
    for idx, method in enumerate(available_methods):
        method_points = {int(r["horizon"]): r for r in results if r["method"] == method.name}
        if not method_points:
            continue
        accuracies = [method_points[h]["metrics"]["accuracy"] for h in horizons if h in method_points]
        offsets = x + (idx - (num_methods - 1) / 2) * width
        plt.bar(offsets[: len(accuracies)], accuracies, width=width, label=method.name)

    abs_values = [r["metrics"]["accuracy"] for r in results]
    if abs_values:
        lower = min(abs_values)
        upper = max(abs_values)
        margin = max(abs(upper), abs(lower)) * 0.1
        plt.ylim(lower - margin, upper + margin)

    plt.xticks(x, [f"{h}h" for h in horizons])
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Accuracy (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate parking prediction methods.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--methods",
        nargs="*",
        help="Subset of methods to evaluate (e.g., 'PewLSTM', 'Simple LSTM'). Defaults to all.",
    )
    args = parser.parse_args()

    selected_methods = select_methods(args.methods)
    if not selected_methods:
        print("No methods selected; exiting without evaluation.")
        return

    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = args.checkpoint or os.path.join(output_dir, "checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)

    results = evaluate_all_methods(
        methods=selected_methods,
        horizons=[1, 2, 3],
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )

    csv_path = os.path.join(output_dir, "accuracy_summary.json")
    with open(csv_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    plot_path = os.path.join(output_dir, "accuracy_comparison.png")
    plot_results(results, plot_path, selected_methods)

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

    print(f"Saved summary JSON to {csv_path}")
    print(f"Saved metrics table to {text_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
