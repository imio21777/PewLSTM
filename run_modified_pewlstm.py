#!/usr/bin/env python3
"""Custom training script for PewLSTM using modifiedPSTM.pew_LSTM."""

import argparse
import math
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn

from modifiedPSTM import pew_LSTM

try:
    from tqdm.auto import trange
except ImportError:
    trange = None


HIDDEN_DIM = 1
SEQ_SIZE = 24
RECORD_PATH = "./data/record"
WEATHER_PATH = "./data/weather"
PARK_TABLE_IDS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]
PARK_WEATHER_IDX = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]


def read_park_table(index: int) -> pd.DataFrame:
    file_path = os.path.join(RECORD_PATH, f"{PARK_TABLE_IDS[index]}.csv")
    return pd.read_csv(file_path, encoding="ISO-8859-1")


def read_weather_table(index: int) -> pd.DataFrame:
    file_path = os.path.join(WEATHER_PATH, f"{index}.csv")
    return pd.read_csv(file_path, encoding="ISO-8859-1")


def trans_record_to_count(data: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    p_dict: Dict[int, Dict[str, int]] = {}
    for stime, etime in zip(data["Lockdown Time"], data["Lockup Time"]):
        start_tss = time.strptime(stime, "%Y/%m/%d %H:%M")
        end_tss = time.strptime(etime, "%Y/%m/%d %H:%M")
        start_tsp = int(time.mktime(start_tss))
        end_tsp = int(time.mktime(end_tss))
        if end_tsp - start_tsp <= 5 * 60:
            continue
        start_hour = int(start_tsp // (60 * 60))
        end_hour = int(end_tsp // (60 * 60))
        for hour in range(start_hour, end_hour + 1):
            if hour not in p_dict:
                p_dict[hour] = {"cnt": 1}
            else:
                p_dict[hour]["cnt"] += 1
    return p_dict


def calc_park_cnt_from_dict(p_dict: Dict[int, Dict[str, int]]) -> List[int]:
    if not p_dict:
        return []
    start_hour = min(p_dict.keys())
    end_hour = max(p_dict.keys())
    sequence: List[int] = []
    for hour in range(start_hour, end_hour + 1):
        if hour in p_dict:
            sequence.append(p_dict[hour]["cnt"])
        else:
            sequence.append(0)
    return sequence


def process_weather(data: pd.DataFrame) -> List:
    output: List = []
    start_h = int(time.mktime(time.strptime(data["DAY"][0], "%Y/%m/%d %H:%M")) // (60 * 60))
    output.append(start_h)
    for _ in range(5):
        output.append([])
    output.append({})
    for idx in range(len(data["HOUR"])):
        output[1].append(data["TEM"][idx])
        output[2].append(data["RHU"][idx])
        output[3].append(data["WIN_S"][idx])
        output[4].append(data["PRE_1h"][idx])
        output[5].append(time.strptime(data["DAY"][idx], "%Y/%m/%d %H:%M").tm_wday)
        hour_key = int(time.mktime(time.strptime(data["DAY"][idx], "%Y/%m/%d %H:%M")) // (60 * 60))
        output[6][hour_key] = idx
    return output


def is_valid(weather_list: Sequence, idx: int) -> List[int]:
    flags = [1, 1, 1, 1]
    for offset in range(1, 5):
        if weather_list[offset][idx] > 999:
            flags[offset - 1] = 0
    return flags


def valid_weather(weather_list: Sequence, idx: int) -> List[float]:
    flags = is_valid(weather_list, idx)
    values = [0.0, 0.0, 0.0, 0.0]
    for offset in range(1, 5):
        if flags[offset - 1] == 0:
            left = idx - 1
            while left >= 0 and is_valid(weather_list, left)[offset - 1] == 0:
                left -= 1
            right = idx + 1
            while right < len(weather_list[offset]) and is_valid(weather_list, right)[offset - 1] == 0:
                right += 1
            up_value = weather_list[offset][left] if left >= 0 else weather_list[offset][idx]
            down_value = weather_list[offset][right] if right < len(weather_list[offset]) else weather_list[offset][idx]
            values[offset - 1] = 0.5 * (up_value + down_value)
        else:
            values[offset - 1] = weather_list[offset][idx]
    return values


def gen_series(park_cnt: Sequence[int], weather_rec: Sequence, start_hour: int, end_hour: int) -> pd.DataFrame:
    hours = [start_hour + idx for idx in range(len(park_cnt))]
    containers = [[] for _ in range(5)]
    for idx, hour_key in enumerate(hours):
        if hour_key in weather_rec[6]:
            w_idx = weather_rec[6][hour_key]
            containers[0].append(park_cnt[idx])
            if any(weather_rec[offset][w_idx] > 999 for offset in range(1, 5)):
                values = valid_weather(weather_rec, w_idx)
                containers[1].append(values[0])
                containers[2].append(values[1])
                containers[3].append(values[2])
                containers[4].append(values[3])
            else:
                containers[1].append(weather_rec[1][w_idx])
                containers[2].append(weather_rec[2][w_idx])
                containers[3].append(weather_rec[3][w_idx])
                containers[4].append(weather_rec[4][w_idx])
    park_series = pd.Series(containers[0], name="cnt")
    tem = pd.Series(containers[1], name="tem")
    rhu = pd.Series(containers[2], name="rhu")
    wind = pd.Series(containers[3], name="wind_s")
    pre = pd.Series(containers[4], name="pre_ih")
    return pd.concat([tem, rhu, wind, pre, park_series], axis=1)


def series_to_supervised(data: Iterable, n_in: int = 1, n_out: int = 1, dropnan: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(data)
    columns, names = [], []
    for i in range(n_in, 0, -1):
        columns.append(df.shift(i))
        names += [f"var{j + 1}(t-{i})" for j in range(df.shape[1])]
    for i in range(0, n_out):
        columns.append(df.shift(-i))
        if i == 0:
            names += [f"var{j + 1}(t)" for j in range(df.shape[1])]
        else:
            names += [f"var{j + 1}(t+{i})" for j in range(df.shape[1])]
    agg = pd.concat(columns, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def build_dataset(lot_index: int) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    park_table = read_park_table(lot_index)
    weather_table = read_weather_table(PARK_WEATHER_IDX[lot_index])
    p_dict = trans_record_to_count(park_table)
    park_counts = calc_park_cnt_from_dict(p_dict)
    if not park_counts:
        raise ValueError(f"No parking events found for {PARK_TABLE_IDS[lot_index]}")
    start_hour = min(p_dict.keys())
    end_hour = max(p_dict.keys())
    weather_rec = process_weather(weather_table)
    series = gen_series(park_counts, weather_rec, start_hour, end_hour)
    series = series.astype("float32")
    scaler_full = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler_full.fit_transform(series)
    reframed = series_to_supervised(scaled, 1, 1)
    reframed.drop(reframed.columns[[5, 6, 7, 8]], axis=1, inplace=True)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    raw_reframed = series_to_supervised(series, 1, 1)
    raw_reframed.drop(raw_reframed.columns[[5, 6, 7, 8]], axis=1, inplace=True)
    target_values = raw_reframed.values[:, -1].reshape(-1, 1)
    target_scaler.fit(target_values)

    features = reframed.values[:, :-1].astype("float32")
    labels = reframed.values[:, -1].astype("float32")
    return features, labels, target_scaler


class PewSequenceModel(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.lstm1 = pew_LSTM(1, hidden_dim, 4)
        self.lstm2 = pew_LSTM(hidden_dim, hidden_dim, 4)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weather = x[:, :, :-1]
        parking = x[:, :, -1].unsqueeze(2)
        h1, _ = self.lstm1(parking, weather)
        h2, _ = self.lstm2(h1, weather)
        out = h2.contiguous().view(-1, HIDDEN_DIM)
        return self.fc(out).view(-1)


@dataclass
class TrainResult:
    lot_id: str
    rmse: float
    mae: float
    mape: float
    epochs: int
    train_loss: float
    test_loss: float
    samples: int


def reshape_sequences(features: np.ndarray, labels: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    usable = (features.shape[0] // SEQ_SIZE) * SEQ_SIZE
    features = features[:usable]
    labels = labels[:usable]
    sequences = torch.from_numpy(features).reshape(-1, SEQ_SIZE, features.shape[1])
    targets = torch.from_numpy(labels)
    return sequences, targets


def split_sequences(sequences: torch.Tensor, targets: torch.Tensor, train_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    days = sequences.size(0)
    if days < 2:
        raise ValueError("Not enough daily sequences to perform a split.")
    split_idx = max(1, int(days * train_ratio))
    if split_idx >= days:
        split_idx = days - 1
    train_seq = sequences[:split_idx]
    test_seq = sequences[split_idx:]
    train_targets = targets[:split_idx * SEQ_SIZE]
    test_targets = targets[split_idx * SEQ_SIZE:]
    return train_seq, train_targets, test_seq, test_targets


def train_model(
    lot_id: str,
    train_seq: torch.Tensor,
    train_targets: torch.Tensor,
    test_seq: torch.Tensor,
    test_targets: torch.Tensor,
    target_scaler: MinMaxScaler,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cpu")
    train_seq = train_seq.to(device)
    train_targets = train_targets.to(device)
    test_seq = test_seq.to(device)
    test_targets = test_targets.to(device)

    model = PewSequenceModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    last_train_loss = math.nan
    last_test_loss = math.nan

    iterator = trange(epochs, desc=f"Training {lot_id}") if trange else range(epochs)
    for epoch in iterator:
        model.train()
        optimizer.zero_grad()
        outputs = model(train_seq)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()
        last_train_loss = loss.item()

        model.eval()
        with torch.no_grad():
            test_outputs = model(test_seq)
            last_test_loss = criterion(test_outputs, test_targets).item()

    model.eval()
    with torch.no_grad():
        preds = model(test_seq).cpu().numpy().reshape(-1, 1)
    y_true = test_targets.cpu().numpy().reshape(-1, 1)

    original_preds = target_scaler.inverse_transform(preds)
    original_true = target_scaler.inverse_transform(y_true)

    diff = original_preds - original_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    denominator = np.clip(np.abs(original_true), 1e-3, None)
    mape = float(np.mean(np.abs(diff) / denominator) * 100.0)

    return TrainResult(
        lot_id=lot_id,
        rmse=rmse,
        mae=mae,
        mape=mape,
        epochs=epochs,
        train_loss=float(last_train_loss),
        test_loss=float(last_test_loss),
        samples=test_seq.size(0) * SEQ_SIZE,
    )


def run_experiment(
    lot_indices: Sequence[int],
    train_ratio: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> List[TrainResult]:
    results: List[TrainResult] = []
    for idx in lot_indices:
        features, labels, scaler = build_dataset(idx)
        sequences, targets = reshape_sequences(features, labels)
        train_seq, train_targets, test_seq, test_targets = split_sequences(sequences, targets, train_ratio)
        result = train_model(
            lot_id=PARK_TABLE_IDS[idx],
            train_seq=train_seq,
            train_targets=train_targets,
            test_seq=test_seq,
            test_targets=test_targets,
            target_scaler=scaler,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
        )
        results.append(result)
    return results


def write_results_markdown(
    results: Sequence[TrainResult],
    lot_indices: Sequence[int],
    args: argparse.Namespace,
    output_path: str = "res.md",
) -> None:
    is_new_file = not os.path.exists(output_path)
    selected_lots = [PARK_TABLE_IDS[idx] for idx in lot_indices]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    if is_new_file:
        lines.append("# PewLSTM Results")
        lines.append("")
    lines.append(f"## Run {timestamp}")
    lines.append("")
    lines.append(f"- Lots: {', '.join(selected_lots)}")
    lines.append(f"- Train ratio: {args.train_ratio}")
    lines.append(f"- Epochs: {args.epochs}")
    lines.append(f"- Learning rate: {args.lr}")
    lines.append(f"- Weight decay: {args.weight_decay}")
    lines.append(f"- Seed: {args.seed}")
    lines.append("")
    lines.append("| Lot | RMSE | MAE | MAPE (%) | Train Loss | Test Loss | Samples |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for res in results:
        lines.append(
            f"| {res.lot_id} | {res.rmse:.4f} | {res.mae:.4f} | {res.mape:.2f} | "
            f"{res.train_loss:.6f} | {res.test_loss:.6f} | {res.samples} |"
        )
    lines.append("")
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PewLSTM on available parking data with custom split.")
    parser.add_argument(
        "--lots",
        nargs="*",
        default=[str(idx + 1) for idx in range(len(PARK_TABLE_IDS))],
        help="Parking lot numbers to include (e.g., 1 3 5). Defaults to all lots.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of daily sequences used for training.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for Adam.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lot_indices: List[int] = []
    for lot in args.lots:
        if lot.startswith("P"):
            lot = lot[1:]
        if not lot.isdigit():
            raise ValueError(f"Invalid lot identifier: {lot}")
        value = int(lot) - 1
        if value < 0 or value >= len(PARK_TABLE_IDS):
            raise ValueError(f"Lot index out of range: {lot}")
        lot_indices.append(value)
    results = run_experiment(
        lot_indices=lot_indices,
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    print("Lot\tRMSE\tMAE\tMAPE(%)\tTrainLoss\tTestLoss\tSamples")
    for res in results:
        print(
            f"{res.lot_id}\t{res.rmse:.4f}\t{res.mae:.4f}\t{res.mape:.2f}\t"
            f"{res.train_loss:.6f}\t{res.test_loss:.6f}\t{res.samples}"
        )
    write_results_markdown(results, lot_indices, args)


if __name__ == "__main__":
    main()
