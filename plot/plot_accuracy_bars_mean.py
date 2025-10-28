#!/usr/bin/env python3
"""Plot mean accuracy by horizon aggregated across parking lots."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Summary file must contain a list of records.")
    return data


def plot_mean_accuracy(results: List[Dict], title: str, output_path: str) -> None:
    horizons = sorted({int(entry["horizon"]) for entry in results})
    method_names = sorted({entry["method"] for entry in results})
    if not horizons or not method_names:
        raise ValueError("No valid results to plot.")

    acc_matrix = np.zeros((len(method_names), len(horizons)))
    for entry in results:
        method_idx = method_names.index(entry["method"])
        horizon_idx = horizons.index(int(entry["horizon"]))
        acc_matrix[method_idx, horizon_idx] = entry["metrics"].get("accuracy", float("nan"))

    x = np.arange(len(horizons))
    width = 0.8 / max(1, len(method_names))

    plt.figure(figsize=(10, 6))
    for idx, method in enumerate(method_names):
        offsets = x + (idx - (len(method_names) - 1) / 2) * width
        plt.bar(offsets, acc_matrix[idx], width=width, label=method)

    all_vals = acc_matrix[~np.isnan(acc_matrix)]
    if all_vals.size:
        lower = all_vals.min()
        margin = max(abs(lower), abs(all_vals.max())) * 0.1
    else:
        lower, margin = 0, 10

    plt.ylim(lower - margin, 100)
    plt.xticks(x, [f"{h}h" for h in horizons])
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Accuracy (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved mean accuracy chart to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean accuracy per horizon.")
    parser.add_argument("--summary", type=str, default="../results/accuracy_summary.json", help="Path to summary JSON file.")
    parser.add_argument("--output", type=str, default="../results/accuracy_comparison_mean.png", help="Output image file.")
    parser.add_argument("--title", type=str, default="Mean Accuracy", help="Figure title.")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    summary_path = args.summary if os.path.isabs(args.summary) else os.path.abspath(os.path.join(base_dir, args.summary))
    output_path = args.output if os.path.isabs(args.output) else os.path.abspath(os.path.join(base_dir, args.output))

    results = load_results(summary_path)
    plot_mean_accuracy(results, args.title, output_path)


if __name__ == "__main__":
    main()
