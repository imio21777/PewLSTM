#!/usr/bin/env python3
"""Plot per-parking-lot accuracy for each horizon and method."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def natural_sort_keys(keys: List[str]) -> List[str]:
    def key_func(k: str):
        prefix = ''.join(filter(str.isalpha, k))
        suffix = ''.join(filter(str.isdigit, k))
        return (prefix, int(suffix) if suffix.isdigit() else suffix)
    return sorted(keys, key=key_func)


def load_results(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Summary file must contain a list of records.")
    return data


def plot_per_lot_accuracy(results: List[Dict], title: str, output_path: str) -> None:
    horizons = sorted({int(entry["horizon"]) for entry in results})
    lots = set()
    method_names = set()
    for entry in results:
        method_names.add(entry["method"])
        per_lot = entry.get("per_lot_metrics", {})
        lots.update(per_lot.keys())

    lots = natural_sort_keys(list(lots)) or ["P1"]
    method_names = sorted(method_names)

    num_horizons = len(horizons)
    fig, axes = plt.subplots(num_horizons, 1, figsize=(14, 4 * num_horizons), sharey=True)
    if num_horizons == 1:
        axes = [axes]

    all_acc_values = []
    for entry in results:
        per_lot = entry.get("per_lot_metrics", {})
        for lot_metrics in per_lot.values():
            if "accuracy" in lot_metrics:
                all_acc_values.append(lot_metrics["accuracy"])

    if all_acc_values:
        global_min = min(all_acc_values)
        global_max = max(all_acc_values)
        global_margin = max(abs(global_min), abs(global_max)) * 0.1
    else:
        global_min = 0
        global_margin = 10

    for ax, horizon in zip(axes, horizons):
        x = np.arange(len(lots))
        width = 0.8 / max(1, len(method_names))

        for idx, method in enumerate(method_names):
            method_entries = [entry for entry in results if entry["method"] == method and int(entry["horizon"]) == horizon]
            if not method_entries:
                continue
            per_lot = method_entries[0].get("per_lot_metrics", {})
            accuracies = [per_lot.get(lot, {}).get("accuracy", np.nan) for lot in lots]
            offsets = x + (idx - (len(method_names) - 1) / 2) * width
            ax.bar(offsets, accuracies, width=width, label=method)

        ax.set_title(f"Horizon {horizon}h")
        ax.set_xticks(x)
        ax.set_xticklabels(lots)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(global_min - global_margin, 100)

    axes[-1].set_xlabel("Parking Lot")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(method_names), bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved per-lot accuracy chart to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-parking-lot accuracy from evaluation summary.")
    parser.add_argument("--summary", type=str, default="../results/accuracy_summary.json", help="Path to summary JSON file.")
    parser.add_argument("--output", type=str, default="../results/accuracy_comparison_eachP.png", help="Output image file.")
    parser.add_argument("--title", type=str, default="Per-Parking-Lot Accuracy", help="Figure title.")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    summary_path = args.summary if os.path.isabs(args.summary) else os.path.abspath(os.path.join(base_dir, args.summary))
    output_path = args.output if os.path.isabs(args.output) else os.path.abspath(os.path.join(base_dir, args.output))

    results = load_results(summary_path)
    plot_per_lot_accuracy(results, args.title, output_path)


if __name__ == "__main__":
    main()
