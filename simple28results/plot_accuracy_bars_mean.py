#!/usr/bin/env python3
"""Plot grouped bar chart of accuracy by horizon for each method."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

RESULT_PATH = os.path.join(os.path.dirname(__file__), "accuracy_summary.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "accuracy_comparison.png")

with open(RESULT_PATH, "r", encoding="utf-8") as fp:
    results = json.load(fp)

horizons = sorted({int(r["horizon"]) for r in results})
methods = sorted({r["method"] for r in results})

if not horizons or not methods:
    raise SystemExit("No results found to plot.")

acc_matrix = np.zeros((len(methods), len(horizons)))
for r in results:
    i = methods.index(r["method"])
    j = horizons.index(int(r["horizon"]))
    acc_matrix[i, j] = r["metrics"]["accuracy"]

x = np.arange(len(horizons))
width = 0.8 / max(1, len(methods))

plt.figure(figsize=(10, 6))
for idx, method in enumerate(methods):
    offsets = x + (idx - (len(methods) - 1) / 2) * width
    plt.bar(offsets, acc_matrix[idx], width=width, label=method)

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
plt.savefig(OUTPUT_PATH)
print(f"Saved bar chart to {OUTPUT_PATH}")
