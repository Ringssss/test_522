import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
# =========================
# 1. Basic settings
# =========================

methods = [
    "(A)dInfer w/o cache",
    "(B)dInfer w/ cache",
    "(C)SGLang",
    "(D)Ours",
]

short_labels = ["A", "B", "C", "D"]

cols = [
    "GSM8K",
    "HumanEval",
    "MGSM",
    "MT-Bench",
]

rows = [
    "Batchsize 32",
    "Batchsize 128",
    "Batchsize 256",
    "Batchsize 512",
]

colors = [
    "#b7d3e8",  # A
    "#ffd966",  # B
    "#f4b183",  # C
    "#c6e0b4",  # D (ours)
]

hatches = ["", "", "", "//"]

baseline = 1.0
ymax = 4.0


# =========================
# 2. Load data from JSON
# =========================

data_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "plt",
    "llada20_performance_data.json")

with open(data_path, "r") as f:
    _summary = json.load(f)

_raw_data = _summary["data"]
for r in range(len(_raw_data)):
    for c in range(len(_raw_data[r])):
        for m in range(len(_raw_data[r][c])):
            if _raw_data[r][c][m] is None:
                _raw_data[r][c][m] = float("nan")
data = np.array(_raw_data)

speedup_labels = _summary["speedup_labels"]
oom_marks = [tuple(x) for x in _summary["oom_marks"]]


# =========================
# 3. Labels for clipped bars
# If a value > ymax, the bar is clipped to ymax,
# and the real value is shown vertically.
# =========================

big_labels = {}
for r in range(data.shape[0]):
    for c in range(data.shape[1]):
        for m in range(data.shape[2]):
            v = data[r, c, m]
            if not np.isnan(v) and v > ymax:
                big_labels[(r, c, m)] = f"{v:.1f}×"


# =========================
# 6. Plot
# =========================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 11,
    "axes.linewidth": 1.0,
})

fig, axes = plt.subplots(
    nrows=4,
    ncols=4,
    figsize=(8, 5.8),
    sharey=True
)

bar_width = 0.65
x = np.arange(len(methods))

for r in range(4):
    for c in range(4):
        ax = axes[r, c]

        values = data[r, c]
        clipped_values = np.nan_to_num(np.minimum(values, ymax), nan=0.0)

        for i in range(len(methods)):
            if np.isnan(values[i]):
                continue

            ax.bar(
                x[i],
                clipped_values[i],
                width=bar_width,
                color=colors[i],
                edgecolor="black",
                linewidth=1.0,
                hatch=hatches[i],
                zorder=2
            )

            # Label for clipped bars
            if (r, c, i) in big_labels:
                ax.text(
                    x[i],
                    ymax - 0.15,
                    big_labels[(r, c, i)],
                    ha="center",
                    va="top",
                    rotation=90,
                    fontsize=10
                )

        # OOM label
        for rr, cc, mm in oom_marks:
            if rr == r and cc == c:
                ax.text(
                    x[mm],
                    0.75,
                    "OOM",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=10
                )

        # Baseline
        ax.axhline(
            baseline,
            color="red",
            linestyle="--",
            linewidth=1.5,
            zorder=3
        )

        # Speedup text near ours
        ax.text(
            x[-1] + 0.05,
            baseline + 0.05,
            speedup_labels[r][c],
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

        # Axes style
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, 2, 4])
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=11)

        if c == 0:
            ax.set_ylabel(rows[r], fontsize=11)
        else:
            ax.tick_params(axis="y", labelleft=False)

        if r == 0:
            ax.set_title(cols[c], fontsize=14, pad=5)

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)


# Shared y-axis label
fig.text(
    0.02,
    0.50,
    "Relative Exec. Time",
    rotation=90,
    va="center",
    ha="center",
    fontsize=15
)


# Legend
legend_handles = [
    Patch(
        facecolor=colors[i],
        edgecolor="black",
        hatch=hatches[i],
        label=methods[i]
    )
    for i in range(len(methods))
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=4,
    frameon=True,
    fontsize=13,
    bbox_to_anchor=(0.52, 1.02)
)


plt.subplots_adjust(
    left=0.14,
    right=0.995,
    top=0.84,
    bottom=0.12,
    wspace=0.23,
    hspace=0.50
)

plt_results_dir = "/home/wuhang/wuhang/dllm_wh/codex_coding/results/plt"
plt.savefig(os.path.join(plt_results_dir, "llada20_performance.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(plt_results_dir, "llada20_performance.png"), dpi=600, bbox_inches="tight")
plt.show()
