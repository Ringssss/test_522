import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
# =========================
# 1. Basic settings
# =========================

methods = [
    "(A)TensorFlow",
    "(B)TensorFlow-XLA",
    "(D)Nimble",
    "(E)TVM",
    "(C)TensorRT",
    "(F)PET",
    "(G)EinNet",
]

short_labels = ["A", "B", "C", "D", "E", "F", "G"]

cols = [
    "InfoGAN",
    "DCGAN",
    "FSRCNN",
    "GCN",
    "ResNet-18",
    "CSRNet",
    "Longformer",
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
    "#8dd3c7",  # C/D Nimble
    "#f4b183",  # D/E TVM
    "#b4accd",  # E/C TensorRT
    "#ff7f73",  # F
    "#c6e0b4",  # G
]

hatches = ["", "", "", "", "", "", "//"]

baseline = 1.0
ymax = 4.0


# =========================
# 2. Example data
# Shape: rows x cols x methods
# Replace these values with your own data
# =========================

data = np.array([
    # A100 Batch Size 1
    [
        [45.7, 47.3, 2.8, 3.5, 2.2, 2.2, 1.0],   # InfoGAN
        [30.5, 30.9, 1.6, 1.9, 1.6, 2.3, 1.0],   # DCGAN
        [69.0, 75.7, 2.0, 17.9, 1.7, 17.6, 1.0], # FSRCNN
        [10.2, 11.6, 3.0, 3.2, 3.2, 2.7, 1.0],   # GCN
        [7.3, 8.7, 1.1, 1.4, 1.7, 1.3, 1.0],     # ResNet-18
        [4.8, 6.2, 2.6, 2.6, 2.0, 1.1, 1.0],     # CSRNet
        [27.9, 14.8, np.nan, 1.4, 1.6, 1.6, 1.0] # Longformer
    ],

    # A100 Batch Size 16
    [
        [20.1, 20.6, 4.5, 2.4, 1.5, 2.5, 1.0],
        [11.2, 12.9, 1.6, 1.5, 1.4, 2.0, 1.0],
        [40.8, 44.8, 3.5, 18.9, 1.9, 18.9, 1.0],
        [7.3, 7.5, 1.3, 1.4, 1.1, 1.2, 1.0],
        [2.0, 2.8, 0.95, 0.95, 0.95, 1.1, 1.0],
        [2.3, 2.7, 2.8, 2.7, 2.1, 1.0, 1.0],
        [31.6, 25.6, np.nan, 1.6, 1.5, 1.7, 1.0]
    ],

    # V100 Batch Size 1
    [
        [29.0, 33.0, 3.1, 2.9, 2.4, 2.1, 1.0],
        [23.2, 20.7, 1.7, 1.7, 2.0, 2.3, 1.0],
        [50.2, 51.7, 2.0, 16.9, 1.4, 18.9, 1.0],
        [9.2, 10.6, 2.9, 3.2, 3.1, 2.7, 1.0],
        [7.9, 9.2, 1.4, 1.9, 1.2, 1.5, 1.0],
        [3.8, 4.7, 2.3, 2.4, 1.6, 1.1, 1.0],
        [20.5, 12.6, np.nan, 2.2, 2.3, 2.3, 1.0]
    ],

    # V100 Batch Size 16
    [
        [13.7, 13.3, 4.4, 2.8, 1.4, 2.5, 1.0],
        [4.9, 4.2, 1.2, 1.2, 1.0, 1.4, 1.0],
        [30.3, 28.0, 3.1, 18.2, 1.4, 19.9, 1.0],
        [3.7, 5.6, 1.4, 1.5, 1.2, 1.2, 1.0],
        [1.7, 1.8, 1.0, 1.0, 1.0, 1.1, 1.0],
        [2.1, 1.8, 2.1, 2.1, 1.9, 1.1, 1.0],
        [22.5, 23.5, np.nan, 2.2, 2.3, 2.4, 1.0]
    ]
])


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
# 4. Speedup labels near method G
# Usually shown as text such as 2.2x, 1.6x, etc.
# Replace these according to your paper/result.
# =========================

speedup_labels = [
    ["2.2x", "1.6x", "1.7x", "2.7x", "1.1x", "1.1x", "1.4x"],
    ["1.4x", "1.4x", "1.9x", "1.2x", "1.0x", "1.0x", "1.5x"],
    ["2.1x", "1.7x", "1.4x", "2.7x", "1.2x", "1.1x", "2.2x"],
    ["1.4x", "1.0x", "1.4x", "1.3x", "1.0x", "1.1x", "2.2x"],
]


# =========================
# 5. OOM marks
# Format: (row_index, col_index, method_index)
# Here Longformer method C is OOM in all four rows.
# =========================

oom_marks = [
    (0, 6, 2),
    (1, 6, 2),
    (2, 6, 2),
    (3, 6, 2),
]


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
    ncols=7,
    figsize=(18, 5.8),
    sharey=True
)

bar_width = 0.65
x = np.arange(len(methods))

for r in range(4):
    for c in range(7):
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

        # Speedup text near G
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
    0.055,
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
    ncol=7,
    frameon=True,
    fontsize=13,
    bbox_to_anchor=(0.52, 1.02)
)


plt.subplots_adjust(
    left=0.10,
    right=0.995,
    top=0.84,
    bottom=0.12,
    wspace=0.23,
    hspace=0.50
)

plt_results_dir = "/home/wuhang/wuhang/dllm_wh/codex_coding/results/plt"
plt.savefig(os.path.join(plt_results_dir, "performance_0511.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(plt_results_dir, "performance_0511.png"), dpi=300, bbox_inches="tight")
plt.show()
