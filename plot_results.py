"""Turn results/<phase>/metrics.json into the figures used in the README.

Usage (from the repo root):
    python plot_results.py            # every phase found under results/
    python plot_results.py numpy      # just one phase

Writes, per phase:
    results/<phase>/curves.png            train/val loss + val accuracy per epoch
    results/<phase>/confusion_matrix.png  test confusion matrix, row-normalized
    results/<phase>/per_class_f1.png      test F1 per class, sorted
and, when both phases exist:
    results/phase_comparison.png          val accuracy, NumPy vs PyTorch

Plotting only: every number comes from metrics.json, nothing is recomputed.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open windows
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

RESULTS = Path("results")
PHASE_TITLES = {
    "numpy": "Phase 1 - NumPy, manual backprop",
    "torch": "Phase 2 - PyTorch, autograd",
}

# Palette (validated categorical slots 1-2 + single-hue sequential ramp)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e6e5e1"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

SHORT_LABELS = {"Money transfer / virtual currency": "Money transfer / virtual curr."}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
})


def short(label):
    return SHORT_LABELS.get(label, label)


def load(phase):
    with open(RESULTS / phase / "metrics.json") as f:
        return json.load(f)


def loss_name(m):
    ls = m.get("config", {}).get("label_smoothing", 0)
    return f"loss (cross-entropy, label smoothing {ls})" if ls else "loss (cross-entropy)"


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)
    ax.plot([x], [y], "o", ms=5, color=color, mec=SURFACE, mew=1.5, zorder=5)


def plot_curves(phase, m, out):
    h = m["history"]
    epochs = np.arange(1, len(h["train_loss"]) + 1)
    best = m.get("best_epoch")
    marker = "o" if len(epochs) <= 3 else None  # a 1-point "line" is invisible otherwise

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(PHASE_TITLES.get(phase, phase), x=0.01, ha="left",
                 fontsize=13, fontweight="bold")

    ax_l.plot(epochs, h["train_loss"], color=SERIES_1, marker=marker, label="train")
    ax_l.plot(epochs, h["val_loss"], color=SERIES_2, marker=marker, label="validation")
    end_label(ax_l, epochs[-1], h["train_loss"][-1], "train", SERIES_1)
    end_label(ax_l, epochs[-1], h["val_loss"][-1], "validation", SERIES_2)
    ax_l.set_title(loss_name(m))
    ax_l.set_xlabel("epoch")
    ax_l.legend(loc="upper right")

    ax_a.plot(epochs, h["val_acc"], color=SERIES_1, marker=marker)
    end_label(ax_a, epochs[-1], h["val_acc"][-1], f"{h['val_acc'][-1]:.1f}%", SERIES_1)
    test_acc = m["test"]["acc"]
    ax_a.axhline(test_acc, color=INK_3, lw=1.2, ls=(0, (4, 3)))
    ax_a.annotate(f"test {test_acc:.1f}% (best checkpoint, evaluated once)",
                  (0.98, test_acc), xycoords=("axes fraction", "data"), ha="right", va="top",
                  xytext=(0, -5), textcoords="offset points", fontsize=9, color=INK_2)
    ax_a.set_title("validation accuracy (%)")
    ax_a.set_xlabel("epoch")

    if best is not None and len(epochs) > 1:
        for ax in (ax_l, ax_a):
            ax.axvline(best, color=INK_3, lw=1, ls=":")
        ax_l.annotate(f"best val loss\n(epoch {best})", (best, 1), xycoords=("data", "axes fraction"),
                      xytext=(4, -4), textcoords="offset points", va="top",
                      fontsize=8.5, color=INK_2)

    for ax in (ax_l, ax_a):
        ax.set_xlim(0.5, epochs[-1] + max(1.5, len(epochs) * 0.18))  # room for end labels
        if len(epochs) <= 10:
            ax.set_xticks(epochs)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_confusion(phase, m, out):
    cm = np.array(m["test"]["confusion_matrix"], dtype=float)
    labels = [short(l) for l in m["labels"]]
    norm = cm / cm.sum(axis=1, keepdims=True)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(10, 8.6))
    im = ax.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(n), labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n), labels)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)  # 2px surface gap between cells
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            if v < 0.03 and i != j:
                continue  # label only the cells that matter
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=8.5,
                    color=SURFACE if v > 0.5 else INK,
                    fontweight="bold" if i == j else "normal")

    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"{PHASE_TITLES.get(phase, phase)} - test confusion matrix\n"
                 f"% of each true class (rows sum to 100; cells under 3% unlabeled)\n"
                 f"accuracy {m['test']['acc']:.1f}%, macro F1 {m['test']['macro_f1']:.3f}",
                 fontsize=10.5, pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, length=0)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_per_class_f1(phase, m, out):
    pc = m["test"]["per_class"]
    rows = sorted(((pc[l]["f1-score"], l, int(pc[l]["support"])) for l in m["labels"]))
    f1s = [r[0] for r in rows]
    names = [f"{short(r[1])}  (n={r[2]})" for r in rows]
    macro = m["test"]["macro_f1"]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    y = np.arange(len(rows))
    ax.set_axisbelow(True)  # grid behind the bars
    ax.barh(y, f1s, height=0.72, color=SERIES_1, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)
    for yi, v in zip(y, f1s):
        ax.annotate(f"{v:.2f}", (v, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)
    ax.axvline(macro, color=INK_3, lw=1.2, ls=(0, (4, 3)))
    ax.annotate(f"macro F1 {macro:.3f}", (macro, len(rows) - 0.4), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color=INK_2, va="bottom")
    ax.set_xlabel("F1 on test split")
    ax.set_title(f"{PHASE_TITLES.get(phase, phase)}\n"
                 f"per-class F1 on test (n = test rows per class)", pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_comparison(runs, out):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = {"numpy": SERIES_1, "torch": SERIES_2}
    for phase, m in runs.items():
        acc = m["history"]["val_acc"]
        ep = np.arange(1, len(acc) + 1)
        name = PHASE_TITLES.get(phase, phase)
        ax.plot(ep, acc, color=colors.get(phase, INK_2), label=name)
        end_label(ax, ep[-1], acc[-1], f"{acc[-1]:.1f}%", colors.get(phase, INK_2))
    n_max = max(len(m["history"]["val_acc"]) for m in runs.values())
    ax.set_xlim(0.5, n_max + max(1.5, n_max * 0.12))
    ax.set_xlabel("epoch")
    ax.set_title("validation accuracy (%) - same architecture, two implementations")
    ax.legend(loc="lower right")

    smoothing = {p: m.get("config", {}).get("label_smoothing", 0) for p, m in runs.items()}
    if len(set(smoothing.values())) > 1:
        note = ", ".join(f"{p}: {s}" for p, s in smoothing.items())
        ax.annotate(f"Losses are not compared: label smoothing differs ({note}).",
                    (0, -0.2), xycoords="axes fraction", fontsize=8.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    phases = sys.argv[1:] or sorted(p.name for p in RESULTS.iterdir()
                                    if (p / "metrics.json").exists())
    if not phases:
        sys.exit("no results/<phase>/metrics.json found - run a training script first")
    runs = {}
    for phase in phases:
        m = load(phase)
        runs[phase] = m
        d = RESULTS / phase
        plot_curves(phase, m, d / "curves.png")
        plot_confusion(phase, m, d / "confusion_matrix.png")
        plot_per_class_f1(phase, m, d / "per_class_f1.png")
        print(f"wrote {d}/curves.png, confusion_matrix.png, per_class_f1.png")
    if {"numpy", "torch"} <= runs.keys():
        plot_comparison({p: runs[p] for p in ("numpy", "torch")},
                        RESULTS / "phase_comparison.png")
        print(f"wrote {RESULTS}/phase_comparison.png")


if __name__ == "__main__":
    main()
