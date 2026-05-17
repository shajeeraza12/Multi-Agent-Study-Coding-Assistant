"""
ABC Benchmarking MAS — Simplified Defence Figures (v2)
Five clean, audience-friendly plots saved as PNG 300 DPI.

Run from the project root:
    python generate_figures_v2.py
"""

import json, statistics, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
RESULTS = os.path.join(os.path.dirname(__file__), "results")
FIGURES = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES, exist_ok=True)

def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)["results"]

p1     = load("sweep_phase1.json")
t85b   = load("sweep_thresh85_b.json")
t85c   = load("sweep_thresh85_c.json")
jsep_a = load("sweep_judge_sep_a.json")
jsep_b = load("sweep_judge_sep_b.json")

IDS = list(p1.keys())

def short(iid):
    """e.g. 'sympy__sympy-16792'  →  'sympy-16792' """
    return iid.split("__")[-1]          # everything after the double underscore

# ── palette (colour-blind safe, no red-green pairs) ────────────────────────
C_A   = "#1f77b4"   # blue
C_B   = "#ff7f0e"   # orange
C_C   = "#9467bd"   # purple
GREY  = "#aaaaaa"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})
DPI = 300

# ══════════════════════════════════════════════════════════════════════════
# PLOT A — Score distributions: box plot (all conditions at a glance)
# ══════════════════════════════════════════════════════════════════════════
def plot_boxplot():
    """Five box plots side-by-side — the single clearest overview."""

    groups = {
        "A\n(Baseline)":  [p1[i]["a"]["relevancy_score"]    for i in IDS],
        "B @ 0.70\n(Reviewer)":  [p1[i]["b"]["relevancy_score"]    for i in IDS],
        "B @ 0.85\n(Reviewer)":  [t85b[i]["b"]["relevancy_score"]  for i in IDS],
        "C @ 0.70\n(Action)":    [p1[i]["c"]["relevancy_score"]    for i in IDS],
        "C @ 0.85\n(Action)":    [t85c[i]["c"]["relevancy_score"]  for i in IDS],
    }
    colors = [C_A, C_B, C_B, C_C, C_C]
    labels = list(groups.keys())
    data   = list(groups.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2.0),
                    whiskerprops=dict(linewidth=1.3),
                    capprops=dict(linewidth=1.3),
                    flierprops=dict(marker="o", markersize=5, alpha=0.6))

    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)

    # Scatter individual points over each box
    for xi, vals in enumerate(data, start=1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter([xi + j for j in jitter], vals,
                   color=colors[xi - 1], alpha=0.55, s=30, zorder=4)

    ax.axhline(0.85, color="#333333", linestyle="--", linewidth=1.3,
               label="Threshold 0.85", zorder=3)
    ax.axhline(0.70, color=GREY, linestyle=":", linewidth=1.2,
               label="Threshold 0.70", zorder=3)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Relevancy Score")
    ax.set_title("Score Distributions Across All Conditions\n"
                 "Reviewer (B) tightens the spread; Action Agent (C@0.85) shows a dangerous low tail")

    # Annotate means
    for xi, vals in enumerate(data, start=1):
        m = statistics.mean(vals)
        ax.text(xi, m + 0.025, f"μ={m:.2f}",
                ha="center", va="bottom", fontsize=9.5,
                color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, ec="none"))

    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(FIGURES, "v2_A_distributions_boxplot.png")
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"  Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT B — Per-instance lift: how much did the Reviewer improve each case?
# ══════════════════════════════════════════════════════════════════════════
def plot_lift_bars():
    """
    Horizontal bar chart: delta = B@0.85 - A for each instance.
    Green = Reviewer helped, orange = Reviewer didn't add much / regressed.
    Sorted by delta descending so the story reads top to bottom.
    """
    deltas = {i: t85b[i]["b"]["relevancy_score"] - p1[i]["a"]["relevancy_score"]
              for i in IDS}
    order  = sorted(IDS, key=lambda i: deltas[i], reverse=True)
    labels = [short(i) for i in order]
    vals   = [deltas[i] for i in order]
    colors = ["#2196F3" if v > 0 else "#FF5722" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 8))

    bars = ax.barh(range(len(order)), vals, color=colors, alpha=0.80,
                   edgecolor="white", height=0.65, zorder=3)

    # Value labels
    for idx, (v, bar) in enumerate(zip(vals, bars)):
        ha  = "left"  if v >= 0 else "right"
        xoff = 0.005 if v >= 0 else -0.005
        ax.text(v + xoff, idx, f"{v:+.2f}",
                va="center", ha=ha, fontsize=9.5, color="black")

    ax.axvline(0, color="black", linewidth=1.2, zorder=4)

    # Annotate the headline case
    sym_idx = order.index("sympy__sympy-16792")
    ax.annotate("  ← 0.30 → 0.96\n     biggest single lift",
                xy=(vals[sym_idx], sym_idx),
                xytext=(vals[sym_idx] + 0.05, sym_idx - 0.7),
                fontsize=9, color="#1a7a1a",
                arrowprops=dict(arrowstyle="->", color="#1a7a1a", lw=1.2))

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Relevancy Score Change  (B@0.85 − Condition A)")
    ax.set_title("Per-Instance Lift: How Much Did the Reviewer Agent Help?\n"
                 "Positive (blue) = improved; Negative (orange) = regressed")

    # Legend patches
    up   = mpatches.Patch(color="#2196F3", alpha=0.8, label="Reviewer improved the score")
    down = mpatches.Patch(color="#FF5722", alpha=0.8, label="Score regressed")
    ax.legend(handles=[up, down], loc="lower right", framealpha=0.9)

    ax.set_xlim(-0.45, 0.80)
    fig.tight_layout()
    path = os.path.join(FIGURES, "v2_B_per_instance_lift.png")
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"  Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT C — CDF: floor enforcement at a glance
# ══════════════════════════════════════════════════════════════════════════
def plot_cdf():
    """
    Empirical CDFs for A, B@0.7, B@0.85.
    The B@0.85 line starts at x=0.85 — the floor is unmistakable.
    """
    def ecdf(data):
        xs = sorted(data)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        # Prepend a zero so the CDF starts from y=0
        xs = [xs[0] - 0.001] + xs
        ys = np.concatenate([[0], ys])
        return xs, ys

    a_scores   = [p1[i]["a"]["relevancy_score"]    for i in IDS]
    b07_scores = [p1[i]["b"]["relevancy_score"]    for i in IDS]
    b85_scores = [t85b[i]["b"]["relevancy_score"]  for i in IDS]

    fig, ax = plt.subplots(figsize=(8, 6))

    xs, ys = ecdf(a_scores)
    ax.step(xs, ys, where="post", color=C_A, linewidth=2.2,
            label=f"Condition A  (min={min(a_scores):.2f}, μ={statistics.mean(a_scores):.3f})")

    xs, ys = ecdf(b07_scores)
    ax.step(xs, ys, where="post", color=C_B, linewidth=2.2, linestyle="--",
            label=f"Condition B @ 0.70  (min={min(b07_scores):.2f}, μ={statistics.mean(b07_scores):.3f})")

    xs, ys = ecdf(b85_scores)
    ax.step(xs, ys, where="post", color="#e67e00", linewidth=2.5,
            label=f"Condition B @ 0.85  (min={min(b85_scores):.2f}, μ={statistics.mean(b85_scores):.3f})")

    # Shade the "danger zone" below 0.70
    ax.axvspan(0.0, 0.70, alpha=0.07, color="red", zorder=0)
    ax.text(0.35, 0.96, "Danger zone\n(score < 0.70)", ha="center",
            fontsize=9.5, color="#c0392b", style="italic")

    # Reference lines
    ax.axvline(0.70, color="#888888", linestyle=":", linewidth=1.3)
    ax.axvline(0.85, color="#555555", linestyle="--", linewidth=1.3)
    ax.text(0.71, 0.04, "0.70\nthreshold", fontsize=9, color="#888888")
    ax.text(0.86, 0.04, "0.85\nthreshold", fontsize=9, color="#555555")

    # Annotate the A minimum
    ax.annotate("A bottoms out at 0.30\n(sympy-16792)",
                xy=(0.30, 0.05), xytext=(0.08, 0.28),
                arrowprops=dict(arrowstyle="->", color=C_A, lw=1.2),
                fontsize=9.5, color=C_A,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    # Annotate B@0.85 floor
    ax.annotate("B@0.85 floor: every instance ≥ 0.85",
                xy=(0.85, 0.02), xytext=(0.50, 0.15),
                arrowprops=dict(arrowstyle="->", color="#e67e00", lw=1.2),
                fontsize=9.5, color="#e67e00",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Relevancy Score")
    ax.set_ylabel("Cumulative Fraction of Instances")
    ax.set_title("CDF Comparison: Reviewer Agent Enforces a Hard Score Floor\n"
                 "B@0.85 curve starts at x=0.85 — no instance falls below threshold")
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(FIGURES, "v2_C_cdf_floor.png")
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"  Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT D — A vs B@0.85 scatter: did the Reviewer help?
# ══════════════════════════════════════════════════════════════════════════
def plot_scatter():
    """
    Each dot = one SWE-bench instance.
    x = Condition A score, y = Condition B@0.85 score.
    Above the diagonal  →  Reviewer improved the result.
    Below the diagonal  →  Reviewer hurt.
    """
    a_scores   = {i: p1[i]["a"]["relevancy_score"]    for i in IDS}
    b85_scores = {i: t85b[i]["b"]["relevancy_score"]  for i in IDS}

    xs = [a_scores[i]   for i in IDS]
    ys = [b85_scores[i] for i in IDS]

    # Colour by improvement
    colors = ["#2196F3" if b85_scores[i] > a_scores[i] else
              ("#FF5722" if b85_scores[i] < a_scores[i] else "#888888")
              for i in IDS]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Shading
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.06, color="#2196F3",
                    label="Reviewer improved the score", zorder=0)
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.06, color="#FF5722",
                    label="Reviewer hurt the score", zorder=0)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="#888888", linewidth=1.5,
            linestyle="--", label="No change", zorder=2)

    ax.scatter(xs, ys, c=colors, s=90, zorder=5, alpha=0.85, edgecolors="white", linewidth=0.5)

    # Label notable points
    highlights = {
        "sympy__sympy-16792": ("sympy-16792\nA=0.30 → B=0.96", "left", "#1a7a1a"),
        "astropy__astropy-14182": ("astropy-14182\nA=0.88 → B=0.94", "right", "#1a7a1a"),
        "django__django-10914":   ("django-10914\nA=0.90 → B=0.90", "right", "#888888"),
    }
    for iid, (label, ha, col) in highlights.items():
        xi, yi = a_scores[iid], b85_scores[iid]
        xoff = 0.03 if ha == "left" else -0.03
        ax.annotate(label,
                    xy=(xi, yi),
                    xytext=(xi + xoff, yi + 0.04),
                    fontsize=8.5, color=col, ha=ha,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    ax.set_xlim(0.20, 1.05)
    ax.set_ylim(0.80, 1.05)
    ax.set_xlabel("Condition A Score (Baseline — no Reviewer)")
    ax.set_ylabel("Condition B Score @ Threshold 0.85 (with Reviewer)")
    ax.set_title("Reviewer Agent Impact: A vs B@0.85 per Instance\n"
                 "Points above the diagonal = Reviewer added value")

    # Custom legend
    up   = mpatches.Patch(color="#2196F3", alpha=0.7, label="Reviewer improved")
    down = mpatches.Patch(color="#FF5722", alpha=0.7, label="Reviewer regressed")
    diag = plt.Line2D([0], [0], color="#888888", linestyle="--", label="No change")
    ax.legend(handles=[up, down, diag], loc="upper left", framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(FIGURES, "v2_D_scatter_a_vs_b85.png")
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"  Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT E — Judge separation (clean, unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════
def plot_judge_separation():
    p1_a_mean = statistics.mean(p1[i]["a"]["relevancy_score"] for i in IDS)
    p1_b_mean = statistics.mean(p1[i]["b"]["relevancy_score"] for i in IDS)
    js_a_mean = statistics.mean(jsep_a[i]["a"]["relevancy_score"] for i in IDS)
    js_b_mean = statistics.mean(jsep_b[i]["b"]["relevancy_score"] for i in IDS)

    x  = np.array([0.5, 1.7])
    w  = 0.30

    fig, ax = plt.subplots(figsize=(8, 6))

    ba = ax.bar(x - w/2, [p1_a_mean, js_a_mean], w,
                color=C_A, alpha=0.80, label="Condition A (no Reviewer)", zorder=3)
    bb = ax.bar(x + w/2, [p1_b_mean, js_b_mean], w,
                color=C_B, alpha=0.80, label="Condition B (with Reviewer)", zorder=3)

    # Value labels on bars
    for bar in list(ba) + list(bb):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.008,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Delta brackets
    for xi, a_v, b_v in zip(x, [p1_a_mean, js_a_mean], [p1_b_mean, js_b_mean]):
        delta = b_v - a_v
        y_top = max(a_v, b_v)
        ax.annotate("", xy=(xi + w/2, y_top + 0.025),
                    xytext=(xi - w/2, y_top + 0.025),
                    arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.6))
        col  = "#1a7a1a" if delta > 0 else "#c0392b"
        sign = "+" if delta >= 0 else ""
        ax.text(xi, y_top + 0.040,
                f"Δ = {sign}{delta:.3f}",
                ha="center", va="bottom", fontsize=12,
                color=col, fontweight="bold")

    # Context labels below x axis
    ax.set_xticks(x)
    ax.set_xticklabels([
        "Phase 1\n(same strong model both sides)\nqwen3-32B generator + judge",
        "Judge Separation\n(weak generator, strong judge)\nqwen2.5-7B gen · qwen3-32B judge"
    ], fontsize=10.5)

    ax.set_ylim(0.55, 1.05)
    ax.set_ylabel("Mean Relevancy Score  (all 19 instances)")
    ax.set_title("Reviewer Agent Adds the Most Value When the Generator Needs It Most\n"
                 "Δ = +0.003 with a strong generator  vs.  Δ = +0.114 with a weak generator")
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(FIGURES, "v2_E_judge_separation.png")
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"  Saved {path}")

# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating simplified defence figures (v2) …")
    plot_boxplot()
    plot_lift_bars()
    plot_cdf()
    plot_scatter()
    plot_judge_separation()
    print("\nDone — all PNGs saved to ./figures/")
