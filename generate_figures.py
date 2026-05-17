"""
ABC Benchmarking MAS — Defence Figure Generator
Produces 6 PNGs (fig1–fig5 + headline) at 300 DPI.
Run from the project root:
    python generate_figures.py
"""

import json, statistics, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
RESULTS = os.path.join(os.path.dirname(__file__), "results")
FIGURES = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES, exist_ok=True)

def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)["results"]

p1    = load("sweep_phase1.json")
t85b  = load("sweep_thresh85_b.json")
t85c  = load("sweep_thresh85_c.json")
jsep_a = load("sweep_judge_sep_a.json")
jsep_b = load("sweep_judge_sep_b.json")

IDS = list(p1.keys())

# Short labels for x/y axes
def short(iid):
    parts = iid.split("__")
    repo = parts[0].split("-")[0]          # e.g. "astropy" or "django"
    issue = parts[-1].split("-")[-1]       # just the issue number
    return f"{repo}-{issue}"

# ── colour palette (tab10, colour-blind-safe, no red-green pairs) ──────────
C_A   = "#1f77b4"   # blue
C_B   = "#ff7f0e"   # orange
C_C   = "#9467bd"   # purple
C_B85 = "#ff7f0e"
C_C85 = "#9467bd"
ALPHA = 0.80

plt.style.use("seaborn-v0_8-whitegrid")
BASE_FONT = 12
plt.rcParams.update({
    "font.size": BASE_FONT,
    "axes.titlesize": BASE_FONT + 1,
    "axes.labelsize": BASE_FONT,
    "xtick.labelsize": BASE_FONT - 1,
    "ytick.labelsize": BASE_FONT - 1,
    "legend.fontsize": BASE_FONT - 1,
    "figure.dpi": 100,
})

DPI = 300

# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Three-condition comparison @ threshold 0.7
# ══════════════════════════════════════════════════════════════════════════
def fig1():
    a_scores = {i: p1[i]["a"]["relevancy_score"] for i in IDS}
    b_scores = {i: p1[i]["b"]["relevancy_score"] for i in IDS}
    c_scores = {i: p1[i]["c"]["relevancy_score"] for i in IDS}

    # Sort by A score ascending
    order = sorted(IDS, key=lambda i: a_scores[i])
    labels = [short(i) for i in order]

    x = np.arange(len(order))
    w = 0.26

    fig, ax = plt.subplots(figsize=(15, 5.5))
    bars_a = ax.bar(x - w, [a_scores[i] for i in order], w, label="Condition A (Baseline)",
                    color=C_A, alpha=ALPHA, zorder=3)
    bars_b = ax.bar(x,     [b_scores[i] for i in order], w, label="Condition B (Reviewer+Refine)",
                    color=C_B, alpha=ALPHA, zorder=3)
    bars_c = ax.bar(x + w, [c_scores[i] for i in order], w, label="Condition C (Hallucination Bridge)",
                    color=C_C, alpha=ALPHA, zorder=3)

    ax.axhline(0.70, color="#555555", linestyle="--", linewidth=1.4, zorder=4, label="Threshold = 0.70")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Relevancy Score")
    ax.set_xlabel("SWE-bench Instance (sorted by Condition A score, ascending)")
    ax.set_title("Fig 1 — Three-Condition Comparison at Acceptance Threshold 0.70")
    ax.legend(loc="upper left", framealpha=0.9)

    # Annotate the outlier (sympy-16792 has A=0.30)
    idx_sym = order.index("sympy__sympy-16792")
    ax.annotate("A=0.30\n(sympy-16792)",
                xy=(idx_sym - w, 0.30), xytext=(idx_sym - w - 2.5, 0.55),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=9.5, color="black")

    fig.tight_layout()
    path = os.path.join(FIGURES, "fig1_three_conditions.png")
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"  Saved {path}")
    return path

# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Threshold sensitivity for Condition B (slope chart)
# ══════════════════════════════════════════════════════════════════════════
def fig2():
    b07  = {i: p1[i]["b"]["relevancy_score"]   for i in IDS}
    b85  = {i: t85b[i]["b"]["relevancy_score"]  for i in IDS}

    fig, ax = plt.subplots(figsize=(7, 7))

    # Thin lines per instance
    for i in IDS:
        col = "#2196F3" if b85[i] >= b07[i] else "#FF5722"
        ax.plot([0, 1], [b07[i], b85[i]], color=col, alpha=0.45, linewidth=1.2)

    # Mean line
    mean07 = statistics.mean(b07.values())
    mean85 = statistics.mean(b85.values())
    ax.plot([0, 1], [mean07, mean85], color="black", linewidth=3.0,
            label=f"Mean  {mean07:.3f} → {mean85:.3f}", zorder=5)

    # Annotate sympy-16792 (largest improvement highlighted in brief)
    sym = "sympy__sympy-16792"
    ax.annotate(f"sympy-16792\n{b07[sym]:.2f} → {b85[sym]:.2f}",
                xy=(1, b85[sym]), xytext=(0.68, b85[sym] - 0.10),
                arrowprops=dict(arrowstyle="->", color="#1a7a1a", lw=1.3),
                fontsize=9.5, color="#1a7a1a",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # Annotate the largest regression: django-10914 (1.00 → 0.90)
    reg = "django__django-10914"
    ax.annotate(f"django-10914\n{b07[reg]:.2f} → {b85[reg]:.2f}  ▼",
                xy=(1, b85[reg]), xytext=(0.60, b85[reg] + 0.04),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3),
                fontsize=9.5, color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    ax.set_xlim(-0.15, 1.35)
    ax.set_ylim(0.6, 1.08)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["B @ threshold 0.70", "B @ threshold 0.85"], fontsize=11)
    ax.set_ylabel("Relevancy Score")
    ax.set_title("Fig 2 — Threshold Sensitivity for Condition B")

    # Legend patches
    up_patch   = mpatches.Patch(color="#2196F3", alpha=0.7, label="Improved or stable")
    down_patch = mpatches.Patch(color="#FF5722", alpha=0.7, label="Regressed")
    mean_line  = plt.Line2D([0], [0], color="black", linewidth=2.5, label=f"Mean  {mean07:.3f} → {mean85:.3f}")
    ax.legend(handles=[up_patch, down_patch, mean_line], loc="lower right", framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(FIGURES, "fig2_threshold_sensitivity_b.png")
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"  Saved {path}")
    return path

# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Reviewer floor enforcement (horizontal dot plot)
# ══════════════════════════════════════════════════════════════════════════
def fig3():
    a_scores  = {i: p1[i]["a"]["relevancy_score"]   for i in IDS}
    b85_scores = {i: t85b[i]["b"]["relevancy_score"] for i in IDS}

    order = sorted(IDS, key=lambda i: a_scores[i])
    labels = [short(i) for i in order]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter([a_scores[i]   for i in order], y, color=C_A, s=70, zorder=4,
               label="Condition A (Baseline)", alpha=0.9)
    ax.scatter([b85_scores[i] for i in order], y, color=C_B, s=70, marker="D", zorder=4,
               label="Condition B @ 0.85", alpha=0.9)

    # Connect with thin lines for easy pairing
    for idx, i in enumerate(order):
        ax.plot([a_scores[i], b85_scores[i]], [idx, idx],
                color="gray", linewidth=0.8, alpha=0.5, zorder=3)

    ax.axvline(0.70, color="#555555", linestyle="--", linewidth=1.4,
               label="Threshold = 0.70", zorder=5)

    # Annotate sympy-16792
    sym_idx = order.index("sympy__sympy-16792")
    ax.annotate("A=0.30 → B=0.96\n(sympy-16792)",
                xy=(a_scores["sympy__sympy-16792"], sym_idx),
                xytext=(0.38, sym_idx + 1.8),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0.0, 1.08)
    ax.set_xlabel("Relevancy Score")
    ax.set_title("Fig 3 — Reviewer Floor Enforcement:\nAll B@0.85 Scores ≥ 0.85 vs. A Scores Spanning 0.30–0.96")
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(FIGURES, "fig3_floor_enforcement.png")
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"  Saved {path}")
    return path

# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Condition C bimodality (overlaid histograms)
# ══════════════════════════════════════════════════════════════════════════
def fig4():
    c07  = [p1[i]["c"]["relevancy_score"]   for i in IDS]
    c85  = [t85c[i]["c"]["relevancy_score"]  for i in IDS]

    bins = np.arange(0.0, 1.1, 0.10)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(c07, bins=bins, color=C_A, alpha=0.60, label=f"C @ threshold 0.70  (mean {statistics.mean(c07):.3f})",
            edgecolor="white", linewidth=0.5, zorder=3)
    ax.hist(c85, bins=bins, color=C_C, alpha=0.60, label=f"C @ threshold 0.85  (mean {statistics.mean(c85):.3f})",
            edgecolor="white", linewidth=0.5, zorder=3)

    ax.axvline(0.85, color="#555555", linestyle="--", linewidth=1.3, label="Threshold = 0.85", zorder=4)

    ax.set_xlabel("Relevancy Score")
    ax.set_ylabel("Number of Instances")
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks(bins)
    ax.set_title("Fig 4 — Condition C Bimodality:\nAggressive Refinement Creates a Low-Score Tail at 0.85 Threshold")
    ax.legend(framealpha=0.9)

    # Annotate the tail region
    ax.annotate("Regression tail\n(≤ 0.40 at C@0.85)",
                xy=(0.15, 2.85), xytext=(0.28, 4.3),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
                fontsize=9.5, color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

    fig.tight_layout()
    path = os.path.join(FIGURES, "fig4_c_bimodality.png")
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"  Saved {path}")
    return path

# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Judge separation amplifies architectural lift
# ══════════════════════════════════════════════════════════════════════════
def fig5():
    # Phase 1 (same model both sides, thresh 0.7)
    p1_a_mean = statistics.mean(p1[i]["a"]["relevancy_score"] for i in IDS)
    p1_b_mean = statistics.mean(p1[i]["b"]["relevancy_score"] for i in IDS)
    # Judge separation (weak generator, strong judge, B@0.85)
    js_a_mean = statistics.mean(jsep_a[i]["a"]["relevancy_score"] for i in IDS)
    js_b_mean = statistics.mean(jsep_b[i]["b"]["relevancy_score"] for i in IDS)

    groups = ["Phase 1\n(same strong model)", "Judge Separation\n(7B gen · 32B judge)"]
    a_vals = [p1_a_mean, js_a_mean]
    b_vals = [p1_b_mean, js_b_mean]
    deltas = [p1_b_mean - p1_a_mean, js_b_mean - js_a_mean]

    x = np.array([0, 1])
    w = 0.28

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars_a = ax.bar(x - w/2, a_vals, w, color=C_A, alpha=ALPHA, label="Condition A", zorder=3)
    bars_b = ax.bar(x + w/2, b_vals, w, color=C_B, alpha=ALPHA, label="Condition B", zorder=3)

    # Value labels on bars
    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    # Delta annotations (bracket-style arrow)
    for xi, delta in zip(x, deltas):
        y_top = max(a_vals[list(x).index(xi)], b_vals[list(x).index(xi)])
        ax.annotate("", xy=(xi + w/2, y_top + 0.03),
                    xytext=(xi - w/2, y_top + 0.03),
                    arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5))
        col = "#1a7a1a" if delta > 0 else "#c0392b"
        sign = "+" if delta >= 0 else ""
        ax.text(xi, y_top + 0.045, f"Δ = {sign}{delta:.3f}",
                ha="center", va="bottom", fontsize=11, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Mean Relevancy Score (all 19 instances)")
    ax.set_title("Fig 5 — Reviewer Adds the Most Value When the Generator Needs It Most:\nΔ = +0.003 (same model) vs. Δ = +0.114 (judge separation)")
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(FIGURES, "fig5_judge_separation.png")
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"  Saved {path}")
    return path

# ══════════════════════════════════════════════════════════════════════════
# HEADLINE — Multipanel (Fig1 top, Fig3 middle, Fig5 bottom)
# ══════════════════════════════════════════════════════════════════════════
def headline():
    from matplotlib.gridspec import GridSpec

    # ── data ──
    a_scores   = {i: p1[i]["a"]["relevancy_score"]    for i in IDS}
    b_scores   = {i: p1[i]["b"]["relevancy_score"]    for i in IDS}
    c_scores   = {i: p1[i]["c"]["relevancy_score"]    for i in IDS}
    b85_scores = {i: t85b[i]["b"]["relevancy_score"]  for i in IDS}

    order = sorted(IDS, key=lambda i: a_scores[i])
    labels = [short(i) for i in order]
    x_bar = np.arange(len(order))
    w = 0.26

    fig = plt.figure(figsize=(18, 14))
    gs  = GridSpec(3, 2, figure=fig,
                   height_ratios=[1, 1, 1],
                   width_ratios=[2, 1],
                   hspace=0.55, wspace=0.30)

    # ── Row 0: Fig 1 (spans both columns) ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(x_bar - w, [a_scores[i]   for i in order], w, color=C_A, alpha=ALPHA, label="A", zorder=3)
    ax1.bar(x_bar,     [b_scores[i]   for i in order], w, color=C_B, alpha=ALPHA, label="B", zorder=3)
    ax1.bar(x_bar + w, [c_scores[i]   for i in order], w, color=C_C, alpha=ALPHA, label="C", zorder=3)
    ax1.axhline(0.70, color="#555555", linestyle="--", linewidth=1.3, label="Threshold 0.70")
    ax1.set_ylim(0, 1.12)
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax1.set_ylabel("Relevancy Score")
    ax1.set_title("All Three Conditions @ Threshold 0.70", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    sym_idx = order.index("sympy__sympy-16792")
    ax1.annotate("A=0.30", xy=(sym_idx - w, 0.30), xytext=(sym_idx - w - 2, 0.52),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1),
                 fontsize=8.5, color="black")

    # ── Row 1: Fig 3 (left) ──
    ax3 = fig.add_subplot(gs[1, 0])
    y_dot = np.arange(len(order))
    ax3.scatter([a_scores[i]   for i in order], y_dot, color=C_A, s=55, zorder=4, label="A")
    ax3.scatter([b85_scores[i] for i in order], y_dot, color=C_B, s=55, marker="D", zorder=4, label="B@0.85")
    for idx, i in enumerate(order):
        ax3.plot([a_scores[i], b85_scores[i]], [idx, idx], color="gray", linewidth=0.7, alpha=0.5)
    ax3.axvline(0.70, color="#555555", linestyle="--", linewidth=1.2, label="0.70")
    ax3.set_yticks(y_dot)
    ax3.set_yticklabels(labels, fontsize=7.5)
    ax3.set_xlim(0.0, 1.08)
    ax3.set_xlabel("Relevancy Score")
    ax3.set_title("Reviewer Floor Enforcement", fontsize=12, fontweight="bold")
    ax3.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax3.annotate("A=0.30 → B=0.96",
                 xy=(a_scores["sympy__sympy-16792"], sym_idx),
                 xytext=(0.35, sym_idx + 1.8),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1),
                 fontsize=8, color="black",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # ── Row 1: Fig 4 (right) — bimodality ──
    ax4 = fig.add_subplot(gs[1, 1])
    c07  = [p1[i]["c"]["relevancy_score"]   for i in IDS]
    c85  = [t85c[i]["c"]["relevancy_score"]  for i in IDS]
    bins = np.arange(0.0, 1.1, 0.10)
    ax4.hist(c07, bins=bins, color=C_A, alpha=0.60, label="C@0.70", edgecolor="white")
    ax4.hist(c85, bins=bins, color=C_C, alpha=0.60, label="C@0.85", edgecolor="white")
    ax4.axvline(0.85, color="#555555", linestyle="--", linewidth=1.2)
    ax4.set_xlabel("Relevancy Score")
    ax4.set_ylabel("Count")
    ax4.set_title("Condition C Bimodality", fontsize=12, fontweight="bold")
    ax4.legend(framealpha=0.9, fontsize=9)

    # ── Row 2: Fig 5 (spans both columns) ──
    ax5 = fig.add_subplot(gs[2, :])
    p1_a_mean = statistics.mean(p1[i]["a"]["relevancy_score"] for i in IDS)
    p1_b_mean = statistics.mean(p1[i]["b"]["relevancy_score"] for i in IDS)
    js_a_mean = statistics.mean(jsep_a[i]["a"]["relevancy_score"] for i in IDS)
    js_b_mean = statistics.mean(jsep_b[i]["b"]["relevancy_score"] for i in IDS)
    xg = np.array([0.5, 1.5])
    wg = 0.28
    ba = ax5.bar(xg - wg/2, [p1_a_mean, js_a_mean], wg, color=C_A, alpha=ALPHA, label="Condition A", zorder=3)
    bb = ax5.bar(xg + wg/2, [p1_b_mean, js_b_mean], wg, color=C_B, alpha=ALPHA, label="Condition B", zorder=3)
    for bar in list(ba) + list(bb):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    for xi, a_v, b_v in zip(xg, [p1_a_mean, js_a_mean], [p1_b_mean, js_b_mean]):
        delta = b_v - a_v
        y_top = max(a_v, b_v)
        ax5.annotate("", xy=(xi + wg/2, y_top + 0.025),
                     xytext=(xi - wg/2, y_top + 0.025),
                     arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5))
        col = "#1a7a1a" if delta > 0 else "#c0392b"
        sign = "+" if delta >= 0 else ""
        ax5.text(xi, y_top + 0.038, f"Δ = {sign}{delta:.3f}",
                 ha="center", va="bottom", fontsize=11, color=col, fontweight="bold")
    ax5.set_xticks(xg)
    ax5.set_xticklabels(["Phase 1 (same strong model)", "Judge Separation (7B gen · 32B judge)"], fontsize=11)
    ax5.set_ylim(0.55, 1.0)
    ax5.set_ylabel("Mean Relevancy Score")
    ax5.set_title("Reviewer Adds Most Value When Generator Needs It Most", fontsize=12, fontweight="bold")
    ax5.legend(loc="lower right", framealpha=0.9)

    fig.suptitle("ABC Benchmarking MAS — Thesis Defence Headline Summary",
                 fontsize=14, fontweight="bold", y=0.995)

    path = os.path.join(FIGURES, "headline.png")
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path

# ══════════════════════════════════════════════════════════════════════════
# Markdown summary table
# ══════════════════════════════════════════════════════════════════════════
def print_markdown_table():
    import statistics as st

    def stats(scores):
        return st.mean(scores), st.stdev(scores), min(scores), max(scores)

    rows = {}
    rows["A @ 0.70"]     = [p1[i]["a"]["relevancy_score"]   for i in IDS]
    rows["B @ 0.70"]     = [p1[i]["b"]["relevancy_score"]   for i in IDS]
    rows["B @ 0.85"]     = [t85b[i]["b"]["relevancy_score"] for i in IDS]
    rows["C @ 0.70"]     = [p1[i]["c"]["relevancy_score"]   for i in IDS]
    rows["C @ 0.85"]     = [t85c[i]["c"]["relevancy_score"] for i in IDS]
    rows["jA (7B gen)"]  = [jsep_a[i]["a"]["relevancy_score"] for i in IDS]
    rows["jB (7B gen)"]  = [jsep_b[i]["b"]["relevancy_score"] for i in IDS]

    lats = {
        "A @ 0.70": [p1[i]["a"]["latency_total"] for i in IDS],
        "B @ 0.70": [p1[i]["b"]["latency_total"] for i in IDS],
        "C @ 0.70": [p1[i]["c"]["latency_total"] for i in IDS],
    }

    print("\n\n## Per-Condition Headline Numbers (paste into slides)\n")
    header = "| Condition | Mean ± Stdev | Min | Max | Mean Latency (s) |"
    sep    = "|-----------|-------------|-----|-----|-----------------|"
    print(header)
    print(sep)
    for name, scores in rows.items():
        m, s, mn, mx = stats(scores)
        lat_key = name if name in lats else None
        lat_str = f"{st.mean(lats[lat_key]):.1f}" if lat_key else "—"
        print(f"| {name:<14} | {m:.3f} ± {s:.3f} | {mn:.3f} | {mx:.3f} | {lat_str} |")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating defence figures …")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    headline()
    print_markdown_table()
    print("\nDone. All PNGs saved to ./figures/")
