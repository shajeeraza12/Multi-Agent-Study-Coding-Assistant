"""
analyze_results.py — ABC Benchmark Analytics
=============================================

Parses the JSON output of swe_bench_runner.py and produces:

  1. Per-condition summary (mean +/- stdev score and latency)
  2. Iteration distribution (% of B/C runs triggering refine 1x/2x/3x)
  3. Per-instance A/B/C comparison table
  4. Condition C forensics (step counts, wall time, status breakdown)
  5. Aggregate JSON written to results/aggregate_<date>.json

Usage (WSL, venv activated):
    python analyze_results.py results/sweep_20260503.json
    python analyze_results.py results/sweep_20260503.json --output results/agg.json
    python analyze_results.py results/sweep_*.json        # merge multiple files
"""

import json
import math
import sys
import os
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stdev(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((x - mean) ** 2 for x in values) / n)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_abc_results(filepaths: List[str]) -> Dict[str, Dict[str, Dict]]:
    """Load and merge one or more swe_bench_runner.py output files.

    Returns:
        { instance_id: { variant: run_dict } }
    """
    merged: Dict[str, Dict[str, Dict]] = {}
    for fp in filepaths:
        if not os.path.isfile(fp):
            print(f"[WARN] File not found: {fp}")
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", {})
        for iid, variants in results.items():
            if iid not in merged:
                merged[iid] = {}
            merged[iid].update(variants)
        n_runs = sum(len(v) for v in results.values())
        print(f"[LOAD] {fp}: {n_runs} runs across {len(results)} instances")
    return merged


# ---------------------------------------------------------------------------
# Condition-level summary
# ---------------------------------------------------------------------------

def condition_summary(results: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
    """Per-condition: mean score, stdev, mean latency, n runs."""
    stats: Dict[str, Dict] = {}
    for variant in ["a", "b", "c"]:
        scores = []
        latencies = []
        for variants in results.values():
            r = variants.get(variant)
            if r:
                scores.append(r.get("relevancy_score", 0.0))
                latencies.append(r.get("latency_total", 0.0))
        if scores:
            stats[variant] = {
                "n": len(scores),
                "mean_score": _mean(scores),
                "stdev_score": _stdev(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "mean_latency_s": _mean(latencies),
                "stdev_latency_s": _stdev(latencies),
            }
    return stats


# ---------------------------------------------------------------------------
# Iteration distribution
# ---------------------------------------------------------------------------

def iteration_distribution(results: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
    """For conditions B and C: how many runs triggered refine 0x, 1x, 2x, 3x?

    iteration_count in ABCMetrics:
      0 = single shot, accepted first attempt
      1 = one refine  (two total LLM calls)
      2 = two refines
      3 = hit the cap (three refines)
    """
    dist: Dict[str, Dict] = {}
    for variant in ["b", "c"]:
        counts: Dict[int, int] = defaultdict(int)
        n = 0
        for variants in results.values():
            r = variants.get(variant)
            if r:
                n += 1
                counts[r.get("iterations", 0)] += 1
        if n:
            dist[variant] = {
                "n": n,
                "distribution": dict(sorted(counts.items())),
                "pct_no_refine":   counts.get(0, 0) / n * 100,
                "pct_refine_1x":   counts.get(1, 0) / n * 100,
                "pct_refine_2x":   counts.get(2, 0) / n * 100,
                "pct_refine_cap":  counts.get(3, 0) / n * 100,
            }
    return dist


# ---------------------------------------------------------------------------
# Per-instance comparison table
# ---------------------------------------------------------------------------

def per_instance_table(results: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    """One row per instance with scores for all three conditions and deltas.
    Sorted by Condition C score descending so outliers are visible.
    """
    rows = []
    for iid, variants in results.items():
        a = variants.get("a", {}).get("relevancy_score")
        b = variants.get("b", {}).get("relevancy_score")
        c = variants.get("c", {}).get("relevancy_score")
        row = {
            "instance_id": iid,
            "score_a": a,
            "score_b": b,
            "score_c": c,
            "delta_b_minus_a": round(b - a, 3) if (b is not None and a is not None) else None,
            "delta_c_minus_a": round(c - a, 3) if (c is not None and a is not None) else None,
            "delta_c_minus_b": round(c - b, 3) if (c is not None and b is not None) else None,
        }
        rows.append(row)
    rows.sort(key=lambda r: (r["score_c"] or -1), reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Condition C forensics
# ---------------------------------------------------------------------------

def condition_c_forensics(results: Dict[str, Dict[str, Dict]]) -> Dict:
    """Condition-C-specific metrics: step counts, wall times, status, finish rate."""
    step_counts = []
    wall_times = []
    status_counts: Dict[str, int] = defaultdict(int)
    reached_finish = 0
    n = 0

    for variants in results.values():
        r = variants.get("c")
        if not r:
            continue
        n += 1
        status = r.get("openhands_status") or "unknown"
        status_counts[status] += 1

        steps = r.get("openhands_step_count")
        if steps is not None:
            step_counts.append(steps)

        wt = r.get("openhands_wall_time_s")
        if wt is not None:
            wall_times.append(wt)

        trace = r.get("openhands_action_trace") or []
        if any(
            "finish" in str(step.get("action_summary", "")).lower() or
            "finish" in str(step.get("tool_name", "")).lower()
            for step in trace
        ):
            reached_finish += 1

    forensics: Dict = {"n": n}
    if step_counts:
        forensics["mean_steps"] = _mean(step_counts)
        forensics["stdev_steps"] = _stdev(step_counts)
        forensics["min_steps"] = min(step_counts)
        forensics["max_steps"] = max(step_counts)
    if wall_times:
        forensics["mean_wall_time_s"] = _mean(wall_times)
        forensics["stdev_wall_time_s"] = _stdev(wall_times)
    if n:
        forensics["status_breakdown"] = dict(status_counts)
        forensics["pct_reached_finish"] = reached_finish / n * 100
    return forensics


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

VARIANT_NAMES = {
    "a": "A — Baseline",
    "b": "B — Reviewer+Refine",
    "c": "C — Hallucination Bridge",
}


def print_summary(cond_stats, iter_dist, table, c_forensics):
    sep  = "=" * 72
    thin = "-" * 72

    print(f"\n{sep}")
    print("  ABC BENCHMARK RESULTS SUMMARY")
    print(sep)

    # 1. Condition summary
    print("\n[1] PER-CONDITION SCORE & LATENCY")
    print(thin)
    print(f"  {'Condition':<28}  {'N':>4}  {'Mean Score':>10}  {'Stdev':>6}  {'Mean Latency':>13}")
    print(thin)
    for v in ["a", "b", "c"]:
        s = cond_stats.get(v)
        if not s:
            print(f"  {VARIANT_NAMES[v]:<28}  (no data)")
            continue
        print(
            f"  {VARIANT_NAMES[v]:<28}  {s['n']:>4}  "
            f"{s['mean_score']:>10.3f}  {s['stdev_score']:>6.3f}  "
            f"{s['mean_latency_s']:>11.1f}s"
        )

    # 2. Iteration distribution
    print(f"\n[2] REFINE-LOOP ITERATION DISTRIBUTION  (B and C only)")
    print(thin)
    for v in ["b", "c"]:
        d = iter_dist.get(v)
        if not d:
            print(f"  Condition {v.upper()}: no data")
            continue
        print(f"  Condition {v.upper()} (n={d['n']}):")
        print(f"    0 refines  (accepted 1st attempt)  : {d['pct_no_refine']:5.1f}%")
        print(f"    1 refine                            : {d['pct_refine_1x']:5.1f}%")
        print(f"    2 refines                           : {d['pct_refine_2x']:5.1f}%")
        print(f"    3 refines (hit cap)                 : {d['pct_refine_cap']:5.1f}%")

    # 3. Per-instance table
    print(f"\n[3] PER-INSTANCE COMPARISON  (sorted by score_c desc)")
    print(thin)
    print(f"  {'Instance ID':<42}  {'A':>5}  {'B':>5}  {'C':>5}  {'B-A':>5}  {'C-A':>5}")
    print(thin)

    def _fmt(v):
        return f"{v:.2f}" if v is not None else "  --"

    def _fmtd(v):
        if v is None:
            return "  --"
        return f"{v:+.2f}"

    for row in table:
        print(
            f"  {row['instance_id']:<42}  {_fmt(row['score_a']):>5}  "
            f"{_fmt(row['score_b']):>5}  {_fmt(row['score_c']):>5}  "
            f"{_fmtd(row['delta_b_minus_a']):>5}  {_fmtd(row['delta_c_minus_a']):>5}"
        )

    # 4. Condition C forensics
    if c_forensics.get("n", 0) > 0:
        print(f"\n[4] CONDITION C — OPENHANDS FORENSICS")
        print(thin)
        cf = c_forensics
        print(f"  Runs analyzed     : {cf['n']}")
        if "mean_steps" in cf:
            print(
                f"  Steps per run     : {cf['mean_steps']:.1f} +/- {cf['stdev_steps']:.1f}"
                f"  (range {cf['min_steps']}-{cf['max_steps']})"
            )
        if "mean_wall_time_s" in cf:
            print(f"  Wall time per run : {cf['mean_wall_time_s']:.0f}s +/- {cf['stdev_wall_time_s']:.0f}s")
        if "pct_reached_finish" in cf:
            print(f"  Reached finish()  : {cf['pct_reached_finish']:.1f}%")
        if "status_breakdown" in cf:
            print(f"  Status breakdown  : {cf['status_breakdown']}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze ABC benchmark results")
    parser.add_argument(
        "input_files", nargs="+", metavar="RESULTS_JSON",
        help="One or more swe_bench_runner.py output JSON files to analyze"
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Path for aggregate JSON output. Defaults to results/aggregate_<date>.json"
    )
    args = parser.parse_args()

    results = load_abc_results(args.input_files)
    if not results:
        print("[ERROR] No results loaded. Check the input file paths.")
        sys.exit(1)

    total_runs = sum(len(v) for v in results.values())
    print(f"\n[INFO] {total_runs} runs across {len(results)} instances loaded")

    cond_stats  = condition_summary(results)
    iter_dist   = iteration_distribution(results)
    table       = per_instance_table(results)
    c_forensics = condition_c_forensics(results)

    print_summary(cond_stats, iter_dist, table, c_forensics)

    out_path = args.output or os.path.join(
        "results", f"aggregate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    aggregate = {
        "timestamp": datetime.now().isoformat(),
        "total_runs": total_runs,
        "total_instances": len(results),
        "condition_summary": cond_stats,
        "iteration_distribution": iter_dist,
        "per_instance_table": table,
        "condition_c_forensics": c_forensics,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)
    print(f"[EXPORT] Aggregate written to: {out_path}")


if __name__ == "__main__":
    main()
