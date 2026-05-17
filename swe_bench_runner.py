"""
SWE-bench Runner with ABC Variant Support
======================================

Runs SWE-bench benchmarks on all three graph variants:
- app_a: Baseline (no reviewer)
- app_b: + Reviewer (current MAS)
- app_c: + OpenHands code helper

Usage:
    python swe_bench_runner.py --variant b --output results_b.json
    python swe_bench_runner.py --variant all --output results_abc.json

    # Targeted sweep with resume-on-crash:
    python swe_bench_runner.py --swe-lite --variant all \
        --instance-ids django__django-11099,django__django-11583 \
        --output results/sweep_$(date +%Y%m%d).json --resume

    # Load instance IDs from sweep_config.json:
    python swe_bench_runner.py --swe-lite --variant all \
        --instance-ids @sweep_config.json \
        --output results/sweep_$(date +%Y%m%d).json --resume
"""

import json
import time
import argparse
import os
import sys
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from graph_variants import app_a, app_b, app_c
from evaluate_swe import (
    SWEBenchInstance,
    load_instances,
    create_initial_state,
    VRAMManager,
    load_swe_bench_lite,
)

print(f"[INFO] Loaded apps: app_a={type(app_a).__name__}, app_b={type(app_b).__name__}, app_c={type(app_c).__name__}")


@dataclass
class ABCMetrics:
    """Metrics for ABC benchmark comparison."""
    instance_id: str
    variant: str
    success: bool
    relevancy_score: float
    latency_total: float
    iterations: int
    code_answer: str
    timestamp: str
    # OpenHands forensic data — only populated for variant='c' with sdk/cloud backend.
    openhands_status: Optional[str] = None
    openhands_backend: Optional[str] = None
    openhands_step_count: Optional[int] = None
    openhands_wall_time_s: Optional[float] = None
    openhands_action_trace: Optional[List[Dict[str, Any]]] = None
    # Per-iteration Reviewer scores — Phase 4 instrumentation.
    # first_iter_score is the Reviewer's score on iteration 1, *before* any refinement.
    # For Condition B this is the Condition-A-equivalent score (single attempt, no refine).
    # all_iter_scores is the full sequence across the refine trajectory.
    first_iter_score: Optional[float] = None
    all_iter_scores: Optional[List[float]] = None


VARIANT_APPS = {
    "a": ("Baseline", app_a),
    "b": ("MAS + Reviewer", app_b),
    "c": ("OpenHands + Reviewer", app_c),
}


def run_single_variant(instance: SWEBenchInstance, variant: str, max_steps: int = 10) -> ABCMetrics:
    """Run evaluation for a single variant."""
    name, app = VARIANT_APPS[variant]

    print(f"\n{'='*60}")
    print(f"Variant {variant.upper()}: {name}")
    print(f"Instance: {instance.instance_id}")
    print('='*60)

    state = create_initial_state(instance, condition=variant)
    config = {"recursion_limit": max_steps}

    final_answer = ""
    relevancy_score = 0.0
    iterations = 0
    oh_status = None
    oh_backend = None
    oh_step_count = None
    oh_wall_time = None
    oh_trace = None
    final_state: Dict[str, Any] = {}  # captured for per-iteration score extraction below

    start_time = time.time()

    try:
        for step in app.stream(state, config=config):
            node_name = list(step.keys())[0]
            node_state = step[node_name]
            final_state.update(node_state)  # keep accumulating; relevancy_checks uses operator.add reducer

            if node_name == "code_helper":
                final_answer = node_state.get("code_answer", "")
                relevancy_score = node_state.get("relevancy_score", 0.0)
                iterations = node_state.get("iteration_count", 0)

                if node_state.get("openhands_status") is not None:
                    oh_status = node_state.get("openhands_status")
                if node_state.get("openhands_backend") is not None:
                    oh_backend = node_state.get("openhands_backend")
                if node_state.get("openhands_step_count") is not None:
                    oh_step_count = node_state.get("openhands_step_count")
                if node_state.get("openhands_wall_time_s") is not None:
                    oh_wall_time = node_state.get("openhands_wall_time_s")
                if node_state.get("openhands_action_trace") is not None:
                    oh_trace = node_state.get("openhands_action_trace")

            print(f"[{node_name}] completed")
            VRAMManager.sleep_with_countdown(3)

    except Exception as e:
        print(f"[ERROR] {e}")
        final_answer = f"Error: {str(e)}"

    total_latency = time.time() - start_time
    success = relevancy_score >= 0.7 if variant != "a" else True
    truncated_answer = final_answer[:2000] if final_answer else ""

    # Phase 4: extract per-iteration Reviewer scores so a single B run yields
    # both an A-equivalent (iter 1, pre-refine) and a B-final data point.
    code_checks = [c for c in (final_state.get("relevancy_checks") or [])
                   if c.get("agent_name") == "code_helper"]
    first_iter_score = code_checks[0].get("confidence") if code_checks else None
    all_iter_scores = [c.get("confidence") for c in code_checks] if code_checks else None

    return ABCMetrics(
        instance_id=instance.instance_id,
        variant=variant,
        success=success,
        relevancy_score=relevancy_score,
        latency_total=total_latency,
        iterations=iterations,
        code_answer=truncated_answer,
        timestamp=datetime.now().isoformat(),
        openhands_status=oh_status,
        openhands_backend=oh_backend,
        openhands_step_count=oh_step_count,
        openhands_wall_time_s=oh_wall_time,
        openhands_action_trace=oh_trace,
        first_iter_score=first_iter_score,
        all_iter_scores=all_iter_scores,
    )


# ---------------------------------------------------------------------------
# Incremental (crash-safe) output writer
# ---------------------------------------------------------------------------

def _write_results_incremental(results_by_instance: Dict[str, Dict], filepath: str) -> None:
    """Atomically write the accumulated results dict to filepath.

    Uses a .tmp sidecar + os.replace() so a crash mid-write never corrupts
    the existing output file.  Safe to call after every (instance, variant) run.
    """
    all_runs: List[Dict] = []
    for variants in results_by_instance.values():
        all_runs.extend(variants.values())

    output: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(all_runs),
        "results": results_by_instance,
    }

    for variant in ["a", "b", "c"]:
        vr = [r for r in all_runs if r.get("variant") == variant]
        if vr:
            scores = [r["relevancy_score"] for r in vr]
            latencies = [r["latency_total"] for r in vr]
            output[f"summary_{variant}"] = {
                "avg_score": sum(scores) / len(scores),
                "avg_latency": sum(latencies) / len(latencies),
                "total_runs": len(vr),
            }

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp = filepath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    os.replace(tmp, filepath)
    print(f"[CHECKPOINT] {filepath} updated ({len(all_runs)} runs total)")


# ---------------------------------------------------------------------------
# Main sweep loop (with resume support)
# ---------------------------------------------------------------------------

def run_abc_comparison(
    instances: List[SWEBenchInstance],
    variants: List[str] = None,
    max_instances: Optional[int] = None,
    completed_pairs: Optional[set] = None,
    existing_results: Optional[Dict[str, Dict]] = None,
    output_filepath: Optional[str] = None,
) -> List[ABCMetrics]:
    """Run ABC comparison across all variants with optional resume support."""
    if variants is None:
        variants = ["a", "b", "c"]
    if max_instances:
        instances = instances[:max_instances]

    completed_pairs = completed_pairs or set()
    merged: Dict[str, Dict] = {k: dict(v) for k, v in (existing_results or {}).items()}
    new_results: List[ABCMetrics] = []

    for instance in instances:
        print(f"\n{'#'*60}")
        print(f"# INSTANCE: {instance.instance_id}")
        print(f"{'#'*60}")

        for variant in variants:
            if (instance.instance_id, variant) in completed_pairs:
                print(f"[SKIP] {instance.instance_id} variant={variant} — already in output")
                continue

            result = run_single_variant(instance, variant)
            new_results.append(result)

            if instance.instance_id not in merged:
                merged[instance.instance_id] = {}
            merged[instance.instance_id][variant] = asdict(result)

            if output_filepath:
                _write_results_incremental(merged, output_filepath)

            time.sleep(2)

    return new_results


# ---------------------------------------------------------------------------
# Legacy export (backward compat)
# ---------------------------------------------------------------------------

def export_abc_results(results: List[ABCMetrics], filepath: str) -> Dict:
    """Export a fresh results list to JSON (non-incremental, end-of-run)."""
    by_instance: Dict[str, Dict] = {}
    for r in results:
        if r.instance_id not in by_instance:
            by_instance[r.instance_id] = {}
        by_instance[r.instance_id][r.variant] = asdict(r)
    _write_results_incremental(by_instance, filepath)
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def print_abc_summary(results: List[ABCMetrics]):
    """Print formatted ABC summary."""
    print("\n" + "="*70)
    print("ABC BENCHMARK SUMMARY")
    print("="*70)

    variant_names = {"a": "Baseline", "b": "MAS+Reviewer", "c": "OpenHands+Reviewer"}

    for variant in ["a", "b", "c"]:
        variant_results = [r for r in results if r.variant == variant]
        if not variant_results:
            continue
        scores = [r.relevancy_score for r in variant_results]
        latencies = [r.latency_total for r in variant_results]
        print(f"\nVariant {variant} ({variant_names[variant]}):")
        print(f"  Avg Score: {sum(scores)/len(scores):.2f}")
        print(f"  Avg Latency: {sum(latencies)/len(latencies):.1f}s")

    print("\n" + "-"*70)
    print("INSTANCE COMPARISON")
    print("-"*70)

    by_instance: Dict[str, Dict] = {}
    for r in results:
        if r.instance_id not in by_instance:
            by_instance[r.instance_id] = {}
        by_instance[r.instance_id][r.variant] = r.relevancy_score

    for instance_id, scores in by_instance.items():
        print(f"{instance_id:42} | A: {scores.get('a', 0):.2f} | "
              f"B: {scores.get('b', 0):.2f} | C: {scores.get('c', 0):.2f}")

    print("="*70)


# ---------------------------------------------------------------------------
# Instance ID helpers
# ---------------------------------------------------------------------------

def _parse_instance_ids(raw: str) -> Optional[List[str]]:
    """Parse --instance-ids value.

    Accepts:
      - Comma-separated:  "django__django-11099,sympy__sympy-13437"
      - @file.txt:        one ID per line, # comments skipped
      - @sweep_config.json:  reads the 'instance_ids' list
    Returns None if raw is empty.
    """
    if not raw:
        return None

    if raw.startswith("@"):
        path = raw[1:]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--instance-ids file not found: {path}")
        if path.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            ids = data.get("instance_ids", [])
            if not ids:
                raise ValueError(f"sweep config JSON has no 'instance_ids' list: {path}")
            return [str(x).strip() for x in ids if str(x).strip()]
        else:
            with open(path, encoding="utf-8") as f:
                return [
                    line.strip() for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]

    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_completed_pairs(output_filepath: str):
    """Load (completed_pairs, existing_results) from an existing output file."""
    if not os.path.isfile(output_filepath):
        return set(), {}
    try:
        with open(output_filepath, encoding="utf-8") as f:
            data = json.load(f)
        existing = data.get("results", {})
        pairs = set()
        for iid, variants in existing.items():
            for v in variants:
                pairs.add((iid, v))
        print(f"[RESUME] Loaded {len(pairs)} completed (instance, variant) pairs from {output_filepath}")
        return pairs, existing
    except Exception as e:
        print(f"[RESUME] WARNING: could not parse existing output ({e}); starting fresh")
        return set(), {}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABC Benchmark Runner")
    parser.add_argument("--variant", type=str, default="all",
                        choices=["a", "b", "c", "all"],
                        help="Variant to run (a=baseline, b=MAS, c=OpenHands)")
    parser.add_argument("--instances", type=str, default="",
                        help="JSON file with SWE-bench instances")
    parser.add_argument("--output", type=str, default="abc_results.json",
                        help="Output file")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument(
        "--instance-ids", type=str, default="",
        help=(
            "Comma-separated instance IDs, OR '@filepath' to read from "
            ".txt (one per line) or .json with 'instance_ids' list. "
            "Example: --instance-ids @sweep_config.json"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume an existing --output file: skip (instance, variant) pairs "
            "already present. Safe to re-run overnight without duplicating work."
        ),
    )
    parser.add_argument("--swe-lite", action="store_true",
                        help="Load real SWE-bench Lite from HuggingFace + clone repos.")
    parser.add_argument("--repos", type=str, default="",
                        help="Comma-separated repo filter for --swe-lite.")
    parser.add_argument("--no-clone", action="store_true",
                        help="Skip git clone (use for A/B-only runs).")

    args = parser.parse_args()
    print("[INFO] Starting SWE-bench runner...")

    # Parse --instance-ids
    try:
        instance_ids = _parse_instance_ids(args.instance_ids)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] --instance-ids: {e}")
        sys.exit(1)

    if instance_ids:
        print(f"[INFO] Targeting {len(instance_ids)} specific instance IDs")

    # Load resume state
    completed_pairs: set = set()
    existing_results: Dict = {}
    if args.resume:
        completed_pairs, existing_results = _load_completed_pairs(args.output)

    # Load instances
    try:
        if args.instances:
            print(f"[INFO] Loading instances from JSON file: {args.instances}")
            instances = load_instances(args.instances)
            if instance_ids:
                id_set = set(instance_ids)
                instances = [i for i in instances if i.instance_id in id_set]
                print(f"[INFO] Filtered to {len(instances)} instances matching --instance-ids")
        elif args.swe_lite:
            print("[INFO] Loading real SWE-bench Lite from HuggingFace")
            repo_filter = (
                [r.strip() for r in args.repos.split(",") if r.strip()]
                if args.repos else None
            )
            instances = load_swe_bench_lite(
                max_instances=args.max_instances,
                repo_filter=repo_filter,
                clone_repos=not args.no_clone,
                workspace_root="workspace",
                instance_ids=instance_ids,
            )
        else:
            print("[WARN] Using toy sample instances (no --swe-lite or --instances).")
            print("[WARN] Condition C will likely fail without a real repo to operate on.")
            from evaluate_swe import create_sample_instances
            instances = create_sample_instances()
            if instance_ids:
                id_set = set(instance_ids)
                instances = [i for i in instances if i.instance_id in id_set]
    except Exception as e:
        print(f"[ERROR] Failed to load instances: {e}")
        sys.exit(1)

    print(f"[INFO] Loaded {len(instances)} instances")

    variants = ["a", "b", "c"] if args.variant == "all" else [args.variant]
    print(f"[INFO] Running variants: {variants}")

    remaining = sum(
        1 for inst in instances for v in variants
        if (inst.instance_id, v) not in completed_pairs
    )
    total_pairs = len(instances) * len(variants)
    print(f"[INFO] {remaining}/{total_pairs} (instance, variant) pairs to run "
          f"({'resuming' if completed_pairs else 'fresh start'})")

    if remaining == 0:
        print("[INFO] All requested runs already completed. Nothing to do.")
        sys.exit(0)

    new_results = run_abc_comparison(
        instances,
        variants,
        max_instances=args.max_instances,
        completed_pairs=completed_pairs,
        existing_results=existing_results,
        output_filepath=args.output,
    )

    if new_results:
        print_abc_summary(new_results)
    else:
        print("[INFO] No new runs this session (all were skipped via --resume).")

    print(f"\n[DONE] Results at: {args.output}")
