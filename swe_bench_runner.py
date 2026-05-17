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

# Verify apps are properly loaded
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
    # The action trace is the per-step (action, observation) sequence the agent
    # produced. Required input for the Hallucination Bridge audit and the
    # main evidence trail for thesis Condition C analysis.
    openhands_status: Optional[str] = None
    openhands_backend: Optional[str] = None
    openhands_step_count: Optional[int] = None
    openhands_wall_time_s: Optional[float] = None
    openhands_action_trace: Optional[List[Dict[str, Any]]] = None


# Map variants to apps
VARIANT_APPS = {
    "a": ("Baseline", app_a),
    "b": ("MAS + Reviewer", app_b),
    "c": ("OpenHands + Reviewer", app_c),
}


def run_single_variant(
    instance: SWEBenchInstance,
    variant: str,
    max_steps: int = 10
) -> ABCMetrics:
    """Run evaluation for a single variant."""
    name, app = VARIANT_APPS[variant]

    print(f"\n{'='*60}")
    print(f"Variant {variant.upper()}: {name}")
    print(f"Instance: {instance.instance_id}")
    print('='*60)

    # Create initial state with variant
    state = create_initial_state(instance, condition=variant)
    config = {"recursion_limit": max_steps}

    latencies = {}
    final_answer = ""
    relevancy_score = 0.0
    iterations = 0
    success = False

    # OpenHands forensic fields — populated only when variant='c' returns SDK metadata.
    oh_status = None
    oh_backend = None
    oh_step_count = None
    oh_wall_time = None
    oh_trace = None

    start_time = time.time()

    try:
        for step in app.stream(state, config=config):
            node_name = list(step.keys())[0]
            node_state = step[node_name]

            # Track code answer + reviewer score + refine count from code_helper node.
            if node_name == "code_helper":
                final_answer = node_state.get("code_answer", "")
                relevancy_score = node_state.get("relevancy_score", 0.0)
                iterations = node_state.get("iteration_count", 0)

                # OpenHands-specific keys flow through ChatState only for variant C
                # (set by openhands_agent._invoke_sdk / _invoke_cloud / _invoke_stub).
                # Capture the latest non-None values across re-entries so the refine
                # loop's final attempt is what we save.
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

            # VRAM cooldown between nodes
            VRAMManager.sleep_with_countdown(3)

    except Exception as e:
        print(f"[ERROR] {e}")
        final_answer = f"Error: {str(e)}"

    total_latency = time.time() - start_time
    success = relevancy_score >= 0.7 if variant != "a" else True

    # Truncate code_answer for top-level field readability. The full text is in
    # the trace if it was generated as part of an action.
    truncated_answer = final_answer[:2000] if final_answer else ""

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
    )


def run_abc_comparison(
    instances: List[SWEBenchInstance],
    variants: List[str] = ["a", "b", "c"],
    max_instances: int = None
) -> List[ABCMetrics]:
    """Run ABC comparison across all variants."""
    if max_instances:
        instances = instances[:max_instances]

    results = []

    for instance in instances:
        print(f"\n{'#'*60}")
        print(f"# INSTANCE: {instance.instance_id}")
        print(f"{'#'*60}")

        for variant in variants:
            result = run_single_variant(instance, variant)
            results.append(result)

            # Cooldown between variants
            time.sleep(2)

    return results


def export_abc_results(results: List[ABCMetrics], filepath: str):
    """Export ABC results to JSON."""
    # Group by instance
    by_instance = {}
    for r in results:
        if r.instance_id not in by_instance:
            by_instance[r.instance_id] = {}
        by_instance[r.instance_id][r.variant] = asdict(r)

    output = {
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(results),
        "results": by_instance,
    }

    # Calculate summary stats
    for variant in ["a", "b", "c"]:
        variant_results = [r for r in results if r.variant == variant]
        if variant_results:
            scores = [r.relevancy_score for r in variant_results]
            latencies = [r.latency_total for r in variant_results]

            output[f"summary_{variant}"] = {
                "avg_score": sum(scores) / len(scores),
                "avg_latency": sum(latencies) / len(latencies),
                "total_runs": len(variant_results),
            }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[EXPORT] Results saved to: {filepath}")
    return output


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

    # Group by instance
    by_instance = {}
    for r in results:
        if r.instance_id not in by_instance:
            by_instance[r.instance_id] = {}
        by_instance[r.instance_id][r.variant] = r.relevancy_score

    for instance_id, scores in by_instance.items():
        print(f"{instance_id:30} | A: {scores.get('a', 0):.2f} | "
              f"B: {scores.get('b', 0):.2f} | C: {scores.get('c', 0):.2f}")

    print("="*70)


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

    # NEW: real SWE-bench Lite mode (replaces the toy create_sample_instances)
    parser.add_argument("--swe-lite", action="store_true",
                        help="Load real SWE-bench Lite from HuggingFace + clone repos. "
                             "Required for thesis-grade Condition C runs.")
    parser.add_argument("--repos", type=str, default="",
                        help="Comma-separated repo filter for --swe-lite "
                             "(e.g. 'django/django,pallets/flask'). Empty = all repos.")
    parser.add_argument("--no-clone", action="store_true",
                        help="With --swe-lite, skip git clone (use only if running A/B "
                             "without action agents).")

    args = parser.parse_args()

    print("[INFO] Starting SWE-bench runner...")

    # Load instances - precedence: explicit --instances JSON > --swe-lite > toy sample
    try:
        if args.instances:
            print(f"[INFO] Loading instances from JSON file: {args.instances}")
            instances = load_instances(args.instances)
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
            )
        else:
            print("[WARN] Using toy sample instances (no --swe-lite or --instances).")
            print("[WARN] Condition C will likely fail without a real repo to operate on.")
            from evaluate_swe import create_sample_instances
            instances = create_sample_instances()
    except Exception as e:
        print(f"[ERROR] Failed to load instances: {e}")
        sys.exit(1)

    print(f"[INFO] Loaded {len(instances)} instances")

    # Determine variants to run
    if args.variant == "all":
        variants = ["a", "b", "c"]
    else:
        variants = [args.variant]

    print(f"[INFO] Running variants: {variants}")

    # Run ABC comparison
    results = run_abc_comparison(instances, variants, args.max_instances)

    # Print and export
    print_abc_summary(results)
    export_abc_results(results, args.output)
