"""
run_ab_only.py - Run Conditions A and B on a single SWE-bench Lite instance.

Companion to the existing Condition C result. Together they produce the
first complete A/B/C row on a real SWE-bench Lite instance using the
unified `minimax/minimax-m2.5:free` LLM.

Why this script exists:
    swe_bench_runner.py only accepts --variant {a, b, c, all} - one or all,
    no two-of-three. This script runs exactly A and B on the specified
    instance using the same internal machinery (run_single_variant from
    swe_bench_runner.py), so the output JSON format is identical and can
    be merged with the C result for the thesis comparison table.

Usage:
    # Default: runs django__django-10914 (matches the existing C result)
    python run_ab_only.py

    # A different instance:
    python run_ab_only.py --instance-id flask__flask-2462

    # Explicit output path:
    python run_ab_only.py --output results/my_ab_run.json

Notes:
    - clone_repos=False is set deliberately. Conditions A and B are text
      generators (they ask the chat LLM to write code as a markdown
      response). They don't need an actual repo checkout to function -
      that's only required for Condition C's OpenHands action agent.
      Setting clone_repos=False saves several minutes per Django instance
      (skips the ~500 MB git clone) and avoids any workspace-state
      interactions with the C run that produced your earlier results.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv, find_dotenv

# Load .env BEFORE importing project modules. agents.py and openhands_agent.py
# read env vars at import time (LLM provider, model, API keys), so the order
# matters - if we import first, those modules pick up empty values.
load_dotenv(find_dotenv(usecwd=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Conditions A and B on a single SWE-bench Lite instance",
    )
    parser.add_argument(
        "--instance-id",
        default="django__django-10914",
        help="SWE-bench Lite instance ID to run. "
             "Default: django__django-10914 (matches existing C result).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. "
             "Default: results/swe_ab_<instance_id>_<timestamp>.json",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="LangGraph recursion limit per variant. Default: 10. "
             "Should match the value used for the C result for fair comparison.",
    )
    args = parser.parse_args()

    # Lazy imports so dotenv expansion has happened first.
    # graph_variants.py compiles app_a/b/c, which trips the env-var reads.
    from evaluate_swe import load_swe_bench_lite
    from swe_bench_runner import (
        run_single_variant,
        export_abc_results,
        print_abc_summary,
    )

    # Narrow the loader to just the relevant repo if we can guess it from the
    # instance_id prefix. Cheap optimization - 14 Django instances loaded
    # instead of 300. Falls through to a full load if the prefix isn't
    # recognizable.
    repo_filter = None
    iid_lower = args.instance_id.lower()
    if iid_lower.startswith("django__"):
        repo_filter = ["django/django"]
    elif iid_lower.startswith("flask__"):
        repo_filter = ["pallets/flask"]
    elif iid_lower.startswith("sympy__"):
        repo_filter = ["sympy/sympy"]
    elif iid_lower.startswith("scikit-learn__"):
        repo_filter = ["scikit-learn/scikit-learn"]
    elif iid_lower.startswith("requests__"):
        repo_filter = ["psf/requests"]

    print(f"[INFO] Loading SWE-bench Lite (repo_filter={repo_filter}, clone_repos=False)")
    instances = load_swe_bench_lite(
        max_instances=None,
        repo_filter=repo_filter,
        clone_repos=False,  # A and B don't need a real repo - text-only
    )

    target = next((i for i in instances if i.instance_id == args.instance_id), None)
    if target is None:
        print(f"[ERROR] Instance {args.instance_id!r} not found.")
        if instances:
            print(f"[HINT] Available IDs in this filter: "
                  f"{[i.instance_id for i in instances[:5]]}{'...' if len(instances) > 5 else ''}")
        sys.exit(1)

    print(f"[INFO] Found {target.instance_id} (repo={target.repo}, base={target.base_commit[:8]})")
    print(f"[INFO] problem_statement (first 200 chars): {target.problem_statement[:200]!r}")
    print(f"[INFO] Running Conditions A and B (skipping C - already completed separately)")

    # Run A then B sequentially. Same internal code path as swe_bench_runner.py
    # uses for variant='all', just without the C iteration. Results carry the
    # same dataclass fields, so the output JSON merges cleanly with the
    # standalone C run.
    results = []
    for variant in ("a", "b"):
        print(f"\n{'#'*60}\n# Variant {variant.upper()}\n{'#'*60}")
        result = run_single_variant(target, variant, max_steps=args.max_steps)
        results.append(result)
        time.sleep(2)  # cooldown - keeps OpenRouter from rate-limiting on close-spaced calls

    # Export
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / f"swe_ab_{args.instance_id}_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print_abc_summary(results)
    export_abc_results(results, str(output_path))

    print(f"\n[OK] Wrote A and B results for {target.instance_id} to {output_path}")
    print(f"[HINT] To merge with the existing C result, see results/swe_c_real_clean_*.json")


if __name__ == "__main__":
    main()
