"""
SWE-bench Evaluation Script
========================

Evaluates the Multi-Agent Coding Assistant on SWE-bench instances.
Compares Condition A (baseline, no reviewer) vs Condition B (with reviewer).

Usage:
    python evaluate_swe.py --instances swe_instances.json --output results.json
    python evaluate_swe.py --instances swe_instances.json --output results.json --max-instances 5
"""

import json
import time
import argparse
import os
import sys
import threading
from datetime import datetime
from dataclasses import dataclass, asdict, field

# Fix Unicode output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from typing import List, Dict, Optional, Any
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph import app, ChatState


@dataclass
class SWEBenchInstance:
    """SWE-bench instance representation."""
    instance_id: str
    repo: str
    problem_statement: str
    patch: str = ""
    gold_patch: str = ""
    base_commit: str = ""  # SHA needed to checkout the codebase OpenHands operates on
    test_patch: str = ""   # Optional: tests that should pass after the fix
    hints_text: str = ""   # Optional: developer hints from the original PR/issue


@dataclass
class EvaluationResult:
    """Result for a single SWE-bench instance evaluation."""
    instance_id: str
    condition: str  # "A" or "B"
    success: bool
    relevancy_score: float
    latency_total: float
    latencies: Dict[str, float]
    token_usage: Dict[str, int]
    iterations: int
    hallucination_detected: bool
    final_answer: str
    timestamp: str


@dataclass
class ComparisonResult:
    """Comparison between Condition A and B for a single instance."""
    instance_id: str
    condition_a: Optional[EvaluationResult] = None
    condition_b: Optional[EvaluationResult] = None
    score_improvement: float = 0.0
    baseline_has_reviewer: bool = False


class VRAMManager:
    """Manages VRAM between nodes for GTX 1650."""

    @staticmethod
    def sleep_with_countdown(seconds: int = 3):
        """Sleep with countdown display."""
        for i in range(seconds, 0, -1):
            print(f"[VRAM] Cooldown {i}s...", end="\r")
            time.sleep(1)
        print(" " * 20, end="\r")


def load_instances(filepath: str) -> List[SWEBenchInstance]:
    """Load SWE-bench instances from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    instances = []
    if isinstance(data, list):
        items = data
    else:
        items = data.get("instances", [])

    for item in items:
        instances.append(SWEBenchInstance(
            instance_id=item.get("instance_id", item.get("id", "unknown")),
            repo=item.get("repo", ""),
            problem_statement=item.get("problem_statement", item.get("problem", "")),
            patch=item.get("patch", ""),
            gold_patch=item.get("gold_patch", item.get("patch", "")),
        ))

    return instances


def create_initial_state(instance: SWEBenchInstance, condition: str) -> Dict:
    """Create initial state for SWE-bench evaluation."""
    return {
        "messages": [{"role": "user", "content": instance.problem_statement}],
        "intent": "code",  # SWE-bench is code-focused
        "answer_mode": "long",
        "main_task": instance.problem_statement[:2000],  # Truncate if needed
        "research_findings": [],
        "draft": "",
        "critique_notes": "",
        "revision_number": 0,
        "next_step": "",
        "current_sub_task": "",
        "code_question": instance.problem_statement[:2000],
        "code_snippet": "",
        "code_answer": "",
        "quiz_output": "",
        "relevancy_checks": [],
        "total_checks": 0,
        "relevant_count": 0,
        "irrelevant_count": 0,
        "agent_type_relevance": {},
        # Phase 1 SWE-bench fields
        "condition": condition,
        "relevancy_score": 0.0,
        "hallucination_detected": False,
        "iteration_count": 0,
        "swe_instance_id": instance.instance_id,
        "latencies": {},
        "token_usage": {},
    }


def run_evaluation(instance: SWEBenchInstance, condition: str, max_steps: int = 10) -> EvaluationResult:
    """Run evaluation on a single instance for a given condition."""
    print(f"\n{'='*60}")
    print(f"Evaluating {instance.instance_id} (Condition {condition})")
    print(f"Problem: {instance.problem_statement[:100]}...")
    print('='*60)

    initial_state = create_initial_state(instance, condition)
    config = {"recursion_limit": max_steps}

    latencies = {}
    token_usage = {}
    final_answer = ""
    relevancy_score = 0.0
    hallucination_detected = False
    iterations = 0

    start_time = time.time()

    try:
        for step in app.stream(initial_state, config=config):
            node_name = list(step.keys())[0]
            node_state = step[node_name]

            # Extract latencies from state
            if "latencies" in node_state and node_state["latencies"]:
                latencies = node_state["latencies"]

            # Extract token usage
            if "token_usage" in node_state and node_state["token_usage"]:
                token_usage = node_state["token_usage"]

            # Get final answer from code_helper output
            if node_name == "code_helper":
                final_answer = node_state.get("code_answer", "")
                relevancy_score = node_state.get("relevancy_score", 0.0)
                hallucination_detected = node_state.get("hallucination_detected", False)
                iterations = node_state.get("iteration_count", 0)

            print(f"[{node_name}] completed")

            # VRAM management: sleep between nodes
            if condition == "B":
                VRAMManager.sleep_with_countdown(3)

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        final_answer = f"Error: {str(e)}"

    total_latency = time.time() - start_time

    return EvaluationResult(
        instance_id=instance.instance_id,
        condition=condition,
        success=relevancy_score >= 0.7 if condition == "B" else True,
        relevancy_score=relevancy_score,
        latency_total=total_latency,
        latencies=latencies,
        token_usage=token_usage,
        iterations=iterations,
        hallucination_detected=hallucination_detected,
        final_answer=final_answer[:500] if final_answer else "",
        timestamp=datetime.now().isoformat()
    )


def evaluate_single_instance(instance: SWEBenchInstance, max_steps: int = 10) -> ComparisonResult:
    """Evaluate a single instance under both conditions."""
    print(f"\n{'='*60}")
    print(f"INSTANCE: {instance.instance_id}")
    print('='*60)

    # Condition A: Baseline (no reviewer)
    print("\n>>> Running Condition A (baseline)...")
    result_a = run_evaluation(instance, "A", max_steps)

    # Brief pause between conditions
    time.sleep(2)

    # Condition B: With reviewer
    print("\n>>> Running Condition B (with reviewer)...")
    result_b = run_evaluation(instance, "B", max_steps)

    # Calculate improvement
    score_improvement = result_b.relevancy_score - result_a.relevancy_score

    return ComparisonResult(
        instance_id=instance.instance_id,
        condition_a=result_a,
        condition_b=result_b,
        score_improvement=score_improvement,
        baseline_has_reviewer=(result_a.relevancy_score < 1.0)
    )


def run_batch_evaluation(instances: List[SWEBenchInstance], max_instances: int = None) -> List[ComparisonResult]:
    """Run batch evaluation on multiple instances."""
    if max_instances:
        instances = instances[:max_instances]

    results = []
    total = len(instances)

    for i, instance in enumerate(instances, 1):
        print(f"\n{'#'*60}")
        print(f"# INSTANCE {i}/{total}")
        print(f"{'#'*60}")

        result = evaluate_single_instance(instance)
        results.append(result)

        print(f"\n[RESULT] {instance.instance_id}")
        print(f"  Condition A: score={result.condition_a.relevancy_score:.2f}, "
              f"latency={result.condition_a.latency_total:.1f}s")
        print(f"  Condition B: score={result.condition_b.relevancy_score:.2f}, "
              f"latency={result.condition_b.latency_total:.1f}s, "
              f"iterations={result.condition_b.iterations}")
        print(f"  Improvement: {result.score_improvement:+.2f}")

    return results


def export_results(results: List[ComparisonResult], filepath: str):
    """Export results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_instances": len(results),
        "results": []
    }

    for r in results:
        result_dict = {
            "instance_id": r.instance_id,
            "score_improvement": r.score_improvement,
            "condition_a": asdict(r.condition_a) if r.condition_a else None,
            "condition_b": asdict(r.condition_b) if r.condition_b else None,
        }
        output["results"].append(result_dict)

    # Calculate aggregate statistics
    if results:
        total_a_latency = sum(r.condition_a.latency_total for r in results if r.condition_a)
        total_b_latency = sum(r.condition_b.latency_total for r in results if r.condition_b)
        avg_a_score = sum(r.condition_a.relevancy_score for r in results if r.condition_a) / len(results)
        avg_b_score = sum(r.condition_b.relevancy_score for r in results if r.condition_b) / len(results)

        output["summary"] = {
            "avg_relevancy_score_a": avg_a_score,
            "avg_relevancy_score_b": avg_b_score,
            "avg_latency_a": total_a_latency / len(results),
            "avg_latency_b": total_b_latency / len(results),
            "total_instances": len(results),
        }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[EXPORT] Results saved to: {filepath}")


def create_sample_instances() -> List[SWEBenchInstance]:
    """Create sample SWE-bench instances for testing.

    LEGACY: stub problem statements with no repo checkout. Useful only for
    pipeline smoke tests. For real Condition C runs use load_swe_bench_lite()
    below, which fetches actual SWE-bench Lite instances and clones the
    target repo at base_commit so OpenHands has code to operate on.
    """
    return [
        SWEBenchInstance(
            instance_id="django__django-11099",
            repo="django/django",
            problem_statement="ForeignObject.get() lookup fails when using custom pk field with non-integer values",
            gold_patch="def get(self, rawness=NOT_PROVIDED, *args, **kwargs): ..."
        ),
        SWEBenchInstance(
            instance_id="flask__flask-2462",
            repo="flask/flask",
            problem_statement="JSON encoded response missing Content-Type header when returning dict from view",
            gold_patch="def jsonify(*args, **kwargs): ..."
        ),
    ]


# ===========================================================================
# Real SWE-bench Lite loader (added 2026-04-25)
# ===========================================================================
# Replaces the stub create_sample_instances() for thesis-grade Condition C
# runs. Pulls real instances from HuggingFace and clones each instance's
# target repo at base_commit into workspace/openhands_<instance_id>/ so the
# OpenHands action agent has actual source to inspect, edit, and test.
#
# Required for Condition C to produce meaningful results - without a real
# repository checkout, OpenHands flails (see progress.md section 6.3).
#
# Dependencies (install once in the WSL venv):
#   uv pip install datasets
#   sudo apt install -y git
# ===========================================================================

def load_swe_bench_lite(
    max_instances: Optional[int] = None,
    repo_filter: Optional[List[str]] = None,
    workspace_root: str = "workspace",
    clone_repos: bool = True,
    cache_dataset_dir: Optional[str] = None,
    skip_existing_clones: bool = True,
    instance_ids: Optional[List[str]] = None,
) -> List[SWEBenchInstance]:
    """
    Load real SWE-bench Lite instances from HuggingFace.

    Args:
        max_instances: Cap on instances loaded (None = all 300 in test split).
        repo_filter: Only include instances whose `repo` field matches one of
                     these strings (e.g. ['django/django', 'pallets/flask']).
                     None = all repos.
        workspace_root: Directory under which per-instance workspaces are
                        created. Must match openhands_agent's
                        OPENHANDS_WORKSPACE_ROOT (default 'workspace').
        clone_repos: If True, git clone each instance's repo at base_commit
                     into workspace_root/openhands_<instance_id>/. Required
                     for Condition C with the SDK backend. Set False if you
                     only run Conditions A/B (no action agent needs files).
        cache_dataset_dir: HuggingFace dataset cache directory. None uses
                           the default (~/.cache/huggingface/datasets).
        skip_existing_clones: If True, do not re-clone if the workspace dir
                              already contains a .git directory. Saves time
                              on re-runs.
        instance_ids: Allowlist of specific instance IDs to include. All
                      others are skipped before cloning, so no unnecessary
                      git work is done. None = include all instances (subject
                      to repo_filter / max_instances). Pass the list from
                      sweep_config.json for deterministic curated sweeps.

    Returns:
        List of SWEBenchInstance with full multi-paragraph problem_statement
        from the real dataset, ready for swe_bench_runner.py.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "load_swe_bench_lite requires the HuggingFace `datasets` library.\n"
            "Install with: uv pip install datasets"
        )

    print("[SWE-BENCH] Loading princeton-nlp/SWE-bench_Lite (test split)...")
    ds = load_dataset(
        "princeton-nlp/SWE-bench_Lite",
        split="test",
        cache_dir=cache_dataset_dir,
    )
    print(f"[SWE-BENCH] Loaded {len(ds)} total instances")

    # Filter by repo if requested
    if repo_filter:
        repo_set = set(repo_filter)
        ds = ds.filter(lambda x: x["repo"] in repo_set)
        print(f"[SWE-BENCH] Filtered to {len(ds)} instances matching repos: {sorted(repo_set)}")

    # Filter by explicit instance ID allowlist (takes precedence over max_instances cap)
    if instance_ids:
        id_set = set(instance_ids)
        ds = ds.filter(lambda x: x["instance_id"] in id_set)
        print(f"[SWE-BENCH] Filtered to {len(ds)} instances matching --instance-ids allowlist")
        # Warn about any requested IDs not found in the dataset
        found = {item["instance_id"] for item in ds}
        missing = id_set - found
        if missing:
            print(f"[SWE-BENCH] WARNING: {len(missing)} requested ID(s) not found in dataset:")
            for m in sorted(missing):
                print(f"             - {m}")

    # Cap count if requested
    if max_instances and max_instances < len(ds):
        ds = ds.select(range(max_instances))
        print(f"[SWE-BENCH] Capped to first {max_instances} instances")

    workspace_root_path = Path(workspace_root)
    instances: List[SWEBenchInstance] = []

    for idx, item in enumerate(ds, start=1):
        instance_id = item["instance_id"]
        repo = item["repo"]
        base_commit = item["base_commit"]
        problem_statement = item["problem_statement"]
        gold_patch = item.get("patch", "")

        print(f"\n[SWE-BENCH] [{idx}/{len(ds)}] {instance_id} ({repo} @ {base_commit[:8]})")

        if clone_repos:
            workspace_path = workspace_root_path / f"openhands_{instance_id}"
            git_dir = workspace_path / ".git"

            if git_dir.exists():
                # CRITICAL: workspace exists from a prior run. Reset to base_commit
                # and wipe untracked files so the agent always starts from the
                # pristine buggy state. Without this, prior agent commits leak
                # across runs and the agent sees its own past work as "task
                # already done", scoring as a false negative. (See progress.md
                # findings 1-3 from the django__django-10914 50-iter run.)
                print(f"[SWE-BENCH] Workspace exists; resetting to {base_commit[:8]}")
                try:
                    _reset_workspace_to_base_commit(workspace_path, base_commit)
                except Exception as e:
                    print(f"[SWE-BENCH] WARNING: reset failed for {instance_id}: {e}")
                    print(f"[SWE-BENCH] Falling back to fresh clone")
                    try:
                        _clone_repo_at_commit(repo, base_commit, workspace_path)
                    except Exception as e2:
                        print(f"[SWE-BENCH] WARNING: clone also failed: {e2}")
            else:
                try:
                    _clone_repo_at_commit(repo, base_commit, workspace_path)
                except Exception as e:
                    print(f"[SWE-BENCH] WARNING: clone failed for {instance_id}: {e}")
                    print(f"[SWE-BENCH] Continuing without workspace - C will likely fail on this instance")

        instances.append(SWEBenchInstance(
            instance_id=instance_id,
            repo=repo,
            problem_statement=problem_statement,
            patch="",  # leave empty; gold_patch is the ground-truth fix
            gold_patch=gold_patch,
            base_commit=base_commit,  # required for repo checkout in Condition C
            test_patch=item.get("test_patch", ""),
            hints_text=item.get("hints_text", ""),
        ))

    print(f"\n[SWE-BENCH] Prepared {len(instances)} instances ready for runner")
    return instances


def _reset_workspace_to_base_commit(workspace: Path, base_commit: str) -> None:
    """Restore an existing workspace to the pristine base_commit state.

    Performs `git fetch` (to ensure we know about base_commit) followed by
    `git reset --hard <base_commit>` and `git clean -fdx`. This wipes:
      - any commits the agent made on top of base_commit in a prior run
      - any uncommitted edits left over after a crashed run
      - any untracked files (build artifacts, .pytest_cache, etc.)

    Result: the workspace is byte-identical to a fresh `git clone` followed
    by `git checkout <base_commit>`, but ~100x faster because we skip the
    network clone.

    Raises RuntimeError on git failure so the caller can fall back to a
    fresh clone.
    """
    import subprocess

    # Make sure base_commit is reachable (in case prior run pruned refs)
    fetch = subprocess.run(
        ["git", "fetch", "--all", "--tags", "--prune"],
        cwd=str(workspace), capture_output=True, text=True, timeout=120,
    )
    if fetch.returncode != 0:
        # fetch failures aren't always fatal (commit may already be present)
        print(f"[RESET] git fetch warning: {fetch.stderr[:300]}")

    reset = subprocess.run(
        ["git", "reset", "--hard", base_commit],
        cwd=str(workspace), capture_output=True, text=True, timeout=60,
    )
    if reset.returncode != 0:
        raise RuntimeError(f"git reset --hard {base_commit[:8]} failed: {reset.stderr[:500]}")

    clean = subprocess.run(
        ["git", "clean", "-fdx"],
        cwd=str(workspace), capture_output=True, text=True, timeout=60,
    )
    if clean.returncode != 0:
        raise RuntimeError(f"git clean -fdx failed: {clean.stderr[:500]}")

    # Verify we're actually at the right commit
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(workspace), capture_output=True, text=True, timeout=10,
    )
    if head.returncode != 0 or head.stdout.strip() != base_commit:
        raise RuntimeError(
            f"post-reset HEAD ({head.stdout.strip()[:8]}) does not match "
            f"base_commit ({base_commit[:8]}); workspace may be corrupt"
        )

    print(f"[RESET] {workspace.name} restored to {base_commit[:8]} (clean tree)")


def _clone_repo_at_commit(repo: str, commit: str, target_path: Path) -> None:
    """
    Clone a GitHub repo and check out a specific commit.

    Uses subprocess + git CLI (no GitPython dependency). Requires `git`
    available on PATH (apt install git).

    On failure, raises RuntimeError with the git stderr output. Caller
    is expected to log and continue with remaining instances.
    """
    import subprocess
    import shutil

    # Clean any partial/wrong state at target
    if target_path.exists():
        print(f"[CLONE] Removing existing dir at {target_path}")
        shutil.rmtree(target_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    clone_url = f"https://github.com/{repo}.git"
    print(f"[CLONE] git clone {clone_url} -> {target_path}")

    # Full clone (need history for checkout). For very large repos, consider
    # using --filter=blob:none for partial clone, but that needs git >=2.22
    # and may break OpenHands operations that read history.
    result = subprocess.run(
        ["git", "clone", clone_url, str(target_path)],
        capture_output=True,
        text=True,
        timeout=600,  # 10 min - large repos like django can be slow
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr[:1000]}")

    print(f"[CLONE] git checkout {commit[:8]}")
    result = subprocess.run(
        ["git", "checkout", commit],
        cwd=str(target_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git checkout failed: {result.stderr[:500]}")

    print(f"[CLONE] OK - {target_path} ready at {commit[:8]}")


def print_summary(results: List[ComparisonResult]):
    """Print formatted summary."""
    print("\n" + "="*70)
    print("SWE-bench EVALUATION SUMMARY")
    print("="*70)

    if not results:
        print("No results to display.")
        return

    total = len(results)
    avg_a = sum(r.condition_a.relevancy_score for r in results) / total
    avg_b = sum(r.condition_b.relevancy_score for r in results) / total
    avg_improvement = sum(r.score_improvement for r in results) / total

    print(f"Total Instances: {total}")
    print(f"\nCondition A (baseline):")
    print(f"  Avg Relevancy Score: {avg_a:.2f}")
    print(f"\nCondition B (with reviewer):")
    print(f"  Avg Relevancy Score: {avg_b:.2f}")
    print(f"\nImprovement: {avg_improvement:+.2f}")

    print("\n" + "-"*70)
    print("PER-INSTANCE DETAILS")
    print("-"*70)
    for r in results:
        print(f"{r.instance_id:30} | A: {r.condition_a.relevancy_score:.2f} | "
              f"B: {r.condition_b.relevancy_score:.2f} | "
              f"DIFF: {r.score_improvement:+.2f}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on SWE-bench")
    parser.add_argument("--instances", type=str, default="",
                        help="JSON file with SWE-bench instances")
    parser.add_argument("--output", type=str, default="swe_results.json",
                        help="Output file for results")
    parser.add_argument("--max-instances", type=int, default=None,
                        help="Maximum number of instances to evaluate")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Maximum workflow steps per instance")
    parser.add_argument("--sample", action="store_true",
                        help="Use sample instances for testing")

    args = parser.parse_args()

    # Load instances
    if args.sample or not args.instances:
        print("[INFO] Using sample instances")
        instances = create_sample_instances()
    else:
        print(f"[INFO] Loading instances from: {args.instances}")
        instances = load_instances(args.instances)

    print(f"[INFO] Loaded {len(instances)} instances")

    # Run evaluation
    results = run_batch_evaluation(instances, args.max_instances)

    # Print and export results
    print_summary(results)
    export_results(results, args.output)