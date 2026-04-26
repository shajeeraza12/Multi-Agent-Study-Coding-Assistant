"""
run_benchmark.py - ABC Benchmarking harness for Conditions A, B, C.

Runs the same input prompt through three compiled graph variants and writes
one comparable JSONL row per (instance_id, condition) tuple. Used both for
mock/wiring tests and real SWE-bench Lite runs.

Usage (mock test):
    python run_benchmark.py
    # -> runs the toy reverse_string prompt through A, B, C using stub
    #    OpenHands. Writes results/mock_<timestamp>.jsonl.

Usage (real SWE-bench, future):
    python run_benchmark.py --instances swe_bench_lite_subset.json --backend cloud
    # -> runs each instance through A, B, C using OpenHands Cloud for C.

Design notes:
    * Never imports `app` from graph.py (which is variant='b' by default).
      Always calls build_graph(variant) explicitly per loop.
    * Sets OPENHANDS_BACKEND env var BEFORE importing graph so the backend
      decision propagates through to create_openhands_chain.
    * Pre-flight checks (Ollama reachable, model loaded, env vars present)
      run before any graph compile to fail fast and save your time.
    * Captures the same metric set per condition for apples-to-apples
      comparison: code_answer, relevancy_score, hallucination_detected,
      iteration_count, latencies, token_usage, total/relevant/irrelevant
      review counts, agent_type_relevance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env BEFORE any other project import that might read env vars at module
# scope (agents.py reads TAVILY_API_KEY at import time).
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

REQUIRED_ENV = ["TAVILY_API_KEY"]  # MODEL_NAME / OLLAMA_BASE_URL have defaults


def preflight(skip_ollama_check: bool = False) -> None:
    """Fail fast if the env is not ready. Print actionable messages."""
    print("=== Pre-flight checks ===")

    # 1. Required env vars
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        print(f"[FAIL] Missing required env vars: {missing}")
        print("       Set them in your .env file at the project root.")
        sys.exit(1)
    print(f"[OK] Env vars present: {REQUIRED_ENV}")

    # 2. LLM provider config
    llm_provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    model_name = os.environ.get("MODEL_NAME", "qwen2.5:7b")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    print(f"[OK] LLM_PROVIDER={llm_provider}, MODEL_NAME={model_name}, BASE_URL={base_url}")

    # 3. OpenHands backend selection
    oh_backend = os.environ.get("OPENHANDS_BACKEND", "cloud")
    print(f"[OK] OPENHANDS_BACKEND={oh_backend}")
    if oh_backend == "sdk":
        print("[WARN] sdk backend is a Phase 2 placeholder; Condition C will raise.")

    # 4. Ollama reachability + model loaded
    if not skip_ollama_check and llm_provider == "ollama":
        try:
            import httpx
            # /api/tags lists installed models; OLLAMA_BASE_URL ends with /v1
            # so we strip that suffix to hit the native API.
            tags_url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
            r = httpx.get(tags_url, timeout=5.0)
            if r.status_code != 200:
                print(f"[FAIL] Ollama at {tags_url} returned HTTP {r.status_code}")
                print("       Start Ollama (Desktop app or `ollama serve`).")
                sys.exit(1)
            installed = [m.get("name", "") for m in r.json().get("models", [])]
            if not any(model_name == m or model_name in m for m in installed):
                print(f"[FAIL] Model '{model_name}' not loaded in Ollama.")
                print(f"       Installed: {installed}")
                print(f"       Run: ollama pull {model_name}")
                sys.exit(1)
            print(f"[OK] Ollama reachable, model '{model_name}' loaded.")
        except Exception as e:
            print(f"[FAIL] Ollama check failed: {e}")
            print("       Start Ollama and ensure the base URL is correct.")
            sys.exit(1)

    print()


# ---------------------------------------------------------------------------
# Initial state builder
# ---------------------------------------------------------------------------

def make_initial_state(prompt: str, condition: str, instance_id: str) -> Dict[str, Any]:
    """
    Build a fresh ChatState seed for one (instance, condition) run.

    The router_node in graph.py has a SWE-bench shortcut: if `intent="code"`
    AND `swe_instance_id` is set, the router preserves the intent and skips
    the LLM-based intent classifier. We exploit that here so all three
    conditions enter the code_helper path deterministically.
    """
    return {
        "messages": [{"role": "user", "content": prompt}],
        "intent": "code",                  # router preserves this when swe_instance_id is set
        "answer_mode": "long",
        "main_task": prompt,
        "code_question": prompt,
        "code_snippet": "",
        "swe_instance_id": instance_id,
        "condition": condition,            # "A" | "B" | "C"
        "research_findings": [],
        "draft": "",
        "critique_notes": "",
        "revision_number": 0,
        "next_step": "",
        "current_sub_task": "",
        "code_answer": "",
        "quiz_output": "",
        "relevancy_checks": [],
        "total_checks": 0,
        "relevant_count": 0,
        "irrelevant_count": 0,
        "agent_type_relevance": {},
        "relevancy_score": 0.0,
        "hallucination_detected": False,
        "iteration_count": 0,
        "latencies": {},
        "token_usage": {},
    }


# ---------------------------------------------------------------------------
# Single-condition run
# ---------------------------------------------------------------------------

def run_one(prompt: str, condition: str, instance_id: str) -> Dict[str, Any]:
    """
    Compile graph for one variant, invoke once, return a flat result dict.

    Imports graph lazily inside the function so OPENHANDS_BACKEND is read
    AFTER the env var has been set by main(). Otherwise the import-time
    side effects in agents.py / graph.py would freeze the backend choice.
    """
    from graph import build_graph  # lazy import; see docstring

    variant = condition.lower()
    print(f"\n>>> Compiling graph for variant='{variant}' (Condition {condition})")
    graph_app = build_graph(variant=variant)

    state = make_initial_state(prompt, condition, instance_id)

    print(f">>> Invoking graph (Condition {condition})...")
    t0 = time.time()
    try:
        final_state = graph_app.invoke(state)
        elapsed = time.time() - t0
        error = None
    except Exception as e:
        elapsed = time.time() - t0
        final_state = state
        error = f"{type(e).__name__}: {e}"
        print(f"[FAIL] Condition {condition} raised: {error}")

    code_answer = final_state.get("code_answer") or ""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "instance_id": instance_id,
        "condition": condition,
        "variant": variant,
        "wall_time_s": round(elapsed, 2),
        "error": error,
        "code_answer_length": len(code_answer),
        "code_answer_preview": code_answer[:300],
        "relevancy_score": final_state.get("relevancy_score", 0.0),
        "hallucination_detected": final_state.get("hallucination_detected", False),
        "iteration_count": final_state.get("iteration_count", 0),
        "total_checks": final_state.get("total_checks", 0),
        "relevant_count": final_state.get("relevant_count", 0),
        "irrelevant_count": final_state.get("irrelevant_count", 0),
        "latencies": final_state.get("latencies", {}),
        "token_usage": final_state.get("token_usage", {}),
        "agent_type_relevance": final_state.get("agent_type_relevance", {}),
        # Diagnostic field only set when condition C runs through the stub /
        # cloud / sdk dispatcher. Useful for the thesis writeup.
        "openhands_backend": (
            os.environ.get("OPENHANDS_BACKEND", "cloud") if variant == "c" else None
        ),
    }


# ---------------------------------------------------------------------------
# Multi-condition driver
# ---------------------------------------------------------------------------

DEFAULT_TOY_PROMPT = (
    "Write a Python function reverse_string(s: str) -> str that returns the "
    "input string reversed. Include type hints, a docstring, and a runtime "
    "check that the input is actually a str."
)


def run_all(
    prompt: str,
    instance_id: str,
    conditions: List[str],
    output_path: Path,
) -> List[Dict[str, Any]]:
    """Run prompt through each condition and write a JSONL row per run."""
    results: List[Dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for cond in conditions:
            row = run_one(prompt, cond, instance_id)
            results.append(row)
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()

    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Compact human-readable summary, one line per condition."""
    print("\n=== SUMMARY ===")
    header = f"{'Cond':<6}{'Time(s)':<10}{'Score':<8}{'Halluc':<10}{'Iters':<8}{'Checks':<8}{'AnsLen':<8}{'Backend':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        backend = r.get("openhands_backend") or "-"
        line = (
            f"{r['condition']:<6}"
            f"{r['wall_time_s']:<10}"
            f"{r['relevancy_score']:<8.2f}"
            f"{str(r['hallucination_detected']):<10}"
            f"{r['iteration_count']:<8}"
            f"{r['total_checks']:<8}"
            f"{r['code_answer_length']:<8}"
            f"{backend:<10}"
        )
        print(line)
        if r.get("error"):
            print(f"      ERROR: {r['error']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ABC Benchmarking harness")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_TOY_PROMPT,
        help="The code prompt to run through all conditions. Default: toy reverse_string task.",
    )
    parser.add_argument(
        "--instance-id",
        default="MOCK_001",
        help="Identifier for this run (used in JSONL output and to trigger router shortcut).",
    )
    parser.add_argument(
        "--conditions",
        default="A,B,C",
        help="Comma-separated list of conditions to run. Default: A,B,C.",
    )
    parser.add_argument(
        "--openhands-backend",
        default=None,
        choices=["stub", "cloud", "sdk"],
        help="Which OpenHands backend Condition C uses. Overrides OPENHANDS_BACKEND env var.",
    )
    parser.add_argument(
        "--skip-ollama-check",
        action="store_true",
        help="Skip the Ollama reachability check (use only if you know what you are doing).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to JSONL results file. Default: results/mock_<timestamp>.jsonl",
    )
    args = parser.parse_args()

    # CLI flag wins over .env. Set BEFORE any graph import.
    if args.openhands_backend:
        os.environ["OPENHANDS_BACKEND"] = args.openhands_backend
        print(f"[CLI] OPENHANDS_BACKEND set to '{args.openhands_backend}'")

    preflight(skip_ollama_check=args.skip_ollama_check)

    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    if not all(c in ("A", "B", "C") for c in conditions):
        print(f"[FAIL] --conditions must contain only A/B/C, got {conditions}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else (
        Path("results") / f"mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    print(f">>> Prompt: {args.prompt!r}")
    print(f">>> Instance: {args.instance_id}")
    print(f">>> Conditions: {conditions}")
    print(f">>> Output: {output_path}")

    results = run_all(args.prompt, args.instance_id, conditions, output_path)
    print_summary(results)

    print(f"\n[OK] Wrote {len(results)} rows to {output_path}")


if __name__ == "__main__":
    main()
