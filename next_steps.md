# Next Steps — Testing, Data Collection, and Analysis

**Project:** A Lightweight Hallucination Detection Algorithm for the Reviewer Agent in a Code Generation Multi-Agent System
**Author:** Syed Muhammad Shajee Raza, ITMO University, Group J4232
**Document purpose:** Self-contained handoff so a fresh agent session can resume testing and data collection without prior conversation context.
**Last updated:** 2026-04-27

---

## 1. What This Project Is

This is a Master's thesis project at ITMO University extending a multi-agent question-answering system built last semester. The current semester's contributions are:

1. A **Reviewer Agent** that scores any agent output along a five-dimension rubric (Correctness, Edge Cases, Security, Code Quality, Relevance) using an LLM-judge prompt with Pydantic-validated output.
2. A **score-driven refine loop** that re-invokes the generator when the score falls below 0.7, capped at 3 iterations.
3. A **Hallucination Bridge** that extends the Reviewer to audit per-step action traces from an autonomous OpenHands action agent, not just text outputs.
4. A **three-condition controlled experiment** (A: baseline, B: + Reviewer + refine, C: + OpenHands action agent) over real SWE-bench Lite instances, all driven by the same LLM for fair comparison.

Pipeline status: **end-to-end working**. First publishable result obtained on `django__django-10914`.

---

## 2. Current State — What Works

### Environment
- **Python 3.12** in **WSL2 Ubuntu 24.04** (project at `~/projects/Multi-Agent-Study-Coding-Assistant/`)
- Windows backup at `D:\ITMO\Semester 4\Project Root\Multi-Agent-Study-Coding-Assistant\`
- Ollama runs on **Windows host** (GPU access), reached from WSL via `${WINDOWS_HOST_IP}` resolved at every shell start through `~/.bashrc`
- Hosted LLM: **OpenRouter** (`minimax/minimax-m2.5:free`) — same model drives all three conditions
- venv at `venv312/`, dependencies in `pyproject-py312.toml`, lockfile in `requirements-py312.txt`
- Setup script: `setup_py312.ps1` (Windows legacy), WSL setup walkthrough in `setup_wsl.md`

### Validated runs

**Toy mock test** (run_benchmark.py with stub OpenHands):
| Condition | Score | Latency | Iterations |
|---|---|---|---|
| A | 0.98 | 87s | 0 |
| B | 0.95 | 64s | 1 |
| C (stub) | 0.98 | 13s | 1 |

**Real SWE-bench Lite — `django__django-10914`** (FILE_UPLOAD_PERMISSIONS bug):
| Condition | Score | Latency | Iterations | Outcome |
|---|---|---|---|---|
| A | 0.58 | 192s | 1 | Markdown explanation of fix |
| B | 0.62 | 611s | 3 | Refine fired, slight improvement |
| C | 0.70 | 1672s | 2 | Real Django patch applied, tests pass, 25-step action trace captured |

These results files live in `results/`:
- `results/swe_c_real_clean_20260426_025943.json` — Condition C with action trace
- `results/swe_ab_django__django-10914_20260426_072941.json` — Conditions A and B

### Key engineering bugs already fixed
1. `pip` could not resolve dependencies on py3.12 → solved by switching to `uv sync --active`
2. `uv sync` without `--active` creates `.venv` instead of using activated `venv312` → fixed by always using `--active`
3. OpenHands SDK throws `NotImplementedError` on Windows → solved by WSL2 migration
4. OpenHands SDK rejects `reasoning_effort=high` on non-frontier models → disabled in `openhands_agent.py`
5. `code_node_c` hardcoded `iteration_count=1` → wrapped in `code_node_c_with_iter` to mirror Condition B
6. Workspace state leaked between runs → `_reset_workspace_to_base_commit()` now wipes uncommitted edits and resets HEAD before each run
7. `swe_bench_runner.py` discarded the OpenHands action trace → `ABCMetrics` extended and `code_node_c` now forwards the data through ChatState

---

## 3. Code Structure

### Source modules

| File | Purpose |
|---|---|
| `graph.py` | LangGraph state machine. `build_graph(variant)` compiles app_a / app_b / app_c. Defines `ChatState` TypedDict, `code_node` (variants A/B), `code_node_c_with_iter` (variant C), `route_from_reviewer` (refine routing), `_review_output` helper. |
| `agents.py` | Agent factories: supervisor, researcher, writer, critique, router, code_helper, quiz_helper, **reviewer**. The `check_relevancy` function is the lightweight detection algorithm — it returns a Pydantic-validated `RelevancyDecision`. |
| `openhands_agent.py` | Three-backend dispatcher (`OPENHANDS_BACKEND` env: stub / cloud / sdk). `_invoke_sdk` is the Hallucination Bridge entry point — it builds the OpenHands LLM, registers the trace callback, runs the agent, returns the action trace alongside the code answer. |
| `openhands_client.py` | HTTP client for OpenHands Cloud API (legacy, kept for backward compatibility) |
| `prompts.py` | Prompt templates for supervisor, writer, critique |
| `tools.py` | Subprocess-based code execution (Python/C/C++), used by Code Helper for sandboxed validation |
| `memory/rag.py` | PDF ingestion + Chroma vectorstore for RAG retrieval |
| `memory/notes_memory.py` | Long-term agent notes (separate Chroma collection) |
| `memory/shared_memory.py` | Façade exposing rag + notes to all agents |

### Harness and benchmark

| File | Purpose |
|---|---|
| `swe_bench_runner.py` | **Primary ABC benchmark runner.** Streams through compiled app_a/b/c, writes per-instance JSON. CLI: `--variant {a,b,c,all}`, `--swe-lite`, `--repos`, `--max-instances`, `--output`. The `ABCMetrics` dataclass captures all per-run metrics including the OpenHands action trace. |
| `graph_variants.py` | Pre-compiles `app_a`, `app_b`, `app_c` for the runner |
| `evaluate_swe.py` | SWE-bench machinery: `SWEBenchInstance` dataclass, `load_swe_bench_lite()` (HuggingFace dataset loader), `_clone_repo_at_commit()` (git clone helper), `_reset_workspace_to_base_commit()` (state-wipe between runs), `create_initial_state()`. |
| `run_benchmark.py` | Newer harness with explicit `--openhands-backend` flag. Used for mock and stub validation. Toy prompts only. |
| `run_ab_only.py` | Companion script to run only Conditions A and B on a single instance, when Condition C has already been done separately. |
| `test_openhands_sdk.py` | Standalone SDK diagnostic — bypasses the graph to isolate SDK + LLM issues. Run this if Condition C ever breaks. |
| `analyze_results.py` | Result aggregation script. **Currently thin — needs extension for the analytics layer (Section 6 below).** |

### Configuration

| File | Purpose |
|---|---|
| `pyproject-py312.toml` | Active py3.12 dependency declaration |
| `pyproject-py310.toml` | Original py3.10 backup |
| `requirements-py312.txt` | Frozen lockfile from `uv pip freeze` |
| `.env` | LLM provider, model, base URL, API keys (OpenRouter, Tavily). **Do not commit.** Contains `LLM_PROVIDER=litellm`, `LITELLM_BASE_URL=https://openrouter.ai/api/v1`, `LITELLM_API_KEY=<openrouter key>`, `MODEL_NAME=minimax/minimax-m2.5:free`, `OPENHANDS_BACKEND=sdk`, `OPENHANDS_MODEL_NAME=openrouter/minimax/minimax-m2.5:free`, `OPENHANDS_API_KEY=<same openrouter key>`, `OPENHANDS_MAX_ITERATIONS=50`, `OLLAMA_BASE_URL=http://${WINDOWS_HOST_IP}:11434/v1`, `TAVILY_API_KEY=<tavily key>`. |
| `setup_py312.ps1` | Windows setup script (legacy after WSL migration) |
| `setup_wsl.md` | WSL2 migration guide |
| `py312_migration_patches.md` | Code patches applied during py3.12 migration |
| `progress.md` | Canonical thesis progress tracker |
| `docs/Shajee_Raza_Sem4_Research_Report.docx` | Research report draft |

### Result data

| File | Purpose |
|---|---|
| `results/swe_c_real_clean_*.json` | Condition C runs on real SWE-bench Lite instances with action trace |
| `results/swe_ab_*.json` | Conditions A/B runs |
| `results/mock_*.jsonl` | Mock pipeline validation runs |
| `cache/repos/<repo>__<repo>/` | Cached full repo clones (one per repo, reused across instances) |
| `cache/swebench/` | Cached HuggingFace dataset parquet |
| `workspace/openhands_<instance_id>/` | Per-instance OpenHands working directory (auto-reset to base_commit on each run) |

### Standard output schema (per run, per condition)

Each row in `swe_bench_runner.py` output JSON contains:
```
{
  "instance_id": "django__django-10914",
  "variant": "c",
  "success": true,
  "relevancy_score": 0.70,
  "latency_total": 1672.4,
  "iterations": 2,
  "code_answer": "<truncated to 2000 chars>",
  "timestamp": "ISO-8601",
  "openhands_status": "success",          # variant C only
  "openhands_backend": "sdk",              # variant C only
  "openhands_step_count": 25,              # variant C only
  "openhands_wall_time_s": 764.4,          # variant C only
  "openhands_action_trace": [              # variant C only — the Hallucination Bridge data
    {"step": 1, "type": "action", "tool_name": "terminal", "action_summary": "..."},
    {"step": 1, "type": "observation", "tool_name": "terminal", "observation_summary": "..."},
    ...
  ]
}
```

---

## 4. How to Run Things

All commands assume WSL Ubuntu, project root at `~/projects/Multi-Agent-Study-Coding-Assistant/`, venv activated with `source venv312/bin/activate`.

### Single instance, single condition (debug mode)
```bash
python swe_bench_runner.py --swe-lite --variant c --max-instances 1 \
  --repos django/django \
  --output results/test_$(date +%Y%m%d_%H%M%S).json
```

### Single instance, all three conditions (one full A/B/C row)
```bash
python swe_bench_runner.py --swe-lite --variant all --max-instances 1 \
  --repos django/django \
  --output results/abc_$(date +%Y%m%d_%H%M%S).json
```

### Multi-instance sweep (the headline data run)
```bash
python swe_bench_runner.py --swe-lite --variant all --max-instances 5 \
  --repos django/django,pallets/flask \
  --output results/sweep_$(date +%Y%m%d_%H%M%S).json
```

Expected total time on minimax/minimax-m2.5:free: ~20-30 minutes per (instance, Condition C) tuple, ~3-5 min per (instance, Condition A or B) tuple.

### Diagnostic when Condition C breaks
```bash
python test_openhands_sdk.py
```

### Toy validation (mock, fast)
```bash
python run_benchmark.py --openhands-backend stub
```

---

## 5. Critical Gotchas — Things That Will Bite You

1. **Always run from WSL, not Windows.** OpenHands SDK throws on Windows. The PowerShell venv works for A/B but is not the canonical environment.

2. **Always activate the venv before running.** If `python swe_bench_runner.py` fails to import openhands or langgraph, the venv isn't active. Check with `which python` — should show `~/projects/.../venv312/bin/python`.

3. **`OPENHANDS_BACKEND=sdk` must be in `.env`** before importing graph_variants. The runner reads it at module-load time. If missing, Condition C silently uses the cloud backend (which has separate rate limits).

4. **OpenRouter free-tier rate limits will throttle long runs.** Add $10-20 of credit at openrouter.ai/credits before scaling. Free-tier upstream pools are shared across all users and unpredictable.

5. **The `.env` file contains a real API key.** Never commit it. `.gitignore` excludes it but verify before any `git push`.

6. **Ollama on Windows must bind 0.0.0.0:11434**, not 127.0.0.1, for WSL to reach it. Set `OLLAMA_HOST=0.0.0.0:11434` in Windows env vars and restart Ollama from the tray.

7. **Workspace reset is automatic** but only fires when `--swe-lite` is used. Manual runs against `--instances some_file.json` do not reset.

8. **Condition C's iteration_count=1 is hardcoded by the SDK** — we wrapped it to increment on refine re-entry. This wrapper is in `graph.py`, function `code_node_c_with_iter`. If you ever rebuild graph.py, preserve this wrapper.

---

## 6. Next Steps — What To Do

### Phase 1: Scale up data collection

- Run **15-25 instances total across 3-4 repos** (Django 5-7, Flask 4-5, sympy/sklearn 4-5, edge-case repos 2-3)
- ~60 runs total (20 instances × 3 conditions), ~20-30 hours wall time spread over 3-5 days
- Add **$10-20 OpenRouter credit first** to escape free-tier rate limits
- Schedule overnight, log to dated JSON, never re-run completed instances

### Phase 2: Strengthen the methodology

- **Judge-generator separation** — re-run a few instances with a different LLM as judge (e.g., qwen-2.5-72b judge, minimax generator), prove results are robust to judge choice
- **Patch verification** — apply Condition C's diff to a fresh checkout, run the instance's `test_patch`, record true SWE-bench pass/fail (the literature-comparable metric)
- **Threshold sensitivity** — re-score one instance at acceptance thresholds 0.5, 0.7, 0.9, show iteration count and final score shifts in one table

### Phase 3: Build the analytics layer (`analyze_results.py`)

Extend the existing thin script to produce:

- Mean ± stdev score per condition
- Per-rubric breakdown (Correctness, Edge Cases, Security, Quality, Relevance) — the signature chart of the thesis
- Iteration distribution: % of B/C runs hitting refine 1×, 2×, 3×
- Latency vs score scatter (cost-quality trade-off)
- For Condition C: tool usage frequency, avg steps per instance, % of runs that reach `finish()`
- Per-instance comparison heatmap (spots outliers)
- **Output**: one `results/aggregate_<date>.json` + matplotlib PNGs to `results/figures/`

Per-rubric breakdown requires re-prompting the judge (or storing per-dimension scores during the original run). Easiest implementation: extend `agents.py:check_relevancy` to return per-dimension scores in the JSON, and propagate through ChatState.

### Phase 4: Hallucination case studies (qualitative spine)

Pick 3-4 traces, write a 1-page narrative each:

- **One success** — agent solves correctly, Reviewer accepts, highlight verification step
- **One detected hallucination** — Reviewer flags it, refine fires, annotate which rubric dimension caught it
- **One missed hallucination** — Reviewer accepts an actual error (limitations evidence)
- **One refine-loop trajectory** — show how iteration N+1 differs from N

Stored in `docs/case_studies/<instance_id>.md`.

### Phase 5: Visualizations (defense-grade)

- **Grouped bar chart** — per-instance A/B/C scores sorted by C
- **Radial chart of the 5-dim rubric** — average per dimension, per condition
- **Trace timeline** — one representative C run as horizontal timeline of tool actions
- matplotlib + seaborn for charts 1-2; static PNG output to `results/figures/`

---

## 7. Suggested Recipe For The New Chat Session

If a new chat is opened to do scale-up testing, the suggested first actions are:

1. Read this `next_steps.md` and `progress.md` end-to-end.
2. Confirm environment: WSL, venv activated, Ollama reachable, OpenRouter key live.
3. Decide whether to add OpenRouter credit before starting (recommended).
4. Pick the instance set for the multi-instance run (curate from SWE-bench Lite, list IDs in advance).
5. Pre-register hyperparameters in `progress.md`: acceptance threshold, max iterations, judge model, generator model, dataset instance IDs.
6. Run a single-instance smoke test first to confirm the pipeline still works after any environment change.
7. Schedule the multi-instance sweep with dated JSON output.
8. While the sweep runs, extend `analyze_results.py` for the per-condition aggregates and rubric breakdown.
9. After the sweep completes, generate figures, write case studies, update the thesis report.

### What the new agent will need from the user
- Confirmation that OpenRouter credit was added (or tolerance for free-tier flakiness)
- The list of SWE-bench Lite instance IDs to include in the sweep, OR delegation to pick a representative subset
- Decision on whether to also do judge-LLM separation (adds 1 day) and patch verification (adds 1 day)
- Decision on whether to update progress.md and the thesis report incrementally or only at the end

---

## 8. Reproducibility Checklist (For The Defense)

Before submitting the thesis, verify each of the following can be reproduced from scratch:

- [ ] Fresh Ubuntu 24.04 + WSL2 install can run `setup_wsl.md` end-to-end and reach the smoke-test passing state
- [ ] `python swe_bench_runner.py --swe-lite --variant c --max-instances 1 --repos django/django` produces a non-empty action trace and a relevancy score within 30-45 minutes
- [ ] All instance IDs cited in the thesis are reachable via `load_swe_bench_lite(instance_ids=[...])`
- [ ] `analyze_results.py` regenerates every figure and table in the thesis from the raw JSONs
- [ ] `.env.example` exists with placeholder values, real `.env` is in `.gitignore`
- [ ] README documents the exact commands and expected outputs

---

*End of next_steps.md. This document plus `progress.md` is the canonical handoff. Update both whenever the work state shifts.*
