# Remote Linux Server Setup Guide
**Project:** ABC Benchmarking MAS — ITMO University  
**Use case:** Running Phase 1 data collection (Conditions A/B/C) on a remote Linux server via VS Code SSH Remote.  
**Last updated:** 2026-05-03

---

## Why the remote server is better than WSL

- OpenHands SDK runs natively — no `NotImplementedError("Windows is not supported")` workaround needed
- No WINDOWS_HOST_IP bridging — Ollama is not used for the sweep (OpenRouter handles all LLM calls)
- VS Code SSH Remote gives you a full terminal + file explorer on the server
- Results JSON files are visible directly in the VS Code file browser; no extra copy step needed

---

## Prerequisites on the remote server

Check these before starting:

```bash
python3 --version        # need 3.12.x
git --version            # need any recent version
uv --version             # install if missing (see below)
```

Install `uv` if not present:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc          # or restart terminal
```

---

## Step 1 — Copy the project to the server

**Option A — rsync from your Windows machine (recommended, preserves .env)**
```bash
# Run this in WSL on your local machine (not on the server):
rsync -avz --exclude='.git' --exclude='venv312' --exclude='cache' --exclude='workspace' \
  /mnt/d/ITMO/Semester\ 4/Project\ Root/Multi-Agent-Study-Coding-Assistant/ \
  user@your-server-ip:~/projects/Multi-Agent-Study-Coding-Assistant/
```

**Option B — git clone (if the repo is on GitHub, then copy .env manually)**
```bash
git clone https://github.com/YOUR_USERNAME/Multi-Agent-Study-Coding-Assistant.git \
  ~/projects/Multi-Agent-Study-Coding-Assistant
# Then copy .env separately (it is gitignored):
scp /mnt/d/ITMO/.../\.env user@your-server-ip:~/projects/Multi-Agent-Study-Coding-Assistant/.env
```

---

## Step 2 — Create the Python 3.12 virtual environment

```bash
cd ~/projects/Multi-Agent-Study-Coding-Assistant

# Create venv using uv (same name as WSL setup)
uv venv venv312 --python 3.12
source venv312/bin/activate

# Install all dependencies from the lockfile
uv sync --active

# Verify
which python          # should show .../venv312/bin/python
python --version      # should show 3.12.x
python -c "import openhands; print('openhands OK')"
python -c "import langgraph; print('langgraph OK')"
```

---

## Step 3 — Verify the .env file

```bash
cat .env   # confirm these keys are present and correct:
```

Required keys:
```
LLM_PROVIDER=litellm
LITELLM_BASE_URL=https://openrouter.ai/api/v1
LITELLM_API_KEY=<your openrouter key>
MODEL_NAME=minimax/minimax-m2.5:free
OPENHANDS_BACKEND=sdk
OPENHANDS_MODEL_NAME=openrouter/minimax/minimax-m2.5:free
OPENHANDS_API_KEY=<same openrouter key>
OPENHANDS_MAX_ITERATIONS=50
TAVILY_API_KEY=<your tavily key>
```

**Note:** `OLLAMA_BASE_URL` is NOT needed on the remote server. The sweep uses OpenRouter exclusively. You can leave it in .env or remove it — it won't be called.

---

## Step 4 — Create required directories

```bash
mkdir -p results workspace cache/repos cache/swebench
```

---

## Step 5 — Verify instance IDs (run once)

```bash
source venv312/bin/activate
python list_swe_instances.py --check sweep_config.json
```

Expected output: `All IDs verified ✓`  
If any IDs are flagged as missing, report them and get replacements before Day 1.

---

## Step 6 — Run the daily commands

Open the project in VS Code SSH Remote, open a terminal, then each day:

```bash
cd ~/projects/Multi-Agent-Study-Coding-Assistant
source venv312/bin/activate

# Paste the line for today from daily_runs.sh, e.g. Day 2:
python swe_bench_runner.py --swe-lite --variant all \
  --instance-ids django__django-11099 \
  --output results/sweep_phase1.json --resume
```

The full ordered list of all 20 daily commands is in `daily_runs.sh`.

---

## Step 7 — Monitor progress mid-run

The runner prints a `[CHECKPOINT]` line after every single (instance, variant) pair completes, so you can see progress in real time. You can also open `results/sweep_phase1.json` directly in VS Code to inspect accumulated results.

To check how many runs are done so far:
```bash
python -c "
import json
with open('results/sweep_phase1.json') as f: d=json.load(f)
for iid, v in d['results'].items():
    print(f'{iid}: {sorted(v.keys())}')
print(f\"Total: {d['total_runs']} runs\")
"
```

---

## Step 8 — Final analysis (after Day 20)

```bash
python analyze_results.py results/sweep_phase1.json
```

This prints the full summary and writes `results/aggregate_<date>.json`.

---

## Keeping the run alive when you close VS Code

If you want the run to continue after closing your laptop:

```bash
# Start a named tmux session before running:
tmux new -s sweep

# Inside tmux, run the daily command as normal.
# Detach with Ctrl+B then D.
# Re-attach later with:
tmux attach -t sweep
```

Or use `nohup`:
```bash
nohup python swe_bench_runner.py --swe-lite --variant all \
  --instance-ids django__django-11099 \
  --output results/sweep_phase1.json --resume \
  > logs/day2.log 2>&1 &
echo $! > logs/day2.pid   # save PID to check later
tail -f logs/day2.log      # follow live output
```

Create the logs directory first: `mkdir -p logs`

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: openhands` | venv not activated | `source venv312/bin/activate` |
| `NotImplementedError: Windows is not supported` | Running on Windows by mistake | You're on the right track — this should never appear on a real Linux server |
| OpenRouter 429 rate limit | Too many requests | Wait 60s and re-run with `--resume`; the completed variants are saved |
| `git clone` timeout during instance load | Slow network or large repo | Increase timeout in `evaluate_swe.py:_clone_repo_at_commit` (default 600s) |
| Condition C score = 0 / agent flails | Workspace not reset cleanly | The reset is automatic; check `[RESET]` lines in output |
