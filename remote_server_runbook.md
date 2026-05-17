# Remote Server Runbook — Phase 1 Data Collection
**Project:** ABC Benchmarking MAS, ITMO University  
**Author:** Shajee  
**Purpose:** Complete command-by-command guide from first sync to final analysis.

---

## PART 1 — ONE-TIME SETUP (do this once, never again)

### 1.1 — Connect VS Code to the remote server

1. Open VS Code on your Windows machine
2. Press `Ctrl+Shift+P` → type **Remote-SSH: Connect to Host** → Enter
3. Type `user@your-server-ip` (replace with your actual username and IP)
4. VS Code will open a new window connected to the server
5. Confirm the bottom-left corner shows **SSH: your-server-ip** in green

> If you have never added the server before, VS Code will ask to add it to `~/.ssh/config`. Click **Continue**.

---

### 1.2 — Open a terminal on the server

Inside the VS Code window connected to the server:

```
Ctrl + ` (backtick)
```

You now have a terminal running directly on the remote Linux server. All commands below are run here unless stated otherwise.

---

### 1.3 — Verify Python 3.12 is available

```bash
python3 --version
```

Expected: `Python 3.12.x`

If you see 3.10 or 3.11, install 3.12:
```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

---

### 1.4 — Install uv (fast package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

Expected: `uv 0.x.x`

---

### 1.5 — Install git

```bash
sudo apt install -y git
git --version
```

---

### 1.6 — Create the project directory on the server

```bash
mkdir -p ~/projects
```

---

### 1.7 — Sync the project from WSL to the remote server

Open a **new terminal on your local WSL machine** (not the server).  
Run this rsync command from WSL — it pushes the project to the server:

```bash
rsync -av --progress \
  --exclude='venv*' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.egg-info' \
  --exclude='workspace' \
  --exclude='.git' \
  --exclude='.pytest_cache' \
  --exclude='cache' \
  ~/projects/Multi-Agent-Study-Coding-Assistant/ \
  user@your-server-ip:~/projects/Multi-Agent-Study-Coding-Assistant/
```

> **Important:** The trailing slash on the source path (`...Assistant/`) is required.  
> Without it, rsync creates a nested folder instead of syncing the contents.

Wait for it to finish. You will see a list of files being copied.

---

### 1.8 — Go back to the server terminal and verify files arrived

```bash
ls ~/projects/Multi-Agent-Study-Coding-Assistant/
```

Expected output (you should see these files):
```
agents.py        evaluate_swe.py    graph.py          swe_bench_runner.py
analyze_results.py  graph_variants.py  daily_runs.sh     sweep_config.json
list_swe_instances.py  openhands_agent.py  results/  ...
```

---

### 1.9 — Create the Python virtual environment

```bash
cd ~/projects/Multi-Agent-Study-Coding-Assistant
python3.12 -m venv venv312
```

Or with uv (faster):
```bash
cd ~/projects/Multi-Agent-Study-Coding-Assistant
uv venv venv312 --python 3.12
```

---

### 1.10 — Activate the venv and install dependencies

```bash
source venv312/bin/activate
uv sync --active
```

This reads `pyproject-py312.toml` and installs everything. Takes 2–5 minutes.

Verify:
```bash
which python
```
Expected: `/home/your-user/projects/Multi-Agent-Study-Coding-Assistant/venv312/bin/python`

```bash
python --version
```
Expected: `Python 3.12.x`

```bash
python -c "import openhands; print('openhands OK')"
python -c "import langgraph; print('langgraph OK')"
```

Both should print OK with no errors.

---

### 1.11 — Create the .env file on the server

The `.env` file is not synced (it is gitignored). Create it manually:

```bash
nano ~/projects/Multi-Agent-Study-Coding-Assistant/.env
```

Paste exactly this content (fill in your real API keys):

```
LLM_PROVIDER=litellm
LITELLM_BASE_URL=https://openrouter.ai/api/v1
LITELLM_API_KEY=your_openrouter_key_here
MODEL_NAME=minimax/minimax-m2.5:free
OPENHANDS_BACKEND=sdk
OPENHANDS_MODEL_NAME=openrouter/minimax/minimax-m2.5:free
OPENHANDS_API_KEY=your_openrouter_key_here
OPENHANDS_MAX_ITERATIONS=50
TAVILY_API_KEY=your_tavily_key_here
```

Save: `Ctrl+O` → Enter → `Ctrl+X`

Verify it was saved:
```bash
cat .env
```

---

### 1.12 — Create required directories

```bash
cd ~/projects/Multi-Agent-Study-Coding-Assistant
mkdir -p results workspace cache/repos cache/swebench logs
```

---

### 1.13 — Verify all 20 instance IDs exist in the dataset

```bash
cd ~/projects/Multi-Agent-Study-Coding-Assistant
source venv312/bin/activate
python list_swe_instances.py --check sweep_config.json
```

Expected output:
```
[INFO] Loading princeton-nlp/SWE-bench_Lite (test split) ...
[INFO] 300 total instances in dataset
[CHECK] Verifying 20 IDs from sweep_config.json against dataset ...
  Found   : 20/20
  All IDs verified ✓
```

If any IDs show as MISSING, report them before starting Day 1 so replacements can be found.

---

### 1.14 — Install tmux (keeps runs alive if your connection drops)

```bash
sudo apt install -y tmux
tmux -V
```

Expected: `tmux 3.x`

---

## PART 2 — DAILY RUN WORKFLOW

Run one of the 20 commands below each day. Each run takes roughly **60–90 minutes** (A + B together ~10–20 min, Condition C ~30–45 min).

### Every day — standard startup

```bash
# 1. Connect VS Code to the server (Ctrl+Shift+P → Remote-SSH: Connect to Host)
# 2. Open terminal (Ctrl+`)
# 3. Navigate and activate venv:

cd ~/projects/Multi-Agent-Study-Coding-Assistant
source venv312/bin/activate
```

---

### Start a tmux session (do this BEFORE running the command)

```bash
tmux new -s sweep
```

You are now inside a tmux session. If your VS Code connection drops, the run keeps going. 

To detach safely (run keeps going in background):
```
Ctrl+B  then  D
```

To re-attach later and see the output:
```bash
tmux attach -t sweep
```

---

### The 20 daily commands

All commands write to the same file `results/sweep_phase1.json`.  
`--resume` means each command skips whatever was already completed.

```bash
# Day 1 — django__django-10914
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-10914 --output results/sweep_phase1.json --resume

# Day 2 — django__django-11099
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-11099 --output results/sweep_phase1.json --resume

# Day 3 — django__django-11583
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-11583 --output results/sweep_phase1.json --resume

# Day 4 — django__django-12113
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-12113 --output results/sweep_phase1.json --resume

# Day 5 — django__django-13230
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-13230 --output results/sweep_phase1.json --resume

# Day 6 — django__django-14534
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-14534 --output results/sweep_phase1.json --resume

# Day 7 — django__django-15738
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-15738 --output results/sweep_phase1.json --resume

# Day 8 — sympy__sympy-13437
python swe_bench_runner.py --swe-lite --variant all --instance-ids sympy__sympy-13437 --output results/sweep_phase1.json --resume

# Day 9 — sympy__sympy-14024
python swe_bench_runner.py --swe-lite --variant all --instance-ids sympy__sympy-14024 --output results/sweep_phase1.json --resume

# Day 10 — sympy__sympy-15308
python swe_bench_runner.py --swe-lite --variant all --instance-ids sympy__sympy-15308 --output results/sweep_phase1.json --resume

# Day 11 — sympy__sympy-16792
python swe_bench_runner.py --swe-lite --variant all --instance-ids sympy__sympy-16792 --output results/sweep_phase1.json --resume

# Day 12 — sympy__sympy-17022
python swe_bench_runner.py --swe-lite --variant all --instance-ids sympy__sympy-17022 --output results/sweep_phase1.json --resume

# Day 13 — scikit-learn__scikit-learn-10297
python swe_bench_runner.py --swe-lite --variant all --instance-ids scikit-learn__scikit-learn-10297 --output results/sweep_phase1.json --resume

# Day 14 — scikit-learn__scikit-learn-13142
python swe_bench_runner.py --swe-lite --variant all --instance-ids scikit-learn__scikit-learn-13142 --output results/sweep_phase1.json --resume

# Day 15 — scikit-learn__scikit-learn-14894
python swe_bench_runner.py --swe-lite --variant all --instance-ids scikit-learn__scikit-learn-14894 --output results/sweep_phase1.json --resume

# Day 16 — scikit-learn__scikit-learn-25500
python swe_bench_runner.py --swe-lite --variant all --instance-ids scikit-learn__scikit-learn-25500 --output results/sweep_phase1.json --resume

# Day 17 — astropy__astropy-12907
python swe_bench_runner.py --swe-lite --variant all --instance-ids astropy__astropy-12907 --output results/sweep_phase1.json --resume

# Day 18 — astropy__astropy-13033
python swe_bench_runner.py --swe-lite --variant all --instance-ids astropy__astropy-13033 --output results/sweep_phase1.json --resume

# Day 19 — astropy__astropy-14182
python swe_bench_runner.py --swe-lite --variant all --instance-ids astropy__astropy-14182 --output results/sweep_phase1.json --resume

# Day 20 — astropy__astropy-14365
python swe_bench_runner.py --swe-lite --variant all --instance-ids astropy__astropy-14365 --output results/sweep_phase1.json --resume
```

---

### What to expect while a run is in progress

The terminal will print something like this:

```
[INFO] Loading real SWE-bench Lite from HuggingFace
[SWE-BENCH] Filtered to 1 instances matching --instance-ids allowlist
[RESUME] Loaded 6 completed (instance, variant) pairs from results/sweep_phase1.json
[INFO] 3/3 (instance, variant) pairs to run

############################################################
# INSTANCE: django__django-11583
############################################################

============================================================
Variant A: Baseline
Instance: django__django-11583
============================================================
[code_helper] completed
[CHECKPOINT] results/sweep_phase1.json updated (7 runs total)

============================================================
Variant B: MAS + Reviewer
Instance: django__django-11583
============================================================
[code_helper] completed
[CHECKPOINT] results/sweep_phase1.json updated (8 runs total)

============================================================
Variant C: OpenHands + Reviewer
Instance: django__django-11583
============================================================
[code_helper] completed
[CHECKPOINT] results/sweep_phase1.json updated (9 runs total)
```

Each `[CHECKPOINT]` line means that run is safely saved to disk.

---

### Checking progress without interrupting the run

Open a second terminal tab (`Ctrl+Shift+`` ` or split terminal) and run:

```bash
cd ~/projects/Multi-Agent-Study-Coding-Assistant
python -c "
import json
with open('results/sweep_phase1.json') as f:
    d = json.load(f)
print(f'Total runs saved: {d[\"total_runs\"]}')
print()
for iid, v in d['results'].items():
    variants_done = sorted(v.keys())
    print(f'  {iid}: {variants_done}')
"
```

---

## PART 3 — AFTER ALL 20 DAYS: FINAL ANALYSIS

### Run the analytics script

```bash
cd ~/projects/Multi-Agent-Study-Coding-Assistant
source venv312/bin/activate
python analyze_results.py results/sweep_phase1.json
```

This prints a full summary table to the terminal and writes:
```
results/aggregate_YYYYMMDD_HHMMSS.json
```

### Copy results back to your Windows machine

From your WSL terminal (not the server):

```bash
rsync -av \
  user@your-server-ip:~/projects/Multi-Agent-Study-Coding-Assistant/results/ \
  ~/projects/Multi-Agent-Study-Coding-Assistant/results/
```

The results then sync to `D:\ITMO\...` on the next Windows backup.

---

## PART 4 — TROUBLESHOOTING

### "venv not found" or import errors
```bash
# Always activate before running anything:
source venv312/bin/activate
which python   # must show venv312/bin/python, not /usr/bin/python3
```

### Run was interrupted (connection drop, server restart, etc.)
Just re-run the exact same command for that day. The `--resume` flag checks what is already in `results/sweep_phase1.json` and skips completed variants automatically. You lose at most one (instance, variant) pair — the one that was mid-run when it died.

### OpenRouter 429 rate limit error
Wait 60 seconds, then re-run the same command with `--resume`. The completed variants are already saved.

### "All IDs verified" check fails (instance ID not in dataset)
```bash
# See what IDs ARE available in that repo:
python list_swe_instances.py --repos django/django
python list_swe_instances.py --repos sympy/sympy
```
Pick a replacement ID and update `sweep_config.json` and `daily_runs.sh` before running.

### See live output from a detached tmux run
```bash
tmux attach -t sweep
# Ctrl+B then D to detach again without stopping it
```

### Check if the run is still alive
```bash
tmux ls
# Shows: sweep: 1 windows (created ...) [attached/detached]
```
