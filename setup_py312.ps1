# ============================================================================
# setup_py312.ps1
# ----------------------------------------------------------------------------
# One-shot migration script: py3.10 -> py3.12 venv for the ABC Benchmarking MAS
# project. Uses `uv` as the resolver to avoid pip's backtracking death spiral
# on the openhands-sdk + langchain + chromadb stack.
#
# Safe to re-run. Idempotent except for the venv312/uv.lock recreation step.
#
# Preconditions:
#   - Python 3.12 installed and on PATH as `py -3.12` (standard on Windows)
#   - You are in the project root (the folder containing pyproject.toml)
#   - pyproject-py312.toml exists in the project root (shipped alongside this
#     script). The script swaps it in as pyproject.toml for the resolve.
#
# What it does:
#   1. Deactivates any active venv.
#   2. Deletes the poisoned venv312/ and the py3.10-era uv.lock.
#   3. Backs up the original pyproject.toml -> pyproject-py310.toml
#      and activates pyproject-py312.toml as the new pyproject.toml.
#   4. Creates a fresh py3.12 venv at .\venv312
#   5. Installs uv into it.
#   6. Runs `uv sync` -> resolves + installs the full dep tree in one shot.
#   7. Installs CPU-only torch explicitly (the GTX 1650 + 4GB VRAM is too small
#      for CUDA torch anyway; Ollama owns the GPU for inference).
#   8. Freezes the final resolved set to requirements-py312.txt for the thesis
#      reproducibility record.
#
# Usage:
#   PS> cd "D:\ITMO\Semester 4\Project Root\Multi-Agent-Study-Coding-Assistant"
#   PS> .\setup_py312.ps1
#
# If PowerShell blocks execution, run once:
#   PS> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# ============================================================================

$ErrorActionPreference = "Stop"

Write-Host "=== ABC Benchmarking MAS :: py3.12 migration ===" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# 0. Sanity checks
# ---------------------------------------------------------------------------
if (-not (Test-Path ".\pyproject.toml")) {
    Write-Error "pyproject.toml not found. Run this from the project root."
    exit 1
}
if (-not (Test-Path ".\pyproject-py312.toml")) {
    Write-Error "pyproject-py312.toml not found. It must ship alongside this script."
    exit 1
}

# Verify py 3.12 is available
try {
    $pyVersion = & py -3.12 --version 2>&1
    Write-Host "[OK] Found: $pyVersion" -ForegroundColor Green
} catch {
    Write-Error "Python 3.12 not found. Install it from python.org and ensure 'py -3.12' works."
    exit 1
}

# ---------------------------------------------------------------------------
# 1. Deactivate any active venv
# ---------------------------------------------------------------------------
if ($env:VIRTUAL_ENV) {
    Write-Host "[..] Active venv detected: $env:VIRTUAL_ENV" -ForegroundColor Yellow
    Write-Host "     Please run 'deactivate' and re-run this script." -ForegroundColor Yellow
    exit 1
}

# ---------------------------------------------------------------------------
# 2. Nuke the poisoned venv and old lockfile
# ---------------------------------------------------------------------------
if (Test-Path ".\venv312") {
    Write-Host "[..] Removing old venv312/ ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".\venv312"
}
if (Test-Path ".\.venv") {
    Write-Host "[..] Removing stray .venv/ (uv's default target) ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".\.venv"
}
if (Test-Path ".\uv.lock") {
    Write-Host "[..] Removing py3.10-era uv.lock ..." -ForegroundColor Yellow
    Remove-Item -Force ".\uv.lock"
}

# ---------------------------------------------------------------------------
# 3. Swap pyproject.toml -> back up original, activate the py3.12 variant
# ---------------------------------------------------------------------------
if (-not (Test-Path ".\pyproject-py310.toml")) {
    Write-Host "[..] Backing up original pyproject.toml -> pyproject-py310.toml" -ForegroundColor Yellow
    Copy-Item ".\pyproject.toml" ".\pyproject-py310.toml"
}
Write-Host "[..] Activating pyproject-py312.toml as pyproject.toml" -ForegroundColor Yellow
Copy-Item ".\pyproject-py312.toml" ".\pyproject.toml" -Force

# ---------------------------------------------------------------------------
# 4. Fresh py3.12 venv
# ---------------------------------------------------------------------------
Write-Host "[..] Creating fresh py3.12 venv at .\venv312 ..." -ForegroundColor Yellow
& py -3.12 -m venv venv312

# Activate it for the rest of the script
$venvActivate = ".\venv312\Scripts\Activate.ps1"
. $venvActivate

Write-Host "[OK] Activated: $env:VIRTUAL_ENV" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 5. Install uv
# ---------------------------------------------------------------------------
Write-Host "[..] Upgrading pip + installing uv ..." -ForegroundColor Yellow
& python -m pip install --upgrade pip wheel setuptools | Out-Null
& python -m pip install uv

# ---------------------------------------------------------------------------
# 6. uv sync -> one-shot resolve + install
# ---------------------------------------------------------------------------
Write-Host "[..] Running 'uv sync --active' (resolves + installs into venv312) ..." -ForegroundColor Yellow
Write-Host "     Expected time: 30-90 seconds on a decent connection." -ForegroundColor DarkGray
# --active forces uv to target the currently activated venv (venv312) instead
# of its default .venv. Without this flag, uv silently creates a sibling .venv
# and installs everything there, bypassing our venv312.
& uv sync --active

if ($LASTEXITCODE -ne 0) {
    Write-Error "uv sync failed. See output above."
    exit 1
}

# ---------------------------------------------------------------------------
# 7. Force CPU-only torch
# ---------------------------------------------------------------------------
# uv sync may have pulled CUDA torch depending on resolver choices. On a
# GTX 1650 with 4GB VRAM + 16GB RAM, CPU torch is safer and smaller. Ollama
# handles the actual LLM inference on GPU separately.
Write-Host "[..] Pinning torch to CPU-only build ..." -ForegroundColor Yellow
& uv pip install --reinstall "torch" --index-url https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# 8. Freeze lockfile
# ---------------------------------------------------------------------------
Write-Host "[..] Freezing resolved set -> requirements-py312.txt" -ForegroundColor Yellow
& uv pip freeze | Out-File -Encoding utf8 ".\requirements-py312.txt"

# ---------------------------------------------------------------------------
# 9. Smoke test
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[..] Smoke test: import critical packages ..." -ForegroundColor Yellow

$smokeTest = @"
import sys
from importlib.metadata import version, PackageNotFoundError

def v(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return 'MISSING'

print(f'Python: {sys.version.split()[0]}')
# Import each package (fails loudly if anything is actually missing)
import langgraph, langchain, langchain_core, streamlit, chromadb
import sentence_transformers, openhands, torch

# Version reporting via importlib.metadata (robust across versions)
print(f'langgraph:             {v(\"langgraph\")}')
print(f'langchain:             {v(\"langchain\")}')
print(f'langchain-core:        {v(\"langchain-core\")}')
print(f'streamlit:             {v(\"streamlit\")}')
print(f'chromadb:              {v(\"chromadb\")}')
print(f'sentence-transformers: {v(\"sentence-transformers\")}')
print(f'openhands-sdk:         {v(\"openhands-sdk\")}')
print(f'torch:                 {torch.__version__} (cuda={torch.cuda.is_available()})')
print('[OK] All critical packages importable')
"@

& python -c $smokeTest

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Apply the code patches listed in py312_migration_patches.md" -ForegroundColor White
Write-Host "  2. Test Condition A/B:  streamlit run app.py" -ForegroundColor White
Write-Host "  3. Build openhands_agent.py for Condition C" -ForegroundColor White
Write-Host ""
Write-Host "Lockfile saved: requirements-py312.txt" -ForegroundColor DarkGray
Write-Host "Original py3.10 pyproject backed up: pyproject-py310.toml" -ForegroundColor DarkGray
