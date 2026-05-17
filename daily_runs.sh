#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Daily Run Commands
# Project: ABC Benchmarking MAS, ITMO University
#
# HOW TO USE:
#   1. cd ~/projects/Multi-Agent-Study-Coding-Assistant
#   2. source venv312/bin/activate
#   3. Copy and run ONE command per day (Day 1, Day 2, ...)
#   4. Results accumulate in results/sweep_phase1.json automatically
#   5. After all 20 days: python analyze_results.py results/sweep_phase1.json
#
# SAFE TO RE-RUN: --resume skips any instance already in the output file.
# OUTPUT FILE:    results/sweep_phase1.json  (same file every day)
# =============================================================================

OUTPUT="results/sweep_phase1.json"

# --- Django (7 instances) ---

# Day 1
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-10914 --output $OUTPUT --resume

# Day 2
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-11099 --output $OUTPUT --resume

# Day 3
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-11583 --output $OUTPUT --resume

# Day 4
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-12113 --output $OUTPUT --resume

# Day 5
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-13230 --output $OUTPUT --resume

# Day 6
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-14534 --output $OUTPUT --resume

# Day 7
python swe_bench_runner.py --swe-lite --variant all --instance-ids django__django-15738 --output $OUTPUT --resume

# --- Sympy (5 instances) ---

# Day 8
python swe_bench_runner.py --swe-lite --variant all --instance-ids sympy__sympy-13437 --output $OUTPUT --resume

# Day 9
python swe_bench_runner.py --swe-lite --variant all --instance-ids sympy__sympy-14024 --output $OUTPUT --resume

# Day 10
python swe_bench_runner.py --swe-lite --variant all --instance-ids sympy__sympy-15308 --output $OUTPUT --resume

# Day 11
python swe_bench_runner.py --swe-lite --variant all --instance-ids sympy__sympy-16792 --output $OUTPUT --resume

# Day 12
python swe_bench_runner.py --swe-lite --variant all --instance-ids sympy__sympy-17022 --output $OUTPUT --resume

# --- Scikit-learn (4 instances) ---

# Day 13
python swe_bench_runner.py --swe-lite --variant all --instance-ids scikit-learn__scikit-learn-10297 --output $OUTPUT --resume

# Day 14
python swe_bench_runner.py --swe-lite --variant all --instance-ids scikit-learn__scikit-learn-13142 --output $OUTPUT --resume

# Day 15
python swe_bench_runner.py --swe-lite --variant all --instance-ids scikit-learn__scikit-learn-14894 --output $OUTPUT --resume

# Day 16
python swe_bench_runner.py --swe-lite --variant all --instance-ids scikit-learn__scikit-learn-25500 --output $OUTPUT --resume

# --- Astropy (4 instances) ---

# Day 17
python swe_bench_runner.py --swe-lite --variant all --instance-ids astropy__astropy-12907 --output $OUTPUT --resume

# Day 18
python swe_bench_runner.py --swe-lite --variant all --instance-ids astropy__astropy-13033 --output $OUTPUT --resume

# Day 19
python swe_bench_runner.py --swe-lite --variant all --instance-ids astropy__astropy-14182 --output $OUTPUT --resume

# Day 20
python swe_bench_runner.py --swe-lite --variant all --instance-ids astropy__astropy-14365 --output $OUTPUT --resume

# =============================================================================
# FINAL ANALYSIS (run after Day 20)
# =============================================================================
# python analyze_results.py results/sweep_phase1.json
