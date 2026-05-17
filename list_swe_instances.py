"""
list_swe_instances.py — SWE-bench Lite instance browser
========================================================

Run this in WSL before any sweep to see which instance IDs are available
in the dataset (grouped by repo), verify an existing config, or emit a
ready-to-use JSON manifest for the next sweep.

Usage (WSL, venv activated):
    python list_swe_instances.py
    python list_swe_instances.py --repos django/django sympy/sympy
    python list_swe_instances.py --repos django/django --max 10
    python list_swe_instances.py --check sweep_config.json

    # Phase 4 — emit a manifest of the next N untested instances,
    # excluding everything already covered by sweep_config.json:
    python list_swe_instances.py --emit-manifest sweep_phase4_night1.json \
                                  --max 50 \
                                  --exclude sweep_config.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_dataset_safe():
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] `datasets` package not installed.")
        print("        Run: uv pip install datasets  (inside WSL venv)")
        sys.exit(1)
    print("[INFO] Loading princeton-nlp/SWE-bench_Lite (test split) …")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"[INFO] {len(ds)} total instances in dataset\n")
    return ds


def cmd_list(args):
    ds = load_dataset_safe()

    # Group by repo
    by_repo = defaultdict(list)
    for item in ds:
        by_repo[item["repo"]].append(item["instance_id"])

    # Apply repo filter
    target_repos = set(args.repos) if args.repos else set(by_repo.keys())

    print(f"{'REPO':<45}  COUNT  INSTANCE IDs")
    print("=" * 90)
    grand_total = 0
    for repo in sorted(target_repos):
        ids = sorted(by_repo.get(repo, []))
        if not ids:
            print(f"  {repo:<43}  (not found in dataset)")
            continue
        cap = args.max if args.max else len(ids)
        shown = ids[:cap]
        suffix = f"  (+{len(ids)-cap} more)" if len(ids) > cap else ""
        print(f"\n{repo}  ({len(ids)} instances){suffix}")
        for iid in shown:
            print(f"    {iid}")
        grand_total += len(ids)

    print(f"\n{'='*90}")
    print(f"Total across selected repos: {grand_total}")


def _load_exclude_set(exclude_paths):
    """Read 'instance_ids' from one or more sweep config JSONs and return a set."""
    excluded = set()
    if not exclude_paths:
        return excluded
    for path in exclude_paths:
        p = Path(path)
        if not p.exists():
            print(f"[WARN] --exclude file not found: {p} (skipped)")
            continue
        try:
            with open(p, encoding="utf-8") as f:
                cfg = json.load(f)
            ids = cfg.get("instance_ids", [])
            excluded.update(ids)
            print(f"[INFO] --exclude {p}: {len(ids)} IDs added to skip-list")
        except Exception as e:
            print(f"[WARN] could not parse {p}: {e}")
    print(f"[INFO] total exclude set size: {len(excluded)}")
    return excluded


def cmd_emit_manifest(args):
    """Write a sweep_config-shaped JSON containing the next N candidate IDs,
    excluding everything in --exclude files. Intended for Phase 4 nightly runs."""
    ds = load_dataset_safe()
    excluded = _load_exclude_set(args.exclude)

    candidates = []
    for item in ds:
        iid = item["instance_id"]
        repo = item["repo"]
        if iid in excluded:
            continue
        if args.repos and repo not in args.repos:
            continue
        candidates.append((repo, iid))

    if args.max:
        # Round-robin across repos so a night's manifest isn't dominated by one repo.
        by_repo = defaultdict(list)
        for repo, iid in candidates:
            by_repo[repo].append(iid)
        ordered_repos = sorted(by_repo.keys())
        picked = []
        while len(picked) < args.max and any(by_repo.values()):
            for repo in ordered_repos:
                if by_repo[repo] and len(picked) < args.max:
                    picked.append(by_repo[repo].pop(0))
        chosen_ids = picked
    else:
        chosen_ids = [iid for _, iid in candidates]

    # Tally per-repo distribution for the log
    chosen_by_repo = defaultdict(int)
    for iid in chosen_ids:
        chosen_by_repo[iid.split("__")[0].replace("-", "/")] += 1

    output = {
        "_comment": f"Auto-generated manifest by list_swe_instances.py --emit-manifest "
                    f"on {Path('/tmp').name}",
        "_generated": True,
        "_source_dataset": "princeton-nlp/SWE-bench_Lite",
        "_excluded_from": list(args.exclude or []),
        "_count": len(chosen_ids),
        "_distribution_by_repo_prefix": dict(chosen_by_repo),
        "instance_ids": chosen_ids,
    }

    out_path = Path(args.emit_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Wrote {len(chosen_ids)} IDs to {out_path}")
    print(f"     Distribution: {dict(chosen_by_repo)}")
    print(f"     Use with:  --instance-ids @{out_path}")


def cmd_check(args):
    """Verify every ID in a sweep_config.json is present in the dataset."""
    cfg_path = Path(args.check)
    if not cfg_path.exists():
        print(f"[ERROR] File not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    requested = cfg.get("instance_ids", [])
    if not requested:
        print("[ERROR] No 'instance_ids' key in config.")
        sys.exit(1)
    print(f"[CHECK] Verifying {len(requested)} IDs from {cfg_path} against dataset …\n")

    ds = load_dataset_safe()
    available = {item["instance_id"] for item in ds}

    ok = [x for x in requested if x in available]
    missing = [x for x in requested if x not in available]

    print(f"  Found   : {len(ok)}/{len(requested)}")
    if missing:
        print(f"  MISSING : {len(missing)} ID(s) not in dataset:")
        for m in missing:
            print(f"    - {m}")
        print("\nRemove or replace these IDs from sweep_config.json before running the sweep.")
    else:
        print("  All IDs verified ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Browse and verify SWE-bench Lite instance IDs."
    )
    parser.add_argument(
        "--repos", nargs="*", metavar="REPO",
        help="Filter output to these repos (e.g. django/django sympy/sympy)"
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Max IDs to show per repo (default: all)"
    )
    parser.add_argument(
        "--check", metavar="SWEEP_CONFIG_JSON",
        help="Verify all instance_ids in a sweep_config.json against the dataset"
    )
    parser.add_argument(
        "--emit-manifest", metavar="OUTPUT_JSON",
        help="Write a sweep_config-shaped JSON containing the next N candidate IDs "
             "(use with --max and --exclude)"
    )
    parser.add_argument(
        "--exclude", nargs="*", metavar="SWEEP_CONFIG_JSON",
        help="Skip any instance_id already present in these sweep_config files. "
             "Use to avoid re-running instances covered by earlier phases."
    )
    args = parser.parse_args()

    if args.check:
        cmd_check(args)
    elif args.emit_manifest:
        cmd_emit_manifest(args)
    else:
        cmd_list(args)
