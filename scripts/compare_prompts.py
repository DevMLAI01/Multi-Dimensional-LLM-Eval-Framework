"""
Phase 9.2 — Prompt Version Comparator.

Shows which prompt files changed between two eval runs by comparing the
prompt hashes stored in eval_runs.prompt_versions.

Usage:
    uv run python scripts/compare_prompts.py --run-a baseline --run-b pr-42
    uv run python scripts/compare_prompts.py --run-a baseline --run-b pr-42 --db reports/eval_results.db
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def compare_prompt_versions(run_a_id: str, run_b_id: str, db_path: Path = None) -> dict:
    """Compare prompt hashes between two eval runs.

    Returns:
        {
            "changed": [{"prompt": name, "hash_a": str, "hash_b": str}],
            "only_in_a": [name],
            "only_in_b": [name],
            "unchanged": [name],
        }
    """
    from eval_runner.results_store import ResultsStore

    store = ResultsStore(db_path) if db_path else ResultsStore()

    run_a = store.get_run(run_a_id)
    run_b = store.get_run(run_b_id)

    if not run_a:
        raise ValueError(f"Run '{run_a_id}' not found in database")
    if not run_b:
        raise ValueError(f"Run '{run_b_id}' not found in database")

    versions_a = json.loads(run_a.get("prompt_versions") or "{}")
    versions_b = json.loads(run_b.get("prompt_versions") or "{}")

    all_prompts = set(versions_a) | set(versions_b)
    changed, only_a, only_b, unchanged = [], [], [], []

    for prompt in sorted(all_prompts):
        if prompt in versions_a and prompt not in versions_b:
            only_a.append(prompt)
        elif prompt in versions_b and prompt not in versions_a:
            only_b.append(prompt)
        elif versions_a[prompt]["hash"] != versions_b[prompt]["hash"]:
            changed.append({
                "prompt": prompt,
                "hash_a": versions_a[prompt]["hash"],
                "hash_b": versions_b[prompt]["hash"],
                "file": versions_a[prompt].get("file", f"{prompt}.yaml"),
            })
        else:
            unchanged.append(prompt)

    return {
        "run_a": run_a_id,
        "run_b": run_b_id,
        "changed": changed,
        "only_in_a": only_a,
        "only_in_b": only_b,
        "unchanged": unchanged,
    }


def print_comparison(diff: dict):
    print(f"\nPrompt comparison: {diff['run_a']} → {diff['run_b']}")
    print("=" * 55)

    if diff["changed"]:
        print(f"\nCHANGED ({len(diff['changed'])}):")
        for c in diff["changed"]:
            print(f"  {c['file']}")
            print(f"    {c['hash_a']} → {c['hash_b']}")
    else:
        print("\nNo prompt changes detected.")

    if diff["only_in_a"]:
        print(f"\nOnly in {diff['run_a']} (removed):")
        for p in diff["only_in_a"]:
            print(f"  {p}")

    if diff["only_in_b"]:
        print(f"\nOnly in {diff['run_b']} (added):")
        for p in diff["only_in_b"]:
            print(f"  {p}")

    if diff["unchanged"]:
        print(f"\nUnchanged: {', '.join(diff['unchanged'])}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare prompt versions between two eval runs")
    parser.add_argument("--run-a", required=True, help="Baseline run ID")
    parser.add_argument("--run-b", required=True, help="Comparison run ID")
    parser.add_argument("--db", default=None, help="Path to SQLite DB (default: reports/eval_results.db)")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None

    try:
        diff = compare_prompt_versions(args.run_a, args.run_b, db_path)
        print_comparison(diff)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
