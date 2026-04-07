"""
Phase 9.3 — Prompt A/B Test Helper.

Runs the eval suite against two versions of a prompt and determines whether
the quality difference is statistically significant.

Workflow:
  1. Copy prompt_b over the active prompt file
  2. Run eval suite for dimension on N cases → scores_b
  3. Restore prompt_a (already active), run same cases → scores_a
  4. Paired t-test: is b significantly different from a?
  5. Print decision + restore original prompt

Usage:
    uv run python eval_runner/ab_test.py \\
        --prompt-a agent/prompts/reasoner_v1.yaml \\
        --prompt-b agent/prompts/reasoner_v2.yaml \\
        --dimension correctness \\
        --cases 20

Output:
    "Prompt B is significantly better (p=0.02, +8% quality). Safe to deploy."
    or
    "No significant difference (p=0.31). Keep Prompt A."
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parents[1] / "data" / "golden_dataset"
_DIMENSION_FILES = {
    "correctness":  "correctness_cases.json",
    "faithfulness": "faithfulness_cases.json",
    "robustness":   "robustness_cases.json",
    "safety":       "safety_cases.json",
}


def _load_cases(dimension: str, n: int) -> list[dict]:
    path = _DATA_DIR / _DIMENSION_FILES[dimension]
    if not path.exists():
        raise FileNotFoundError(f"Cases not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)[:n]


def _run_eval_for_prompt(
    prompt_file: Path,
    active_prompt_path: Path,
    cases: list[dict],
    dimension: str,
    run_id: str,
    db_path: Path,
) -> list[float]:
    """Swap in prompt_file, run eval, restore original, return quality scores."""
    from eval_runner.runner import _run_dimension
    from eval_runner.results_store import ResultsStore

    # Back up current prompt
    backup = active_prompt_path.with_suffix(".yaml.bak")
    shutil.copy2(active_prompt_path, backup)

    try:
        # Swap in the test prompt
        shutil.copy2(prompt_file, active_prompt_path)
        log.info("Using prompt: %s → %s", prompt_file.name, active_prompt_path.name)

        store = ResultsStore(db_path=db_path)
        store.create_run(run_id, model_config="hybrid")

        results = _run_dimension(dimension, cases, store, run_id, "hybrid")
        scores = [r.score for r in results if r.score is not None]
        return scores

    finally:
        # Always restore original prompt
        shutil.copy2(backup, active_prompt_path)
        backup.unlink(missing_ok=True)
        log.info("Restored original prompt: %s", active_prompt_path.name)


def run_ab_test(
    prompt_a_path: Path,
    prompt_b_path: Path,
    dimension: str,
    n_cases: int = 20,
    alpha: float = 0.05,
    min_effect: float = 0.05,
) -> dict:
    """Run A/B test comparing two prompt files on a given dimension.

    Args:
        prompt_a_path: Path to prompt A (treated as the baseline / "current").
        prompt_b_path: Path to prompt B (the candidate to test).
        dimension:     Which eval dimension to use for scoring.
        n_cases:       How many cases to run (subset of golden dataset).
        alpha:         Significance level for t-test (default 0.05).
        min_effect:    Minimum quality delta to call "meaningful" (default 0.05).

    Returns:
        Result dict with decision, p_value, delta, and scores.
    """
    from evaluators.statistical_significance import paired_ttest

    if dimension not in _DIMENSION_FILES:
        raise ValueError(f"A/B test only supports: {list(_DIMENSION_FILES)}")

    cases = _load_cases(dimension, n_cases)
    actual_n = len(cases)
    log.info("Loaded %d cases for dimension '%s'", actual_n, dimension)

    # The active prompt path is determined by which YAML the evaluator reads.
    # For a prompt with path like agent/prompts/reasoner_v1.yaml, the active
    # file is that exact path.
    active_path = prompt_a_path   # A is the current active prompt

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "ab_test.db"

        log.info("=== Running Prompt A: %s ===", prompt_a_path.name)
        ts_a = f"ab_a_{int(time.time())}"
        scores_a = _run_eval_for_prompt(
            prompt_file=prompt_a_path,
            active_prompt_path=active_path,
            cases=cases,
            dimension=dimension,
            run_id=ts_a,
            db_path=db_path,
        )

        log.info("=== Running Prompt B: %s ===", prompt_b_path.name)
        ts_b = f"ab_b_{int(time.time())}"
        scores_b = _run_eval_for_prompt(
            prompt_file=prompt_b_path,
            active_prompt_path=active_path,
            cases=cases,
            dimension=dimension,
            run_id=ts_b,
            db_path=db_path,
        )

    # Align to pairs (both must succeed for the same case index)
    pairs_a, pairs_b = [], []
    for a, b in zip(scores_a, scores_b):
        pairs_a.append(a)
        pairs_b.append(b)

    ttest = paired_ttest(pairs_a, pairs_b, "prompt_a", "prompt_b", alpha, min_effect)

    mean_a = ttest.mean_score_a
    mean_b = ttest.mean_score_b
    delta  = ttest.quality_delta

    if ttest.significant and delta > 0:
        decision = (
            f"Prompt B is significantly BETTER "
            f"(p={ttest.p_value:.4f}, delta={delta:+.3f} / "
            f"{delta/max(mean_a,0.001)*100:+.1f}%). "
            f"Safe to deploy."
        )
    elif ttest.significant and delta < 0:
        decision = (
            f"Prompt B is significantly WORSE "
            f"(p={ttest.p_value:.4f}, delta={delta:+.3f}). "
            f"Do NOT deploy."
        )
    else:
        decision = (
            f"No significant difference (p={ttest.p_value:.4f}, "
            f"delta={delta:+.3f}). Keep Prompt A."
        )

    result = {
        "dimension":    dimension,
        "n_cases":      actual_n,
        "n_pairs":      ttest.n_pairs,
        "prompt_a":     prompt_a_path.name,
        "prompt_b":     prompt_b_path.name,
        "mean_score_a": round(mean_a, 4),
        "mean_score_b": round(mean_b, 4),
        "delta":        round(delta, 4),
        "p_value":      ttest.p_value,
        "ci_95":        [ttest.ci_lower, ttest.ci_upper],
        "significant":  ttest.significant,
        "decision":     decision,
    }

    print(f"\n{'='*60}")
    print("PROMPT A/B TEST RESULTS")
    print(f"{'='*60}")
    print(f"Dimension:  {dimension}  (n={actual_n} cases)")
    print(f"Prompt A:   {prompt_a_path.name}  mean={mean_a:.3f}")
    print(f"Prompt B:   {prompt_b_path.name}  mean={mean_b:.3f}")
    print(f"Delta:      {delta:+.4f}")
    print(f"p-value:    {ttest.p_value:.6f}  (α={alpha})")
    print(f"95% CI:     [{ttest.ci_lower:.4f}, {ttest.ci_upper:.4f}]")
    print(f"\n>>> {decision}")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B test two prompt versions")
    parser.add_argument("--prompt-a", required=True, help="Path to prompt A (baseline)")
    parser.add_argument("--prompt-b", required=True, help="Path to prompt B (candidate)")
    parser.add_argument(
        "--dimension", default="correctness",
        choices=list(_DIMENSION_FILES.keys()),
        help="Eval dimension to use for scoring (default: correctness)",
    )
    parser.add_argument("--cases", type=int, default=20,
                        help="Number of cases to run (default: 20)")
    args = parser.parse_args()

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    run_ab_test(
        prompt_a_path=Path(args.prompt_a),
        prompt_b_path=Path(args.prompt_b),
        dimension=args.dimension,
        n_cases=args.cases,
    )
