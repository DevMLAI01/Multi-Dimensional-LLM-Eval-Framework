"""
Phase 8.2 — Eval Suite Runner.

Single entry point for running the full evaluation suite across all dimensions.
Writes results to SQLite incrementally (each result saved immediately — partial
runs are recoverable by checking existing results).

Usage:
    # Python API
    from eval_runner.runner import run_eval_suite
    summary = run_eval_suite(run_id="baseline")

    # CLI
    uv run python eval_runner/runner.py --run-id baseline
    uv run python eval_runner/runner.py --run-id pr-42 --compare-to baseline
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parents[1] / "data" / "golden_dataset"
_REPORTS_DIR = Path(__file__).parents[1] / "reports"

# Which JSON file backs each dimension
_DIMENSION_FILES = {
    "correctness":  "correctness_cases.json",
    "faithfulness": "faithfulness_cases.json",
    "robustness":   "robustness_cases.json",
    "safety":       "safety_cases.json",
    "latency":      "latency_cases.json",
}

_MAX_RETRIES = 3
_RETRY_DELAY = 5.0   # seconds between retries on API timeout


@dataclass
class EvalRunSummary:
    run_id: str
    dimension_scores: dict[str, dict]  # {dimension: {mean_score, pass_rate, ...}}
    overall_score: float
    safety_cap_applied: bool
    total_cases: int
    passed_cases: int
    total_errors: int
    duration_seconds: float
    total_cost_usd: float
    regressions: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    regression_passed: bool = True


def _load_cases(dimension: str) -> list[dict]:
    path = _DATA_DIR / _DIMENSION_FILES[dimension]
    if not path.exists():
        raise FileNotFoundError(f"Cases file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_evaluator(dimension: str, model_config: str = "hybrid"):
    """Return the evaluator instance for a given dimension."""
    if dimension == "correctness":
        from evaluators.correctness_evaluator import CorrectnessEvaluator
        return CorrectnessEvaluator()
    elif dimension == "faithfulness":
        from evaluators.faithfulness_evaluator import FaithfulnessEvaluator
        return FaithfulnessEvaluator()
    elif dimension == "robustness":
        from evaluators.robustness_evaluator import RobustnessEvaluator
        return RobustnessEvaluator()
    elif dimension == "safety":
        from evaluators.safety_evaluator import SafetyEvaluator
        return SafetyEvaluator()
    elif dimension == "latency":
        from evaluators.latency_quality_evaluator import LatencyQualityEvaluator
        return LatencyQualityEvaluator()
    else:
        raise ValueError(f"Unknown dimension: {dimension}")


def _run_agent_with_retry(alarm, max_retries: int = _MAX_RETRIES):
    """Run agent with exponential backoff on API errors."""
    from agent.noc_agent import run_agent
    last_exc = None
    for attempt in range(max_retries):
        try:
            return run_agent(alarm)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = _RETRY_DELAY * (2 ** attempt)
                log.warning("Agent error (attempt %d/%d): %s. Retrying in %.0fs...",
                             attempt + 1, max_retries, exc, wait)
                time.sleep(wait)
    raise last_exc


def _run_dimension(
    dimension: str,
    cases: list[dict],
    store,
    run_id: str,
    model_config: str,
    start_index: int = 0,
) -> list:
    """Run all cases for one dimension, returning EvalResult list."""
    from evaluators.base_evaluator import EvalResult

    evaluator = _get_evaluator(dimension, model_config)
    results = []
    n = len(cases)

    for i, case in enumerate(cases):
        global_idx = start_index + i
        case_id = case.get("case_id", f"{dimension}_{i}")

        try:
            if dimension == "robustness":
                # Robustness evaluator runs the agent itself on both inputs
                result = evaluator.evaluate(case)

            elif dimension == "latency":
                # Latency evaluator runs all 3 configs — pick the model_config result
                record = evaluator.run_single(case, model_config)
                result = record.to_eval_result()

            elif dimension == "safety":
                # Safety evaluator runs the agent itself
                result = evaluator.evaluate(case)

            else:
                # Correctness and faithfulness: run agent first, then evaluate
                from agent.models import AlarmEvent
                inp = case.get("input", {})
                alarm_data = inp.get("alarm_event", inp)
                alarm = AlarmEvent(**alarm_data)
                diagnosis = _run_agent_with_retry(alarm)
                result = evaluator.evaluate(case, diagnosis)

        except Exception as exc:
            log.error("[%s/%s] Unexpected error: %s", dimension, case_id, exc)
            from evaluators.base_evaluator import EvalResult
            result = EvalResult(
                case_id=case_id,
                dimension=dimension,
                evaluator_version="1.0",
                score=None,
                passed=False,
                error=f"runner_error: {exc}",
            )

        results.append(result)
        store.save_result(run_id, result)

        # Progress line
        valid_scores = [r.score for r in results if r.score is not None]
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        tick = "✓" if result.passed else "✗"
        print(
            f"  [{global_idx + 1:>3}/{start_index + n}] "
            f"{dimension}: {avg:.2f} avg {tick}",
            flush=True,
        )

    return results


def run_eval_suite(
    run_id: str,
    dimensions: list[str] | None = None,
    model_config: str = "hybrid",
    git_commit: Optional[str] = None,
    compare_to: Optional[str] = None,
    triggered_by: str = "manual",
    db_path: Optional[Path] = None,
) -> EvalRunSummary:
    """Run the full eval suite and return an EvalRunSummary.

    Args:
        run_id:       Unique identifier for this run (e.g. "baseline", "pr-42")
        dimensions:   Which dimensions to evaluate. Default: all 5.
        model_config: Model config for latency and robustness agent runs.
        git_commit:   Git commit SHA (from CI env).
        compare_to:   run_id to compare against for regression check.
        triggered_by: "manual" | "CI" | "scheduled"
        db_path:      Override the default SQLite path.

    Returns:
        EvalRunSummary with all scores, costs, and regression info.
    """
    if dimensions is None:
        dimensions = list(_DIMENSION_FILES.keys())

    from eval_runner.results_store import ResultsStore
    from eval_runner.scorer import compute_overall_score, scores_from_results, DimensionScore
    from eval_runner.regression_checker import RegressionChecker
    from agent.prompts.prompt_registry import hash_prompts

    store = ResultsStore(db_path) if db_path else ResultsStore()

    # Collect prompt versions for tracking
    try:
        prompt_versions = hash_prompts()
    except Exception:
        prompt_versions = {}

    store.create_run(
        run_id=run_id,
        model_config=model_config,
        git_commit=git_commit,
        prompt_versions=prompt_versions,
        triggered_by=triggered_by,
    )

    t_start = time.monotonic()
    all_results = []
    total_cases = sum(len(_load_cases(d)) for d in dimensions if
                      (_DATA_DIR / _DIMENSION_FILES[d]).exists())

    print(f"\n{'='*60}")
    print(f"Eval Suite — run_id={run_id}  config={model_config}")
    print(f"Dimensions: {', '.join(dimensions)}")
    print(f"Total cases: {total_cases}")
    print(f"{'='*60}")

    case_offset = 0
    for dimension in dimensions:
        print(f"\n[{dimension.upper()}]")
        try:
            cases = _load_cases(dimension)
        except FileNotFoundError as exc:
            log.warning("%s — skipping dimension", exc)
            continue

        dim_results = _run_dimension(
            dimension=dimension,
            cases=cases,
            store=store,
            run_id=run_id,
            model_config=model_config,
            start_index=case_offset,
        )
        all_results.extend(dim_results)
        case_offset += len(cases)

        # Summarise dimension
        valid = [r for r in dim_results if r.score is not None]
        passed = [r for r in dim_results if r.passed]
        mean_score = sum(r.score for r in valid) / len(valid) if valid else 0.0
        pass_rate  = len(passed) / len(dim_results) if dim_results else 0.0
        errors     = sum(1 for r in dim_results if r.error)

        from eval_runner.regression_checker import THRESHOLDS
        threshold = THRESHOLDS.get(dimension, 0.0)

        store.save_dimension_summary(
            run_id=run_id,
            dimension=dimension,
            mean_score=round(mean_score, 4),
            pass_rate=round(pass_rate, 4),
            threshold=threshold,
            cases_run=len(dim_results),
            cases_passed=len(passed),
            cases_failed=len(dim_results) - len(passed),
        )

        status = "PASS" if mean_score >= threshold else "FAIL"
        print(
            f"  → {dimension}: score={mean_score:.3f} pass_rate={pass_rate:.1%} "
            f"errors={errors} [{status}]"
        )

    # Overall score
    dim_score_objs = scores_from_results(all_results)
    overall = compute_overall_score(dim_score_objs)
    duration = time.monotonic() - t_start

    # Estimate total cost from metadata
    total_cost = sum(
        r.metadata.get("cost_usd", 0.0) or r.metadata.get("estimated_cost_usd", 0.0)
        for r in all_results
        if r.metadata
    )

    store.finalize_run(
        run_id=run_id,
        overall_score=overall.weighted_score,
        duration_seconds=duration,
        total_cost_usd=total_cost,
    )

    # Regression check
    regressions, warnings_list, regression_passed = [], [], True
    if compare_to:
        checker = RegressionChecker(store)
        current_summaries = store.get_dimension_summaries(run_id)
        report = checker.check(current_summaries, compare_to, run_id)
        regressions = [
            {"dimension": r.dimension, "severity": r.severity, "delta": r.delta}
            for r in report.regressions
        ]
        warnings_list = [
            {"dimension": w.dimension, "severity": w.severity, "delta": w.delta}
            for w in report.warnings
        ]
        regression_passed = report.passed

    # Print final summary
    total_passed = sum(1 for r in all_results if r.passed)
    total_errors = sum(1 for r in all_results if r.error)

    print(f"\n{'='*60}")
    print(f"SUMMARY — run_id={run_id}")
    print(f"Overall score:  {overall.weighted_score:.3f}"
          + (" [SAFETY CAP APPLIED]" if overall.safety_cap_applied else ""))
    print(f"Cases: {len(all_results)} total, {total_passed} passed, {total_errors} errors")
    print(f"Duration: {duration:.0f}s  Cost: ${total_cost:.4f}")
    if regressions:
        print(f"REGRESSIONS: {regressions}")
    if warnings_list:
        print(f"WARNINGS: {warnings_list}")
    print(f"{'='*60}\n")

    # Save summary JSON
    _REPORTS_DIR.mkdir(exist_ok=True)
    summary_path = _REPORTS_DIR / f"eval_summary_{run_id}.json"
    summary_data = {
        "run_id": run_id,
        "overall_score": overall.weighted_score,
        "safety_cap_applied": overall.safety_cap_applied,
        "dimension_scores": {
            d: {
                "mean_score": s.mean_score,
                "pass_rate": s.pass_rate,
                "cases_run": s.cases_run,
                "cases_passed": s.cases_passed,
                "threshold": s.threshold,
            }
            for d, s in dim_score_objs.items()
        },
        "regressions": regressions,
        "warnings": warnings_list,
        "regression_passed": regression_passed,
        "duration_seconds": round(duration, 1),
        "total_cost_usd": round(total_cost, 4),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return EvalRunSummary(
        run_id=run_id,
        dimension_scores={d: vars(s) for d, s in dim_score_objs.items()},
        overall_score=overall.weighted_score,
        safety_cap_applied=overall.safety_cap_applied,
        total_cases=len(all_results),
        passed_cases=total_passed,
        total_errors=total_errors,
        duration_seconds=duration,
        total_cost_usd=total_cost,
        regressions=regressions,
        warnings=warnings_list,
        regression_passed=regression_passed,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Run the LLM eval suite")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument(
        "--dimensions", nargs="+",
        choices=list(_DIMENSION_FILES.keys()),
        default=None,
        help="Which dimensions to evaluate (default: all)",
    )
    parser.add_argument(
        "--model-config", default="hybrid",
        choices=["haiku-all", "sonnet-all", "hybrid"],
        help="Model configuration for agent (default: hybrid)",
    )
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--compare-to", default=None,
                        help="run_id of baseline to compare against")
    parser.add_argument("--triggered-by", default="manual")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    args = _parse_args()
    summary = run_eval_suite(
        run_id=args.run_id,
        dimensions=args.dimensions,
        model_config=args.model_config,
        git_commit=args.git_commit,
        compare_to=args.compare_to,
        triggered_by=args.triggered_by,
    )

    # Exit 1 if regressions detected (for CI gating)
    if not summary.regression_passed:
        print("REGRESSION DETECTED — exiting with code 1", file=sys.stderr)
        sys.exit(1)
