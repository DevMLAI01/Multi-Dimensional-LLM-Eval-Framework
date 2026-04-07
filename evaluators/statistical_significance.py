"""
Phase 7 — Statistical Significance Testing.

Paired t-tests on quality scores across model configurations.
Uses scipy.stats.ttest_rel for paired comparisons (same cases, different configs).

Decision rule:
    Significant if p_value < 0.05 AND abs(quality_delta) > 0.05

Usage:
    from evaluators.statistical_significance import compare_configs
    results = compare_configs(records)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from evaluators.latency_quality_evaluator import LatencyRunRecord

log = logging.getLogger(__name__)


@dataclass
class PairwiseTestResult:
    config_a: str
    config_b: str
    n_pairs: int
    mean_score_a: float
    mean_score_b: float
    quality_delta: float          # mean_b - mean_a
    p_value: float
    ci_lower: float               # 95% CI on delta
    ci_upper: float
    significant: bool             # p < 0.05 AND |delta| > 0.05
    interpretation: str


def _align_pairs(
    records: list[LatencyRunRecord],
    config_a: str,
    config_b: str,
) -> tuple[list[float], list[float]]:
    """Return aligned quality score lists for two configs (same case_id order)."""
    a_map = {r.case_id: r.quality_score for r in records
             if r.config_name == config_a and r.error is None}
    b_map = {r.case_id: r.quality_score for r in records
             if r.config_name == config_b and r.error is None}

    # Only include cases where both configs succeeded
    shared_ids = sorted(set(a_map) & set(b_map))
    scores_a = [a_map[cid] for cid in shared_ids]
    scores_b = [b_map[cid] for cid in shared_ids]
    return scores_a, scores_b


def paired_ttest(
    scores_a: list[float],
    scores_b: list[float],
    config_a: str,
    config_b: str,
    alpha: float = 0.05,
    min_effect_size: float = 0.05,
) -> PairwiseTestResult:
    """Paired t-test: is config_b significantly better/worse than config_a?

    Args:
        scores_a: Quality scores for config A (baseline)
        scores_b: Quality scores for config B
        config_a: Name of baseline config
        config_b: Name of comparison config
        alpha: Significance level (default 0.05)
        min_effect_size: Minimum quality delta to call "meaningful" (default 0.05)

    Returns:
        PairwiseTestResult with p-value, CI, and significance verdict.
    """
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return PairwiseTestResult(
            config_a=config_a, config_b=config_b, n_pairs=len(scores_a),
            mean_score_a=0.0, mean_score_b=0.0, quality_delta=0.0,
            p_value=1.0, ci_lower=0.0, ci_upper=0.0, significant=False,
            interpretation="Insufficient data for statistical test",
        )

    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)

    t_stat, p_value = stats.ttest_rel(arr_b, arr_a)  # H0: b - a == 0

    # 95% confidence interval on the mean difference
    diffs = arr_b - arr_a
    n = len(diffs)
    se = stats.sem(diffs)
    ci = stats.t.interval(0.95, df=n - 1, loc=np.mean(diffs), scale=se)

    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    delta = round(mean_b - mean_a, 4)
    p_val = round(float(p_value), 6)

    significant = (p_val < alpha) and (abs(delta) > min_effect_size)

    # Build human-readable interpretation
    if significant:
        direction = "better" if delta > 0 else "worse"
        interpretation = (
            f"{config_b} is significantly {direction} than {config_a} "
            f"(Δ={delta:+.3f}, p={p_val:.4f}). Effect is real — not noise."
        )
    elif p_val < alpha:
        interpretation = (
            f"Statistically significant (p={p_val:.4f}) but effect too small "
            f"(Δ={delta:+.3f} < {min_effect_size}). Difference is negligible."
        )
    else:
        interpretation = (
            f"No significant difference between {config_a} and {config_b} "
            f"(Δ={delta:+.3f}, p={p_val:.4f}). Configs are statistically equivalent."
        )

    return PairwiseTestResult(
        config_a=config_a,
        config_b=config_b,
        n_pairs=n,
        mean_score_a=round(mean_a, 4),
        mean_score_b=round(mean_b, 4),
        quality_delta=delta,
        p_value=p_val,
        ci_lower=round(float(ci[0]), 4),
        ci_upper=round(float(ci[1]), 4),
        significant=significant,
        interpretation=interpretation,
    )


def compare_configs(
    records: list[LatencyRunRecord],
    baseline: str = "haiku-all",
    comparisons: list[tuple[str, str]] | None = None,
) -> dict[str, PairwiseTestResult]:
    """Run all pairwise significance tests.

    Args:
        records:     All LatencyRunRecords from LatencyQualityEvaluator.run_all_configs()
        baseline:    Reference config for naming (not used for directionality)
        comparisons: List of (config_a, config_b) pairs to test.
                     Defaults to the three Phase 7 pairs.

    Returns:
        Dict keyed as "configA_vs_configB" → PairwiseTestResult
    """
    if comparisons is None:
        comparisons = [
            ("haiku-all", "hybrid"),
            ("haiku-all", "sonnet-all"),
            ("hybrid",    "sonnet-all"),
        ]

    results: dict[str, PairwiseTestResult] = {}
    for (a, b) in comparisons:
        scores_a, scores_b = _align_pairs(records, a, b)
        log.info("t-test: %s vs %s  n=%d", a, b, len(scores_a))
        key = f"{a}_vs_{b}"
        results[key] = paired_ttest(scores_a, scores_b, a, b)

    return results


def format_test_results(test_results: dict[str, PairwiseTestResult]) -> str:
    """Return a human-readable summary of all pairwise tests."""
    lines = ["Statistical Significance Tests (paired t-test, α=0.05, min_effect=0.05)", "=" * 70]
    for key, res in test_results.items():
        lines.append(f"\n{res.config_a} vs {res.config_b} (n={res.n_pairs} pairs):")
        lines.append(f"  Mean scores: {res.mean_score_a:.3f} vs {res.mean_score_b:.3f}")
        lines.append(f"  Delta:       {res.quality_delta:+.4f}")
        lines.append(f"  p-value:     {res.p_value:.6f}  {'*SIGNIFICANT*' if res.significant else ''}")
        lines.append(f"  95% CI:      [{res.ci_lower:.4f}, {res.ci_upper:.4f}]")
        lines.append(f"  Verdict:     {res.interpretation}")
    return "\n".join(lines)
