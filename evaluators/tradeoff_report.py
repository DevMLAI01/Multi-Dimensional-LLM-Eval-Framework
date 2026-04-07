"""
Phase 7 — Tradeoff Report Generator.

Generates reports/latency_quality_tradeoff.json summarising the cost/quality
tradeoff across haiku-all / sonnet-all / hybrid model configurations.

Also provides a production recommendation based on:
  1. Statistical significance tests
  2. Cost efficiency (quality per dollar)
  3. Whether hybrid is significantly worse than sonnet-all

Usage:
    uv run python evaluators/tradeoff_report.py
"""

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parents[1] / "reports"
OUTPUT_FILE = REPORTS_DIR / "latency_quality_tradeoff.json"
DATA_DIR    = Path(__file__).parents[1] / "data" / "golden_dataset"


def _pick_recommendation(
    agg: dict[str, dict],
    stat_tests: dict,
) -> tuple[str, str]:
    """Choose production recommendation based on stats and cost.

    Strategy:
    - If hybrid is NOT significantly worse than sonnet-all AND
      hybrid is significantly better than haiku-all → recommend hybrid
    - If sonnet-all is significantly better than hybrid → recommend sonnet-all
    - Otherwise → recommend hybrid (cost/quality compromise)

    Returns (config_name, reasoning_text).
    """
    hybrid_vs_sonnet = stat_tests.get("hybrid_vs_sonnet-all")
    hybrid_vs_haiku  = stat_tests.get("haiku-all_vs_hybrid")

    hybrid_mean  = agg.get("hybrid", {}).get("mean_quality", 0.0)
    sonnet_mean  = agg.get("sonnet-all", {}).get("mean_quality", 0.0)
    haiku_mean   = agg.get("haiku-all", {}).get("mean_quality", 0.0)
    hybrid_cost  = agg.get("hybrid", {}).get("mean_cost_usd", 0.0)
    sonnet_cost  = agg.get("sonnet-all", {}).get("mean_cost_usd", 0.0)
    haiku_cost   = agg.get("haiku-all", {}).get("mean_cost_usd", 0.0)

    # Cost savings of hybrid vs sonnet-all
    cost_saving_pct = 0.0
    if sonnet_cost > 0:
        cost_saving_pct = round((sonnet_cost - hybrid_cost) / sonnet_cost * 100, 1)

    # Quality fraction of sonnet-all
    quality_fraction = 0.0
    if sonnet_mean > 0:
        quality_fraction = round(hybrid_mean / sonnet_mean * 100, 1)

    # Check if hybrid is significantly worse than sonnet (delta < -0.05 and p < 0.05)
    hybrid_worse_than_sonnet = False
    if hybrid_vs_sonnet:
        # hybrid_vs_sonnet: config_a=hybrid, config_b=sonnet-all
        # negative delta means hybrid < sonnet
        if hybrid_vs_sonnet.get("significant") and hybrid_vs_sonnet.get("quality_delta", 0) > 0:
            hybrid_worse_than_sonnet = True

    if hybrid_worse_than_sonnet:
        recommendation = "sonnet-all"
        reasoning = (
            f"Sonnet-all is statistically significantly better than hybrid "
            f"(Δ={hybrid_vs_sonnet.get('quality_delta', 0):+.3f}, "
            f"p={hybrid_vs_sonnet.get('p_value', 1.0):.4f}). "
            f"Despite {cost_saving_pct:.0f}% higher cost, the quality gain justifies it."
        )
    else:
        recommendation = "hybrid"
        reasoning = (
            f"Hybrid achieves {quality_fraction:.0f}% of sonnet-all quality "
            f"({hybrid_mean:.3f} vs {sonnet_mean:.3f}) at {cost_saving_pct:.0f}% lower cost "
            f"(${hybrid_cost:.5f} vs ${sonnet_cost:.5f} per call). "
            f"Difference from sonnet-all is not statistically significant "
            f"(p={hybrid_vs_sonnet.get('p_value', 'N/A') if hybrid_vs_sonnet else 'N/A'}). "
            f"Hybrid is the optimal production configuration."
        )

    return recommendation, reasoning


def generate_report(records=None, cases_path: Path | None = None) -> dict:
    """Generate the full tradeoff report.

    Args:
        records:    Pre-computed LatencyRunRecord list. If None, runs the full eval.
        cases_path: Path to latency_cases.json. Defaults to golden dataset.

    Returns:
        Report dict (also written to reports/latency_quality_tradeoff.json).
    """
    from evaluators.latency_quality_evaluator import LatencyQualityEvaluator
    from evaluators.statistical_significance import compare_configs, format_test_results

    if records is None:
        if cases_path is None:
            cases_path = DATA_DIR / "latency_cases.json"

        if not cases_path.exists():
            raise FileNotFoundError(f"latency_cases.json not found at {cases_path}")

        with open(cases_path, encoding="utf-8") as f:
            cases = json.load(f)

        log.info("Running %d cases × 3 configs = %d agent calls...", len(cases), len(cases) * 3)
        evaluator = LatencyQualityEvaluator()
        records = evaluator.run_all_configs(cases, delay_between=0.5)

    # Aggregate per-config stats
    evaluator = LatencyQualityEvaluator()
    agg = evaluator.aggregate(records)

    # Statistical tests
    stat_results = compare_configs(records)

    # Serialise stat tests for JSON
    stat_json = {}
    for key, res in stat_results.items():
        stat_json[key] = {
            "config_a":       res.config_a,
            "config_b":       res.config_b,
            "n_pairs":        res.n_pairs,
            "mean_score_a":   res.mean_score_a,
            "mean_score_b":   res.mean_score_b,
            "quality_delta":  res.quality_delta,
            "p_value":        res.p_value,
            "ci_95":          [res.ci_lower, res.ci_upper],
            "significant":    res.significant,
            "interpretation": res.interpretation,
        }

    recommendation, rec_reasoning = _pick_recommendation(agg, stat_json)

    errors = [r for r in records if r.error is not None]

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_cases_per_config": len(records) // max(
            len(set(r.config_name for r in records)), 1
        ),
        "total_agent_calls": len(records),
        "total_errors": len(errors),
        "configs": {
            config: {
                "mean_quality":   stats.get("mean_quality", 0.0),
                "p50_latency_ms": stats.get("p50_latency_ms", 0),
                "p95_latency_ms": stats.get("p95_latency_ms", 0),
                "p99_latency_ms": stats.get("p99_latency_ms", 0),
                "mean_latency_ms": stats.get("mean_latency_ms", 0),
                "mean_cost_usd":  stats.get("mean_cost_usd", 0.0),
                "total_cost_usd": stats.get("total_cost_usd", 0.0),
                "quality_per_dollar": stats.get("quality_per_dollar", 0.0),
            }
            for config, stats in agg.items()
        },
        "recommendation": recommendation,
        "recommendation_reasoning": rec_reasoning,
        "statistical_tests": stat_json,
    }

    # Print summary
    print("\n" + "=" * 65)
    print("LATENCY / QUALITY TRADEOFF REPORT")
    print("=" * 65)
    print(f"{'Config':<15} {'Quality':>8} {'p95 ms':>8} {'$/call':>9} {'Q/$':>8}")
    print("-" * 55)
    for cfg, s in sorted(report["configs"].items(), key=lambda x: -x[1]["mean_quality"]):
        print(
            f"{cfg:<15} {s['mean_quality']:>8.3f} "
            f"{s['p95_latency_ms']:>8} "
            f"${s['mean_cost_usd']:>8.5f} "
            f"{s['quality_per_dollar']:>8.1f}"
        )
    print()
    print(format_test_results(stat_results))
    print()
    print(f"RECOMMENDATION: {recommendation.upper()}")
    print(f"Reasoning: {rec_reasoning}")
    print("=" * 65)

    # Save report
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved → %s", OUTPUT_FILE)

    return report


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    generate_report()
