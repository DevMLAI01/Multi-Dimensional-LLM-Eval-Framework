"""
Phase 5.2 — Robustness Stress Test.

Generates 20 additional adversarial perturbations on-the-fly using Claude Haiku,
then runs the robustness evaluator on all of them.

Reports which perturbation type causes the largest average score drop —
this tells you where to focus prompt engineering effort.

Usage:
    uv run python evaluators/robustness_stress_test.py
"""

import json
import logging
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PERTURBATION_TYPES = [
    "PARAPHRASE",
    "TYPO",
    "TERMINOLOGY",
    "SEVERITY_MISLABEL",
    "NOISE",
]

# 4 stress cases per type × 5 types = 20 total
STRESS_CASES_PER_TYPE = 4

# Canonical alarm seeds — one per perturbation type
SEED_ALARMS = [
    {
        "alarm_id": "STRESS_C001",
        "device_id": "RTR-OSL-042",
        "alarm_type": "LINK_DOWN",
        "severity": "CRITICAL",
        "timestamp": "2024-11-14T03:22:00Z",
        "raw_message": "Interface GigabitEthernet0/0/1 went down unexpectedly on core router",
        "affected_site": "Oslo-DC-North",
    },
    {
        "alarm_id": "STRESS_C002",
        "device_id": "RTR-BER-019",
        "alarm_type": "HIGH_CPU",
        "severity": "MAJOR",
        "timestamp": "2024-11-14T04:00:00Z",
        "raw_message": "CPU utilisation reached 96% for 10 consecutive minutes on BGP router",
        "affected_site": "Berlin-DC-East",
    },
    {
        "alarm_id": "STRESS_C003",
        "device_id": "RTR-AMS-007",
        "alarm_type": "BGP_SESSION_DOWN",
        "severity": "CRITICAL",
        "timestamp": "2024-11-14T05:00:00Z",
        "raw_message": "BGP session to peer 192.168.1.1 AS65001 dropped, hold timer expired",
        "affected_site": "Amsterdam-DC-West",
    },
    {
        "alarm_id": "STRESS_C004",
        "device_id": "RTR-LDN-001",
        "alarm_type": "PACKET_LOSS",
        "severity": "MAJOR",
        "timestamp": "2024-11-14T06:00:00Z",
        "raw_message": "Packet loss of 15% detected on uplink interface to core network",
        "affected_site": "London-DC-Central",
    },
    {
        "alarm_id": "STRESS_C005",
        "device_id": "RTR-PAR-003",
        "alarm_type": "OPTICAL_DEGRADATION",
        "severity": "MAJOR",
        "timestamp": "2024-11-14T07:00:00Z",
        "raw_message": "Optical receive power on port 1 dropped to -28dBm, threshold is -25dBm",
        "affected_site": "Paris-DC-South",
    },
]

PERTURBATION_PROMPTS = {
    "PARAPHRASE": (
        "Rewrite this network alarm message in completely different words with the same meaning. "
        "Return ONLY the rewritten message, nothing else."
    ),
    "TYPO": (
        "Add 3-5 realistic typos to this network alarm message (swap letters, drop letters, "
        "common keyboard miskeys). Return ONLY the message with typos, nothing else."
    ),
    "TERMINOLOGY": (
        "Rewrite this network alarm message using alternative vendor terminology "
        "(e.g. GigabitEthernet→ge-0/0/1, BGP peer→BGP neighbour, dropped→flapped). "
        "Return ONLY the rewritten message, nothing else."
    ),
    "SEVERITY_MISLABEL": (
        "Keep this alarm message exactly as-is but I will set the severity to WARNING even though "
        "the content implies something more serious. Return the message unchanged, nothing else."
    ),
    "NOISE": (
        "Wrap this alarm message in realistic NMS boilerplate noise. Add a system header, "
        "correlation ID, timestamp prefix, and footer around it. "
        "Return ONLY the wrapped message, nothing else."
    ),
}


def _generate_perturbation(
    client: anthropic.Anthropic,
    alarm: dict,
    perturbation_type: str,
) -> str:
    """Generate a perturbed version of alarm's raw_message using Claude Haiku."""
    instruction = PERTURBATION_PROMPTS[perturbation_type]
    prompt = f"{instruction}\n\nOriginal message: {alarm['raw_message']}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _make_stress_case(
    canonical: dict,
    perturbed_message: str,
    perturbation_type: str,
    case_num: int,
) -> dict:
    perturbed = dict(canonical)
    perturbed["alarm_id"] = f"STRESS_P{case_num:03d}"
    perturbed["raw_message"] = perturbed_message

    if perturbation_type == "SEVERITY_MISLABEL":
        perturbed["severity"] = "WARNING"  # intentionally wrong

    return {
        "case_id": f"STRESS_{case_num:03d}",
        "perturbation_type": perturbation_type,
        "canonical_input": canonical,
        "perturbed_input": perturbed,
        "expected": {
            "classification_should_match_canonical": True,
            "acceptable_score_delta": 0.15,
        },
    }


def generate_stress_cases(client: anthropic.Anthropic) -> list[dict]:
    """Generate 20 stress test cases (4 per perturbation type)."""
    cases = []
    case_num = 1

    for p_idx, pert_type in enumerate(PERTURBATION_TYPES):
        seed = SEED_ALARMS[p_idx % len(SEED_ALARMS)]
        log.info("Generating %d %s cases...", STRESS_CASES_PER_TYPE, pert_type)

        for i in range(STRESS_CASES_PER_TYPE):
            perturbed_msg = _generate_perturbation(client, seed, pert_type)
            case = _make_stress_case(seed, perturbed_msg, pert_type, case_num)
            cases.append(case)
            case_num += 1
            log.debug("  Case %d: '%s' → '%s'", case_num - 1,
                      seed["raw_message"][:40], perturbed_msg[:40])
            time.sleep(0.2)

    return cases


def run_stress_test() -> dict:
    """Run the full stress test and return a summary report."""
    from evaluators.robustness_evaluator import RobustnessEvaluator

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    evaluator = RobustnessEvaluator()

    # 1. Generate stress cases
    log.info("=== Robustness Stress Test ===")
    log.info("Generating %d stress cases via Claude Haiku...", STRESS_CASES_PER_TYPE * len(PERTURBATION_TYPES))
    cases = generate_stress_cases(client)
    log.info("Generated %d cases.", len(cases))

    # 2. Run evaluator
    log.info("Running robustness evaluator on all %d cases...", len(cases))
    results = evaluator.evaluate_batch(cases, delay_between=0.3)

    # 3. Aggregate by perturbation type
    by_type = evaluator.score_by_perturbation_type(results)

    # 4. Build report
    all_scores = [r.score for r in results if r.score is not None]
    overall_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    errors = [r for r in results if r.error is not None]

    # Find worst perturbation type
    worst_type = min(by_type, key=lambda t: by_type[t]["mean_score"]) if by_type else None
    best_type  = max(by_type, key=lambda t: by_type[t]["mean_score"]) if by_type else None

    report = {
        "total_cases": len(cases),
        "total_errors": len(errors),
        "overall_mean_score": round(overall_mean, 4),
        "overall_pass_rate": round(
            sum(1 for r in results if r.passed) / len(results), 4
        ) if results else 0.0,
        "by_perturbation_type": by_type,
        "worst_perturbation_type": worst_type,
        "best_perturbation_type": best_type,
        "recommendation": (
            f"Focus prompt engineering on {worst_type} perturbations "
            f"(mean score: {by_type[worst_type]['mean_score']:.3f})"
            if worst_type else "No recommendation available"
        ),
    }

    # 5. Print report
    print("\n" + "=" * 60)
    print("ROBUSTNESS STRESS TEST REPORT")
    print("=" * 60)
    print(f"Total cases:     {report['total_cases']}")
    print(f"Errors:          {report['total_errors']}")
    print(f"Overall score:   {report['overall_mean_score']:.3f}")
    print(f"Overall pass:    {report['overall_pass_rate']:.1%}")
    print()
    print(f"{'Perturbation Type':<25} {'Mean':>6} {'Min':>6} {'Pass%':>7} {'N':>4}")
    print("-" * 55)
    for pt, stats in sorted(by_type.items(), key=lambda x: x[1]["mean_score"]):
        print(
            f"{pt:<25} {stats['mean_score']:>6.3f} {stats['min_score']:>6.3f} "
            f"{stats['pass_rate']:>7.1%} {stats['n']:>4}"
        )
    print()
    print(f"Worst type: {report['worst_perturbation_type']}")
    print(f"Recommendation: {report['recommendation']}")
    print("=" * 60)

    # 6. Save report
    reports_dir = Path(__file__).parents[1] / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "robustness_stress_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved → %s", out_path)

    return report


if __name__ == "__main__":
    run_stress_test()
