"""
Phase 2.7 — Validate the golden dataset.

Checks:
  - Total count = 200
  - All required fields present per dimension
  - Alarm type distribution: no single type > 20% of cases
  - Severity distribution: all 4 levels represented
  - Prints summary report with estimated API cost

Usage:
    uv run python data/golden_dataset/validate_dataset.py
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

GOLDEN_DIR = Path(__file__).parent

# (filename, dimension, target_count)
DATASETS = [
    ("correctness_cases.json",  "correctness",  50),
    ("faithfulness_cases.json", "faithfulness", 40),
    ("robustness_cases.json",   "robustness",   40),
    ("safety_cases.json",       "safety",       30),
    ("latency_cases.json",      "latency",      40),
]

# Required fields per dimension
REQUIRED_FIELDS = {
    "correctness": {
        "top": ["case_id", "input", "expected", "failure_mode_tested"],
        "input": ["alarm_id", "device_id", "alarm_type", "severity", "raw_message", "affected_site"],
        "expected": ["correct_classification", "correct_root_cause_category",
                     "required_actions_include", "severity_should_be", "expert_reasoning"],
    },
    "faithfulness": {
        "top": ["case_id", "case_type", "input", "injected_context", "expected"],
        "input": ["alarm_event"],
        "expected": [],  # structure varies by case_type
    },
    "robustness": {
        "top": ["case_id", "perturbation_type", "canonical_input", "perturbed_input", "expected"],
        "canonical_input": ["alarm_id", "device_id", "alarm_type", "severity", "raw_message"],
        "expected": ["classification_should_match_canonical", "acceptable_score_delta"],
    },
    "safety": {
        "top": ["case_id", "attack_type", "input", "expected"],
        "input": ["alarm_id", "device_id", "alarm_type", "raw_message"],
        "expected": ["should_refuse_or_ignore_injection", "forbidden_outputs"],
    },
    "latency": {
        "top": ["case_id", "complexity", "input", "expected"],
        "input": ["alarm_id", "device_id", "alarm_type", "severity", "raw_message"],
        "expected": ["expected_confidence_range"],
    },
}

ALARM_TYPES = [
    "LINK_DOWN", "HIGH_CPU", "PACKET_LOSS", "BGP_SESSION_DOWN",
    "INTERFACE_ERROR", "MEMORY_THRESHOLD", "POWER_SUPPLY_FAIL",
    "FAN_FAILURE", "OPTICAL_DEGRADATION", "SPANNING_TREE_CHANGE",
]
SEVERITIES = ["CRITICAL", "MAJOR", "MINOR", "WARNING"]

# Rough cost estimates (Sonnet used for generation)
SONNET_INPUT_PER_MTOK  = 3.00   # USD per million input tokens
SONNET_OUTPUT_PER_MTOK = 15.00  # USD per million output tokens
AVG_TOKENS_PER_EVAL_CALL = 2000  # judge call tokens (input+output)
HAIKU_INPUT_PER_MTOK   = 0.25
HAIKU_OUTPUT_PER_MTOK  = 1.25


def _load(filename: str) -> list[dict]:
    path = GOLDEN_DIR / filename
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _check_fields(case: dict, rules: dict, case_id: str, errors: list[str]) -> None:
    for field in rules.get("top", []):
        if field not in case:
            errors.append(f"{case_id}: missing top-level field '{field}'")
    for section, fields in rules.items():
        if section == "top":
            continue
        if section in case and isinstance(case[section], dict):
            for field in fields:
                if field not in case[section]:
                    errors.append(f"{case_id}[{section}]: missing field '{field}'")


def _extract_alarm_type(case: dict, dimension: str) -> str | None:
    if dimension == "correctness":
        return case.get("input", {}).get("alarm_type")
    if dimension == "faithfulness":
        ev = case.get("input", {}).get("alarm_event", {})
        return ev.get("alarm_type")
    if dimension == "robustness":
        return case.get("canonical_input", {}).get("alarm_type")
    if dimension == "safety":
        return case.get("input", {}).get("alarm_type")
    if dimension == "latency":
        return case.get("input", {}).get("alarm_type")
    return None


def _extract_severity(case: dict, dimension: str) -> str | None:
    if dimension == "correctness":
        return case.get("input", {}).get("severity")
    if dimension == "faithfulness":
        ev = case.get("input", {}).get("alarm_event", {})
        return ev.get("severity")
    if dimension == "robustness":
        return case.get("canonical_input", {}).get("severity")
    if dimension == "safety":
        return case.get("input", {}).get("severity")
    if dimension == "latency":
        return case.get("input", {}).get("severity")
    return None


def validate() -> bool:
    print("=" * 60)
    print("GOLDEN DATASET VALIDATION REPORT")
    print("=" * 60)

    all_alarm_types: list[str] = []
    all_severities: list[str] = []
    total_cases = 0
    all_errors: list[str] = []
    dimension_summary: list[dict] = []

    for filename, dimension, target in DATASETS:
        cases = _load(filename)
        count = len(cases)
        errors: list[str] = []
        rules = REQUIRED_FIELDS.get(dimension, {})

        for case in cases:
            case_id = case.get("case_id", "UNKNOWN")
            _check_fields(case, rules, case_id, errors)

            at = _extract_alarm_type(case, dimension)
            sv = _extract_severity(case, dimension)
            if at:
                all_alarm_types.append(at)
            if sv:
                all_severities.append(sv)

        total_cases += count
        all_errors.extend(errors)

        status = "OK" if count >= target and not errors else "FAIL"
        dimension_summary.append({
            "dimension": dimension,
            "count": count,
            "target": target,
            "errors": len(errors),
            "status": status,
        })

    # --- Print dimension table ---
    print(f"\n{'Dimension':<16} {'Count':>6} {'Target':>7} {'Errors':>7} {'Status':>7}")
    print("-" * 50)
    for row in dimension_summary:
        print(
            f"{row['dimension']:<16} {row['count']:>6} {row['target']:>7} "
            f"{row['errors']:>7} {row['status']:>7}"
        )
    print("-" * 50)
    print(f"{'TOTAL':<16} {total_cases:>6} {'200':>7}")

    # --- Schema errors ---
    if all_errors:
        print(f"\n⚠ SCHEMA ERRORS ({len(all_errors)}):")
        for e in all_errors[:20]:
            print(f"  • {e}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")
    else:
        print("\nNo schema errors found.")

    # --- Alarm type distribution ---
    print(f"\nAlarm type distribution ({len(all_alarm_types)} cases with alarm_type):")
    at_counter = Counter(all_alarm_types)
    distribution_ok = True
    for alarm_type in ALARM_TYPES:
        count = at_counter.get(alarm_type, 0)
        pct = count / len(all_alarm_types) * 100 if all_alarm_types else 0
        flag = " ⚠ OVER 20%" if pct > 20 else ""
        if pct > 20:
            distribution_ok = False
        print(f"  {alarm_type:<25} {count:>4} ({pct:.1f}%){flag}")
    missing = [t for t in ALARM_TYPES if t not in at_counter]
    if missing:
        print(f"  ⚠ Missing alarm types: {', '.join(missing)}")
        distribution_ok = False

    # --- Severity distribution ---
    print(f"\nSeverity distribution ({len(all_severities)} cases with severity):")
    sv_counter = Counter(all_severities)
    severity_ok = True
    for sev in SEVERITIES:
        count = sv_counter.get(sev, 0)
        pct = count / len(all_severities) * 100 if all_severities else 0
        flag = " ⚠ MISSING" if count == 0 else ""
        if count == 0:
            severity_ok = False
        print(f"  {sev:<10} {count:>4} ({pct:.1f}%){flag}")

    # --- Cost estimate ---
    print("\nEstimated API cost to run full eval suite:")
    # Correctness + faithfulness: Sonnet judge per case
    sonnet_cases = 50 + 40
    haiku_cases = 40 + 30 + 40  # robustness, safety, latency
    sonnet_cost = sonnet_cases * AVG_TOKENS_PER_EVAL_CALL / 1_000_000 * (SONNET_INPUT_PER_MTOK + SONNET_OUTPUT_PER_MTOK)
    haiku_cost = haiku_cases * AVG_TOKENS_PER_EVAL_CALL / 1_000_000 * (HAIKU_INPUT_PER_MTOK + HAIKU_OUTPUT_PER_MTOK)
    agent_cost = total_cases * 3 * 1000 / 1_000_000 * (HAIKU_INPUT_PER_MTOK + HAIKU_OUTPUT_PER_MTOK)  # 3 nodes × 1k tokens
    total_cost = sonnet_cost + haiku_cost + agent_cost
    print(f"  Agent runs ({total_cases} cases × 3 nodes × Haiku): ${agent_cost:.3f}")
    print(f"  Sonnet judge ({sonnet_cases} cases):                 ${sonnet_cost:.3f}")
    print(f"  Haiku eval ({haiku_cases} cases):                    ${haiku_cost:.3f}")
    print(f"  TOTAL estimated:                               ${total_cost:.3f}")

    # --- Final verdict ---
    total_ok = total_cases >= 200
    schema_ok = len(all_errors) == 0
    passed = total_ok and schema_ok and distribution_ok and severity_ok

    print("\n" + "=" * 60)
    if passed:
        print("VALIDATION PASSED — dataset is ready for evaluation")
    else:
        print("VALIDATION FAILED — issues found above")
        if not total_ok:
            print(f"  • Total cases {total_cases} < 200 target")
        if not schema_ok:
            print(f"  • {len(all_errors)} schema errors")
        if not distribution_ok:
            print("  • Alarm type distribution issues")
        if not severity_ok:
            print("  • Severity distribution missing levels")
    print("=" * 60)

    return passed


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
