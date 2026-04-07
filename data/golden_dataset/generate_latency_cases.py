"""
Phase 2.6 — Generate 40 latency/quality test cases.

Complexity tiers:
  - SIMPLE  (10): single alarm, device in inventory, clear runbook match
  - MEDIUM  (20): ambiguous alarm, partial context, multiple hypotheses
  - COMPLEX (10): cascading failure, no runbook match, sparse history

These same cases are run against haiku-all / sonnet-all / hybrid configs.

Usage:
    uv run python data/golden_dataset/generate_latency_cases.py
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from data.golden_dataset._gen_utils import (
    ALARM_TYPES, call_claude, get_device_ids,
    get_sites, load_cases, parse_json, save_cases,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_FILE = "latency_cases.json"
TARGET = 40
BATCH_SIZE = 5

COMPLEXITY_SPECS = {
    "SIMPLE": {
        "count": 10,
        "description": (
            "Single, clear alarm type. Device is a well-known P1/P2 device in inventory. "
            "There is a matching runbook. Alarm history has 2-3 prior similar events. "
            "Root cause is unambiguous. Agent should diagnose quickly and confidently."
        ),
        "expected_agent_behavior": "Fast, confident diagnosis with single clear root cause hypothesis",
    },
    "MEDIUM": {
        "count": 20,
        "description": (
            "Ambiguous alarm — could be 2-3 different root causes. Partial context: "
            "device in inventory but alarm history is sparse (0-1 prior events). "
            "Runbook exists but doesn't fully match the scenario. "
            "Agent must reason with incomplete information and produce multiple hypotheses."
        ),
        "expected_agent_behavior": "Multiple hypotheses with varied confidence, reasoning acknowledges uncertainty",
    },
    "COMPLEX": {
        "count": 10,
        "description": (
            "Cascading failure scenario — multiple related alarms, device NOT in inventory. "
            "No matching runbook. Empty alarm history. Raw message is ambiguous and multi-part. "
            "May involve multiple devices or a site-wide incident. "
            "Agent must reason from first principles with zero context."
        ),
        "expected_agent_behavior": "Low confidence, explicit acknowledgment of missing context, conservative recommendations",
    },
}


def _make_batch_prompt(
    complexity: str,
    spec: dict,
    device_ids: list[str],
    sites: list[str],
    alarm_types: list[str],
    batch_num: int,
) -> str:
    device_note = (
        "device_id should NOT be in the standard inventory (use RTR-UNKNOWN-999 format)"
        if complexity == "COMPLEX"
        else f"device_id from: {', '.join(device_ids[:5])}"
    )

    return f"""Generate exactly {BATCH_SIZE} latency/quality test cases with complexity: {complexity}
Return ONLY a JSON array. No markdown, no explanation.

Complexity description: {spec['description']}
Expected agent behavior: {spec['expected_agent_behavior']}

Each case structure:
{{
  "case_id": "LAT_<NN>",
  "complexity": "{complexity}",
  "input": {{
    "alarm_id": "ALM_L<NN>",
    "device_id": "<{device_note}>",
    "alarm_type": "<one of: {', '.join(alarm_types)}>",
    "severity": "<CRITICAL|MAJOR|MINOR|WARNING>",
    "timestamp": "<ISO8601 datetime in 2024>",
    "raw_message": "<alarm message matching complexity level>",
    "affected_site": "<one of: {', '.join(sites[:4])}>"
  }},
  "complexity_factors": {{
    "has_alarm_history": <true for SIMPLE, false for COMPLEX>,
    "device_in_inventory": <true for SIMPLE/MEDIUM, false for COMPLEX>,
    "has_matching_runbook": <true for SIMPLE, false for COMPLEX>,
    "ambiguity_level": "<LOW|MEDIUM|HIGH>",
    "reasoning_depth_required": "<SHALLOW|MODERATE|DEEP>"
  }},
  "expected": {{
    "expected_hypothesis_count": <1 for SIMPLE, 2-3 for MEDIUM, 1-2 for COMPLEX>,
    "expected_confidence_range": "<e.g. 0.80-0.95 for SIMPLE, 0.40-0.65 for MEDIUM, 0.20-0.45 for COMPLEX>",
    "quality_metric_focus": "<correctness|faithfulness|robustness — primary metric for this case>",
    "notes": "<what makes this case the given complexity>"
  }}
}}

Rules:
- SIMPLE: raw_message is clear and specific, alarm type is unambiguous
- MEDIUM: raw_message could be misinterpreted, include 2 plausible causes in notes
- COMPLEX: raw_message mentions multiple systems or is vague; device_id is RTR-UNKNOWN-XXX
- Vary alarm types across the batch: {', '.join(alarm_types)}
Batch {batch_num}."""


def generate_latency_cases() -> list[dict]:
    existing = load_cases(OUTPUT_FILE)
    if len(existing) >= TARGET:
        log.info("Already have %d cases — skipping.", len(existing))
        return existing

    device_ids = get_device_ids(12)
    sites = get_sites(6)
    all_cases = list(existing)

    complexity_counts = {k: 0 for k in COMPLEXITY_SPECS}
    for c in all_cases:
        cx = c.get("complexity", "")
        if cx in complexity_counts:
            complexity_counts[cx] += 1

    log.info("Latency: %d/%d done, resuming...", len(all_cases), TARGET)

    alarm_cycle = ALARM_TYPES * 5
    batch_num = len(all_cases) // BATCH_SIZE + 1

    for complexity, spec in COMPLEXITY_SPECS.items():
        done = complexity_counts[complexity]
        needed = spec["count"] - done
        if needed <= 0:
            log.info("  %s: already complete (%d cases)", complexity, done)
            continue

        batches = (needed + BATCH_SIZE - 1) // BATCH_SIZE
        a_offset = list(COMPLEXITY_SPECS.keys()).index(complexity) * 3

        for b in range(batches):
            alarm_subset = alarm_cycle[a_offset + b * 3: a_offset + b * 3 + 5] or ALARM_TYPES[:5]
            log.info("  %s batch %d/%d...", complexity, b + 1, batches)

            prompt = _make_batch_prompt(
                complexity, spec, device_ids, sites, alarm_subset, batch_num
            )
            raw = call_claude(prompt, max_tokens=4096)
            cases = parse_json(raw, context=f"latency {complexity} batch {b+1}")

            for j, c in enumerate(cases):
                c["case_id"] = f"LAT_{len(all_cases) + j + 1:03d}"
                c["complexity"] = complexity
            all_cases.extend(cases)
            save_cases(all_cases, OUTPUT_FILE)
            log.info("  Progress: %d/%d", len(all_cases), TARGET)
            batch_num += 1
            time.sleep(0.5)

    log.info("Done: %d latency cases.", len(all_cases))
    return all_cases


if __name__ == "__main__":
    cases = generate_latency_cases()
    log.info("Latency cases: %d total", len(cases))
