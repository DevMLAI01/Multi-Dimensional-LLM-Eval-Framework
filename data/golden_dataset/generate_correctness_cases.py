"""
Phase 2.2 — Generate 50 correctness test cases.

Each case is an alarm + expert-labelled diagnosis. Uses Claude Sonnet as the
expert to label ground truth. Covers all 10 alarm types and all 4 severities.
Includes 10 tricky cases where the obvious classification is wrong.

Usage:
    uv run python data/golden_dataset/generate_correctness_cases.py
"""

import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from data.golden_dataset._gen_utils import (
    ALARM_TYPES, SEVERITIES, call_claude, get_device_ids,
    get_sites, load_cases, parse_json, save_cases,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_FILE = "correctness_cases.json"
TARGET = 50
BATCH_SIZE = 5  # cases per API call


# Tricky scenarios where the obvious classification is wrong
TRICKY_SCENARIOS = [
    {
        "alarm_type": "LINK_DOWN",
        "twist": "caused by BGP misconfiguration, not physical failure",
        "hint": "Recent BGP config change preceded the link drop",
    },
    {
        "alarm_type": "HIGH_CPU",
        "twist": "caused by a routing table explosion from a BGP peer sending full table",
        "hint": "CPU spike correlated with sudden BGP route count increase",
    },
    {
        "alarm_type": "PACKET_LOSS",
        "twist": "caused by spanning tree topology change, not congestion",
        "hint": "Packet loss started immediately after a spanning tree recalculation",
    },
    {
        "alarm_type": "BGP_SESSION_DOWN",
        "twist": "caused by physical fiber cut on the peering link, not BGP config",
        "hint": "Physical interface also shows down state",
    },
    {
        "alarm_type": "INTERFACE_ERROR",
        "twist": "caused by mismatched duplex settings after maintenance, not hardware fault",
        "hint": "Errors started after a maintenance window last night",
    },
    {
        "alarm_type": "MEMORY_THRESHOLD",
        "twist": "caused by a memory leak in a specific routing process, not traffic growth",
        "hint": "Memory usage is growing linearly over 6 hours despite stable traffic",
    },
    {
        "alarm_type": "OPTICAL_DEGRADATION",
        "twist": "caused by dirty fiber connector, not fiber break or hardware fault",
        "hint": "Signal degraded gradually over days, not suddenly",
    },
    {
        "alarm_type": "SPANNING_TREE_CHANGE",
        "twist": "caused by a rogue switch being plugged in, not legitimate topology change",
        "hint": "New MAC address appeared on access port just before topology change",
    },
    {
        "alarm_type": "POWER_SUPPLY_FAIL",
        "twist": "caused by facility UPS bypass mode, not PSU hardware failure",
        "hint": "Multiple devices in the same rack reported power anomalies simultaneously",
    },
    {
        "alarm_type": "FAN_FAILURE",
        "twist": "false alarm from faulty sensor — fans are operational",
        "hint": "Device temperature is normal and redundant fans show no issue",
    },
]


def _make_batch_prompt(
    alarm_types_subset: list[str],
    severities_subset: list[str],
    device_ids: list[str],
    sites: list[str],
    tricky_scenario: dict | None,
    batch_num: int,
    total_batches: int,
) -> str:
    tricky_instruction = ""
    if tricky_scenario:
        tricky_instruction = f"""
IMPORTANT — this batch must include at least one TRICKY case:
- Alarm type: {tricky_scenario['alarm_type']}
- Twist: {tricky_scenario['twist']}
- Hint in the raw_message: {tricky_scenario['hint']}
- failure_mode_tested must explicitly say this is a tricky/counterintuitive case
"""

    return f"""Generate exactly {BATCH_SIZE} correctness test cases for a telecom NOC diagnostic agent evaluation.
Return ONLY a JSON array. No markdown, no explanation.

Each case structure:
{{
  "case_id": "CORR_<NN>" (sequential number),
  "input": {{
    "alarm_id": "ALM_C<NN>",
    "device_id": "<one of: {', '.join(device_ids[:6])}>",
    "alarm_type": "<one of: {', '.join(alarm_types_subset)}>",
    "severity": "<one of: {', '.join(severities_subset)}>",
    "timestamp": "<ISO8601 datetime in 2024>",
    "raw_message": "<realistic NMS alarm message, 10-30 words, specific and technical>",
    "affected_site": "<one of: {', '.join(sites[:4])}>"
  }},
  "expected": {{
    "correct_classification": "<specific classification, e.g. Physical layer failure / Software configuration error>",
    "correct_root_cause_category": "<pipe-separated categories, e.g. FIBER_CUT | HARDWARE_FAILURE | CONFIG_ERROR>",
    "required_actions_include": ["<action keyword 1>", "<action keyword 2>", "<action keyword 3>"],
    "severity_should_be": "<CRITICAL|MAJOR|MINOR|WARNING — what it SHOULD be, may differ from input>",
    "expert_reasoning": "<2-3 sentences: why this classification is correct, what NOC engineer looks for>"
  }},
  "failure_mode_tested": "<specific failure mode or edge case this tests, 1 sentence>"
}}

Rules:
- Cover a mix of alarm types from: {', '.join(alarm_types_subset)}
- Cover all severity levels across the batch
- raw_message must be specific and technical (include interface names, IPs, percentages)
- expert_reasoning must reference specific observable evidence
- required_actions_include: 3 keywords an agent MUST recommend (e.g. check_physical, escalate_p1, contact_vendor)
{tricky_instruction}
Batch {batch_num} of {total_batches}."""


def generate_correctness_cases() -> list[dict]:
    existing = load_cases(OUTPUT_FILE)
    if len(existing) >= TARGET:
        log.info("Already have %d cases — skipping generation.", len(existing))
        return existing

    start_idx = len(existing)
    log.info("Generating correctness cases: %d/%d done, resuming...", start_idx, TARGET)

    device_ids = get_device_ids(15)
    sites = get_sites(8)
    all_cases = list(existing)

    total_batches = (TARGET - start_idx + BATCH_SIZE - 1) // BATCH_SIZE
    # Distribute alarm types and severities evenly
    alarm_cycle = ALARM_TYPES * 10  # enough to cover all batches
    severity_cycle = SEVERITIES * 15

    tricky_iter = iter(TRICKY_SCENARIOS)

    for b in range(total_batches):
        global_batch = start_idx // BATCH_SIZE + b + 1
        a_start = b * BATCH_SIZE % len(ALARM_TYPES)
        alarm_subset = ALARM_TYPES[a_start:a_start + 5] or ALARM_TYPES[:5]
        sev_subset = severity_cycle[b * 2: b * 2 + 4] or SEVERITIES

        # Inject a tricky scenario every ~5 batches
        tricky = next(tricky_iter, None) if b % 5 == 0 else None

        log.info("  Batch %d/%d (alarms: %s)...", b + 1, total_batches, ", ".join(alarm_subset[:3]))

        prompt = _make_batch_prompt(
            alarm_subset, sev_subset, device_ids, sites, tricky, global_batch, total_batches
        )

        raw = call_claude(prompt, max_tokens=4096)
        cases = parse_json(raw, context=f"correctness batch {b+1}")

        # Renumber case IDs sequentially
        for j, case in enumerate(cases):
            case["case_id"] = f"CORR_{len(all_cases) + j + 1:03d}"
            if "input" in case:
                case["input"]["alarm_id"] = f"ALM_C{len(all_cases) + j + 1:03d}"

        all_cases.extend(cases)
        save_cases(all_cases, OUTPUT_FILE)
        log.info("  Progress: %d/%d cases", len(all_cases), TARGET)
        time.sleep(0.5)

    log.info("Done: %d correctness cases generated.", len(all_cases))
    return all_cases


if __name__ == "__main__":
    cases = generate_correctness_cases()
    log.info("Correctness cases: %d total in %s", len(cases), OUTPUT_FILE)
