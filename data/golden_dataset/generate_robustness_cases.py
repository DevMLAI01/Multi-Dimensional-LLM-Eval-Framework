"""
Phase 2.4 — Generate 40 robustness test cases.

Each case is a PAIR: canonical clean input + perturbed variant.
Perturbation types (8 cases each):
  - PARAPHRASE:         same alarm, different wording
  - TYPO:              realistic operator typos in raw_message
  - TERMINOLOGY:       vendor-specific vs generic interface names
  - SEVERITY_MISLABEL: alarm_type says MINOR but message content implies CRITICAL
  - NOISE:             raw_message has irrelevant boilerplate around the real content

Usage:
    uv run python data/golden_dataset/generate_robustness_cases.py
"""

import logging
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

OUTPUT_FILE = "robustness_cases.json"
TARGET = 40
CASES_PER_TYPE = 8
BATCH_SIZE = 4

PERTURBATION_TYPES = [
    "PARAPHRASE",
    "TYPO",
    "TERMINOLOGY",
    "SEVERITY_MISLABEL",
    "NOISE",
]

PERTURBATION_INSTRUCTIONS = {
    "PARAPHRASE": (
        "Create a paraphrased version of the raw_message: same meaning, completely different wording. "
        "Example: 'Interface GigE0/0/1 is down' → 'Port GigabitEthernet0/0/1 is no longer active'. "
        "Classification should remain identical."
    ),
    "TYPO": (
        "Create a version with realistic operator typos in raw_message. "
        "Example: 'Interface GigE0/0/1 went down' → 'Interfce Gig0/0/1 wnet dwon'. "
        "Swap letters, drop letters, add common miskeys. 3-5 typos total."
    ),
    "TERMINOLOGY": (
        "Create a version using vendor-specific or alternative interface terminology. "
        "Example: 'GigabitEthernet0/0/1' → 'ge-0/0/1' (Juniper style) or 'Gi0/0/1' (Cisco shorthand). "
        "Also varies: 'router' vs 'device', 'session' vs 'adjacency', 'link' vs 'interface'."
    ),
    "SEVERITY_MISLABEL": (
        "The alarm_type severity field says WARNING or MINOR, but the raw_message content clearly "
        "describes a CRITICAL or MAJOR incident. The agent should override the labeled severity based "
        "on message content. Example: severity=MINOR but message='Core P1 router completely unreachable'."
    ),
    "NOISE": (
        "Wrap the real alarm message with irrelevant NMS boilerplate. "
        "Example: 'SYSTEM: Auto-generated alert from NMS v4.2.1 [timestamp=...] [node=NMS-01] "
        "ALERT_TEXT: <actual alarm here> END_OF_ALERT [correlation_id=abc123] [suppressed=false]'. "
        "The agent must extract the real alarm content from the noise."
    ),
}


def _make_batch_prompt(
    perturbation_type: str,
    device_ids: list[str],
    sites: list[str],
    alarm_types: list[str],
    batch_num: int,
) -> str:
    instruction = PERTURBATION_INSTRUCTIONS[perturbation_type]

    severity_note = ""
    if perturbation_type == "SEVERITY_MISLABEL":
        severity_note = (
            "\nFor SEVERITY_MISLABEL: canonical_input has correct severity, "
            "perturbed_input.severity is set to WARNING or MINOR but message implies CRITICAL/MAJOR. "
            "expected.severity_override_required must be true."
        )

    return f"""Generate exactly {BATCH_SIZE} robustness test cases with perturbation type: {perturbation_type}
Return ONLY a JSON array. No markdown, no explanation.

Perturbation instruction: {instruction}

Each case structure:
{{
  "case_id": "ROB_<NN>",
  "perturbation_type": "{perturbation_type}",
  "canonical_input": {{
    "alarm_id": "ALM_R<NN>_C",
    "device_id": "<one of: {', '.join(device_ids[:6])}>",
    "alarm_type": "<one of: {', '.join(alarm_types)}>",
    "severity": "<CRITICAL|MAJOR|MINOR|WARNING>",
    "timestamp": "2024-11-14T03:22:00Z",
    "raw_message": "<clean, well-formed alarm message>",
    "affected_site": "<one of: {', '.join(sites[:4])}>"
  }},
  "perturbed_input": {{
    "alarm_id": "ALM_R<NN>_P",
    "device_id": "<same as canonical>",
    "alarm_type": "<same as canonical>",
    "severity": "<same OR intentionally wrong for SEVERITY_MISLABEL>",
    "timestamp": "2024-11-14T03:22:00Z",
    "raw_message": "<perturbed version per instruction above>",
    "affected_site": "<same as canonical>"
  }},
  "expected": {{
    "classification_should_match_canonical": true,
    "acceptable_score_delta": 0.15,
    "severity_override_required": <true only for SEVERITY_MISLABEL, else false>,
    "correct_classification": "<what both canonical and perturbed should classify as>",
    "notes": "<what makes this perturbation challenging>"
  }}
}}{severity_note}

Rules:
- canonical and perturbed MUST describe the same underlying incident
- Make 8 different alarm types across the {BATCH_SIZE} cases in this batch
- Perturbations must be realistic, not exaggerated
Batch {batch_num}."""


def generate_robustness_cases() -> list[dict]:
    existing = load_cases(OUTPUT_FILE)
    if len(existing) >= TARGET:
        log.info("Already have %d cases — skipping.", len(existing))
        return existing

    device_ids = get_device_ids(12)
    sites = get_sites(6)
    all_cases = list(existing)

    log.info("Robustness: %d/%d done, resuming...", len(all_cases), TARGET)

    # Track how many cases per type we've generated
    type_counts: dict[str, int] = {t: 0 for t in PERTURBATION_TYPES}
    for c in all_cases:
        pt = c.get("perturbation_type", "")
        if pt in type_counts:
            type_counts[pt] += 1

    alarm_cycle = ALARM_TYPES * 5
    batch_num = len(all_cases) // BATCH_SIZE + 1

    for pert_type in PERTURBATION_TYPES:
        done = type_counts[pert_type]
        needed = CASES_PER_TYPE - done
        if needed <= 0:
            log.info("  %s: already complete (%d cases)", pert_type, done)
            continue

        batches = (needed + BATCH_SIZE - 1) // BATCH_SIZE
        a_offset = PERTURBATION_TYPES.index(pert_type) * 2

        for b in range(batches):
            alarm_subset = alarm_cycle[a_offset + b * 2: a_offset + b * 2 + 5] or ALARM_TYPES[:5]
            log.info("  %s batch %d/%d...", pert_type, b + 1, batches)

            prompt = _make_batch_prompt(
                pert_type, device_ids, sites, alarm_subset, batch_num
            )
            raw = call_claude(prompt, max_tokens=4096)
            cases = parse_json(raw, context=f"robustness {pert_type} batch {b+1}")

            for j, c in enumerate(cases):
                c["case_id"] = f"ROB_{len(all_cases) + j + 1:03d}"
                c["perturbation_type"] = pert_type
            all_cases.extend(cases)
            save_cases(all_cases, OUTPUT_FILE)
            log.info("  Progress: %d/%d", len(all_cases), TARGET)
            batch_num += 1
            time.sleep(0.5)

    log.info("Done: %d robustness cases.", len(all_cases))
    return all_cases


if __name__ == "__main__":
    cases = generate_robustness_cases()
    log.info("Robustness cases: %d total", len(cases))
