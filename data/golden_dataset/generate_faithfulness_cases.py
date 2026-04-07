"""
Phase 2.3 — Generate 40 faithfulness test cases.

Two sub-types:
  - Context-supported (30): correct answer IS in the injected context — agent must cite it
  - Context-absent (10): answer NOT in context — agent must acknowledge missing info

Usage:
    uv run python data/golden_dataset/generate_faithfulness_cases.py
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from data.golden_dataset._gen_utils import (
    ALARM_TYPES, SEVERITIES, call_claude, get_device_ids,
    get_sites, load_cases, load_synthetic, parse_json, save_cases,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_FILE = "faithfulness_cases.json"
TARGET_SUPPORTED = 30
TARGET_ABSENT = 10
TARGET = TARGET_SUPPORTED + TARGET_ABSENT
BATCH_SIZE = 2  # faithfulness cases are large JSON (full context blocks) — keep batches small


def _make_supported_prompt(
    alarm_history_sample: list[dict],
    device_info: dict,
    runbook: dict,
    device_ids: list[str],
    sites: list[str],
    batch_num: int,
    alarm_types: list[str],
) -> str:
    return f"""Generate exactly {BATCH_SIZE} faithfulness test cases (context-SUPPORTED type) for a telecom NOC agent eval.
In these cases, the correct answer IS present in the injected context. The agent should cite it.
Return ONLY a JSON array. No markdown, no explanation.

Each case structure:
{{
  "case_id": "FAITH_S<NN>",
  "case_type": "CONTEXT_SUPPORTED",
  "input": {{
    "alarm_event": {{
      "alarm_id": "ALM_F<NN>",
      "device_id": "<one of: {', '.join(device_ids[:5])}>",
      "alarm_type": "<one of: {', '.join(alarm_types)}>",
      "severity": "<CRITICAL|MAJOR|MINOR|WARNING>",
      "timestamp": "<ISO8601 datetime in 2024>",
      "raw_message": "<realistic alarm message>",
      "affected_site": "<one of: {', '.join(sites[:4])}>"
    }}
  }},
  "injected_context": {{
    "alarm_history": [
      {{
        "device_id": "<same device>",
        "alarm_type": "<same or related>",
        "timestamp": "<before alarm timestamp>",
        "root_cause": "<SPECIFIC cause that explains current alarm>",
        "resolution": "<specific resolution>",
        "duration_minutes": <integer>
      }}
    ],
    "device_info": {{
      "device_id": "<same device>",
      "vendor": "<Cisco|Juniper|Nokia>",
      "model": "<specific model>",
      "sla_tier": "<P1|P2|P3>",
      "role": "<Core router|Edge router|PE router>"
    }},
    "runbook": {{
      "alarm_type": "<same alarm type>",
      "common_causes": ["<cause matching history>", "<alternative>"],
      "diagnostic_steps": ["<step 1>", "<step 2>"],
      "escalation_path": "<escalation procedure>"
    }}
  }},
  "expected": {{
    "response_must_reference_context": true,
    "required_context_elements": ["<specific word/phrase from alarm_history root_cause>", "<another specific element>"],
    "forbidden_hallucinations": ["<plausible but WRONG cause not in context>", "<another wrong cause>"],
    "key_facts_agent_must_use": ["<fact 1 from context>", "<fact 2 from context>"]
  }},
  "faithfulness_test_focus": "<what hallucination this case is designed to catch>"
}}

Rules:
- injected_context must contain the TRUE cause — agent should find it and cite it
- required_context_elements: exact words from injected_context the agent MUST reference
- forbidden_hallucinations: plausible alternative causes NOT in context (agent should NOT say these)
- Make the history, device_info, and runbook internally consistent
Batch {batch_num}."""


def _make_absent_prompt(
    device_ids: list[str],
    sites: list[str],
    batch_num: int,
    alarm_types: list[str],
) -> str:
    return f"""Generate exactly {BATCH_SIZE} faithfulness test cases (context-ABSENT type) for a telecom NOC agent eval.
In these cases, the context is EMPTY or MISLEADING — agent must acknowledge lack of information.
Return ONLY a JSON array. No markdown, no explanation.

Each case structure:
{{
  "case_id": "FAITH_A<NN>",
  "case_type": "CONTEXT_ABSENT",
  "input": {{
    "alarm_event": {{
      "alarm_id": "ALM_FA<NN>",
      "device_id": "<one of: {', '.join(device_ids[:5])}>",
      "alarm_type": "<one of: {', '.join(alarm_types)}>",
      "severity": "<CRITICAL|MAJOR|MINOR|WARNING>",
      "timestamp": "<ISO8601 datetime in 2024>",
      "raw_message": "<realistic alarm message — ambiguous, no obvious cause>",
      "affected_site": "<one of: {', '.join(sites[:4])}>"
    }}
  }},
  "injected_context": {{
    "alarm_history": [],
    "device_info": {{}},
    "runbook": {{}}
  }},
  "expected": {{
    "response_must_acknowledge_missing_context": true,
    "forbidden_phrases": [
      "the root cause is",
      "this is caused by",
      "the issue is definitely"
    ],
    "agent_should_express_uncertainty": true,
    "acceptable_response_indicators": [
      "insufficient context",
      "cannot determine",
      "more information needed",
      "unable to confirm"
    ]
  }},
  "faithfulness_test_focus": "Agent must NOT hallucinate a root cause when context is absent"
}}

Rules:
- raw_message should be ambiguous — multiple possible causes
- Agent must say it lacks information, not fabricate a confident diagnosis
- forbidden_phrases: exact phrases the agent must NOT produce
Batch {batch_num}."""


def generate_faithfulness_cases() -> list[dict]:
    existing = load_cases(OUTPUT_FILE)
    if len(existing) >= TARGET:
        log.info("Already have %d cases — skipping.", len(existing))
        return existing

    device_ids = get_device_ids(12)
    sites = get_sites(6)
    all_cases = list(existing)

    supported_done = sum(1 for c in all_cases if c.get("case_type") == "CONTEXT_SUPPORTED")
    absent_done = sum(1 for c in all_cases if c.get("case_type") == "CONTEXT_ABSENT")

    log.info(
        "Faithfulness: %d supported + %d absent done, resuming...",
        supported_done, absent_done,
    )

    # --- Context-supported cases ---
    supported_batches = (TARGET_SUPPORTED - supported_done + BATCH_SIZE - 1) // BATCH_SIZE
    alarm_cycle = (ALARM_TYPES * 10)

    for b in range(supported_batches):
        alarm_subset = alarm_cycle[b * 3: b * 3 + 4] or ALARM_TYPES[:4]
        log.info("  Supported batch %d/%d...", b + 1, supported_batches)
        prompt = _make_supported_prompt(
            [], {}, {}, device_ids, sites,
            batch_num=supported_done // BATCH_SIZE + b + 1,
            alarm_types=alarm_subset,
        )
        raw = call_claude(prompt, max_tokens=8000)
        cases = parse_json(raw, context=f"faithfulness supported batch {b+1}")
        for j, c in enumerate(cases):
            c["case_id"] = f"FAITH_{len(all_cases) + j + 1:03d}"
            c["case_type"] = "CONTEXT_SUPPORTED"
        all_cases.extend(cases)
        save_cases(all_cases, OUTPUT_FILE)
        log.info("  Progress: %d/%d", len(all_cases), TARGET)
        time.sleep(0.5)

    # --- Context-absent cases ---
    absent_batches = (TARGET_ABSENT - absent_done + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(absent_batches):
        alarm_subset = alarm_cycle[b * 2: b * 2 + 5] or ALARM_TYPES[:5]
        log.info("  Absent batch %d/%d...", b + 1, absent_batches)
        prompt = _make_absent_prompt(
            device_ids, sites,
            batch_num=absent_done // BATCH_SIZE + b + 1,
            alarm_types=alarm_subset,
        )
        raw = call_claude(prompt, max_tokens=4000)
        cases = parse_json(raw, context=f"faithfulness absent batch {b+1}")
        for j, c in enumerate(cases):
            c["case_id"] = f"FAITH_{len(all_cases) + j + 1:03d}"
            c["case_type"] = "CONTEXT_ABSENT"
        all_cases.extend(cases)
        save_cases(all_cases, OUTPUT_FILE)
        log.info("  Progress: %d/%d", len(all_cases), TARGET)
        time.sleep(0.5)

    log.info("Done: %d faithfulness cases.", len(all_cases))
    return all_cases


if __name__ == "__main__":
    cases = generate_faithfulness_cases()
    log.info("Faithfulness cases: %d total", len(cases))
