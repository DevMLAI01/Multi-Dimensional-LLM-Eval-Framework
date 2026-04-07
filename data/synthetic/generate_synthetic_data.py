"""
Generate synthetic supporting data for the NOC diagnostic agent.

Produces:
  - alarm_history.json    — 500 historical alarm records
  - device_inventory.json — 100 device records
  - runbooks.json         — 30 runbook entries (3 per alarm type × 10 types)

Usage:
    uv run python data/synthetic/generate_synthetic_data.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

ALARM_TYPES = [
    "LINK_DOWN",
    "HIGH_CPU",
    "PACKET_LOSS",
    "BGP_SESSION_DOWN",
    "INTERFACE_ERROR",
    "MEMORY_THRESHOLD",
    "POWER_SUPPLY_FAIL",
    "FAN_FAILURE",
    "OPTICAL_DEGRADATION",
    "SPANNING_TREE_CHANGE",
]

SITES = [
    "Oslo-DC-North", "Berlin-DC-East", "Amsterdam-DC-West",
    "London-DC-Central", "Paris-DC-South", "Stockholm-DC-North",
    "Frankfurt-DC-Central", "Copenhagen-DC-East", "Helsinki-DC-North",
    "Warsaw-DC-Central",
]

VENDORS = ["Cisco", "Juniper", "Nokia", "Huawei", "Arista"]
MODELS = {
    "Cisco":   ["ASR-9001", "ASR-9006", "NCS-5500", "Catalyst-9300"],
    "Juniper": ["MX-480", "MX-960", "PTX-5000", "QFX-10002"],
    "Nokia":   ["7750-SR", "7250-IXR", "7210-SAS"],
    "Huawei":  ["NE40E", "NE9000", "CE-12800"],
    "Arista":  ["7050X3", "7280R3", "7368X4"],
}

OUTPUT_DIR = Path(__file__).parent


def call_claude(prompt: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 4096) -> str:
    """Call Claude API and return text content."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from model output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # drop first line (```json or ```)
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _parse_json_with_retry(raw: str, context: str = "") -> list | dict:
    """Parse JSON, retrying with a repair prompt if truncated."""
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("JSON parse failed (%s) — attempting repair: %s", context, exc)
        repair_prompt = (
            f"The following JSON is malformed or truncated. "
            f"Fix it and return ONLY valid JSON, nothing else:\n\n{cleaned}"
        )
        repaired = _strip_fences(call_claude(repair_prompt))
        return json.loads(repaired)


def generate_alarm_history() -> list[dict]:
    """Generate 500 historical alarm records in batches of 25.

    Saves progress after each batch — safe to resume if interrupted.
    Already-saved records are loaded and batches already completed are skipped.
    """
    target = 500
    batch_size = 25
    batches = target // batch_size
    output_path = OUTPUT_DIR / "alarm_history.json"

    # Resume: load existing records if file already present
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            all_records = json.load(f)
        batches_done = len(all_records) // batch_size
        log.info(
            "Resuming alarm_history: %d records already saved, starting at batch %d/%d",
            len(all_records), batches_done + 1, batches,
        )
    else:
        all_records = []
        batches_done = 0

    log.info("Generating alarm_history.json (%d records total)...", target)

    for i in range(batches_done, batches):
        alarm_types_slice = ALARM_TYPES[i % len(ALARM_TYPES)]
        log.info("  Batch %d/%d...", i + 1, batches)
        prompt = f"""Generate exactly {batch_size} realistic historical network alarm records for a telecom NOC.
Return ONLY a JSON array with no markdown, no explanation.

Each record must have exactly these fields:
{{
  "device_id": "RTR-<CITY3>-<NNN>",
  "alarm_type": "<one of: {', '.join(ALARM_TYPES)}>",
  "timestamp": "<ISO8601 datetime between 2024-01-01 and 2024-12-31>",
  "duration_minutes": <integer 1-480>,
  "root_cause": "<specific root cause, 5-15 words>",
  "resolution": "<specific resolution action, 5-20 words>",
  "severity": "<CRITICAL|MAJOR|MINOR|WARNING>",
  "affected_site": "<one of: {', '.join(SITES)}>"
}}

Vary alarm types, severities, and sites realistically. Include {alarm_types_slice} frequently.
Make root causes and resolutions specific and realistic for telecom networks.
Batch {i+1} of {batches}. Output exactly {batch_size} records, no more, no less."""

        raw = call_claude(prompt, max_tokens=6000)
        records = _parse_json_with_retry(raw, context=f"alarm_history batch {i+1}")
        all_records.extend(records)

        # Save after every batch — partial runs are recoverable
        save(all_records, "alarm_history.json")
        time.sleep(0.3)

    log.info("  Generated %d alarm history records.", len(all_records))
    return all_records


def generate_device_inventory() -> list[dict]:
    """Generate 100 device inventory records in batches of 25.

    Saves progress after each batch — safe to resume if interrupted.
    """
    target = 100
    batch_size = 25
    batches = target // batch_size
    output_path = OUTPUT_DIR / "device_inventory.json"

    # Resume: load existing records
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            all_records = json.load(f)
        batches_done = len(all_records) // batch_size
        log.info(
            "Resuming device_inventory: %d records already saved, starting at batch %d/%d",
            len(all_records), batches_done + 1, batches,
        )
    else:
        all_records = []
        batches_done = 0

    log.info("Generating device_inventory.json (%d records total)...", target)

    # Assign site slices so each batch covers ~2-3 sites
    site_groups = [SITES[i:i+3] for i in range(0, len(SITES), 3)]

    for i in range(batches_done, batches):
        sites_slice = site_groups[i % len(site_groups)]
        log.info("  Batch %d/%d (sites: %s)...", i + 1, batches, ", ".join(sites_slice))
        prompt = f"""Generate exactly {batch_size} realistic network device inventory records for a European telecom network.
Return ONLY a JSON array with no markdown, no explanation.

Each record must have exactly these fields:
{{
  "device_id": "RTR-<CITY3>-<NNN>",
  "vendor": "<one of: {', '.join(VENDORS)}>",
  "model": "<realistic model for that vendor>",
  "site": "<one of: {', '.join(sites_slice)}>",
  "role": "<Core router|Edge router|Aggregation switch|Access switch|PE router|P router>",
  "firmware": "<realistic version string, e.g. 7.3.2>",
  "last_maintenance": "<ISO date between 2023-01-01 and 2024-12-01>",
  "sla_tier": "<P1|P2|P3>",
  "connected_devices": ["<device_id>", "<device_id>"]
}}

Rules:
- device_id format: RTR-<3-letter city code>-<3-digit number>, e.g. RTR-OSL-042
- P1 = Core/PE router (25%), P2 = Edge/Aggregation (50%), P3 = Access (25%)
- connected_devices: 2-4 device_ids
- Vendor model must match vendor (Cisco → ASR-9001, Juniper → MX-480, etc.)
- Batch {i+1} of {batches}. Output exactly {batch_size} records."""

        raw = call_claude(prompt, max_tokens=4096)
        records = _parse_json_with_retry(raw, context=f"device_inventory batch {i+1}")
        all_records.extend(records)

        save(all_records, "device_inventory.json")
        time.sleep(0.3)

    log.info("  Generated %d device inventory records.", len(all_records))
    return all_records


def generate_runbooks() -> list[dict]:
    """Generate 30 runbook entries — 3 per alarm type."""
    log.info("Generating runbooks.json (30 entries)...")
    all_runbooks: list[dict] = []

    for idx, alarm_type in enumerate(ALARM_TYPES):
        prompt = f"""Generate exactly 3 runbook entries for alarm type: {alarm_type}
Return ONLY a JSON array with no markdown, no explanation.

Each entry must have exactly these fields:
{{
  "runbook_id": "RB-{idx*3+1:03d}" through "RB-{idx*3+3:03d}",
  "alarm_type": "{alarm_type}",
  "title": "<descriptive title for this specific scenario>",
  "trigger_conditions": "<specific conditions that trigger this alarm, 2-3 sentences>",
  "diagnostic_steps": [
    "<step 1: specific CLI command or check>",
    "<step 2>",
    "<step 3>",
    "<step 4>",
    "<step 5>"
  ],
  "common_causes": [
    "<cause 1>",
    "<cause 2>",
    "<cause 3>"
  ],
  "escalation_path": "<specific escalation procedure, who to contact and when>",
  "estimated_resolution_time": "<e.g. 15-30 minutes>",
  "severity_typical": "<CRITICAL|MAJOR|MINOR|WARNING>"
}}

Make runbooks realistic and actionable for a senior NOC engineer.
Include vendor-specific commands where relevant (Cisco IOS-XR, Juniper JUNOS).
The 3 entries should cover different sub-scenarios for {alarm_type}."""

        raw = call_claude(prompt, max_tokens=3000)
        entries = _parse_json_with_retry(raw, context=f"runbook {alarm_type}")
        # Fix runbook IDs to be sequential
        for j, entry in enumerate(entries):
            entry["runbook_id"] = f"RB-{idx*3+j+1:03d}"
            entry["alarm_type"] = alarm_type
        all_runbooks.extend(entries)
        log.info(f"  {alarm_type}: {len(entries)} runbooks")
        time.sleep(0.3)

    log.info(f"  Generated {len(all_runbooks)} runbook entries.")
    return all_runbooks


def save(data: list[dict], filename: str) -> None:
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"  Saved {len(data)} records → {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # All generators save incrementally and self-resume
    alarm_history = generate_alarm_history()
    device_inventory = generate_device_inventory()

    # runbooks: skip if already complete (30 entries = small, one-shot is fine)
    rb_path = OUTPUT_DIR / "runbooks.json"
    if rb_path.exists():
        with open(rb_path, encoding="utf-8") as f:
            runbooks = json.load(f)
        log.info("Skipping runbooks — already generated (%d entries).", len(runbooks))
    else:
        runbooks = generate_runbooks()
        save(runbooks, "runbooks.json")

    log.info("Done. All synthetic data generated.")
    log.info(f"  alarm_history:    {len(alarm_history)} records")
    log.info(f"  device_inventory: {len(device_inventory)} records")
    log.info(f"  runbooks:         {len(runbooks)} entries")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill in the key.")
        sys.exit(1)
    main()
