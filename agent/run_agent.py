"""
CLI entry point for the NOC diagnostic agent.

Usage:
    uv run python agent/run_agent.py \\
      --alarm-id ALM001 \\
      --device-id RTR-OSL-042 \\
      --alarm-type LINK_DOWN \\
      --severity CRITICAL \\
      --message "Interface GigE0/0/1 went down unexpectedly" \\
      --site Oslo-DC-North
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parents[1]))

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="NOC Diagnostic Agent CLI")
    parser.add_argument("--alarm-id", required=True, help="Alarm identifier, e.g. ALM001")
    parser.add_argument("--device-id", required=True, help="Device ID, e.g. RTR-OSL-042")
    parser.add_argument(
        "--alarm-type",
        required=True,
        choices=[
            "LINK_DOWN", "HIGH_CPU", "PACKET_LOSS", "BGP_SESSION_DOWN",
            "INTERFACE_ERROR", "MEMORY_THRESHOLD", "POWER_SUPPLY_FAIL",
            "FAN_FAILURE", "OPTICAL_DEGRADATION", "SPANNING_TREE_CHANGE",
        ],
        help="Alarm type",
    )
    parser.add_argument(
        "--severity",
        required=True,
        choices=["CRITICAL", "MAJOR", "MINOR", "WARNING"],
        help="Reported alarm severity",
    )
    parser.add_argument("--message", required=True, help="Raw alarm message from NMS")
    parser.add_argument("--site", default="Unknown", help="Affected site name")
    parser.add_argument(
        "--timestamp",
        default=None,
        help="ISO8601 timestamp (default: now)",
    )
    parser.add_argument(
        "--output",
        choices=["json", "pretty"],
        default="pretty",
        help="Output format",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill in the key.")
        sys.exit(1)

    from datetime import datetime, timezone

    from agent.models import AlarmEvent
    from agent.noc_agent import run_agent

    alarm = AlarmEvent(
        alarm_id=args.alarm_id,
        device_id=args.device_id,
        alarm_type=args.alarm_type,
        severity=args.severity,
        timestamp=args.timestamp or datetime.now(tz=timezone.utc).isoformat(),
        raw_message=args.message,
        affected_site=args.site,
    )

    print(f"\nProcessing alarm {alarm.alarm_id} ({alarm.alarm_type} on {alarm.device_id})...\n")
    diagnosis = run_agent(alarm)

    if args.output == "json":
        print(diagnosis.model_dump_json(indent=2))
    else:
        _pretty_print(diagnosis)


def _pretty_print(diagnosis) -> None:
    print("=" * 60)
    print("NOC AGENT DIAGNOSIS")
    print("=" * 60)
    print(f"Alarm ID:          {diagnosis.alarm_id}")
    print(f"Classification:    {diagnosis.classification}")
    print(f"Severity:          {diagnosis.severity_assessment}")
    print(f"Confidence:        {diagnosis.confidence_score:.0%}")
    print()
    print(f"Most likely cause: {diagnosis.most_likely_cause}")
    print()

    if diagnosis.root_cause_hypotheses:
        print("Root Cause Hypotheses:")
        for h in sorted(diagnosis.root_cause_hypotheses, key=lambda x: -x.confidence):
            print(f"  [{h.confidence:.0%}] {h.hypothesis}")
    print()

    if diagnosis.recommended_actions:
        print("Recommended Actions:")
        for a in sorted(diagnosis.recommended_actions, key=lambda x: x.priority):
            print(f"  {a.priority}. {a.action}")
            print(f"     → {a.rationale}")
    print()

    if diagnosis.supporting_evidence:
        print("Supporting Evidence:")
        for e in diagnosis.supporting_evidence:
            print(f"  • {e}")
    print()

    print("Reasoning Trace:")
    print(f"  {diagnosis.reasoning_trace}")

    if diagnosis.error:
        print()
        print(f"⚠ Partial result — error: {diagnosis.error}")

    print("=" * 60)


if __name__ == "__main__":
    main()
