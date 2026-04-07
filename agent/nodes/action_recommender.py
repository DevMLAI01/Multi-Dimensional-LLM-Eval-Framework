"""
Node: action_recommender

Calls Claude Haiku with the recommender prompt to produce ranked remediation
actions. Final node before END.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import anthropic
import yaml

from agent.models import RecommendedAction

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "recommender_v1.yaml"
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _load_prompt() -> dict:
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise


def action_recommender(state: dict[str, Any]) -> dict[str, Any]:
    """Generate ranked remediation actions.

    Reads from state:
        alarm_event, classification, severity_assessment,
        most_likely_cause, device_info, runbook

    Writes to state:
        recommended_actions: list[RecommendedAction]
        confidence_score: float
    """
    alarm = state["alarm_event"]
    device_info = state.get("device_info", {})
    runbook = state.get("runbook", {})
    prompt = _load_prompt()

    user_msg = prompt["user"].format(
        device_id=alarm.device_id,
        alarm_type=alarm.alarm_type,
        affected_site=alarm.affected_site,
        classification=state.get("classification", alarm.alarm_type),
        most_likely_cause=state.get("most_likely_cause", "Unknown"),
        severity_assessment=state.get("severity_assessment", alarm.severity),
        sla_tier=device_info.get("sla_tier", "Unknown"),
        escalation_path=runbook.get("escalation_path", "Contact NOC duty manager."),
    )

    try:
        client = _get_client()
        response = client.messages.create(
            model=os.getenv("NOC_RECOMMENDER_MODEL", prompt["model"]),
            max_tokens=prompt["max_tokens"],
            system=prompt["system"],
            messages=[{"role": "user", "content": user_msg}],
        )
        result = _extract_json(response.content[0].text)

        actions = [
            RecommendedAction(**a)
            for a in result.get("recommended_actions", [])
        ]
        confidence = float(result.get("confidence_score", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        log.info(
            "Recommender: %d actions, confidence %.2f",
            len(actions),
            confidence,
        )

        return {
            "recommended_actions": actions,
            "confidence_score": confidence,
        }

    except Exception as exc:
        log.error("action_recommender failed: %s", exc)
        return {
            "recommended_actions": [
                RecommendedAction(
                    action="Escalate to senior NOC engineer",
                    priority=1,
                    rationale="Recommender failed — manual escalation required",
                )
            ],
            "confidence_score": 0.0,
        }
