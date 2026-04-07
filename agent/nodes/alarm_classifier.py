"""
Node: alarm_classifier

Calls Claude Haiku with the classifier prompt to classify the alarm type
and assess severity. First node in the graph.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import anthropic
import yaml

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "classifier_v1.yaml"
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
    """Extract JSON from model response, handling markdown fences."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Find first { ... } block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def alarm_classifier(state: dict[str, Any]) -> dict[str, Any]:
    """Classify the alarm and assess severity.

    Reads from state:
        alarm_event: AlarmEvent

    Writes to state:
        classification: str
        severity_assessment: str
        error: str | None (on failure)
    """
    alarm = state["alarm_event"]
    prompt = _load_prompt()

    user_msg = prompt["user"].format(
        device_id=alarm.device_id,
        alarm_type=alarm.alarm_type,
        severity=alarm.severity,
        raw_message=alarm.raw_message,
        affected_site=alarm.affected_site,
    )

    try:
        client = _get_client()
        response = client.messages.create(
            model=os.getenv("NOC_CLASSIFIER_MODEL", prompt["model"]),
            max_tokens=prompt["max_tokens"],
            system=prompt["system"],
            messages=[{"role": "user", "content": user_msg}],
        )
        result = _extract_json(response.content[0].text)

        log.info(
            "Classifier: %s → %s (confidence %.2f)",
            alarm.alarm_type,
            result.get("classification"),
            result.get("confidence", 0),
        )

        return {
            "classification": result.get("classification", alarm.alarm_type),
            "severity_assessment": result.get("severity_assessment", alarm.severity),
            "reasoning_trace": f"Classifier: {result.get('reasoning', '')}",
            "error": None,
        }

    except Exception as exc:
        log.error("alarm_classifier failed: %s", exc)
        return {
            "classification": alarm.alarm_type,
            "severity_assessment": alarm.severity,
            "reasoning_trace": "Classifier failed — using raw alarm data.",
            "error": f"classifier_error: {exc}",
        }
