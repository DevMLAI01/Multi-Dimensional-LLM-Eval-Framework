"""
Node: root_cause_reasoner

Calls Claude Haiku with the reasoner prompt + full context fetched by
context_fetcher to produce ranked root cause hypotheses.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import anthropic
import yaml

from agent.models import RootCauseHypothesis

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "reasoner_v1.yaml"
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


def _format_context(data: list | dict) -> str:
    """Format context data as compact JSON string for the prompt."""
    if not data:
        return "No data available."
    return json.dumps(data, indent=2, default=str)


def root_cause_reasoner(state: dict[str, Any]) -> dict[str, Any]:
    """Perform root cause analysis using fetched context.

    Reads from state:
        alarm_event, classification, severity_assessment,
        alarm_history, device_info, runbook

    Writes to state:
        root_cause_hypotheses: list[RootCauseHypothesis]
        most_likely_cause: str
        supporting_evidence: list[str]
        reasoning_trace: str (appended)
    """
    alarm = state["alarm_event"]
    prompt = _load_prompt()

    user_msg = prompt["user"].format(
        device_id=alarm.device_id,
        alarm_type=alarm.alarm_type,
        severity_assessment=state.get("severity_assessment", alarm.severity),
        raw_message=alarm.raw_message,
        alarm_history=_format_context(state.get("alarm_history", [])),
        device_info=_format_context(state.get("device_info", {})),
        runbook=_format_context(state.get("runbook", {})),
    )

    try:
        client = _get_client()
        response = client.messages.create(
            model=os.getenv("NOC_REASONER_MODEL", prompt["model"]),
            max_tokens=prompt["max_tokens"],
            system=prompt["system"],
            messages=[{"role": "user", "content": user_msg}],
        )
        result = _extract_json(response.content[0].text)

        hypotheses = [
            RootCauseHypothesis(**h)
            for h in result.get("root_cause_hypotheses", [])
        ]

        prior_trace = state.get("reasoning_trace", "")
        new_trace = result.get("reasoning_trace", "")
        combined_trace = f"{prior_trace}\nReasoner: {new_trace}".strip()

        log.info(
            "Reasoner: %d hypotheses, most likely: %s",
            len(hypotheses),
            result.get("most_likely_cause", "")[:60],
        )

        return {
            "root_cause_hypotheses": hypotheses,
            "most_likely_cause": result.get("most_likely_cause", "Unknown"),
            "supporting_evidence": result.get("supporting_evidence", []),
            "reasoning_trace": combined_trace,
        }

    except Exception as exc:
        log.error("root_cause_reasoner failed: %s", exc)
        prior_trace = state.get("reasoning_trace", "")
        return {
            "root_cause_hypotheses": [],
            "most_likely_cause": "Unknown — reasoner failed",
            "supporting_evidence": [],
            "reasoning_trace": f"{prior_trace}\nReasoner failed: {exc}".strip(),
        }
