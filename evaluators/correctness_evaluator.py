"""
Phase 3 — Correctness Evaluator.

Uses Claude Sonnet as an LLM-as-judge to score whether the agent's diagnosis
matches the expert-labelled expected output in the golden dataset.

Scoring weights:
    classification_accuracy  × 0.30
    root_cause_accuracy      × 0.30
    action_completeness      × 0.25
    severity_accuracy        × 0.15
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import anthropic
import yaml
from dotenv import load_dotenv

from agent.models import AgentDiagnosis
from evaluators.base_evaluator import BaseEvaluator, EvalResult

load_dotenv()

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "judge_correctness.yaml"

WEIGHTS = {
    "classification_accuracy": 0.30,
    "root_cause_accuracy":     0.30,
    "action_completeness":     0.25,
    "severity_accuracy":       0.15,
}

# Haiku per-token costs (USD per token) for metadata tracking
_COST_PER_INPUT_TOKEN  = 0.25 / 1_000_000   # Sonnet: $3/M
_COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000   # Sonnet: $15/M

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    return _client


def _load_prompt() -> dict:
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _extract_json(text: str) -> dict:
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _format_actions(actions: list) -> str:
    """Format recommended actions list for the judge prompt."""
    if not actions:
        return "None"
    parts = []
    for a in actions:
        if hasattr(a, "action"):
            parts.append(f"{a.priority}. {a.action}")
        elif isinstance(a, dict):
            parts.append(f"{a.get('priority', '?')}. {a.get('action', str(a))}")
        else:
            parts.append(str(a))
    return "; ".join(parts)


def _weighted_score(sub_scores: dict[str, float]) -> float:
    """Compute weighted overall score from sub-scores."""
    total = 0.0
    for key, weight in WEIGHTS.items():
        total += sub_scores.get(key, 0.0) * weight
    return round(min(1.0, max(0.0, total)), 4)


class CorrectnessEvaluator(BaseEvaluator):
    """LLM-as-judge evaluator for diagnosis correctness."""

    dimension = "correctness"
    version = "1.0"
    threshold = 0.75    # score below this = FAIL

    def evaluate(self, test_case: dict, agent_output: AgentDiagnosis) -> EvalResult:
        """Score the agent's diagnosis against the expert-labelled expected output.

        Args:
            test_case:    A case from correctness_cases.json
            agent_output: AgentDiagnosis returned by run_agent()

        Returns:
            EvalResult with score 0.0–1.0. Never raises.
        """
        case_id = test_case.get("case_id", "UNKNOWN")
        expected = test_case.get("expected", {})
        inp = test_case.get("input", {})

        try:
            prompt = _load_prompt()
            user_msg = prompt["user"].format(
                device_id=inp.get("device_id", ""),
                alarm_type=inp.get("alarm_type", ""),
                severity=inp.get("severity", ""),
                affected_site=inp.get("affected_site", ""),
                raw_message=inp.get("raw_message", ""),
                expected_classification=expected.get("correct_classification", ""),
                expected_root_cause_category=expected.get("correct_root_cause_category", ""),
                required_actions=", ".join(expected.get("required_actions_include", [])),
                expert_reasoning=expected.get("expert_reasoning", ""),
                agent_classification=agent_output.classification,
                agent_severity=agent_output.severity_assessment,
                agent_most_likely_cause=agent_output.most_likely_cause,
                agent_recommended_actions=_format_actions(agent_output.recommended_actions),
                agent_confidence=f"{agent_output.confidence_score:.2f}",
                agent_reasoning_trace=agent_output.reasoning_trace[:500],  # truncate for cost
            )

            t_start = time.monotonic()
            client = _get_client()
            response = client.messages.create(
                model=prompt["model"],
                max_tokens=prompt["max_tokens"],
                system=prompt["system"],
                messages=[{"role": "user", "content": user_msg}],
            )
            latency_ms = int((time.monotonic() - t_start) * 1000)

            usage = response.usage
            cost_usd = (
                usage.input_tokens  * _COST_PER_INPUT_TOKEN +
                usage.output_tokens * _COST_PER_OUTPUT_TOKEN
            )

            judge = _extract_json(response.content[0].text)

            # Recompute weighted score ourselves (don't trust model arithmetic)
            sub_scores = {k: float(judge.get(k, 0.0)) for k in WEIGHTS}
            overall = _weighted_score(sub_scores)

            passed = overall >= self.threshold

            log.info(
                "[%s] correctness=%.3f (%s) cls=%.2f rc=%.2f act=%.2f sev=%.2f",
                case_id, overall, "PASS" if passed else "FAIL",
                sub_scores["classification_accuracy"],
                sub_scores["root_cause_accuracy"],
                sub_scores["action_completeness"],
                sub_scores["severity_accuracy"],
            )

            # Rate-limit courtesy pause
            time.sleep(0.5)

            return EvalResult(
                case_id=case_id,
                dimension=self.dimension,
                evaluator_version=self.version,
                score=overall,
                passed=passed,
                reasoning=judge.get("reasoning", ""),
                sub_scores={
                    **sub_scores,
                    "critical_errors": judge.get("critical_errors", []),
                },
                metadata={
                    "model": prompt["model"],
                    "latency_ms": latency_ms,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cost_usd": round(cost_usd, 6),
                },
                agent_run_id=agent_output.alarm_id,
            )

        except json.JSONDecodeError as exc:
            log.error("[%s] Judge JSON parse failed: %s", case_id, exc)
            return self._make_error_result(case_id, f"judge_json_parse_error: {exc}")

        except anthropic.APIError as exc:
            log.error("[%s] Judge API error: %s", case_id, exc)
            return self._make_error_result(case_id, f"judge_api_error: {exc}")

        except Exception as exc:
            log.error("[%s] Unexpected error: %s", case_id, exc)
            return self._make_error_result(case_id, f"unexpected_error: {exc}")
