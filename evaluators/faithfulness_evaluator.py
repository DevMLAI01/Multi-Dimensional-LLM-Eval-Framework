"""
Phase 4 — Faithfulness Evaluator.

Measures whether the agent's response is grounded in the retrieved context,
or contains hallucinated details not present in the context.

Two sub-types (driven by test_case["case_type"]):

  CONTEXT_SUPPORTED:
    - grounding_score: fraction of claims backed by context
    - coverage_score:  fraction of key context facts the agent used
    - overall = 0.7 * grounding + 0.3 * coverage
    - Any CONTRADICTED claim caps overall at 0.50

  CONTEXT_ABSENT:
    - Measures epistemic humility — did the agent admit it doesn't know?
    - Score = epistemic_score from judge
    - Passes if agent acknowledged missing context and didn't hallucinate
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

_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "judge_faithfulness.yaml"

_COST_PER_INPUT_TOKEN  = 3.0  / 1_000_000   # Sonnet
_COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000

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


def _format_list(items: list) -> str:
    if not items:
        return "None"
    parts = []
    for item in items:
        if hasattr(item, "action"):
            parts.append(f"{item.priority}. {item.action}")
        elif isinstance(item, dict):
            parts.append(str(item.get("action", item)))
        else:
            parts.append(str(item))
    return "; ".join(parts)


def _format_context(data: Any) -> str:
    if not data:
        return "None"
    return json.dumps(data, indent=2, default=str)[:1000]  # cap at 1k chars for cost


class FaithfulnessEvaluator(BaseEvaluator):
    """LLM-as-judge evaluator measuring response grounding in retrieved context."""

    dimension = "faithfulness"
    version = "1.0"
    threshold = 0.70    # below this = FAIL

    # Score cap when any claim CONTRADICTS the context
    CONTRADICTION_CAP = 0.50

    def evaluate(self, test_case: dict, agent_output: AgentDiagnosis) -> EvalResult:
        """Score faithfulness of the agent's response to the injected context.

        Args:
            test_case:    A case from faithfulness_cases.json
            agent_output: AgentDiagnosis returned by run_agent()

        Returns:
            EvalResult with score 0.0–1.0. Never raises.
        """
        case_id = test_case.get("case_id", "UNKNOWN")
        case_type = test_case.get("case_type", "CONTEXT_SUPPORTED")

        try:
            if case_type == "CONTEXT_ABSENT":
                return self._evaluate_absent(test_case, agent_output)
            return self._evaluate_supported(test_case, agent_output)
        except Exception as exc:
            log.error("[%s] Unexpected error in faithfulness eval: %s", case_id, exc)
            return self._make_error_result(case_id, f"unexpected_error: {exc}")

    # ------------------------------------------------------------------
    # Context-supported path
    # ------------------------------------------------------------------

    def _evaluate_supported(self, test_case: dict, agent_output: AgentDiagnosis) -> EvalResult:
        case_id = test_case.get("case_id", "UNKNOWN")
        ctx = test_case.get("injected_context", {})
        prompt = _load_prompt()

        user_msg = prompt["user_context_supported"].format(
            alarm_history=_format_context(ctx.get("alarm_history", [])),
            device_info=_format_context(ctx.get("device_info", {})),
            runbook=_format_context(ctx.get("runbook", {})),
            agent_classification=agent_output.classification,
            agent_most_likely_cause=agent_output.most_likely_cause,
            agent_recommended_actions=_format_list(agent_output.recommended_actions),
            agent_reasoning_trace=agent_output.reasoning_trace[:600],
            agent_supporting_evidence=_format_list(agent_output.supporting_evidence),
        )

        try:
            t_start = time.monotonic()
            response = _get_client().messages.create(
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
        except json.JSONDecodeError as exc:
            return self._make_error_result(case_id, f"judge_json_parse_error: {exc}")
        except anthropic.APIError as exc:
            return self._make_error_result(case_id, f"judge_api_error: {exc}")

        grounding = float(judge.get("grounding_score", 0.0))
        coverage  = float(judge.get("coverage_score", 0.0))
        has_contradictions = bool(judge.get("has_contradictions", False))

        overall = 0.7 * grounding + 0.3 * coverage

        # Cap score if any claim directly contradicts the context
        if has_contradictions:
            overall = min(overall, self.CONTRADICTION_CAP)
            log.warning("[%s] Contradicted claim found — capping score at %.2f", case_id, self.CONTRADICTION_CAP)

        overall = round(min(1.0, max(0.0, overall)), 4)
        passed = overall >= self.threshold

        log.info(
            "[%s] faithfulness=%.3f (%s) grounding=%.2f coverage=%.2f contradictions=%s",
            case_id, overall, "PASS" if passed else "FAIL",
            grounding, coverage, has_contradictions,
        )

        time.sleep(0.5)

        return EvalResult(
            case_id=case_id,
            dimension=self.dimension,
            evaluator_version=self.version,
            score=overall,
            passed=passed,
            reasoning=judge.get("reasoning", ""),
            sub_scores={
                "grounding_score": grounding,
                "coverage_score": coverage,
                "has_contradictions": has_contradictions,
                "hallucinated_claims": judge.get("hallucinated_claims", []),
                "missed_key_facts": judge.get("missed_key_facts", []),
                "claims": judge.get("claims", []),
            },
            metadata={
                "model": prompt["model"],
                "case_type": "CONTEXT_SUPPORTED",
                "latency_ms": latency_ms,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": round(cost_usd, 6),
                "contradiction_cap_applied": has_contradictions,
            },
            agent_run_id=agent_output.alarm_id,
        )

    # ------------------------------------------------------------------
    # Context-absent path
    # ------------------------------------------------------------------

    def _evaluate_absent(self, test_case: dict, agent_output: AgentDiagnosis) -> EvalResult:
        case_id = test_case.get("case_id", "UNKNOWN")
        prompt = _load_prompt()

        user_msg = prompt["user_context_absent"].format(
            agent_classification=agent_output.classification,
            agent_most_likely_cause=agent_output.most_likely_cause,
            agent_recommended_actions=_format_list(agent_output.recommended_actions),
            agent_reasoning_trace=agent_output.reasoning_trace[:600],
        )

        try:
            t_start = time.monotonic()
            response = _get_client().messages.create(
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
        except json.JSONDecodeError as exc:
            return self._make_error_result(case_id, f"judge_json_parse_error: {exc}")
        except anthropic.APIError as exc:
            return self._make_error_result(case_id, f"judge_api_error: {exc}")

        epistemic = float(judge.get("epistemic_score", 0.0))
        hallucinated = bool(judge.get("hallucinated_specific_facts", False))
        overall = round(min(1.0, max(0.0, epistemic)), 4)
        passed = overall >= self.threshold

        log.info(
            "[%s] faithfulness(absent)=%.3f (%s) epistemic=%.2f hallucinated=%s",
            case_id, overall, "PASS" if passed else "FAIL",
            epistemic, hallucinated,
        )

        time.sleep(0.5)

        return EvalResult(
            case_id=case_id,
            dimension=self.dimension,
            evaluator_version=self.version,
            score=overall,
            passed=passed,
            reasoning=judge.get("reasoning", ""),
            sub_scores={
                "epistemic_score": epistemic,
                "acknowledged_missing_context": judge.get("acknowledged_missing_context", False),
                "expressed_uncertainty": judge.get("expressed_uncertainty", False),
                "hallucinated_specific_facts": hallucinated,
                "forbidden_phrase_found": judge.get("forbidden_phrase_found"),
            },
            metadata={
                "model": prompt["model"],
                "case_type": "CONTEXT_ABSENT",
                "latency_ms": latency_ms,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": round(cost_usd, 6),
            },
            agent_run_id=agent_output.alarm_id,
        )
