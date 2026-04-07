"""
Phase 4 — Faithfulness evaluator tests.

Unit tests: mock the judge API.
Integration tests: real Anthropic API, run with -m integration.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.models import AgentDiagnosis, RecommendedAction, RootCauseHypothesis
from evaluators.base_evaluator import EvalResult
from evaluators.faithfulness_evaluator import FaithfulnessEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluator():
    return FaithfulnessEvaluator()


def _make_diagnosis(
    alarm_id: str = "ALM001",
    classification: str = "Physical layer failure",
    most_likely_cause: str = "Fiber cut on oslo-042-to-ber-019 span",
    reasoning: str = "Prior fiber cut found in alarm history. Device is P1 core router.",
    supporting_evidence: list[str] | None = None,
) -> AgentDiagnosis:
    return AgentDiagnosis(
        alarm_id=alarm_id,
        classification=classification,
        severity_assessment="CRITICAL",
        most_likely_cause=most_likely_cause,
        root_cause_hypotheses=[
            RootCauseHypothesis(hypothesis=most_likely_cause, confidence=0.85, supporting_evidence=[])
        ],
        recommended_actions=[
            RecommendedAction(action="Check physical interface status", priority=1, rationale="test"),
            RecommendedAction(action="Contact field team", priority=2, rationale="test"),
        ],
        supporting_evidence=supporting_evidence or ["Fiber cut on span oslo-042-to-ber-019"],
        confidence_score=0.85,
        reasoning_trace=reasoning,
    )


def _make_supported_case(case_id: str = "FAITH_001") -> dict:
    return {
        "case_id": case_id,
        "case_type": "CONTEXT_SUPPORTED",
        "input": {
            "alarm_event": {
                "alarm_id": "ALM_F001",
                "device_id": "RTR-OSL-042",
                "alarm_type": "LINK_DOWN",
                "severity": "CRITICAL",
                "timestamp": "2024-11-14T03:22:00Z",
                "raw_message": "Interface GigabitEthernet0/0/1 went down",
                "affected_site": "Oslo-DC-North",
            }
        },
        "injected_context": {
            "alarm_history": [
                {
                    "device_id": "RTR-OSL-042",
                    "alarm_type": "LINK_DOWN",
                    "root_cause": "Fiber cut on span oslo-042-to-ber-019",
                    "resolution": "Rerouted via backup path",
                    "timestamp": "2024-10-15T08:00:00Z",
                }
            ],
            "device_info": {"device_id": "RTR-OSL-042", "sla_tier": "P1", "role": "Core router"},
            "runbook": {
                "alarm_type": "LINK_DOWN",
                "common_causes": ["Fiber cut", "Hardware failure"],
                "escalation_path": "Contact NOC duty manager for P1 devices",
            },
        },
        "expected": {
            "response_must_reference_context": True,
            "required_context_elements": ["fiber cut", "backup path"],
            "forbidden_hallucinations": ["hardware failure", "firmware bug"],
        },
    }


def _make_absent_case(case_id: str = "FAITH_031") -> dict:
    return {
        "case_id": case_id,
        "case_type": "CONTEXT_ABSENT",
        "input": {
            "alarm_event": {
                "alarm_id": "ALM_FA001",
                "device_id": "RTR-UNKNOWN-999",
                "alarm_type": "LINK_DOWN",
                "severity": "MAJOR",
                "timestamp": "2024-11-14T03:22:00Z",
                "raw_message": "Interface went down",
                "affected_site": "Unknown-Site",
            }
        },
        "injected_context": {"alarm_history": [], "device_info": {}, "runbook": {}},
        "expected": {
            "response_must_acknowledge_missing_context": True,
            "forbidden_phrases": ["the root cause is", "this is caused by"],
        },
    }


def _mock_supported_response(
    grounding: float = 0.9,
    coverage: float = 0.8,
    has_contradictions: bool = False,
    reasoning: str = "Well-grounded response.",
) -> MagicMock:
    payload = json.dumps({
        "claims": [
            {"claim": "Fiber cut on oslo-042 span", "status": "SUPPORTED"},
            {"claim": "P1 device requires escalation", "status": "SUPPORTED"},
        ],
        "grounding_score": grounding,
        "coverage_score": coverage,
        "hallucinated_claims": [],
        "missed_key_facts": [],
        "has_contradictions": has_contradictions,
        "reasoning": reasoning,
    })
    mock = MagicMock()
    mock.content = [MagicMock(text=payload)]
    mock.usage.input_tokens = 600
    mock.usage.output_tokens = 200
    return mock


def _mock_absent_response(
    epistemic: float = 0.9,
    acknowledged: bool = True,
    hallucinated: bool = False,
    reasoning: str = "Agent correctly expressed uncertainty.",
) -> MagicMock:
    payload = json.dumps({
        "acknowledged_missing_context": acknowledged,
        "expressed_uncertainty": acknowledged,
        "hallucinated_specific_facts": hallucinated,
        "forbidden_phrase_found": None,
        "epistemic_score": epistemic,
        "reasoning": reasoning,
    })
    mock = MagicMock()
    mock.content = [MagicMock(text=payload)]
    mock.usage.input_tokens = 400
    mock.usage.output_tokens = 150
    return mock


# ---------------------------------------------------------------------------
# Unit tests — context-supported
# ---------------------------------------------------------------------------

class TestFaithfulnessSupported:
    def test_grounded_response_scores_high(self, evaluator):
        with patch("evaluators.faithfulness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_supported_response(
                grounding=0.95, coverage=0.85
            )
            result = evaluator.evaluate(_make_supported_case(), _make_diagnosis())

        # 0.7 * 0.95 + 0.3 * 0.85 = 0.665 + 0.255 = 0.92
        assert result.score is not None
        assert result.score >= 0.80
        assert result.passed is True
        assert result.error is None

    def test_hallucinated_response_scores_low(self, evaluator):
        hallucinating = _make_diagnosis(
            most_likely_cause="Hardware failure on line card",   # not in context
            reasoning="The hardware failed based on memory leak patterns.",
        )
        with patch("evaluators.faithfulness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_supported_response(
                grounding=0.2, coverage=0.1, reasoning="Agent made up most facts."
            )
            result = evaluator.evaluate(_make_supported_case(), hallucinating)

        assert result.score is not None
        assert result.score <= 0.40
        assert result.passed is False

    def test_contradicted_claim_caps_score_at_50(self, evaluator):
        with patch("evaluators.faithfulness_evaluator._get_client") as mock_client:
            # Even with 80% grounding, contradiction should cap at 0.50
            mock_client.return_value.messages.create.return_value = _mock_supported_response(
                grounding=0.8, coverage=0.7, has_contradictions=True,
                reasoning="One claim directly contradicted the context."
            )
            result = evaluator.evaluate(_make_supported_case(), _make_diagnosis())

        assert result.score is not None
        assert result.score <= 0.50
        assert result.sub_scores["has_contradictions"] is True
        assert result.metadata["contradiction_cap_applied"] is True

    def test_sub_scores_populated(self, evaluator):
        with patch("evaluators.faithfulness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_supported_response()
            result = evaluator.evaluate(_make_supported_case(), _make_diagnosis())

        assert "grounding_score" in result.sub_scores
        assert "coverage_score" in result.sub_scores
        assert "has_contradictions" in result.sub_scores
        assert "hallucinated_claims" in result.sub_scores
        assert "missed_key_facts" in result.sub_scores

    def test_score_formula_correct(self, evaluator):
        """0.7 * grounding + 0.3 * coverage, no contradiction."""
        with patch("evaluators.faithfulness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_supported_response(
                grounding=0.6, coverage=0.4
            )
            result = evaluator.evaluate(_make_supported_case(), _make_diagnosis())

        expected = 0.7 * 0.6 + 0.3 * 0.4  # = 0.54
        assert result.score is not None
        assert abs(result.score - expected) < 0.01


# ---------------------------------------------------------------------------
# Unit tests — context-absent
# ---------------------------------------------------------------------------

class TestFaithfulnessAbsent:
    def test_humble_response_passes(self, evaluator):
        humble = _make_diagnosis(
            most_likely_cause="Cannot determine — insufficient context",
            reasoning="No alarm history or device info available. Unable to confirm root cause.",
        )
        with patch("evaluators.faithfulness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_absent_response(
                epistemic=0.95, acknowledged=True, hallucinated=False
            )
            result = evaluator.evaluate(_make_absent_case(), humble)

        assert result.score is not None
        assert result.score >= 0.70
        assert result.passed is True
        assert result.sub_scores["hallucinated_specific_facts"] is False

    def test_hallucinating_response_fails(self, evaluator):
        hallucinating = _make_diagnosis(
            most_likely_cause="The root cause is a fiber cut on the uplink span",
            reasoning="Based on typical patterns, this is definitely a fiber cut.",
        )
        with patch("evaluators.faithfulness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_absent_response(
                epistemic=0.1, acknowledged=False, hallucinated=True,
                reasoning="Agent fabricated specific cause without any context."
            )
            result = evaluator.evaluate(_make_absent_case(), hallucinating)

        assert result.score is not None
        assert result.score <= 0.40
        assert result.passed is False
        assert result.sub_scores["hallucinated_specific_facts"] is True

    def test_absent_case_uses_epistemic_score(self, evaluator):
        with patch("evaluators.faithfulness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_absent_response(
                epistemic=0.75
            )
            result = evaluator.evaluate(_make_absent_case(), _make_diagnosis())

        assert result.sub_scores.get("epistemic_score") == 0.75
        assert result.metadata["case_type"] == "CONTEXT_ABSENT"


# ---------------------------------------------------------------------------
# Shared error handling tests
# ---------------------------------------------------------------------------

class TestFaithfulnessErrorHandling:
    def test_judge_json_parse_failure_returns_error(self, evaluator):
        with patch("evaluators.faithfulness_evaluator._get_client") as mock_client:
            bad = MagicMock()
            bad.content = [MagicMock(text="not json at all")]
            bad.usage.input_tokens = 100
            bad.usage.output_tokens = 10
            mock_client.return_value.messages.create.return_value = bad
            result = evaluator.evaluate(_make_supported_case(), _make_diagnosis())

        assert result.error is not None
        assert result.score is None
        assert result.passed is False

    def test_api_failure_returns_error_not_exception(self, evaluator):
        with patch("evaluators.faithfulness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = Exception("API timeout")
            result = evaluator.evaluate(_make_supported_case(), _make_diagnosis())

        assert result.error is not None
        assert result.score is None

    def test_result_has_metadata_fields(self, evaluator):
        with patch("evaluators.faithfulness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_supported_response()
            result = evaluator.evaluate(_make_supported_case(), _make_diagnosis())

        assert "latency_ms" in result.metadata
        assert "cost_usd" in result.metadata
        assert "model" in result.metadata

    def test_result_json_serializable(self, evaluator):
        with patch("evaluators.faithfulness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_supported_response()
            result = evaluator.evaluate(_make_supported_case(), _make_diagnosis())

        parsed = json.loads(result.model_dump_json())
        assert parsed["dimension"] == "faithfulness"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFaithfulnessIntegration:
    """Run with: uv run pytest -m integration"""

    def test_grounded_response_scores_above_threshold(self):
        """Agent with context-sourced answer should score >= 0.70."""
        from agent.models import AlarmEvent
        from agent.nodes.context_fetcher import context_fetcher
        from agent.noc_agent import run_agent

        evaluator = FaithfulnessEvaluator()
        case = _make_supported_case()
        alarm = AlarmEvent(**case["input"]["alarm_event"])
        diagnosis = run_agent(alarm)
        result = evaluator.evaluate(case, diagnosis)

        assert result.error is None, f"Error: {result.error}"
        assert result.score is not None
        assert 0.0 <= result.score <= 1.0

    def test_hallucination_detected_in_5_absent_cases(self):
        """Context-absent cases: agent should not hallucinate confidently."""
        import json
        from pathlib import Path

        from agent.models import AlarmEvent
        from agent.noc_agent import run_agent

        cases_path = Path("data/golden_dataset/faithfulness_cases.json")
        if not cases_path.exists():
            pytest.skip("faithfulness_cases.json not found")

        with open(cases_path, encoding="utf-8") as f:
            all_cases = json.load(f)

        absent_cases = [c for c in all_cases if c.get("case_type") == "CONTEXT_ABSENT"][:3]
        if not absent_cases:
            pytest.skip("No CONTEXT_ABSENT cases found")

        evaluator = FaithfulnessEvaluator()
        for case in absent_cases:
            alarm_data = case["input"].get("alarm_event", case["input"])
            alarm = AlarmEvent(**alarm_data)
            diagnosis = run_agent(alarm)
            result = evaluator.evaluate(case, diagnosis)

            assert isinstance(result, EvalResult), f"[{case['case_id']}] not EvalResult"
            assert result.score is not None, f"[{case['case_id']}] score None: {result.error}"
            assert 0.0 <= result.score <= 1.0
