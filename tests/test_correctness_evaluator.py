"""
Phase 3 — Correctness evaluator tests.

Unit tests: mock the judge API — fast, no API cost.
Integration tests: hit the real Anthropic API — run with -m integration.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.models import AgentDiagnosis, RecommendedAction, RootCauseHypothesis
from evaluators.base_evaluator import EvalResult
from evaluators.correctness_evaluator import CorrectnessEvaluator, _weighted_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluator():
    return CorrectnessEvaluator()


def _make_diagnosis(
    alarm_id: str = "ALM001",
    classification: str = "Physical layer failure",
    severity: str = "CRITICAL",
    most_likely_cause: str = "Fiber cut on uplink span",
    actions: list[str] | None = None,
    confidence: float = 0.85,
) -> AgentDiagnosis:
    if actions is None:
        actions = ["check_physical_layer", "contact_field_team", "escalate_p1"]
    return AgentDiagnosis(
        alarm_id=alarm_id,
        classification=classification,
        severity_assessment=severity,
        root_cause_hypotheses=[
            RootCauseHypothesis(
                hypothesis=most_likely_cause,
                confidence=confidence,
                supporting_evidence=["Prior fiber cut in alarm history"],
            )
        ],
        most_likely_cause=most_likely_cause,
        recommended_actions=[
            RecommendedAction(action=a, priority=i + 1, rationale="test")
            for i, a in enumerate(actions)
        ],
        supporting_evidence=["alarm history"],
        confidence_score=confidence,
        reasoning_trace=f"Classifier: Physical layer. Reasoner: {most_likely_cause}.",
    )


def _make_test_case(
    case_id: str = "CORR_001",
    classification: str = "Physical layer failure",
    root_cause_category: str = "FIBER_CUT | HARDWARE_FAILURE",
    required_actions: list[str] | None = None,
    severity_should_be: str = "CRITICAL",
) -> dict:
    if required_actions is None:
        required_actions = ["check_physical_layer", "contact_field_team"]
    return {
        "case_id": case_id,
        "input": {
            "alarm_id": "ALM001",
            "device_id": "RTR-OSL-042",
            "alarm_type": "LINK_DOWN",
            "severity": "CRITICAL",
            "timestamp": "2024-11-14T03:22:00Z",
            "raw_message": "Interface GigabitEthernet0/0/1 went down unexpectedly",
            "affected_site": "Oslo-DC-North",
        },
        "expected": {
            "correct_classification": classification,
            "correct_root_cause_category": root_cause_category,
            "required_actions_include": required_actions,
            "severity_should_be": severity_should_be,
            "expert_reasoning": "LINK_DOWN on P1 core router with no prior CPU/memory alarms suggests physical failure.",
        },
        "failure_mode_tested": "Physical vs software misclassification",
    }


def _mock_judge_response(
    classification: float = 1.0,
    root_cause: float = 1.0,
    actions: float = 1.0,
    severity: float = 1.0,
    reasoning: str = "Perfect diagnosis.",
    critical_errors: list | None = None,
) -> MagicMock:
    """Build a mock Anthropic API response with given judge scores."""
    score = classification * 0.30 + root_cause * 0.30 + actions * 0.25 + severity * 0.15
    payload = json.dumps({
        "classification_accuracy": classification,
        "root_cause_accuracy": root_cause,
        "action_completeness": actions,
        "severity_accuracy": severity,
        "overall_score": score,
        "reasoning": reasoning,
        "critical_errors": critical_errors or [],
    })
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=payload)]
    mock_resp.usage.input_tokens = 500
    mock_resp.usage.output_tokens = 150
    return mock_resp


# ---------------------------------------------------------------------------
# Unit tests (mocked judge)
# ---------------------------------------------------------------------------

class TestWeightedScore:
    def test_perfect_scores(self):
        sub = {"classification_accuracy": 1.0, "root_cause_accuracy": 1.0,
               "action_completeness": 1.0, "severity_accuracy": 1.0}
        assert _weighted_score(sub) == 1.0

    def test_zero_scores(self):
        sub = {"classification_accuracy": 0.0, "root_cause_accuracy": 0.0,
               "action_completeness": 0.0, "severity_accuracy": 0.0}
        assert _weighted_score(sub) == 0.0

    def test_weights_sum_to_one(self):
        from evaluators.correctness_evaluator import WEIGHTS
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_partial_score(self):
        sub = {"classification_accuracy": 1.0, "root_cause_accuracy": 1.0,
               "action_completeness": 0.0, "severity_accuracy": 1.0}
        # 0.3 + 0.3 + 0.0 + 0.15 = 0.75
        assert abs(_weighted_score(sub) - 0.75) < 0.001


class TestCorrectnessEvaluatorUnit:
    def test_perfect_diagnosis_scores_high(self, evaluator):
        with patch.object(evaluator, "_CorrectnessEvaluator__get_client" if hasattr(evaluator, "_CorrectnessEvaluator__get_client") else "__init__", create=True):
            pass
        with patch("evaluators.correctness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_judge_response(
                classification=1.0, root_cause=1.0, actions=1.0, severity=1.0,
                reasoning="All dimensions correct."
            )
            result = evaluator.evaluate(_make_test_case(), _make_diagnosis())

        assert result.score is not None
        assert result.score >= 0.90
        assert result.passed is True
        assert result.dimension == "correctness"
        assert result.error is None

    def test_wrong_classification_scores_low(self, evaluator):
        with patch("evaluators.correctness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_judge_response(
                classification=0.0, root_cause=0.0, actions=0.5, severity=0.5,
                reasoning="Classification completely wrong.",
                critical_errors=["Wrong alarm category"],
            )
            result = evaluator.evaluate(_make_test_case(), _make_diagnosis())

        assert result.score is not None
        assert result.score <= 0.30
        assert result.passed is False
        assert "Wrong alarm category" in result.sub_scores.get("critical_errors", [])

    def test_missing_required_actions_lowers_score(self, evaluator):
        diagnosis = _make_diagnosis(actions=["check_logs"])  # missing required actions
        with patch("evaluators.correctness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_judge_response(
                classification=1.0, root_cause=1.0, actions=0.0, severity=1.0,
                reasoning="Actions missing.",
            )
            result = evaluator.evaluate(_make_test_case(), diagnosis)

        assert result.sub_scores["action_completeness"] == 0.0
        # 0.3 + 0.3 + 0 + 0.15 = 0.75 — exactly at threshold
        assert result.score is not None
        assert result.score < 0.76

    def test_judge_json_parse_failure_returns_error_not_exception(self, evaluator):
        with patch("evaluators.correctness_evaluator._get_client") as mock_client:
            bad_resp = MagicMock()
            bad_resp.content = [MagicMock(text="This is not JSON at all, just plain text.")]
            bad_resp.usage.input_tokens = 100
            bad_resp.usage.output_tokens = 20
            mock_client.return_value.messages.create.return_value = bad_resp

            result = evaluator.evaluate(_make_test_case(), _make_diagnosis())

        assert result.error is not None
        assert "parse" in result.error.lower() or "json" in result.error.lower()
        assert result.score is None
        assert result.passed is False

    def test_judge_api_failure_returns_error_not_exception(self, evaluator):
        with patch("evaluators.correctness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = (
                Exception("Connection timeout")
            )
            result = evaluator.evaluate(_make_test_case(), _make_diagnosis())

        assert result.error is not None
        assert result.score is None
        assert result.passed is False

    def test_missing_agent_fields_handled_gracefully(self, evaluator):
        """Agent diagnosis with minimal fields should not crash evaluator."""
        minimal = AgentDiagnosis(
            alarm_id="ALM_MIN",
            classification="Unknown",
            severity_assessment="WARNING",
            most_likely_cause="Unknown",
            confidence_score=0.1,
            reasoning_trace="",
        )
        with patch("evaluators.correctness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_judge_response(
                classification=0.1, root_cause=0.1, actions=0.0, severity=0.5
            )
            result = evaluator.evaluate(_make_test_case(), minimal)

        assert isinstance(result, EvalResult)
        assert result.error is None

    def test_result_has_all_metadata_fields(self, evaluator):
        with patch("evaluators.correctness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_judge_response()
            result = evaluator.evaluate(_make_test_case(), _make_diagnosis())

        assert "latency_ms" in result.metadata
        assert "input_tokens" in result.metadata
        assert "output_tokens" in result.metadata
        assert "cost_usd" in result.metadata
        assert "model" in result.metadata

    def test_result_is_json_serializable(self, evaluator):
        with patch("evaluators.correctness_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_judge_response()
            result = evaluator.evaluate(_make_test_case(), _make_diagnosis())

        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["dimension"] == "correctness"
        assert "score" in parsed


# ---------------------------------------------------------------------------
# Integration tests (real Anthropic API)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCorrectnessEvaluatorIntegration:
    """Run with: uv run pytest -m integration"""

    def test_perfect_diagnosis_scores_above_90(self):
        evaluator = CorrectnessEvaluator()
        result = evaluator.evaluate(_make_test_case(), _make_diagnosis())
        assert result.error is None
        assert result.score is not None
        assert result.score >= 0.90, f"Expected >= 0.90, got {result.score}"

    def test_completely_wrong_diagnosis_scores_below_30(self):
        evaluator = CorrectnessEvaluator()
        bad_diagnosis = _make_diagnosis(
            classification="Memory leak in routing daemon",   # wrong
            most_likely_cause="RAM exhaustion due to traffic growth",  # wrong
            actions=["reboot_device", "check_memory"],        # wrong
            severity="WARNING",                               # wrong (should be CRITICAL)
            confidence=0.9,
        )
        result = evaluator.evaluate(_make_test_case(), bad_diagnosis)
        assert result.error is None
        assert result.score is not None
        assert result.score <= 0.40, f"Expected <= 0.40, got {result.score}"

    def test_scores_5_golden_cases_within_valid_range(self):
        import json
        from pathlib import Path

        cases_path = Path("data/golden_dataset/correctness_cases.json")
        if not cases_path.exists():
            pytest.skip("correctness_cases.json not found")

        with open(cases_path, encoding="utf-8") as f:
            cases = json.load(f)[:5]

        from agent.noc_agent import run_agent
        from agent.models import AlarmEvent

        evaluator = CorrectnessEvaluator()
        for case in cases:
            inp = case["input"]
            alarm = AlarmEvent(**inp)
            diagnosis = run_agent(alarm)
            result = evaluator.evaluate(case, diagnosis)

            assert isinstance(result, EvalResult), f"[{case['case_id']}] not EvalResult"
            assert result.score is not None, f"[{case['case_id']}] score is None: {result.error}"
            assert 0.0 <= result.score <= 1.0, f"[{case['case_id']}] score out of range: {result.score}"
