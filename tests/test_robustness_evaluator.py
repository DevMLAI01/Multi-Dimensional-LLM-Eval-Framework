"""
Phase 5 — Robustness evaluator tests.

Unit tests: mock the agent runs and embed model — fast, no API cost.
Integration tests: real agent + embeddings, run with -m integration.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.models import AgentDiagnosis, RecommendedAction, RootCauseHypothesis
from evaluators.base_evaluator import EvalResult
from evaluators.robustness_evaluator import (
    RobustnessEvaluator,
    _cosine_similarity,
    _diagnosis_to_text,
    SCORE_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluator():
    return RobustnessEvaluator()


def _make_diagnosis(
    alarm_id: str = "ALM_C001",
    classification: str = "Physical layer failure",
    severity: str = "CRITICAL",
    most_likely_cause: str = "Fiber cut on uplink span",
    reasoning: str = "Prior fiber cut found in alarm history.",
) -> AgentDiagnosis:
    return AgentDiagnosis(
        alarm_id=alarm_id,
        classification=classification,
        severity_assessment=severity,
        most_likely_cause=most_likely_cause,
        root_cause_hypotheses=[
            RootCauseHypothesis(hypothesis=most_likely_cause, confidence=0.85,
                                supporting_evidence=[])
        ],
        recommended_actions=[
            RecommendedAction(action="Check physical interface", priority=1, rationale="test"),
        ],
        supporting_evidence=[],
        confidence_score=0.85,
        reasoning_trace=reasoning,
    )


def _make_robustness_case(
    case_id: str = "ROB_001",
    perturbation_type: str = "PARAPHRASE",
    canonical_message: str = "Interface GigabitEthernet0/0/1 went down unexpectedly",
    perturbed_message: str = "Port GigabitEthernet0/0/1 is no longer active",
    canonical_severity: str = "CRITICAL",
    perturbed_severity: str = "CRITICAL",
) -> dict:
    base = {
        "device_id": "RTR-OSL-042",
        "alarm_type": "LINK_DOWN",
        "timestamp": "2024-11-14T03:22:00Z",
        "affected_site": "Oslo-DC-North",
    }
    return {
        "case_id": case_id,
        "perturbation_type": perturbation_type,
        "canonical_input": {
            **base, "alarm_id": "ALM_C001",
            "severity": canonical_severity,
            "raw_message": canonical_message,
        },
        "perturbed_input": {
            **base, "alarm_id": "ALM_P001",
            "severity": perturbed_severity,
            "raw_message": perturbed_message,
        },
        "expected": {
            "classification_should_match_canonical": True,
            "acceptable_score_delta": 0.15,
        },
    }


# ---------------------------------------------------------------------------
# Pure function tests (no mocking)
# ---------------------------------------------------------------------------

class TestCosineSimiliarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.5, 0.3])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.5])
        assert _cosine_similarity(a, b) == 0.0

    def test_similar_vectors_high_score(self):
        a = np.array([1.0, 1.0, 0.9])
        b = np.array([1.0, 0.9, 1.0])
        assert _cosine_similarity(a, b) > 0.98


class TestDiagnosisToText:
    def test_includes_all_key_fields(self):
        diag = _make_diagnosis()
        text = _diagnosis_to_text(diag)
        assert "Classification:" in text
        assert "Severity:" in text
        assert "Most likely cause:" in text
        assert "Reasoning:" in text

    def test_truncates_long_reasoning(self):
        diag = _make_diagnosis(reasoning="x" * 1000)
        text = _diagnosis_to_text(diag)
        # reasoning is capped at 300 chars in the implementation
        assert len(text) < 1200

    def test_includes_hypotheses(self):
        diag = _make_diagnosis()
        text = _diagnosis_to_text(diag)
        assert "Hypotheses:" in text


class TestScoreWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Unit tests (mocked agent + embeddings)
# ---------------------------------------------------------------------------

class TestRobustnessEvaluatorUnit:
    def _mock_agent_and_embeddings(
        self,
        canonical_diag: AgentDiagnosis,
        perturbed_diag: AgentDiagnosis,
        similarity: float = 0.95,
    ):
        """Context manager that mocks _run_agent and SentenceTransformer.encode."""
        from unittest.mock import patch
        import numpy as np

        # Create two embeddings with the given cosine similarity
        e1 = np.array([1.0, 0.0, 0.0])
        # e2 at angle θ where cos(θ) = similarity
        import math
        theta = math.acos(min(1.0, max(-1.0, similarity)))
        e2 = np.array([math.cos(theta), math.sin(theta), 0.0])

        run_patch = patch(
            "evaluators.robustness_evaluator._run_agent",
            side_effect=[canonical_diag, perturbed_diag],
        )
        embed_patch = patch.object(
            _get_model_class(),
            "encode",
            return_value=np.array([e1, e2]),
        )
        return run_patch, embed_patch

    def test_identical_outputs_score_near_one(self, evaluator):
        diag = _make_diagnosis()
        with patch("evaluators.robustness_evaluator._run_agent", return_value=diag), \
             patch("evaluators.robustness_evaluator._get_embed_model") as mock_model:
            mock_model.return_value.encode.return_value = np.array([
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # identical
            ])
            result = evaluator.evaluate(_make_robustness_case())

        assert result.score is not None
        assert result.score >= 0.95
        assert result.passed is True

    def test_different_classification_lowers_score(self, evaluator):
        canonical = _make_diagnosis(classification="Physical layer failure")
        perturbed = _make_diagnosis(alarm_id="ALM_P001", classification="Software bug")
        with patch("evaluators.robustness_evaluator._run_agent",
                   side_effect=[canonical, perturbed]), \
             patch("evaluators.robustness_evaluator._get_embed_model") as mock_model:
            # High semantic similarity but classification diverged
            mock_model.return_value.encode.return_value = np.array([
                [1.0, 0.0, 0.0],
                [0.99, 0.14, 0.0],
            ])
            result = evaluator.evaluate(_make_robustness_case())

        # classification_match = 0, which costs 0.3 × 1 = 0.3 off the score
        assert result.sub_scores["classification_match"] == 0.0
        assert result.score is not None

    def test_severity_mislabel_case_detected(self, evaluator):
        """SEVERITY_MISLABEL case: perturbed has wrong severity label."""
        canonical = _make_diagnosis(severity="CRITICAL")
        perturbed  = _make_diagnosis(alarm_id="ALM_P001", severity="WARNING")
        case = _make_robustness_case(
            perturbation_type="SEVERITY_MISLABEL",
            perturbed_severity="WARNING",
        )
        with patch("evaluators.robustness_evaluator._run_agent",
                   side_effect=[canonical, perturbed]), \
             patch("evaluators.robustness_evaluator._get_embed_model") as mock_model:
            mock_model.return_value.encode.return_value = np.array([
                [1.0, 0.0],
                [1.0, 0.0],  # semantically identical
            ])
            result = evaluator.evaluate(case)

        # Severity mismatch should lower score
        assert result.sub_scores["severity_match"] == 0.0
        assert result.sub_scores["perturbation_type"] == "SEVERITY_MISLABEL"

    def test_low_semantic_similarity_fails(self, evaluator):
        canonical = _make_diagnosis(
            classification="Physical layer failure",
            most_likely_cause="Fiber cut",
        )
        perturbed = _make_diagnosis(
            alarm_id="ALM_P001",
            classification="Physical layer failure",  # same
            most_likely_cause="BGP misconfiguration",  # very different
            reasoning="BGP route leak caused the interface to go admin-down.",
        )
        with patch("evaluators.robustness_evaluator._run_agent",
                   side_effect=[canonical, perturbed]), \
             patch("evaluators.robustness_evaluator._get_embed_model") as mock_model:
            # Low cosine similarity
            mock_model.return_value.encode.return_value = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],  # orthogonal → sim = 0
            ])
            result = evaluator.evaluate(_make_robustness_case())

        # sim=0 → 0.5×0 + 0.3×1 + 0.2×1 = 0.5 (below 0.85 threshold)
        assert result.score is not None
        assert result.score < 0.85
        assert result.passed is False

    def test_agent_failure_returns_error_result(self, evaluator):
        with patch("evaluators.robustness_evaluator._run_agent",
                   side_effect=Exception("Agent crashed")):
            result = evaluator.evaluate(_make_robustness_case())

        assert result.error is not None
        assert "agent_run_error" in result.error
        assert result.score is None

    def test_malformed_case_returns_error_result(self, evaluator):
        bad_case = {"case_id": "BAD_001", "perturbation_type": "PARAPHRASE"}
        result = evaluator.evaluate(bad_case)
        assert result.error is not None
        assert result.score is None

    def test_result_metadata_populated(self, evaluator):
        diag = _make_diagnosis()
        with patch("evaluators.robustness_evaluator._run_agent", return_value=diag), \
             patch("evaluators.robustness_evaluator._get_embed_model") as mock_model:
            mock_model.return_value.encode.return_value = np.array([
                [1.0, 0.0], [1.0, 0.0],
            ])
            result = evaluator.evaluate(_make_robustness_case())

        assert "agent_latency_ms" in result.metadata
        assert "embed_model" in result.metadata
        assert result.metadata["perturbation_type"] == "PARAPHRASE"

    def test_result_json_serializable(self, evaluator):
        diag = _make_diagnosis()
        with patch("evaluators.robustness_evaluator._run_agent", return_value=diag), \
             patch("evaluators.robustness_evaluator._get_embed_model") as mock_model:
            mock_model.return_value.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])
            result = evaluator.evaluate(_make_robustness_case())

        parsed = json.loads(result.model_dump_json())
        assert parsed["dimension"] == "robustness"


class TestScoreByPerturbationType:
    def test_aggregates_correctly(self):
        evaluator = RobustnessEvaluator()
        results = [
            EvalResult(case_id="R1", dimension="robustness", evaluator_version="1.0",
                       score=0.9, passed=True, sub_scores={"perturbation_type": "TYPO"}),
            EvalResult(case_id="R2", dimension="robustness", evaluator_version="1.0",
                       score=0.7, passed=False, sub_scores={"perturbation_type": "TYPO"}),
            EvalResult(case_id="R3", dimension="robustness", evaluator_version="1.0",
                       score=0.95, passed=True, sub_scores={"perturbation_type": "PARAPHRASE"}),
        ]
        summary = evaluator.score_by_perturbation_type(results)

        assert "TYPO" in summary
        assert abs(summary["TYPO"]["mean_score"] - 0.8) < 0.01
        assert summary["TYPO"]["pass_rate"] == 0.5
        assert summary["PARAPHRASE"]["mean_score"] == 0.95


def _get_model_class():
    """Get the SentenceTransformer class for patching."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRobustnessIntegration:
    """Run with: uv run pytest -m integration"""

    def test_paraphrase_case_passes(self):
        """Paraphrased input should produce near-identical classification."""
        evaluator = RobustnessEvaluator()
        case = _make_robustness_case(
            perturbation_type="PARAPHRASE",
            canonical_message="Interface GigabitEthernet0/0/1 went down unexpectedly",
            perturbed_message="Port GigabitEthernet0/0/1 is no longer active and unreachable",
        )
        result = evaluator.evaluate(case)
        assert result.error is None
        assert result.score is not None
        assert 0.0 <= result.score <= 1.0

    def test_3_golden_robustness_cases(self):
        """Run 3 cases from the golden dataset — scores should be in valid range."""
        import json
        from pathlib import Path

        cases_path = Path("data/golden_dataset/robustness_cases.json")
        if not cases_path.exists():
            pytest.skip("robustness_cases.json not found")

        with open(cases_path, encoding="utf-8") as f:
            cases = json.load(f)[:3]

        evaluator = RobustnessEvaluator()
        flagged = []
        for case in cases:
            result = evaluator.evaluate(case)
            assert result.score is not None, f"[{case['case_id']}] score None: {result.error}"
            assert 0.0 <= result.score <= 1.0
            if not result.passed:
                flagged.append(case["case_id"])

        # Log but don't fail — some perturbations legitimately cause drift
        if flagged:
            print(f"\nCases that failed threshold: {flagged}")
