"""
Phase 7 — Latency/Quality Evaluator tests.

Unit tests: mock agent runs and judge calls — fast, no API cost.
Integration tests: real agent + judge, run with -m integration.
"""

import json
import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.models import AgentDiagnosis, RecommendedAction, RootCauseHypothesis
from evaluators.latency_quality_evaluator import (
    MODEL_CONFIGS,
    HAIKU,
    SONNET,
    AgentRunResult,
    ApiCallRecord,
    LatencyQualityEvaluator,
    LatencyRunRecord,
    _cost_usd,
    score_quality,
)
from evaluators.statistical_significance import (
    PairwiseTestResult,
    compare_configs,
    paired_ttest,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_diagnosis(
    alarm_id: str = "ALM_L001",
    classification: str = "Physical layer failure",
    most_likely_cause: str = "Fiber cut on uplink span",
    confidence: float = 0.85,
    reasoning: str = "Physical layer signal lost on core router.",
) -> AgentDiagnosis:
    return AgentDiagnosis(
        alarm_id=alarm_id,
        classification=classification,
        severity_assessment="CRITICAL",
        most_likely_cause=most_likely_cause,
        root_cause_hypotheses=[
            RootCauseHypothesis(hypothesis=most_likely_cause, confidence=confidence,
                                supporting_evidence=[])
        ],
        recommended_actions=[
            RecommendedAction(action="Check physical interface", priority=1, rationale="test"),
        ],
        supporting_evidence=[],
        confidence_score=confidence,
        reasoning_trace=reasoning,
    )


def _make_latency_case(
    case_id: str = "LAT_001",
    complexity: str = "SIMPLE",
    alarm_type: str = "LINK_DOWN",
) -> dict:
    return {
        "case_id": case_id,
        "complexity": complexity,
        "input": {
            "alarm_id": f"ALM_{case_id}",
            "device_id": "RTR-OSL-001",
            "alarm_type": alarm_type,
            "severity": "CRITICAL",
            "timestamp": "2024-03-12T08:14:22Z",
            "raw_message": "Interface GigabitEthernet0/0 on RTR-OSL-001 is DOWN.",
            "affected_site": "Stockholm-DC-North",
        },
        "complexity_factors": {
            "has_alarm_history": True,
            "device_in_inventory": True,
            "has_matching_runbook": True,
            "ambiguity_level": "LOW",
            "reasoning_depth_required": "SHALLOW",
        },
        "expected": {
            "expected_hypothesis_count": 1,
            "expected_confidence_range": "0.85-0.95",
            "quality_metric_focus": "correctness",
            "notes": "Physical layer signal loss on a known P1 device. Root cause is cable fault.",
        },
    }


def _make_agent_run_result(
    latency_ms: int = 1200,
    n_calls: int = 3,
    model: str = HAIKU,
    error: str = None,
) -> AgentRunResult:
    calls = [
        ApiCallRecord(model=model, input_tokens=500, output_tokens=100,
                      latency_ms=latency_ms // n_calls)
        for _ in range(n_calls)
    ]
    return AgentRunResult(
        diagnosis=_make_diagnosis() if not error else None,
        total_latency_ms=latency_ms,
        api_calls=calls if not error else [],
        error=error,
    )


def _mock_judge_response(
    cls_acc: float = 0.9,
    rc_acc:  float = 0.8,
    act_comp: float = 0.7,
    sev_acc: float = 1.0,
) -> MagicMock:
    payload = json.dumps({
        "classification_accuracy": cls_acc,
        "root_cause_accuracy": rc_acc,
        "action_completeness": act_comp,
        "severity_accuracy": sev_acc,
        "overall_score": 0.85,
        "reasoning": "Good diagnosis.",
        "critical_errors": [],
    })
    mock = MagicMock()
    mock.content = [MagicMock(text=payload)]
    mock.usage.input_tokens = 600
    mock.usage.output_tokens = 150
    return mock


# ---------------------------------------------------------------------------
# MODEL_CONFIGS tests
# ---------------------------------------------------------------------------

class TestModelConfigs:
    def test_three_configs_defined(self):
        assert set(MODEL_CONFIGS.keys()) == {"haiku-all", "sonnet-all", "hybrid"}

    def test_haiku_all_uses_haiku(self):
        cfg = MODEL_CONFIGS["haiku-all"]
        assert all(v == HAIKU for v in cfg.values())

    def test_sonnet_all_uses_sonnet(self):
        cfg = MODEL_CONFIGS["sonnet-all"]
        assert all(v == SONNET for v in cfg.values())

    def test_hybrid_uses_sonnet_for_reasoner(self):
        cfg = MODEL_CONFIGS["hybrid"]
        assert cfg["NOC_REASONER_MODEL"] == SONNET
        assert cfg["NOC_CLASSIFIER_MODEL"] == HAIKU
        assert cfg["NOC_RECOMMENDER_MODEL"] == HAIKU


# ---------------------------------------------------------------------------
# Cost calculation tests
# ---------------------------------------------------------------------------

class TestCostCalculation:
    def test_haiku_cheaper_than_sonnet(self):
        haiku_cost  = _cost_usd(HAIKU,  1000, 100)
        sonnet_cost = _cost_usd(SONNET, 1000, 100)
        assert haiku_cost < sonnet_cost

    def test_cost_scales_with_tokens(self):
        cost_small = _cost_usd(HAIKU, 100, 10)
        cost_large = _cost_usd(HAIKU, 1000, 100)
        assert cost_large > cost_small

    def test_zero_tokens_zero_cost(self):
        assert _cost_usd(HAIKU, 0, 0) == 0.0


# ---------------------------------------------------------------------------
# AgentRunResult property tests
# ---------------------------------------------------------------------------

class TestAgentRunResult:
    def test_total_tokens_summed(self):
        run = _make_agent_run_result(n_calls=3)
        assert run.total_input_tokens == 3 * 500
        assert run.total_output_tokens == 3 * 100

    def test_cost_computed(self):
        run = _make_agent_run_result(n_calls=3, model=HAIKU)
        assert run.estimated_cost_usd > 0.0

    def test_node_latencies_labelled(self):
        run = _make_agent_run_result(latency_ms=1500, n_calls=3)
        nl = run.node_latencies
        assert "classifier" in nl
        assert "reasoner" in nl
        assert "recommender" in nl


# ---------------------------------------------------------------------------
# Quality scoring tests
# ---------------------------------------------------------------------------

class TestScoreQuality:
    def test_good_diagnosis_scores_high(self):
        case = _make_latency_case()
        diag = _make_diagnosis()
        with patch("evaluators.latency_quality_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_judge_response(
                cls_acc=0.9, rc_acc=0.9, act_comp=0.8, sev_acc=1.0
            )
            score, sub = score_quality(case, diag)

        assert score > 0.80
        assert "classification_accuracy" in sub

    def test_poor_diagnosis_scores_low(self):
        case = _make_latency_case()
        diag = _make_diagnosis()
        with patch("evaluators.latency_quality_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_judge_response(
                cls_acc=0.1, rc_acc=0.1, act_comp=0.1, sev_acc=0.1
            )
            score, sub = score_quality(case, diag)

        assert score < 0.30

    def test_judge_api_failure_returns_zero(self):
        case = _make_latency_case()
        diag = _make_diagnosis()
        with patch("evaluators.latency_quality_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = Exception("API error")
            score, sub = score_quality(case, diag)

        assert score == 0.0
        assert "error" in sub

    def test_score_formula_correct(self):
        """0.30*cls + 0.30*rc + 0.25*act + 0.15*sev."""
        case = _make_latency_case()
        diag = _make_diagnosis()
        with patch("evaluators.latency_quality_evaluator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _mock_judge_response(
                cls_acc=1.0, rc_acc=1.0, act_comp=0.0, sev_acc=0.0
            )
            score, _ = score_quality(case, diag)

        expected = 0.30 * 1.0 + 0.30 * 1.0 + 0.25 * 0.0 + 0.15 * 0.0  # = 0.60
        assert abs(score - expected) < 0.01


# ---------------------------------------------------------------------------
# LatencyRunRecord tests
# ---------------------------------------------------------------------------

class TestLatencyRunRecord:
    def test_to_eval_result_sets_dimension(self):
        record = LatencyRunRecord(
            case_id="LAT_001", config_name="hybrid", complexity="SIMPLE",
            total_latency_ms=1200, node_latencies={"classifier": 400},
            total_input_tokens=1500, total_output_tokens=300,
            estimated_cost_usd=0.0012, quality_score=0.85,
            quality_sub_scores={"classification_accuracy": 0.9},
        )
        result = record.to_eval_result()
        assert result.dimension == "latency_quality"
        assert result.case_id == "LAT_001_hybrid"
        assert result.score == 0.85
        assert result.passed is True

    def test_pass_threshold_is_70(self):
        record = LatencyRunRecord(
            case_id="LAT_X", config_name="haiku-all", complexity="MEDIUM",
            total_latency_ms=2000, node_latencies={},
            total_input_tokens=1000, total_output_tokens=200,
            estimated_cost_usd=0.0008, quality_score=0.69,
            quality_sub_scores={},
        )
        result = record.to_eval_result()
        assert result.passed is False


# ---------------------------------------------------------------------------
# Evaluator.run_single unit test (mocked agent)
# ---------------------------------------------------------------------------

class TestLatencyQualityEvaluatorUnit:
    def test_run_single_happy_path(self):
        evaluator = LatencyQualityEvaluator()
        case = _make_latency_case()
        diag = _make_diagnosis()

        with patch("evaluators.latency_quality_evaluator.run_agent_with_config",
                   return_value=_make_agent_run_result()), \
             patch("evaluators.latency_quality_evaluator.score_quality",
                   return_value=(0.85, {"classification_accuracy": 0.9})):
            record = evaluator.run_single(case, "hybrid")

        assert record.error is None
        assert record.quality_score == 0.85
        assert record.config_name == "hybrid"
        assert record.total_latency_ms == 1200

    def test_run_single_agent_error(self):
        evaluator = LatencyQualityEvaluator()
        case = _make_latency_case()

        with patch("evaluators.latency_quality_evaluator.run_agent_with_config",
                   return_value=_make_agent_run_result(error="API crashed")):
            record = evaluator.run_single(case, "haiku-all")

        assert record.error == "API crashed"
        assert record.quality_score == 0.0

    def test_run_single_bad_input(self):
        evaluator = LatencyQualityEvaluator()
        bad_case = {"case_id": "BAD", "complexity": "SIMPLE", "input": {}}

        record = evaluator.run_single(bad_case, "hybrid")
        assert record.error is not None
        assert "input_parse_error" in record.error

    def test_aggregate_computes_p95(self):
        evaluator = LatencyQualityEvaluator()
        records = [
            LatencyRunRecord(
                case_id=f"L{i}", config_name="haiku-all", complexity="SIMPLE",
                total_latency_ms=1000 + i * 100,
                node_latencies={}, total_input_tokens=500, total_output_tokens=100,
                estimated_cost_usd=0.001, quality_score=0.80, quality_sub_scores={},
            )
            for i in range(20)
        ]
        agg = evaluator.aggregate(records)
        assert "haiku-all" in agg
        assert agg["haiku-all"]["p95_latency_ms"] >= agg["haiku-all"]["p50_latency_ms"]
        assert agg["haiku-all"]["mean_quality"] == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# Statistical significance tests
# ---------------------------------------------------------------------------

class TestStatisticalSignificance:
    def test_paired_ttest_significant_difference(self):
        # haiku scores around 0.60, hybrid around 0.85 — should be significant
        rng = np.random.default_rng(42)
        scores_a = (rng.normal(0.60, 0.05, 20)).tolist()
        scores_b = (rng.normal(0.85, 0.05, 20)).tolist()
        result = paired_ttest(scores_a, scores_b, "haiku-all", "hybrid")

        assert result.significant is True
        assert result.quality_delta > 0
        assert result.p_value < 0.05

    def test_paired_ttest_no_difference(self):
        # Same distribution — should NOT be significant
        rng = np.random.default_rng(99)
        scores_a = (rng.normal(0.80, 0.02, 30)).tolist()
        scores_b = (rng.normal(0.80, 0.02, 30)).tolist()
        result = paired_ttest(scores_a, scores_b, "hybrid", "sonnet-all")

        assert result.significant is False

    def test_ttest_too_few_samples(self):
        result = paired_ttest([0.8], [0.9], "a", "b")
        assert result.significant is False
        assert "Insufficient" in result.interpretation

    def test_compare_configs_returns_three_tests(self):
        # Build synthetic records for 3 configs
        records = []
        configs = ["haiku-all", "sonnet-all", "hybrid"]
        rng = np.random.default_rng(7)
        means = {"haiku-all": 0.65, "sonnet-all": 0.88, "hybrid": 0.83}
        for config in configs:
            for i in range(10):
                records.append(LatencyRunRecord(
                    case_id=f"LAT_{i:03d}",
                    config_name=config,
                    complexity="MEDIUM",
                    total_latency_ms=1000,
                    node_latencies={},
                    total_input_tokens=500,
                    total_output_tokens=100,
                    estimated_cost_usd=0.001,
                    quality_score=float(rng.normal(means[config], 0.03)),
                    quality_sub_scores={},
                ))

        tests = compare_configs(records)
        assert len(tests) == 3  # 3 pairwise comparisons
        assert "haiku-all_vs_hybrid" in tests
        assert "haiku-all_vs_sonnet-all" in tests
        assert "hybrid_vs_sonnet-all" in tests


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLatencyQualityIntegration:
    """Run with: uv run pytest -m integration"""

    def test_single_case_haiku_all(self):
        """One case, haiku-all config — should complete and return valid data."""
        evaluator = LatencyQualityEvaluator()
        case = _make_latency_case()

        record = evaluator.run_single(case, "haiku-all")

        assert record.error is None, f"Error: {record.error}"
        assert record.total_latency_ms > 0
        assert record.quality_score is not None
        assert 0.0 <= record.quality_score <= 1.0
        assert record.estimated_cost_usd > 0

    def test_single_case_hybrid(self):
        """One case, hybrid config — reasoner should use Sonnet."""
        evaluator = LatencyQualityEvaluator()
        case = _make_latency_case(case_id="LAT_HYBRID_001")

        record = evaluator.run_single(case, "hybrid")

        assert record.error is None, f"Error: {record.error}"
        assert record.config_name == "hybrid"
        assert record.total_latency_ms > 0

    def test_env_vars_restored_after_run(self):
        """Env vars set for model config should be cleaned up after run."""
        evaluator = LatencyQualityEvaluator()
        case = _make_latency_case()

        # Record env state before
        before = {k: os.environ.get(k) for k in
                  ["NOC_CLASSIFIER_MODEL", "NOC_REASONER_MODEL", "NOC_RECOMMENDER_MODEL"]}

        evaluator.run_single(case, "sonnet-all")

        # Env vars should be restored
        after = {k: os.environ.get(k) for k in before}
        assert before == after, f"Env vars not restored: {after}"

    def test_3_cases_all_configs(self):
        """Run 3 latency cases × 3 configs = 9 runs. Aggregate should be valid."""
        import json
        from pathlib import Path

        cases_path = Path("data/golden_dataset/latency_cases.json")
        if not cases_path.exists():
            pytest.skip("latency_cases.json not found")

        with open(cases_path, encoding="utf-8") as f:
            cases = json.load(f)[:3]

        evaluator = LatencyQualityEvaluator()
        records = evaluator.run_all_configs(cases, delay_between=0.3)

        assert len(records) == 9  # 3 cases × 3 configs
        agg = evaluator.aggregate(records)

        for config in ["haiku-all", "sonnet-all", "hybrid"]:
            assert config in agg
            assert agg[config]["p95_latency_ms"] >= 0
            assert 0.0 <= agg[config]["mean_quality"] <= 1.0
