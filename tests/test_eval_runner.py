"""
Phase 8 — Eval Runner tests.

Unit tests: all SQLite and logic tests run fully in-memory or with temp DBs.
Integration tests: real eval suite run, marked with -m integration.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval_runner.coverage_analyzer import CoverageAnalyzer, CoverageGap, _classify_gap
from eval_runner.regression_checker import RegressionChecker, RegressionEvent, THRESHOLDS
from eval_runner.results_store import ResultsStore
from eval_runner.scorer import (
    DimensionScore,
    OverallScore,
    DIMENSION_WEIGHTS,
    SAFETY_CAP,
    compute_overall_score,
    scores_from_results,
)
from evaluators.base_evaluator import EvalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    """A ResultsStore backed by a temp SQLite DB."""
    return ResultsStore(db_path=tmp_path / "test.db")


def _make_result(
    case_id: str = "C001",
    dimension: str = "correctness",
    score: float = 0.85,
    passed: bool = True,
    error: str = None,
    metadata: dict = None,
) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        dimension=dimension,
        evaluator_version="1.0",
        score=score,
        passed=passed,
        error=error,
        metadata=metadata or {},
    )


def _make_dim_score(
    dimension: str,
    mean_score: float,
    pass_rate: float = 1.0,
    cases_run: int = 10,
    threshold: float = 0.75,
) -> DimensionScore:
    cases_passed = int(cases_run * pass_rate)
    return DimensionScore(
        dimension=dimension,
        mean_score=mean_score,
        pass_rate=pass_rate,
        cases_run=cases_run,
        cases_passed=cases_passed,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# ResultsStore tests
# ---------------------------------------------------------------------------

class TestResultsStore:
    def test_create_and_retrieve_run(self, tmp_store):
        tmp_store.create_run("run001", model_config="hybrid")
        run = tmp_store.get_run("run001")
        assert run is not None
        assert run["run_id"] == "run001"
        assert run["model_config"] == "hybrid"

    def test_save_and_retrieve_result(self, tmp_store):
        tmp_store.create_run("run001")
        result = _make_result(case_id="C001", dimension="correctness", score=0.82)
        tmp_store.save_result("run001", result)

        rows = tmp_store.get_results_for_run("run001")
        assert len(rows) == 1
        assert rows[0]["case_id"] == "C001"
        assert rows[0]["score"] == pytest.approx(0.82)

    def test_filter_results_by_dimension(self, tmp_store):
        tmp_store.create_run("run001")
        tmp_store.save_result("run001", _make_result("C001", "correctness", 0.8))
        tmp_store.save_result("run001", _make_result("F001", "faithfulness", 0.9))

        corr = tmp_store.get_results_for_run("run001", dimension="correctness")
        assert len(corr) == 1
        assert corr[0]["dimension"] == "correctness"

    def test_save_dimension_summary(self, tmp_store):
        tmp_store.create_run("run001")
        tmp_store.save_dimension_summary(
            run_id="run001", dimension="correctness",
            mean_score=0.82, pass_rate=0.90, threshold=0.75,
            cases_run=10, cases_passed=9, cases_failed=1,
        )
        summaries = tmp_store.get_dimension_summaries("run001")
        assert "correctness" in summaries
        assert summaries["correctness"]["mean_score"] == pytest.approx(0.82)

    def test_finalize_run_counts_results(self, tmp_store):
        tmp_store.create_run("run001")
        for i in range(5):
            tmp_store.save_result("run001", _make_result(f"C{i}", passed=True, score=0.8))
        for i in range(2):
            tmp_store.save_result("run001", _make_result(f"F{i}", passed=False, score=0.4,
                                                          dimension="faithfulness"))
        tmp_store.finalize_run("run001", overall_score=0.75, duration_seconds=120.0)
        run = tmp_store.get_run("run001")
        assert run["total_cases"] == 7
        assert run["passed_cases"] == 5

    def test_save_regression_event(self, tmp_store):
        tmp_store.create_run("run002")
        tmp_store.save_regression_event(
            run_id="run002", dimension="correctness",
            previous_score=0.85, current_score=0.72,
            delta=-0.13, severity="CRITICAL",
            affected_cases=["C001", "C002"],
        )
        events = tmp_store.get_regression_events("run002")
        assert len(events) == 1
        assert events[0]["severity"] == "CRITICAL"
        assert events[0]["delta"] == pytest.approx(-0.13)

    def test_list_runs(self, tmp_store):
        for rid in ["run_a", "run_b", "run_c"]:
            tmp_store.create_run(rid)
        runs = tmp_store.list_runs()
        assert len(runs) == 3

    def test_duplicate_run_id_ignored(self, tmp_store):
        tmp_store.create_run("run001")
        tmp_store.create_run("run001")   # should not raise
        runs = tmp_store.list_runs()
        assert len(runs) == 1

    def test_result_with_error_stored(self, tmp_store):
        tmp_store.create_run("run001")
        result = _make_result(score=None, passed=False, error="agent_run_error: timeout")
        result.score = None
        tmp_store.save_result("run001", result)
        rows = tmp_store.get_results_for_run("run001")
        assert rows[0]["error"] == "agent_run_error: timeout"
        assert rows[0]["score"] is None


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------

class TestScorer:
    def test_weights_sum_to_one(self):
        assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9

    def test_all_dimensions_perfect_score(self):
        dim_scores = {
            d: _make_dim_score(d, 1.0, threshold=t)
            for d, t in DIMENSION_WEIGHTS.items()
        }
        # Add threshold info
        dim_scores["correctness"] = _make_dim_score("correctness", 1.0, threshold=0.75)
        dim_scores["safety"]      = _make_dim_score("safety", 1.0, pass_rate=1.0, threshold=1.0)
        result = compute_overall_score(dim_scores)
        assert result.weighted_score == pytest.approx(1.0)
        assert result.safety_cap_applied is False

    def test_safety_cap_applied_when_safety_fails(self):
        dim_scores = {
            "correctness":  _make_dim_score("correctness", 0.90),
            "faithfulness": _make_dim_score("faithfulness", 0.88),
            "robustness":   _make_dim_score("robustness", 0.87),
            "safety":       _make_dim_score("safety", 0.80, pass_rate=0.80, threshold=1.0),
            "latency":      _make_dim_score("latency", 0.85),
        }
        result = compute_overall_score(dim_scores)
        assert result.safety_cap_applied is True
        assert result.weighted_score <= SAFETY_CAP

    def test_no_safety_cap_when_safety_passes(self):
        dim_scores = {
            "correctness":  _make_dim_score("correctness", 0.90),
            "safety":       _make_dim_score("safety", 1.0, pass_rate=1.0, threshold=1.0),
        }
        result = compute_overall_score(dim_scores)
        assert result.safety_cap_applied is False

    def test_missing_dimensions_renormalised(self):
        """If only 2 of 5 dimensions present, weights renormalised to sum to 1."""
        dim_scores = {
            "correctness": _make_dim_score("correctness", 0.80),
            "faithfulness": _make_dim_score("faithfulness", 0.70),
        }
        result = compute_overall_score(dim_scores)
        assert 0.0 < result.weighted_score < 1.0
        assert "latency" in result.dimensions_missing

    def test_scores_from_results_groups_by_dimension(self):
        results = [
            _make_result("C1", "correctness", 0.80, True),
            _make_result("C2", "correctness", 0.70, False),
            _make_result("F1", "faithfulness", 0.90, True),
        ]
        scores = scores_from_results(results)
        assert "correctness" in scores
        assert "faithfulness" in scores
        assert scores["correctness"].mean_score == pytest.approx(0.75)
        assert scores["correctness"].cases_run == 2
        assert scores["correctness"].cases_passed == 1
        assert scores["faithfulness"].mean_score == pytest.approx(0.90)

    def test_scores_from_results_ignores_none_scores(self):
        results = [
            _make_result("C1", "correctness", 0.80, True),
            _make_result("C2", "correctness", None, False, error="timeout"),
        ]
        results[1].score = None
        scores = scores_from_results(results)
        # mean computed only over valid scores
        assert scores["correctness"].mean_score == pytest.approx(0.80)
        assert scores["correctness"].cases_run == 2


# ---------------------------------------------------------------------------
# RegressionChecker tests
# ---------------------------------------------------------------------------

class TestRegressionChecker:
    def _setup_baseline(self, store, run_id="baseline"):
        store.create_run(run_id)
        store.save_dimension_summary(run_id, "correctness", 0.85, 0.95, 0.75, 20, 19, 1)
        store.save_dimension_summary(run_id, "faithfulness", 0.82, 0.90, 0.80, 20, 18, 2)
        store.save_dimension_summary(run_id, "safety", 1.0, 1.0, 1.0, 10, 10, 0)

    def test_no_regression_passes(self, tmp_store):
        self._setup_baseline(tmp_store)
        checker = RegressionChecker(tmp_store)
        current = {
            "correctness":  {"mean_score": 0.86, "pass_rate": 0.95},
            "faithfulness": {"mean_score": 0.83, "pass_rate": 0.90},
            "safety":       {"mean_score": 1.0,  "pass_rate": 1.0},
        }
        report = checker.check(current, "baseline", "current_run")
        assert report.passed is True
        assert len(report.regressions) == 0

    def test_critical_regression_detected(self, tmp_store):
        self._setup_baseline(tmp_store)
        checker = RegressionChecker(tmp_store)
        current = {
            "correctness":  {"mean_score": 0.72, "pass_rate": 0.80},  # -0.13 drop
            "faithfulness": {"mean_score": 0.82, "pass_rate": 0.90},
            "safety":       {"mean_score": 1.0,  "pass_rate": 1.0},
        }
        report = checker.check(current, "baseline", "current_run")
        assert report.passed is False
        assert any(r.severity == "CRITICAL" for r in report.regressions)

    def test_minor_regression_is_warning_not_blocking(self, tmp_store):
        self._setup_baseline(tmp_store)
        checker = RegressionChecker(tmp_store)
        current = {
            "correctness":  {"mean_score": 0.83, "pass_rate": 0.93},  # -0.02 drop
            "faithfulness": {"mean_score": 0.82, "pass_rate": 0.90},
            "safety":       {"mean_score": 1.0,  "pass_rate": 1.0},
        }
        report = checker.check(current, "baseline", "current_run")
        assert report.passed is True  # minor = warning only, doesn't block
        assert len(report.warnings) > 0

    def test_safety_regression_is_major(self, tmp_store):
        self._setup_baseline(tmp_store)
        checker = RegressionChecker(tmp_store)
        current = {
            "correctness":  {"mean_score": 0.85, "pass_rate": 0.95},
            "faithfulness": {"mean_score": 0.82, "pass_rate": 0.90},
            "safety":       {"mean_score": 0.90, "pass_rate": 0.90},  # safety dropped
        }
        report = checker.check(current, "baseline", "current_run")
        assert report.passed is False
        safety_reg = [r for r in report.regressions if r.dimension == "safety"]
        assert len(safety_reg) == 1
        assert safety_reg[0].severity in ("MAJOR", "CRITICAL")

    def test_missing_baseline_skips_check(self, tmp_store):
        checker = RegressionChecker(tmp_store)
        current = {"correctness": {"mean_score": 0.80, "pass_rate": 0.90}}
        report = checker.check(current, "nonexistent_baseline", "run001")
        assert report.passed is True
        assert "No baseline" in report.summary

    def test_thresholds_dict_has_all_dimensions(self):
        for dim in ["correctness", "faithfulness", "robustness", "safety", "latency"]:
            assert dim in THRESHOLDS

    def test_regression_saved_to_sqlite(self, tmp_store):
        self._setup_baseline(tmp_store)
        tmp_store.create_run("current_run")
        checker = RegressionChecker(tmp_store)
        current = {"correctness": {"mean_score": 0.70, "pass_rate": 0.80}}
        checker.check(current, "baseline", "current_run")
        events = tmp_store.get_regression_events("current_run")
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# CoverageAnalyzer tests
# ---------------------------------------------------------------------------

class TestCoverageAnalyzer:
    def test_classify_gap_high(self):
        assert _classify_gap(historical_rate=0.10, eval_rate=0.01) == "HIGH"

    def test_classify_gap_medium(self):
        assert _classify_gap(historical_rate=0.05, eval_rate=0.01) == "MEDIUM"

    def test_classify_gap_none_when_well_covered(self):
        assert _classify_gap(historical_rate=0.05, eval_rate=0.05) is None

    def test_analyze_returns_gaps(self, tmp_path):
        # Write synthetic history with a skewed alarm distribution
        history = [{"alarm_type": "LINK_DOWN"} for _ in range(60)]
        history += [{"alarm_type": "HIGH_CPU"} for _ in range(30)]
        history += [{"alarm_type": "BGP_SESSION_DOWN"} for _ in range(10)]
        history_file = tmp_path / "alarm_history.json"
        with open(history_file, "w") as f:
            json.dump(history, f)

        # Write tiny golden dataset — only LINK_DOWN covered
        golden_file = tmp_path / "cases.json"
        golden = [
            {"case_id": f"C{i}", "input": {"alarm_type": "LINK_DOWN"}}
            for i in range(10)
        ]
        with open(golden_file, "w") as f:
            json.dump(golden, f)

        analyzer = CoverageAnalyzer(
            history_file=history_file,
            golden_files=[golden_file],
        )
        gaps = analyzer.analyze()

        # HIGH_CPU and BGP_SESSION_DOWN should be gap — not in eval
        gap_types = {g.alarm_type for g in gaps}
        assert "HIGH_CPU" in gap_types or "BGP_SESSION_DOWN" in gap_types

    def test_analyze_empty_history_returns_empty(self, tmp_path):
        history_file = tmp_path / "empty.json"
        with open(history_file, "w") as f:
            json.dump([], f)

        analyzer = CoverageAnalyzer(history_file=history_file, golden_files=[])
        gaps = analyzer.analyze()
        assert gaps == []

    def test_save_report_creates_json(self, tmp_path):
        gaps = [
            CoverageGap(
                alarm_type="HIGH_CPU",
                historical_count=50,
                historical_rate=0.10,
                eval_count=1,
                eval_coverage_rate=0.01,
                gap_severity="HIGH",
            )
        ]
        analyzer = CoverageAnalyzer.__new__(CoverageAnalyzer)
        # Monkeypatch the reports dir
        import eval_runner.coverage_analyzer as ca_mod
        original = ca_mod._REPORTS_DIR
        ca_mod._REPORTS_DIR = tmp_path
        try:
            out = analyzer.save_report(gaps)
        finally:
            ca_mod._REPORTS_DIR = original

        assert out.exists()
        report = json.loads(out.read_text())
        assert report["high_severity"] == 1


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEvalRunnerIntegration:
    """Run with: uv run pytest -m integration"""

    def test_correctness_single_case(self, tmp_path):
        """Run 1 correctness case end-to-end through the runner."""
        import json
        from pathlib import Path
        from eval_runner.runner import _run_dimension
        from eval_runner.results_store import ResultsStore

        cases_path = Path("data/golden_dataset/correctness_cases.json")
        if not cases_path.exists():
            pytest.skip("correctness_cases.json not found")

        with open(cases_path, encoding="utf-8") as f:
            cases = json.load(f)[:1]

        store = ResultsStore(db_path=tmp_path / "test.db")
        store.create_run("test_run")

        results = _run_dimension("correctness", cases, store, "test_run", "hybrid")
        assert len(results) == 1
        assert results[0].score is not None or results[0].error is not None

    def test_coverage_analyzer_on_real_data(self):
        """Run coverage analyzer on real alarm_history.json."""
        from pathlib import Path
        analyzer = CoverageAnalyzer()
        gaps = analyzer.analyze()
        # Should return a list (may be empty if perfectly covered)
        assert isinstance(gaps, list)
        for gap in gaps:
            assert gap.gap_severity in ("HIGH", "MEDIUM", "LOW")
            assert 0.0 <= gap.historical_rate <= 1.0

    def test_regression_checker_no_baseline(self, tmp_path):
        """When no baseline exists, regression check passes with warning."""
        store = ResultsStore(db_path=tmp_path / "test.db")
        checker = RegressionChecker(store)
        report = checker.check(
            {"correctness": {"mean_score": 0.80, "pass_rate": 0.90}},
            "nonexistent",
            "run001",
        )
        assert report.passed is True
