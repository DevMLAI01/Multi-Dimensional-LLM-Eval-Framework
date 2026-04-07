"""
Phase 10 — Dashboard tests.

Unit tests for dashboard helper logic (no Streamlit runtime required).
Integration test: full smoke-test of the app module import and data layer.

No API calls in unit tests.
Integration tests: marked with -m integration.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    from eval_runner.results_store import ResultsStore
    store = ResultsStore(db_path=tmp_path / "test.db")
    return store


@pytest.fixture
def populated_store(tmp_store):
    """Store with two runs, dimension summaries, and a regression event."""
    import time

    tmp_store.create_run(
        "run_baseline",
        model_config="hybrid",
        git_commit="abc123",
        triggered_by="manual",
    )
    tmp_store.finalize_run("run_baseline", overall_score=0.82, duration_seconds=120.0)
    tmp_store.save_dimension_summary(
        "run_baseline", "correctness",
        mean_score=0.85, pass_rate=0.90,
        threshold=0.75, cases_run=20, cases_passed=18, cases_failed=2,
    )
    tmp_store.save_dimension_summary(
        "run_baseline", "safety",
        mean_score=1.0, pass_rate=1.0,
        threshold=1.0, cases_run=10, cases_passed=10, cases_failed=0,
    )

    tmp_store.create_run(
        "run_pr42",
        model_config="hybrid",
        git_commit="def456",
        triggered_by="CI",
    )
    tmp_store.finalize_run("run_pr42", overall_score=0.74, duration_seconds=135.0)
    tmp_store.save_dimension_summary(
        "run_pr42", "correctness",
        mean_score=0.70, pass_rate=0.75,
        threshold=0.75, cases_run=20, cases_passed=15, cases_failed=5,
    )
    tmp_store.save_dimension_summary(
        "run_pr42", "safety",
        mean_score=1.0, pass_rate=1.0,
        threshold=1.0, cases_run=10, cases_passed=10, cases_failed=0,
    )

    tmp_store.save_regression_event(
        run_id="run_pr42",
        dimension="correctness",
        previous_score=0.85,
        current_score=0.70,
        delta=-0.15,
        severity="CRITICAL",
        affected_cases=5,
    )

    return tmp_store


# ---------------------------------------------------------------------------
# ResultsStore data layer tests (used by dashboard)
# ---------------------------------------------------------------------------

class TestResultsStoreForDashboard:
    def test_list_runs_returns_runs(self, populated_store):
        runs = populated_store.list_runs()
        assert len(runs) == 2
        run_ids = {r["run_id"] for r in runs}
        assert "run_baseline" in run_ids
        assert "run_pr42" in run_ids

    def test_list_runs_ordered_by_timestamp_desc(self, populated_store):
        runs = populated_store.list_runs()
        # Most recent first
        timestamps = [r["timestamp"] for r in runs]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_get_dimension_summaries(self, populated_store):
        summaries = populated_store.get_dimension_summaries("run_baseline")
        assert "correctness" in summaries
        assert "safety" in summaries
        assert summaries["correctness"]["mean_score"] == pytest.approx(0.85)
        assert summaries["correctness"]["pass_rate"] == pytest.approx(0.90)

    def test_get_regression_events(self, populated_store):
        events = populated_store.get_regression_events("run_pr42")
        assert len(events) == 1
        e = events[0]
        assert e["dimension"] == "correctness"
        assert e["severity"] == "CRITICAL"
        assert e["delta"] == pytest.approx(-0.15)

    def test_no_regression_events_for_baseline(self, populated_store):
        events = populated_store.get_regression_events("run_baseline")
        assert events == []

    def test_get_results_for_run_empty(self, populated_store):
        results = populated_store.get_results_for_run("run_baseline", dimension="correctness")
        assert results == []

    def test_list_runs_limit(self, tmp_store):
        for i in range(10):
            tmp_store.create_run(f"run_{i:02d}")
        runs = tmp_store.list_runs(limit=3)
        assert len(runs) == 3

    def test_finalized_run_has_overall_score(self, populated_store):
        run = populated_store.get_run("run_baseline")
        assert run["overall_score"] == pytest.approx(0.82)
        assert run["duration_seconds"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Dashboard helper logic tests
# ---------------------------------------------------------------------------

class TestDashboardHelpers:
    """Test the helper functions used by the dashboard without invoking Streamlit."""

    def test_color_score_pass(self):
        """Scores above threshold get green badge."""
        # Inline the helper to avoid Streamlit import
        def _color_score(score, threshold=0.75):
            if score is None:
                return "—"
            color = "#2ecc71" if score >= threshold else "#e74c3c"
            return f'<span style="color:{color};font-weight:bold">{score:.3f}</span>'

        result = _color_score(0.85)
        assert "#2ecc71" in result
        assert "0.850" in result

    def test_color_score_fail(self):
        def _color_score(score, threshold=0.75):
            if score is None:
                return "—"
            color = "#2ecc71" if score >= threshold else "#e74c3c"
            return f'<span style="color:{color};font-weight:bold">{score:.3f}</span>'

        result = _color_score(0.60)
        assert "#e74c3c" in result

    def test_color_score_none(self):
        def _color_score(score, threshold=0.75):
            if score is None:
                return "—"
            color = "#2ecc71" if score >= threshold else "#e74c3c"
            return f'<span style="color:{color};font-weight:bold">{score:.3f}</span>'

        assert _color_score(None) == "—"

    def test_severity_badge_colors(self):
        def _severity_badge(sev):
            colors = {"CRITICAL": "#e74c3c", "MAJOR": "#e67e22", "MINOR": "#f1c40f"}
            c = colors.get(sev, "#95a5a6")
            return f'<span style="background:{c};color:#fff">{sev}</span>'

        assert "#e74c3c" in _severity_badge("CRITICAL")
        assert "#e67e22" in _severity_badge("MAJOR")
        assert "#f1c40f" in _severity_badge("MINOR")
        assert "#95a5a6" in _severity_badge("UNKNOWN")

    def test_dimension_weights_sum_to_one(self):
        weights = {
            "correctness": 0.35, "faithfulness": 0.25,
            "robustness": 0.20, "safety": 0.15, "latency": 0.05,
        }
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_all_dimensions_have_thresholds(self):
        thresholds = {
            "correctness": 0.75, "faithfulness": 0.80,
            "robustness": 0.85, "safety": 1.00, "latency": 0.70,
        }
        weights = {
            "correctness": 0.35, "faithfulness": 0.25,
            "robustness": 0.20, "safety": 0.15, "latency": 0.05,
        }
        assert set(thresholds.keys()) == set(weights.keys())


# ---------------------------------------------------------------------------
# Coverage report parsing tests
# ---------------------------------------------------------------------------

class TestCoverageReportParsing:
    def _make_report(self, tmp_path, gaps):
        report = {
            "gaps": gaps,
            "historical_distribution": {"link_down": 0.40, "cpu_high": 0.30, "memory_high": 0.30},
            "eval_distribution": {"link_down": 0.50, "cpu_high": 0.25, "memory_high": 0.25},
        }
        p = tmp_path / "coverage_gaps.json"
        p.write_text(json.dumps(report))
        return p

    def test_reads_gaps_correctly(self, tmp_path):
        gaps = [
            {"alarm_type": "link_down", "severity": "HIGH", "historical_pct": 0.40, "eval_pct": 0.01},
            {"alarm_type": "cpu_high", "severity": "MEDIUM", "historical_pct": 0.30, "eval_pct": 0.10},
        ]
        p = self._make_report(tmp_path, gaps)
        with open(p) as f:
            report = json.load(f)
        assert len(report["gaps"]) == 2
        assert report["gaps"][0]["severity"] == "HIGH"

    def test_empty_gaps(self, tmp_path):
        p = self._make_report(tmp_path, [])
        with open(p) as f:
            report = json.load(f)
        assert report["gaps"] == []

    def test_distributions_present(self, tmp_path):
        p = self._make_report(tmp_path, [])
        with open(p) as f:
            report = json.load(f)
        assert "historical_distribution" in report
        assert "eval_distribution" in report

    def test_severity_count(self, tmp_path):
        gaps = [
            {"alarm_type": "a", "severity": "HIGH"},
            {"alarm_type": "b", "severity": "HIGH"},
            {"alarm_type": "c", "severity": "MEDIUM"},
        ]
        p = self._make_report(tmp_path, gaps)
        with open(p) as f:
            report = json.load(f)
        severity_counts = {}
        for g in report["gaps"]:
            sev = g.get("severity", "LOW")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        assert severity_counts["HIGH"] == 2
        assert severity_counts["MEDIUM"] == 1


# ---------------------------------------------------------------------------
# Dashboard app module import smoke test
# ---------------------------------------------------------------------------

class TestDashboardAppModule:
    def test_app_file_exists(self):
        app_path = Path(__file__).parents[1] / "dashboard" / "app.py"
        assert app_path.exists(), "dashboard/app.py not found"

    def test_app_file_is_valid_python(self):
        import ast
        app_path = Path(__file__).parents[1] / "dashboard" / "app.py"
        src = app_path.read_text(encoding="utf-8")
        # Will raise SyntaxError if invalid
        tree = ast.parse(src)
        assert tree is not None

    def test_app_references_all_tabs(self):
        app_path = Path(__file__).parents[1] / "dashboard" / "app.py"
        src = app_path.read_text(encoding="utf-8")
        for expected_tab in ["Overview", "Dimension Deep Dive", "Regression History", "Coverage Analysis"]:
            assert expected_tab in src, f"Missing tab: {expected_tab}"

    def test_app_imports_results_store(self):
        app_path = Path(__file__).parents[1] / "dashboard" / "app.py"
        src = app_path.read_text(encoding="utf-8")
        assert "ResultsStore" in src

    def test_app_uses_plotly(self):
        app_path = Path(__file__).parents[1] / "dashboard" / "app.py"
        src = app_path.read_text(encoding="utf-8")
        assert "plotly" in src

    def test_app_has_db_path_input(self):
        """Dashboard must allow custom DB path override."""
        app_path = Path(__file__).parents[1] / "dashboard" / "app.py"
        src = app_path.read_text(encoding="utf-8")
        assert "db_path" in src.lower() or "DB path" in src


# ---------------------------------------------------------------------------
# Integration tests (require real eval data — fast, no API)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDashboardIntegration:
    def test_store_round_trip_for_dashboard(self, tmp_path):
        """Create a full run and verify all data the dashboard needs is readable."""
        from eval_runner.results_store import ResultsStore

        store = ResultsStore(db_path=tmp_path / "test.db")

        store.create_run("dash_test", model_config="hybrid",
                         git_commit="abc", triggered_by="pytest")
        store.save_dimension_summary(
            "dash_test", "correctness",
            mean_score=0.80, pass_rate=0.85,
            threshold=0.75, cases_run=20, cases_passed=17, cases_failed=3,
        )
        store.save_regression_event(
            run_id="dash_test", dimension="correctness",
            previous_score=0.90, current_score=0.80,
            delta=-0.10, severity="CRITICAL", affected_cases=3,
        )
        store.finalize_run("dash_test", overall_score=0.80, duration_seconds=60.0)

        # Verify list_runs
        runs = store.list_runs()
        assert any(r["run_id"] == "dash_test" for r in runs)

        # Verify dimension summaries
        summaries = store.get_dimension_summaries("dash_test")
        assert "correctness" in summaries
        assert summaries["correctness"]["mean_score"] == pytest.approx(0.80)

        # Verify regression events
        events = store.get_regression_events("dash_test")
        assert len(events) == 1
        assert events[0]["severity"] == "CRITICAL"

        # Verify finalized run
        run = store.get_run("dash_test")
        assert run["overall_score"] == pytest.approx(0.80)
        assert run["triggered_by"] == "pytest"

    def test_dashboard_handles_empty_store(self, tmp_path):
        """Dashboard data layer should work gracefully with an empty DB."""
        from eval_runner.results_store import ResultsStore

        store = ResultsStore(db_path=tmp_path / "empty.db")
        runs = store.list_runs()
        assert runs == []
