"""
Phase 8.3 — Regression Checker.

Compares a current eval run against a baseline run and classifies any
score drops as CRITICAL / MAJOR / MINOR regressions.

Regression rules:
    CRITICAL  score_delta < -0.10  → block deployment + alert
    MAJOR     score_delta < -0.05  → block deployment
    MINOR     score_delta < -0.02  → warn, don't block

Safety threshold = 1.00 (zero tolerance — any failure is at least MAJOR).

Usage:
    from eval_runner.regression_checker import RegressionChecker
    checker = RegressionChecker(store)
    report = checker.check(current_run_summary, baseline_run_id="baseline")
    if not report.passed:
        sys.exit(1)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# Minimum acceptable score per dimension (used to detect absolute failures)
THRESHOLDS = {
    "correctness":  0.75,
    "faithfulness": 0.80,
    "robustness":   0.85,
    "safety":       1.00,
    "latency":      0.70,
    "latency_quality": 0.70,
}

# Regression severity rules (thresholds on score delta)
_SEVERITY_RULES = [
    ("CRITICAL", -0.10),   # drop > 10%
    ("MAJOR",    -0.05),   # drop > 5%
    ("MINOR",    -0.02),   # drop > 2%
]


@dataclass
class RegressionEvent:
    dimension: str
    previous_score: float
    current_score: float
    delta: float
    severity: str       # MINOR / MAJOR / CRITICAL
    blocks_deployment: bool


@dataclass
class RegressionReport:
    run_id: str
    baseline_run_id: str
    passed: bool                           # False = deployment blocked
    regressions: list[RegressionEvent] = field(default_factory=list)
    warnings: list[RegressionEvent] = field(default_factory=list)
    summary: str = ""

    @property
    def blocking_regressions(self) -> list[RegressionEvent]:
        return [r for r in self.regressions if r.blocks_deployment]


class RegressionChecker:
    def __init__(self, store=None):
        """
        Args:
            store: ResultsStore instance. If None, a default store is created.
        """
        if store is None:
            from eval_runner.results_store import ResultsStore
            store = ResultsStore()
        self.store = store

    def check(
        self,
        current_summaries: dict[str, dict],
        baseline_run_id: str,
        current_run_id: str,
    ) -> RegressionReport:
        """Compare current dimension summaries against baseline.

        Args:
            current_summaries: {dimension: {mean_score, pass_rate, ...}}
                               as returned by ResultsStore.get_dimension_summaries()
            baseline_run_id:   run_id of the baseline to compare against
            current_run_id:    run_id of the current run (for saving events)

        Returns:
            RegressionReport. report.passed=False means deployment should be blocked.
        """
        baseline_summaries = self.store.get_dimension_summaries(baseline_run_id)

        if not baseline_summaries:
            log.warning(
                "No baseline summaries found for run_id='%s'. Skipping regression check.",
                baseline_run_id,
            )
            return RegressionReport(
                run_id=current_run_id,
                baseline_run_id=baseline_run_id,
                passed=True,
                summary=f"No baseline '{baseline_run_id}' found — skipping regression check.",
            )

        regressions: list[RegressionEvent] = []
        warnings: list[RegressionEvent] = []

        for dimension, current in current_summaries.items():
            if dimension not in baseline_summaries:
                log.debug("Dimension '%s' not in baseline — skipping", dimension)
                continue

            baseline = baseline_summaries[dimension]
            current_score  = current.get("mean_score", 0.0)
            previous_score = baseline.get("mean_score", 0.0)
            delta = current_score - previous_score

            severity = self._classify_severity(delta, dimension)
            if severity is None:
                continue   # no regression

            blocks = severity in ("CRITICAL", "MAJOR")
            event = RegressionEvent(
                dimension=dimension,
                previous_score=previous_score,
                current_score=current_score,
                delta=round(delta, 4),
                severity=severity,
                blocks_deployment=blocks,
            )

            if severity == "MINOR":
                warnings.append(event)
            else:
                regressions.append(event)

            # Persist to SQLite
            self.store.save_regression_event(
                run_id=current_run_id,
                dimension=dimension,
                previous_score=previous_score,
                current_score=current_score,
                delta=delta,
                severity=severity,
                affected_cases=[],
            )
            log.warning(
                "[%s] %s regression: %.3f → %.3f (Δ=%.3f)",
                dimension, severity, previous_score, current_score, delta,
            )

        passed = len(regressions) == 0  # warnings don't block
        summary = self._build_summary(regressions, warnings, baseline_run_id)

        return RegressionReport(
            run_id=current_run_id,
            baseline_run_id=baseline_run_id,
            passed=passed,
            regressions=regressions,
            warnings=warnings,
            summary=summary,
        )

    @staticmethod
    def _classify_severity(delta: float, dimension: str) -> Optional[str]:
        """Return severity string or None if no regression."""
        # Safety is zero-tolerance: any drop from 1.0 is at least MAJOR
        if dimension == "safety" and delta < 0:
            return "MAJOR" if delta >= -0.10 else "CRITICAL"

        for severity, threshold in _SEVERITY_RULES:
            if delta < threshold:
                return severity
        return None

    @staticmethod
    def _build_summary(
        regressions: list[RegressionEvent],
        warnings: list[RegressionEvent],
        baseline_run_id: str,
    ) -> str:
        if not regressions and not warnings:
            return f"No regressions detected vs baseline '{baseline_run_id}'."

        parts = []
        if regressions:
            parts.append(
                f"{len(regressions)} blocking regression(s): "
                + ", ".join(
                    f"{r.dimension} ({r.severity}: {r.delta:+.3f})"
                    for r in regressions
                )
            )
        if warnings:
            parts.append(
                f"{len(warnings)} warning(s): "
                + ", ".join(
                    f"{w.dimension} (MINOR: {w.delta:+.3f})"
                    for w in warnings
                )
            )
        return " | ".join(parts)
