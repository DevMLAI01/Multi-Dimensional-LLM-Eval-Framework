"""
Phase 8.1 — SQLite Results Store.

Initialises and manages the SQLite database that persists all eval results
across runs. Tables:

    eval_runs          — one row per eval suite run
    eval_results       — one row per EvalResult (individual case result)
    dimension_summaries — aggregated per-dimension stats per run
    regression_events  — regressions detected by RegressionChecker

Usage:
    from eval_runner.results_store import ResultsStore
    store = ResultsStore()               # default: reports/eval_results.db
    store.save_result(run_id, result)    # write a single EvalResult
    store.get_run_summary(run_id)        # fetch dimension summaries
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from evaluators.base_evaluator import EvalResult

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parents[1] / "reports" / "eval_results.db"


class ResultsStore:
    """SQLite-backed persistence for eval run results."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        db_path.parent.mkdir(exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                    run_id          TEXT PRIMARY KEY,
                    timestamp       TEXT NOT NULL,
                    git_commit      TEXT,
                    prompt_versions TEXT,           -- JSON
                    model_config    TEXT,
                    total_cases     INTEGER DEFAULT 0,
                    passed_cases    INTEGER DEFAULT 0,
                    overall_score   REAL,
                    duration_seconds REAL,
                    total_cost_usd  REAL DEFAULT 0.0,
                    triggered_by    TEXT DEFAULT 'manual'
                );

                CREATE TABLE IF NOT EXISTS eval_results (
                    result_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id       TEXT NOT NULL,
                    case_id      TEXT NOT NULL,
                    dimension    TEXT NOT NULL,
                    score        REAL,
                    passed       INTEGER NOT NULL,   -- 0/1
                    reasoning    TEXT,
                    sub_scores   TEXT,               -- JSON
                    metadata     TEXT,               -- JSON
                    agent_run_id TEXT,
                    error        TEXT,
                    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS dimension_summaries (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id       TEXT NOT NULL,
                    dimension    TEXT NOT NULL,
                    mean_score   REAL,
                    pass_rate    REAL,
                    threshold    REAL,
                    cases_run    INTEGER,
                    cases_passed INTEGER,
                    cases_failed INTEGER,
                    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS regression_events (
                    regression_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id         TEXT NOT NULL,
                    dimension      TEXT NOT NULL,
                    previous_score REAL,
                    current_score  REAL,
                    delta          REAL,
                    severity       TEXT,             -- MINOR/MAJOR/CRITICAL
                    affected_cases TEXT,             -- JSON list
                    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_results_run_id
                    ON eval_results(run_id);
                CREATE INDEX IF NOT EXISTS idx_results_dimension
                    ON eval_results(dimension);
                CREATE INDEX IF NOT EXISTS idx_summaries_run_id
                    ON dimension_summaries(run_id);
            """)
        log.debug("Schema initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        model_config: str = "hybrid",
        git_commit: Optional[str] = None,
        prompt_versions: Optional[dict] = None,
        triggered_by: str = "manual",
    ):
        """Insert a new eval_runs row. Call before writing results."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO eval_runs
                    (run_id, timestamp, git_commit, prompt_versions,
                     model_config, triggered_by)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.utcnow().isoformat(),
                    git_commit,
                    json.dumps(prompt_versions or {}),
                    model_config,
                    triggered_by,
                ),
            )

    def save_result(self, run_id: str, result: EvalResult):
        """Write one EvalResult row immediately (not batched)."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO eval_results
                    (run_id, case_id, dimension, score, passed, reasoning,
                     sub_scores, metadata, agent_run_id, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    result.case_id,
                    result.dimension,
                    result.score,
                    int(result.passed),
                    result.reasoning,
                    json.dumps(result.sub_scores, default=str),
                    json.dumps(result.metadata, default=str),
                    result.agent_run_id,
                    result.error,
                ),
            )

    def finalize_run(
        self,
        run_id: str,
        overall_score: float,
        duration_seconds: float,
        total_cost_usd: float = 0.0,
    ):
        """Update eval_runs with final aggregated stats."""
        with self._conn() as conn:
            # Count results
            row = conn.execute(
                "SELECT COUNT(*) as n, SUM(passed) as p FROM eval_results WHERE run_id=?",
                (run_id,),
            ).fetchone()
            total = row["n"] or 0
            passed = int(row["p"] or 0)

            conn.execute(
                """
                UPDATE eval_runs
                SET total_cases=?, passed_cases=?, overall_score=?,
                    duration_seconds=?, total_cost_usd=?
                WHERE run_id=?
                """,
                (total, passed, overall_score, duration_seconds, total_cost_usd, run_id),
            )

    def save_dimension_summary(
        self,
        run_id: str,
        dimension: str,
        mean_score: float,
        pass_rate: float,
        threshold: float,
        cases_run: int,
        cases_passed: int,
        cases_failed: int,
    ):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO dimension_summaries
                    (run_id, dimension, mean_score, pass_rate, threshold,
                     cases_run, cases_passed, cases_failed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, dimension, mean_score, pass_rate, threshold,
                 cases_run, cases_passed, cases_failed),
            )

    def save_regression_event(
        self,
        run_id: str,
        dimension: str,
        previous_score: float,
        current_score: float,
        delta: float,
        severity: str,
        affected_cases: list[str],
    ):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO regression_events
                    (run_id, dimension, previous_score, current_score,
                     delta, severity, affected_cases)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, dimension, previous_score, current_score,
                 delta, severity, json.dumps(affected_cases)),
            )

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM eval_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_dimension_summaries(self, run_id: str) -> dict[str, dict]:
        """Return {dimension: summary_dict} for a run."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM dimension_summaries WHERE run_id=?", (run_id,)
            ).fetchall()
        return {row["dimension"]: dict(row) for row in rows}

    def get_results_for_run(
        self, run_id: str, dimension: Optional[str] = None
    ) -> list[dict]:
        with self._conn() as conn:
            if dimension:
                rows = conn.execute(
                    "SELECT * FROM eval_results WHERE run_id=? AND dimension=?",
                    (run_id, dimension),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM eval_results WHERE run_id=?", (run_id,)
                ).fetchall()
        return [dict(r) for r in rows]

    def get_regression_events(self, run_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM regression_events WHERE run_id=?", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_regression_events(self) -> list[dict]:
        """Return all regression events joined with their run timestamp."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT re.*, er.timestamp
                   FROM regression_events re
                   JOIN eval_runs er USING (run_id)
                   ORDER BY er.timestamp DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def list_runs(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_runs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
