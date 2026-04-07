"""
Phase 10 — Streamlit Dashboard for the Multi-Dimensional LLM Eval Framework.

Four tabs:
  1. Overview         — run history table, overall score trend, latest run scorecard
  2. Dimension Dive   — per-dimension scores + individual case results for any run
  3. Regression History — regression events timeline, severity breakdown
  4. Coverage Analysis  — alarm-type coverage gaps from coverage_analyzer

Run:
    uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make project root importable when launched from any CWD
_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

from eval_runner.scorer import DIMENSION_THRESHOLDS, DIMENSION_WEIGHTS

st.set_page_config(
    page_title="LLM Eval Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SEVERITY_COLORS = {"CRITICAL": "#e74c3c", "MAJOR": "#e67e22", "MINOR": "#f1c40f"}


@st.cache_resource
def _get_store():
    from eval_runner.results_store import ResultsStore
    return ResultsStore()


st.sidebar.title("LLM Eval Suite")
st.sidebar.caption("Multi-Dimensional Eval Framework — Telecom NOC Agent")

_db_path_input = st.sidebar.text_input(
    "DB path",
    value=str(_ROOT / "reports" / "eval_results.db"),
    help="SQLite database produced by eval_runner/runner.py",
)

if _db_path_input.strip():
    from eval_runner.results_store import ResultsStore
    _store = ResultsStore(db_path=Path(_db_path_input.strip()))
else:
    _store = _get_store()

runs = _store.list_runs(limit=50)

tab_overview, tab_dimension, tab_regression, tab_coverage = st.tabs([
    "Overview",
    "Dimension Deep Dive",
    "Regression History",
    "Coverage Analysis",
])

# =============================================================================
# Tab 1 — Overview
# =============================================================================

with tab_overview:
    st.header("Eval Suite Overview")

    if not runs:
        st.info("No eval runs found. Run `uv run python eval_runner/runner.py` to generate data.")
        st.stop()

    df_runs = pd.DataFrame(runs)
    df_runs["timestamp"] = pd.to_datetime(df_runs["timestamp"])
    df_runs = df_runs.sort_values("timestamp")

    latest = df_runs.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Latest Run", latest["run_id"])
    with col2:
        score = latest.get("overall_score")
        st.metric("Overall Score", f"{score:.3f}" if score is not None else "—")
    with col3:
        passed = latest.get("passed_cases", 0)
        total = latest.get("total_cases", 0)
        st.metric("Cases Passed", f"{passed}/{total}")
    with col4:
        dur = latest.get("duration_seconds")
        st.metric("Duration", f"{dur:.0f}s" if dur is not None else "—")

    st.divider()

    st.subheader("Overall Score Trend")
    if df_runs["overall_score"].notna().any():
        fig = px.line(
            df_runs[df_runs["overall_score"].notna()],
            x="timestamp",
            y="overall_score",
            markers=True,
            labels={"overall_score": "Overall Score", "timestamp": "Run time"},
            color_discrete_sequence=["#3498db"],
        )
        fig.add_hline(y=0.75, line_dash="dash", line_color="#e74c3c",
                      annotation_text="baseline threshold")
        fig.update_layout(yaxis_range=[0, 1], height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No finalized runs with overall scores yet.")

    st.subheader("Run History")
    display_cols = ["run_id", "timestamp", "overall_score", "passed_cases",
                    "total_cases", "model_config", "triggered_by", "git_commit"]
    available = [c for c in display_cols if c in df_runs.columns]
    st.dataframe(
        df_runs[available].sort_values("timestamp", ascending=False)
        .rename(columns={
            "run_id": "Run ID", "timestamp": "Timestamp",
            "overall_score": "Score", "passed_cases": "Passed",
            "total_cases": "Total", "model_config": "Model Config",
            "triggered_by": "Triggered By", "git_commit": "Commit",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(f"Latest Run Scorecard — {latest['run_id']}")
    dim_summaries = _store.get_dimension_summaries(latest["run_id"])

    if dim_summaries:
        scorecard_rows = []
        for dim, s in dim_summaries.items():
            threshold = DIMENSION_THRESHOLDS.get(dim, 0.75)
            weight = DIMENSION_WEIGHTS.get(dim, 0.0)
            scorecard_rows.append({
                "Dimension": dim,
                "Mean Score": round(s.get("mean_score", 0), 4),
                "Pass Rate": f"{s.get('pass_rate', 0)*100:.0f}%",
                "Threshold": threshold,
                "Weight": f"{weight:.0%}",
                "Cases": f"{s.get('cases_passed', 0)}/{s.get('cases_run', 0)}",
                "Status": "✅ PASS" if s.get("mean_score", 0) >= threshold else "❌ FAIL",
            })

        st.dataframe(pd.DataFrame(scorecard_rows), use_container_width=True, hide_index=True)

        if len(scorecard_rows) >= 3:
            dims = [r["Dimension"] for r in scorecard_rows]
            scores = [r["Mean Score"] for r in scorecard_rows]
            thresholds = [r["Threshold"] for r in scorecard_rows]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=dims + [dims[0]],
                fill="toself",
                name="Score",
                line_color="#3498db",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=thresholds + [thresholds[0]],
                theta=dims + [dims[0]],
                fill="toself",
                name="Threshold",
                line_color="#e74c3c",
                line_dash="dash",
                fillcolor="rgba(231,76,60,0.1)",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=400,
                title="Dimension Scores vs Thresholds",
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("No dimension summaries yet for this run.")


# =============================================================================
# Tab 2 — Dimension Deep Dive
# =============================================================================

with tab_dimension:
    st.header("Dimension Deep Dive")

    if not runs:
        st.info("No runs yet.")
        st.stop()

    run_ids = [r["run_id"] for r in runs]
    selected_run = st.selectbox("Select run", run_ids, key="dive_run")
    dim_options = list(DIMENSION_THRESHOLDS.keys())
    selected_dim = st.selectbox("Select dimension", dim_options, key="dive_dim")

    dim_summaries = _store.get_dimension_summaries(selected_run)

    if selected_dim in dim_summaries:
        s = dim_summaries[selected_dim]
        threshold = DIMENSION_THRESHOLDS.get(selected_dim, 0.75)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Score", f"{s.get('mean_score', 0):.4f}")
        col2.metric("Pass Rate", f"{s.get('pass_rate', 0)*100:.1f}%")
        col3.metric("Passed / Total", f"{s.get('cases_passed', 0)}/{s.get('cases_run', 0)}")
        col4.metric("Threshold", f"{threshold:.2f}")
    else:
        st.warning(f"No summary found for '{selected_dim}' in run '{selected_run}'.")

    st.divider()

    results = _store.get_results_for_run(selected_run, dimension=selected_dim)

    if results:
        df_results = pd.DataFrame(results)

        if "score" in df_results.columns and df_results["score"].notna().any():
            fig = px.histogram(
                df_results[df_results["score"].notna()],
                x="score",
                nbins=20,
                labels={"score": "Score"},
                color_discrete_sequence=["#3498db"],
                title=f"Score distribution — {selected_dim}",
            )
            fig.add_vline(
                x=DIMENSION_THRESHOLDS.get(selected_dim, 0.75),
                line_dash="dash", line_color="#e74c3c",
                annotation_text="threshold",
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Individual Cases")
        show_cols = ["case_id", "score", "passed", "reasoning", "error"]
        available = [c for c in show_cols if c in df_results.columns]
        df_display = df_results[available].copy()
        if "passed" in df_display.columns:
            df_display["passed"] = df_display["passed"].map({1: "✅", 0: "❌"})

        filter_col, _ = st.columns([1, 3])
        with filter_col:
            show_only = st.selectbox("Filter", ["All", "Passed", "Failed"], key="case_filter")

        if show_only == "Passed" and "passed" in df_display.columns:
            df_display = df_display[df_display["passed"] == "✅"]
        elif show_only == "Failed" and "passed" in df_display.columns:
            df_display = df_display[df_display["passed"] == "❌"]

        st.dataframe(df_display, use_container_width=True, hide_index=True,
                     column_config={
                         "reasoning": st.column_config.TextColumn(width="large"),
                     })

        if "sub_scores" in df_results.columns:
            parsed = df_results["sub_scores"].apply(
                lambda x: json.loads(x) if x else None
            ).dropna()
            if not parsed.empty:
                df_sub = pd.DataFrame(parsed.tolist(), index=df_results.loc[parsed.index, "case_id"])
                numeric_cols = df_sub.select_dtypes("number").columns.tolist()
                if numeric_cols:
                    st.subheader("Sub-Score Breakdown")
                    fig2 = px.box(
                        df_sub[numeric_cols].reset_index().melt(
                            id_vars="case_id", value_vars=numeric_cols
                        ),
                        x="variable", y="value",
                        labels={"variable": "Sub-score", "value": "Score"},
                        title="Sub-score distributions",
                        color="variable",
                    )
                    fig2.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(f"No case results found for dimension '{selected_dim}' in run '{selected_run}'.")


# =============================================================================
# Tab 3 — Regression History
# =============================================================================

with tab_regression:
    st.header("Regression History")

    if not runs:
        st.info("No runs yet.")
        st.stop()

    all_events = _store.get_all_regression_events()

    if not all_events:
        st.info("No regression events recorded. All runs passed or no comparison runs exist yet.")
    else:
        df_ev = pd.DataFrame(all_events)
        df_ev["timestamp"] = pd.to_datetime(df_ev["timestamp"])

        col1, col2, col3 = st.columns(3)
        counts = df_ev["severity"].value_counts()
        col1.metric("Critical", counts.get("CRITICAL", 0))
        col2.metric("Major", counts.get("MAJOR", 0))
        col3.metric("Minor", counts.get("MINOR", 0))

        st.divider()

        fig = px.scatter(
            df_ev,
            x="timestamp",
            y="delta",
            color="severity",
            symbol="dimension",
            hover_data=["run_id", "dimension", "previous_score", "current_score"],
            title="Regression Events Timeline",
            color_discrete_map=SEVERITY_COLORS,
        )
        fig.add_hline(y=0, line_dash="solid", line_color="#95a5a6")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Regression Events")
        show_cols = ["run_id", "timestamp", "dimension", "severity",
                     "previous_score", "current_score", "delta"]
        available = [c for c in show_cols if c in df_ev.columns]
        st.dataframe(
            df_ev[available].sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Regressions by Dimension")
        dim_counts = df_ev.groupby(["dimension", "severity"]).size().reset_index(name="count")
        fig2 = px.bar(
            dim_counts,
            x="dimension",
            y="count",
            color="severity",
            barmode="stack",
            color_discrete_map=SEVERITY_COLORS,
            title="Regression count by dimension and severity",
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# Tab 4 — Coverage Analysis
# =============================================================================

with tab_coverage:
    st.header("Coverage Analysis")

    coverage_path = _ROOT / "reports" / "coverage_gaps.json"

    if not coverage_path.exists():
        st.info(
            "No coverage report found. Generate it by running:\n\n"
            "```\nuv run python eval_runner/coverage_analyzer.py\n```"
        )
    else:
        with open(coverage_path, encoding="utf-8") as f:
            report = json.load(f)

        gaps = report.get("gaps", [])
        hist = report.get("historical_distribution", {})
        eval_dist = report.get("eval_distribution", {})

        severity_counts: dict[str, int] = {}
        for g in gaps:
            sev = g.get("severity", "LOW")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Gaps", len(gaps))
        col2.metric("HIGH severity", severity_counts.get("HIGH", 0))
        col3.metric("MEDIUM severity", severity_counts.get("MEDIUM", 0))
        col4.metric("LOW severity", severity_counts.get("LOW", 0))

        st.divider()

        if hist and eval_dist:
            alarm_types = sorted(set(hist) | set(eval_dist))
            df_dist = pd.DataFrame({
                "alarm_type": alarm_types,
                "historical_%": [hist.get(t, 0) * 100 for t in alarm_types],
                "eval_%": [eval_dist.get(t, 0) * 100 for t in alarm_types],
            })
            fig = px.bar(
                df_dist.melt(id_vars="alarm_type",
                             value_vars=["historical_%", "eval_%"],
                             var_name="source", value_name="percentage"),
                x="alarm_type",
                y="percentage",
                color="source",
                barmode="group",
                title="Alarm-type distribution: Historical vs Eval dataset",
                labels={"alarm_type": "Alarm Type", "percentage": "%"},
                color_discrete_map={"historical_%": "#3498db", "eval_%": "#2ecc71"},
            )
            fig.update_layout(height=380, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        if gaps:
            st.subheader("Coverage Gaps")
            df_gaps = pd.DataFrame(gaps)
            severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            df_gaps = df_gaps.sort_values(
                "severity", key=lambda s: s.map(severity_order).fillna(3)
            )
            st.dataframe(df_gaps, use_container_width=True, hide_index=True)
        else:
            st.success("No coverage gaps detected — eval dataset mirrors production distribution.")

    st.divider()
    if st.button("Re-run coverage analysis"):
        with st.spinner("Analyzing coverage..."):
            from eval_runner.coverage_analyzer import CoverageAnalyzer
            analyzer = CoverageAnalyzer()
            gaps_new = analyzer.analyze()
            out = analyzer.save_report(gaps_new)
        st.success(f"Report saved to `{out}`. Refresh the page to see updates.")
