"""
NOC Diagnostic Agent — LangGraph StateGraph definition.

Graph flow:
    alarm_classifier → context_fetcher → root_cause_reasoner → action_recommender → END
                     ↘ (on error) ─────────────────────────────────────────────────────↗

Usage:
    from agent.noc_agent import run_agent
    from agent.models import AlarmEvent, AgentDiagnosis

    alarm = AlarmEvent(...)
    diagnosis = run_agent(alarm)
"""

import logging
from typing import Any, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agent.models import AgentDiagnosis, AlarmEvent, RecommendedAction, RootCauseHypothesis
from agent.nodes.action_recommender import action_recommender
from agent.nodes.alarm_classifier import alarm_classifier
from agent.nodes.context_fetcher import context_fetcher
from agent.nodes.root_cause_reasoner import root_cause_reasoner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class NOCAgentState(TypedDict, total=False):
    alarm_event: AlarmEvent
    classification: Optional[str]
    severity_assessment: Optional[str]
    alarm_history: list
    device_info: dict
    runbook: dict
    root_cause_hypotheses: list
    most_likely_cause: Optional[str]
    recommended_actions: list
    supporting_evidence: list
    confidence_score: Optional[float]
    reasoning_trace: str
    error: Optional[str]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_classifier(state: NOCAgentState) -> str:
    """If classifier errored hard (non-recoverable), skip to END with partial data."""
    if state.get("error") and "classifier_error" in (state.get("error") or ""):
        log.warning("Routing to END after classifier error: %s", state.get("error"))
        return END
    return "context_fetcher"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    graph = StateGraph(NOCAgentState)

    graph.add_node("alarm_classifier", alarm_classifier)
    graph.add_node("context_fetcher", context_fetcher)
    graph.add_node("root_cause_reasoner", root_cause_reasoner)
    graph.add_node("action_recommender", action_recommender)

    graph.set_entry_point("alarm_classifier")

    graph.add_conditional_edges(
        "alarm_classifier",
        _route_after_classifier,
        {END: END, "context_fetcher": "context_fetcher"},
    )

    graph.add_edge("context_fetcher", "root_cause_reasoner")
    graph.add_edge("root_cause_reasoner", "action_recommender")
    graph.add_edge("action_recommender", END)

    return graph.compile()


# Singleton compiled graph — built once per process
_graph = None


def _get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_agent(alarm: AlarmEvent) -> AgentDiagnosis:
    """Run the full NOC diagnostic pipeline for a given alarm.

    Args:
        alarm: AlarmEvent with alarm details.

    Returns:
        AgentDiagnosis with classification, root causes, and recommended actions.
        Never raises — returns a partial diagnosis with error field on failure.
    """
    log.info("Running NOC agent for alarm %s (%s on %s)", alarm.alarm_id, alarm.alarm_type, alarm.device_id)

    initial_state: NOCAgentState = {
        "alarm_event": alarm,
        "classification": None,
        "severity_assessment": None,
        "alarm_history": [],
        "device_info": {},
        "runbook": {},
        "root_cause_hypotheses": [],
        "most_likely_cause": None,
        "recommended_actions": [],
        "supporting_evidence": [],
        "confidence_score": None,
        "reasoning_trace": "",
        "error": None,
    }

    try:
        graph = _get_graph()
        final_state: NOCAgentState = graph.invoke(initial_state)
    except Exception as exc:
        log.error("Graph execution failed for alarm %s: %s", alarm.alarm_id, exc)
        final_state = {**initial_state, "error": f"graph_error: {exc}"}

    return _state_to_diagnosis(alarm, final_state)


def _state_to_diagnosis(alarm: AlarmEvent, state: NOCAgentState) -> AgentDiagnosis:
    """Convert final graph state to AgentDiagnosis output model."""
    hypotheses = state.get("root_cause_hypotheses") or []
    # Handle case where hypotheses are still dicts (not yet converted to models)
    typed_hypotheses: list[RootCauseHypothesis] = []
    for h in hypotheses:
        if isinstance(h, RootCauseHypothesis):
            typed_hypotheses.append(h)
        elif isinstance(h, dict):
            typed_hypotheses.append(RootCauseHypothesis(**h))

    actions = state.get("recommended_actions") or []
    typed_actions: list[RecommendedAction] = []
    for a in actions:
        if isinstance(a, RecommendedAction):
            typed_actions.append(a)
        elif isinstance(a, dict):
            typed_actions.append(RecommendedAction(**a))

    # Fallbacks for required fields
    classification = state.get("classification") or alarm.alarm_type
    severity_assessment = state.get("severity_assessment") or alarm.severity
    most_likely_cause = state.get("most_likely_cause") or "Could not determine root cause"
    confidence_score = state.get("confidence_score")
    if confidence_score is None:
        confidence_score = 0.0

    if not typed_actions:
        typed_actions = [
            RecommendedAction(
                action="Escalate to senior NOC engineer for manual investigation",
                priority=1,
                rationale="Agent could not generate specific recommendations",
            )
        ]

    return AgentDiagnosis(
        alarm_id=alarm.alarm_id,
        classification=classification,
        severity_assessment=severity_assessment,
        root_cause_hypotheses=typed_hypotheses,
        most_likely_cause=most_likely_cause,
        recommended_actions=typed_actions,
        supporting_evidence=state.get("supporting_evidence") or [],
        confidence_score=confidence_score,
        reasoning_trace=state.get("reasoning_trace") or "No reasoning trace available.",
        error=state.get("error"),
    )
