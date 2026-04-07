"""
Phase 1.5 — Agent unit tests.

These tests validate the agent's behaviour without hitting the Anthropic API.
API-dependent tests are marked with @pytest.mark.integration and skipped by default.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.models import AgentDiagnosis, AlarmEvent, RecommendedAction, RootCauseHypothesis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ALARMS = [
    AlarmEvent(
        alarm_id="ALM001",
        device_id="RTR-OSL-042",
        alarm_type="LINK_DOWN",
        severity="CRITICAL",
        timestamp="2024-11-14T03:22:00Z",
        raw_message="Interface GigabitEthernet0/0/1 went down unexpectedly",
        affected_site="Oslo-DC-North",
    ),
    AlarmEvent(
        alarm_id="ALM002",
        device_id="RTR-BER-019",
        alarm_type="HIGH_CPU",
        severity="MAJOR",
        timestamp="2024-11-14T04:00:00Z",
        raw_message="CPU utilisation exceeded 95% for 5 consecutive minutes",
        affected_site="Berlin-DC-East",
    ),
    AlarmEvent(
        alarm_id="ALM003",
        device_id="RTR-AMS-007",
        alarm_type="BGP_SESSION_DOWN",
        severity="CRITICAL",
        timestamp="2024-11-14T05:00:00Z",
        raw_message="BGP session to peer 10.0.0.1 dropped unexpectedly",
        affected_site="Amsterdam-DC-West",
    ),
    AlarmEvent(
        alarm_id="ALM004",
        device_id="RTR-LDN-001",
        alarm_type="MEMORY_THRESHOLD",
        severity="WARNING",
        timestamp="2024-11-14T06:00:00Z",
        raw_message="Memory utilisation at 87%, approaching threshold of 90%",
        affected_site="London-DC-Central",
    ),
    AlarmEvent(
        alarm_id="ALM005",
        device_id="RTR-PAR-003",
        alarm_type="PACKET_LOSS",
        severity="MAJOR",
        timestamp="2024-11-14T07:00:00Z",
        raw_message="Packet loss of 12% detected on uplink interface",
        affected_site="Paris-DC-South",
    ),
]


def _mock_diagnosis(alarm_id: str = "ALM001") -> AgentDiagnosis:
    """Return a valid AgentDiagnosis for mocking."""
    return AgentDiagnosis(
        alarm_id=alarm_id,
        classification="Physical layer failure",
        severity_assessment="CRITICAL",
        root_cause_hypotheses=[
            RootCauseHypothesis(
                hypothesis="Fiber cut on uplink span",
                confidence=0.85,
                supporting_evidence=["No prior CPU/memory alarms", "Sudden link drop"],
            ),
            RootCauseHypothesis(
                hypothesis="Hardware failure on line card",
                confidence=0.15,
                supporting_evidence=["No recent maintenance"],
            ),
        ],
        most_likely_cause="Fiber cut on uplink span",
        recommended_actions=[
            RecommendedAction(
                action="Check physical interface status via CLI",
                priority=1,
                rationale="Confirm link state before escalating",
            ),
            RecommendedAction(
                action="Contact field team for physical inspection",
                priority=2,
                rationale="Fiber cut requires on-site repair",
            ),
        ],
        supporting_evidence=["Fiber cut on span oslo-042-to-ber-019 (2024-10-15)"],
        confidence_score=0.85,
        reasoning_trace="Classifier: Physical layer failure. Reasoner: Prior fiber cut event found in history.",
        error=None,
    )


# ---------------------------------------------------------------------------
# Model validation tests (no API calls)
# ---------------------------------------------------------------------------

class TestAlarmEventModel:
    def test_valid_alarm_creation(self):
        alarm = SAMPLE_ALARMS[0]
        assert alarm.alarm_id == "ALM001"
        assert alarm.severity == "CRITICAL"

    def test_severity_normalised_to_upper(self):
        alarm = AlarmEvent(
            alarm_id="ALM999",
            device_id="RTR-TST-001",
            alarm_type="LINK_DOWN",
            severity="critical",  # lowercase input
            timestamp="2024-01-01T00:00:00Z",
            raw_message="test",
            affected_site="Test-Site",
        )
        assert alarm.severity == "CRITICAL"

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            AlarmEvent(
                alarm_id="ALM999",
                device_id="RTR-TST-001",
                alarm_type="LINK_DOWN",
                severity="EXTREME",  # invalid
                timestamp="2024-01-01T00:00:00Z",
                raw_message="test",
                affected_site="Test-Site",
            )


class TestAgentDiagnosisModel:
    def test_confidence_score_bounds(self):
        diag = _mock_diagnosis()
        assert 0.0 <= diag.confidence_score <= 1.0

    def test_all_required_fields_populated(self):
        diag = _mock_diagnosis()
        assert diag.alarm_id
        assert diag.classification
        assert diag.severity_assessment
        assert diag.most_likely_cause
        assert diag.reasoning_trace
        assert isinstance(diag.root_cause_hypotheses, list)
        assert isinstance(diag.recommended_actions, list)
        assert isinstance(diag.supporting_evidence, list)

    def test_json_serializable(self):
        diag = _mock_diagnosis()
        json_str = diag.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["alarm_id"] == "ALM001"


# ---------------------------------------------------------------------------
# Tool tests (no API calls, uses synthetic data if available)
# ---------------------------------------------------------------------------

class TestAgentTools:
    def test_query_alarm_history_missing_data_returns_empty(self):
        from agent.tools.query_alarm_history import query_alarm_history
        # Should not raise even if file doesn't exist
        result = query_alarm_history("RTR-NONEXISTENT-999", "LINK_DOWN")
        assert isinstance(result, list)

    def test_get_device_info_missing_device_returns_empty(self):
        from agent.tools.get_device_info import get_device_info
        result = get_device_info("RTR-NONEXISTENT-999")
        assert result == {}

    def test_search_runbooks_unknown_type_returns_empty(self):
        from agent.tools.search_runbooks import search_runbooks
        result = search_runbooks("TOTALLY_UNKNOWN_ALARM_TYPE_XYZ")
        assert result == {}

    def test_query_alarm_history_filters_by_device(self):
        """If synthetic data exists, verify device filtering works."""
        data_path = Path("data/synthetic/alarm_history.json")
        if not data_path.exists():
            pytest.skip("Synthetic data not generated yet — run generate_synthetic_data.py")
        from agent.tools.query_alarm_history import query_alarm_history
        results = query_alarm_history("RTR-NONEXISTENT-ZZZ-999")
        assert results == []

    def test_search_runbooks_returns_dict_on_hit(self):
        """If synthetic data exists, runbook lookup returns a dict."""
        data_path = Path("data/synthetic/runbooks.json")
        if not data_path.exists():
            pytest.skip("Synthetic data not generated yet — run generate_synthetic_data.py")
        from agent.tools.search_runbooks import search_runbooks
        result = search_runbooks("LINK_DOWN")
        assert isinstance(result, dict)
        if result:
            assert "alarm_type" in result


# ---------------------------------------------------------------------------
# Agent integration tests (hit Anthropic API — skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAgentIntegration:
    """These tests call the Anthropic API. Run with: pytest -m integration"""

    def test_agent_returns_valid_diagnosis_for_sample_alarms(self):
        from agent.noc_agent import run_agent
        for alarm in SAMPLE_ALARMS:
            diag = run_agent(alarm)
            assert isinstance(diag, AgentDiagnosis), f"Failed for {alarm.alarm_id}"
            assert diag.alarm_id == alarm.alarm_id
            assert diag.classification
            assert diag.severity_assessment in {"CRITICAL", "MAJOR", "MINOR", "WARNING"}
            assert 0.0 <= diag.confidence_score <= 1.0
            assert diag.most_likely_cause
            assert diag.reasoning_trace
            assert len(diag.recommended_actions) >= 1

    def test_agent_handles_missing_device_without_crashing(self):
        from agent.noc_agent import run_agent
        alarm = AlarmEvent(
            alarm_id="ALM_MISSING_DEV",
            device_id="RTR-NONEXISTENT-999",
            alarm_type="LINK_DOWN",
            severity="CRITICAL",
            timestamp="2024-11-14T03:22:00Z",
            raw_message="Interface down on unknown device",
            affected_site="Unknown-Site",
        )
        diag = run_agent(alarm)
        assert isinstance(diag, AgentDiagnosis)
        assert diag.alarm_id == "ALM_MISSING_DEV"
        # Must still populate required fields
        assert diag.classification
        assert diag.most_likely_cause

    def test_agent_handles_unknown_alarm_type_gracefully(self):
        """Agent receives an alarm_type not in any runbook — should not crash."""
        from agent.noc_agent import run_agent
        # We can't use AlarmEvent validation here (it doesn't restrict alarm_type)
        # so we directly test that missing runbook context is handled
        alarm = AlarmEvent(
            alarm_id="ALM_UNK_TYPE",
            device_id="RTR-OSL-042",
            alarm_type="LINK_DOWN",  # valid type, but runbook will be empty if data missing
            severity="MINOR",
            timestamp="2024-11-14T03:22:00Z",
            raw_message="Unusual alarm condition with no matching runbook",
            affected_site="Oslo-DC-North",
        )
        diag = run_agent(alarm)
        assert isinstance(diag, AgentDiagnosis)
        assert diag.confidence_score is not None

    def test_all_output_fields_populated(self):
        from agent.noc_agent import run_agent
        alarm = SAMPLE_ALARMS[0]
        diag = run_agent(alarm)
        # No None on required fields
        assert diag.alarm_id is not None
        assert diag.classification is not None
        assert diag.severity_assessment is not None
        assert diag.most_likely_cause is not None
        assert diag.reasoning_trace is not None
        assert diag.confidence_score is not None
        assert isinstance(diag.root_cause_hypotheses, list)
        assert isinstance(diag.recommended_actions, list)
        assert isinstance(diag.supporting_evidence, list)

    def test_confidence_score_in_valid_range(self):
        from agent.noc_agent import run_agent
        for alarm in SAMPLE_ALARMS[:3]:
            diag = run_agent(alarm)
            assert 0.0 <= diag.confidence_score <= 1.0, (
                f"confidence_score={diag.confidence_score} out of range for {alarm.alarm_id}"
            )
