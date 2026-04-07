"""
Shared Pydantic models for the NOC diagnostic agent.

AlarmEvent  — input contract (what the agent receives)
AgentDiagnosis — output contract (what the agent returns)
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AlarmEvent(BaseModel):
    """Input to the NOC diagnostic agent."""

    alarm_id: str = Field(..., description="Unique alarm identifier, e.g. ALM_001")
    device_id: str = Field(..., description="Device identifier, e.g. RTR-OSL-042")
    alarm_type: str = Field(
        ...,
        description=(
            "Alarm category: LINK_DOWN | HIGH_CPU | PACKET_LOSS | BGP_SESSION_DOWN | "
            "INTERFACE_ERROR | MEMORY_THRESHOLD | POWER_SUPPLY_FAIL | FAN_FAILURE | "
            "OPTICAL_DEGRADATION | SPANNING_TREE_CHANGE"
        ),
    )
    severity: str = Field(..., description="CRITICAL | MAJOR | MINOR | WARNING")
    timestamp: str = Field(..., description="ISO8601 datetime string")
    raw_message: str = Field(..., description="Free-text alarm description from NMS")
    affected_site: str = Field(..., description="Site name, e.g. Oslo-DC-North")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        allowed = {"CRITICAL", "MAJOR", "MINOR", "WARNING"}
        if v.upper() not in allowed:
            raise ValueError(f"severity must be one of {allowed}, got '{v}'")
        return v.upper()


class RootCauseHypothesis(BaseModel):
    """A single ranked root cause hypothesis."""

    hypothesis: str = Field(..., description="Root cause description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0–1.0")
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Evidence items supporting this hypothesis"
    )


class RecommendedAction(BaseModel):
    """A single ranked recommended action."""

    action: str = Field(..., description="Action description")
    priority: int = Field(..., ge=1, description="Priority rank (1 = highest)")
    rationale: str = Field(..., description="Why this action is recommended")


class AgentDiagnosis(BaseModel):
    """Output from the NOC diagnostic agent."""

    alarm_id: str
    classification: str = Field(..., description="Confirmed alarm category")
    severity_assessment: str = Field(..., description="Agent's severity judgment")
    root_cause_hypotheses: list[RootCauseHypothesis] = Field(
        default_factory=list, description="Ranked list of root cause hypotheses"
    )
    most_likely_cause: str = Field(..., description="Top-ranked root cause summary")
    recommended_actions: list[RecommendedAction] = Field(
        default_factory=list, description="Ranked action list"
    )
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Context items the agent used"
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence 0.0–1.0")
    reasoning_trace: str = Field(..., description="Agent's chain of thought")
    error: Optional[str] = Field(None, description="Error message if agent failed partially")
    created_at: datetime = Field(default_factory=datetime.utcnow)
