"""
Base evaluator abstract class and shared result model.

All five evaluators inherit from BaseEvaluator and return EvalResult.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    """Standardised output from any evaluator dimension."""

    case_id: str
    dimension: str                          # correctness | faithfulness | robustness | safety | latency
    evaluator_version: str
    score: Optional[float] = Field(None, ge=0.0, le=1.0)  # None when judge call failed
    passed: bool = False                    # score >= dimension threshold
    reasoning: str = ""                    # judge's explanation
    sub_scores: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)   # latency_ms, tokens, cost_usd, model
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_run_id: str = ""
    error: Optional[str] = None            # populated when judge call failed


class BaseEvaluator(ABC):
    """Abstract base for all eval dimensions."""

    dimension: str = "base"
    version: str = "1.0"
    threshold: float = 0.0      # subclasses override

    @abstractmethod
    def evaluate(self, test_case: dict, agent_output: Any) -> EvalResult:
        """Run evaluation for a single test case.

        Args:
            test_case:    Golden dataset case dict.
            agent_output: AgentDiagnosis from the NOC agent.

        Returns:
            EvalResult with score, reasoning, and sub_scores.
            Never raises — errors are captured in EvalResult.error.
        """

    def _make_error_result(self, case_id: str, error: str) -> EvalResult:
        """Return a failed EvalResult capturing an unexpected error."""
        return EvalResult(
            case_id=case_id,
            dimension=self.dimension,
            evaluator_version=self.version,
            score=None,
            passed=False,
            reasoning="",
            error=error,
            metadata={"error": error},
        )
