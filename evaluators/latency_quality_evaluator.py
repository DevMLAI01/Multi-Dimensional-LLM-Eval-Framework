"""
Phase 7 — Latency/Quality Evaluator.

Runs the NOC agent under three model configurations on 40 latency test cases,
measuring both quality (correctness score) and performance (latency, cost).

Model configurations:
    haiku-all:  all nodes use claude-haiku-4-5-20251001
    sonnet-all: all nodes use claude-sonnet-4-6
    hybrid:     classifier=haiku, reasoner=sonnet, recommender=haiku

Quality scoring: adapted correctness judge (uses latency case notes as
expert reference). Identical judge model across all configs for fairness.

Per-run data recorded:
    - total_latency_ms
    - node_latencies (via ApiCallRecorder patching the Anthropic client)
    - input_tokens, output_tokens, estimated_cost_usd (per API call and total)
    - quality_score (0.0–1.0, via adapted correctness judge)

Usage:
    from evaluators.latency_quality_evaluator import LatencyQualityEvaluator
    evaluator = LatencyQualityEvaluator()
    results = evaluator.run_all_configs(cases)
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import anthropic
import yaml
from dotenv import load_dotenv

from agent.models import AgentDiagnosis, AlarmEvent
from evaluators.base_evaluator import EvalResult

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "haiku-all": {
        "NOC_CLASSIFIER_MODEL":  HAIKU,
        "NOC_REASONER_MODEL":    HAIKU,
        "NOC_RECOMMENDER_MODEL": HAIKU,
    },
    "sonnet-all": {
        "NOC_CLASSIFIER_MODEL":  SONNET,
        "NOC_REASONER_MODEL":    SONNET,
        "NOC_RECOMMENDER_MODEL": SONNET,
    },
    "hybrid": {
        "NOC_CLASSIFIER_MODEL":  HAIKU,
        "NOC_REASONER_MODEL":    SONNET,
        "NOC_RECOMMENDER_MODEL": HAIKU,
    },
}

# Token cost per 1M tokens (USD)
TOKEN_COSTS = {
    HAIKU:  {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    SONNET: {"input": 3.00 / 1_000_000, "output": 15.0 / 1_000_000},
}

# ---------------------------------------------------------------------------
# Alarm type → expected classification mapping
# ---------------------------------------------------------------------------

_ALARM_CLASS_MAP = {
    "LINK_DOWN":           "Physical layer failure",
    "HIGH_CPU":            "Resource exhaustion",
    "BGP_SESSION_DOWN":    "Routing protocol failure",
    "BGP_FLAP":            "Routing protocol instability",
    "OPTICAL_DEGRADATION": "Physical layer degradation",
    "POWER_SUPPLY_FAIL":   "Power system failure",
    "PACKET_LOSS":         "Network performance degradation",
    "MEMORY_HIGH":         "Resource exhaustion",
    "INTERFACE_ERROR":     "Physical layer failure",
    "OSPF_NEIGHBOR_DOWN":  "Routing protocol failure",
    "HARDWARE_FAILURE":    "Hardware failure",
    "TEMPERATURE_HIGH":    "Environmental alarm",
}

_JUDGE_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "judge_correctness.yaml"

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    return _client


def _load_judge_prompt() -> dict:
    with open(_JUDGE_PROMPT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _extract_json(text: str) -> dict:
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _format_actions(actions: list) -> str:
    if not actions:
        return "None"
    parts = []
    for a in actions:
        if hasattr(a, "action"):
            parts.append(f"{a.priority}. {a.action}")
        elif isinstance(a, dict):
            parts.append(f"{a.get('priority', '?')}. {a.get('action', str(a))}")
        else:
            parts.append(str(a))
    return "; ".join(parts)


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = TOKEN_COSTS.get(model, TOKEN_COSTS[HAIKU])
    return input_tokens * costs["input"] + output_tokens * costs["output"]


# ---------------------------------------------------------------------------
# API call recorder — patches anthropic client to capture per-call telemetry
# ---------------------------------------------------------------------------

@dataclass
class ApiCallRecord:
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    node_hint: str = ""   # set by the env var active at call time


# ---------------------------------------------------------------------------
# Run agent with a specific model config + record telemetry
# ---------------------------------------------------------------------------

@dataclass
class AgentRunResult:
    diagnosis: Optional[AgentDiagnosis]
    total_latency_ms: int
    api_calls: list[ApiCallRecord] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.api_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.api_calls)

    @property
    def estimated_cost_usd(self) -> float:
        return sum(_cost_usd(c.model, c.input_tokens, c.output_tokens) for c in self.api_calls)

    @property
    def node_latencies(self) -> dict[str, int]:
        """Map call_N → latency_ms for the 3 LLM-calling nodes."""
        labels = ["classifier", "reasoner", "recommender"]
        result = {}
        for i, call in enumerate(self.api_calls):
            label = labels[i] if i < len(labels) else f"call_{i}"
            result[label] = call.latency_ms
        return result


def run_agent_with_config(alarm: AlarmEvent, config_name: str) -> AgentRunResult:
    """Run the agent with a named model config, recording telemetry.

    Sets NOC_*_MODEL env vars around the run so nodes pick them up.
    Resets env vars after regardless of outcome.

    API calls are intercepted by patching anthropic.resources.messages.Messages.create
    (the concrete Messages class, not the cached_property descriptor on Anthropic).
    """
    from agent.noc_agent import build_graph, _state_to_diagnosis
    from anthropic.resources.messages.messages import Messages as AnthropicMessages

    env_vars = MODEL_CONFIGS[config_name]
    old_env = {k: os.environ.get(k) for k in env_vars}

    api_calls: list[ApiCallRecord] = []

    try:
        # Set model config env vars
        for k, v in env_vars.items():
            os.environ[k] = v

        # Invalidate cached node singletons so they re-read env vars
        _invalidate_node_caches()

        initial_state = {
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

        _real_create = AnthropicMessages.create

        def _patched_create(self_msgs, *args, **kwargs):
            t_call = time.monotonic()
            resp = _real_create(self_msgs, *args, **kwargs)
            elapsed = int((time.monotonic() - t_call) * 1000)
            model = kwargs.get("model", "unknown")
            api_calls.append(ApiCallRecord(
                model=model,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                latency_ms=elapsed,
            ))
            return resp

        graph = build_graph()
        t0 = time.monotonic()

        with patch.object(AnthropicMessages, "create", _patched_create):
            final_state = graph.invoke(initial_state)

        total_latency_ms = int((time.monotonic() - t0) * 1000)
        diagnosis = _state_to_diagnosis(alarm, final_state)

        return AgentRunResult(
            diagnosis=diagnosis,
            total_latency_ms=total_latency_ms,
            api_calls=api_calls,
        )

    except Exception as exc:
        log.error("Agent run failed for config %s: %s", config_name, exc)
        return AgentRunResult(
            diagnosis=None,
            total_latency_ms=0,
            api_calls=api_calls,
            error=str(exc),
        )

    finally:
        # Restore original env vars
        for k, old_v in old_env.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v
        _invalidate_node_caches()


def _invalidate_node_caches():
    """Clear module-level _client singletons in node modules so they re-init."""
    import importlib
    for mod_name in [
        "agent.nodes.alarm_classifier",
        "agent.nodes.root_cause_reasoner",
        "agent.nodes.action_recommender",
    ]:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "_client"):
                mod._client = None
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Quality scoring (adapted correctness judge)
# ---------------------------------------------------------------------------

def score_quality(
    latency_case: dict,
    diagnosis: AgentDiagnosis,
) -> tuple[float, dict]:
    """Score diagnosis quality using the correctness judge.

    Adapts the latency case format to the correctness judge prompt by
    deriving expected fields from the case's alarm_type and notes.

    Returns: (score 0.0–1.0, sub_scores dict)
    """
    inp = latency_case.get("input", {})
    expected_notes = latency_case.get("expected", {}).get("notes", "")
    alarm_type = inp.get("alarm_type", "")

    expected_classification = _ALARM_CLASS_MAP.get(alarm_type, alarm_type.replace("_", " ").title())
    # Use first 200 chars of notes as root cause reference
    expected_root_cause = expected_notes[:200] if expected_notes else "See alarm type"

    try:
        prompt = _load_judge_prompt()
        user_msg = prompt["user"].format(
            device_id=inp.get("device_id", ""),
            alarm_type=alarm_type,
            severity=inp.get("severity", ""),
            affected_site=inp.get("affected_site", ""),
            raw_message=inp.get("raw_message", ""),
            expected_classification=expected_classification,
            expected_root_cause_category=expected_root_cause,
            required_actions="",
            expert_reasoning=expected_notes[:400],
            agent_classification=diagnosis.classification,
            agent_severity=diagnosis.severity_assessment,
            agent_most_likely_cause=diagnosis.most_likely_cause,
            agent_recommended_actions=_format_actions(diagnosis.recommended_actions),
            agent_confidence=f"{diagnosis.confidence_score:.2f}",
            agent_reasoning_trace=diagnosis.reasoning_trace[:500],
        )

        response = _get_client().messages.create(
            model=prompt["model"],
            max_tokens=prompt["max_tokens"],
            system=prompt["system"],
            messages=[{"role": "user", "content": user_msg}],
        )
        judge = _extract_json(response.content[0].text)

        # Recompute weighted score
        weights = {
            "classification_accuracy": 0.30,
            "root_cause_accuracy":     0.30,
            "action_completeness":     0.25,
            "severity_accuracy":       0.15,
        }
        sub = {k: float(judge.get(k, 0.0)) for k in weights}
        score = sum(sub[k] * w for k, w in weights.items())
        score = round(min(1.0, max(0.0, score)), 4)

        judge_cost = _cost_usd(
            prompt["model"],
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        sub["judge_cost_usd"] = round(judge_cost, 6)

        time.sleep(0.3)  # rate-limit courtesy pause
        return score, sub

    except Exception as exc:
        log.error("Quality scoring failed: %s", exc)
        return 0.0, {"error": str(exc)}


# ---------------------------------------------------------------------------
# Per-case result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LatencyRunRecord:
    case_id: str
    config_name: str
    complexity: str
    total_latency_ms: int
    node_latencies: dict[str, int]
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    quality_score: float
    quality_sub_scores: dict
    error: Optional[str] = None

    def to_eval_result(self) -> EvalResult:
        return EvalResult(
            case_id=f"{self.case_id}_{self.config_name}",
            dimension="latency_quality",
            evaluator_version="1.0",
            score=self.quality_score,
            passed=self.quality_score >= 0.70,
            sub_scores={
                **self.quality_sub_scores,
                "config_name": self.config_name,
                "complexity": self.complexity,
                "total_latency_ms": self.total_latency_ms,
                "node_latencies": self.node_latencies,
            },
            metadata={
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "estimated_cost_usd": round(self.estimated_cost_usd, 6),
                "config_name": self.config_name,
            },
            error=self.error,
        )


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class LatencyQualityEvaluator:
    """Runs 40 latency cases × 3 model configs = 120 agent calls."""

    def run_single(
        self,
        latency_case: dict,
        config_name: str,
        score_quality_flag: bool = True,
    ) -> LatencyRunRecord:
        """Run one case against one config and return a LatencyRunRecord."""
        case_id = latency_case.get("case_id", "UNKNOWN")
        complexity = latency_case.get("complexity", "UNKNOWN")
        inp = latency_case.get("input", {})

        try:
            alarm = AlarmEvent(**inp)
        except Exception as exc:
            return LatencyRunRecord(
                case_id=case_id, config_name=config_name, complexity=complexity,
                total_latency_ms=0, node_latencies={},
                total_input_tokens=0, total_output_tokens=0,
                estimated_cost_usd=0.0, quality_score=0.0,
                quality_sub_scores={}, error=f"input_parse_error: {exc}",
            )

        run = run_agent_with_config(alarm, config_name)

        if run.error or run.diagnosis is None:
            return LatencyRunRecord(
                case_id=case_id, config_name=config_name, complexity=complexity,
                total_latency_ms=run.total_latency_ms,
                node_latencies=run.node_latencies,
                total_input_tokens=run.total_input_tokens,
                total_output_tokens=run.total_output_tokens,
                estimated_cost_usd=run.estimated_cost_usd,
                quality_score=0.0, quality_sub_scores={},
                error=run.error or "no_diagnosis",
            )

        quality_score, quality_sub = 0.0, {}
        if score_quality_flag:
            quality_score, quality_sub = score_quality(latency_case, run.diagnosis)

        log.info(
            "[%s/%s] latency=%dms quality=%.3f cost=$%.4f",
            case_id, config_name, run.total_latency_ms,
            quality_score, run.estimated_cost_usd,
        )

        return LatencyRunRecord(
            case_id=case_id,
            config_name=config_name,
            complexity=complexity,
            total_latency_ms=run.total_latency_ms,
            node_latencies=run.node_latencies,
            total_input_tokens=run.total_input_tokens,
            total_output_tokens=run.total_output_tokens,
            estimated_cost_usd=run.estimated_cost_usd,
            quality_score=quality_score,
            quality_sub_scores=quality_sub,
        )

    def run_all_configs(
        self,
        latency_cases: list[dict],
        configs: list[str] | None = None,
        delay_between: float = 0.5,
    ) -> list[LatencyRunRecord]:
        """Run all cases against all configs (or a subset).

        Returns a flat list: len = len(cases) × len(configs).
        """
        configs = configs or list(MODEL_CONFIGS.keys())
        records = []

        for config_name in configs:
            log.info("=== Config: %s ===", config_name)
            for i, case in enumerate(latency_cases):
                log.info("  [%d/%d] %s", i + 1, len(latency_cases), case.get("case_id"))
                record = self.run_single(case, config_name)
                records.append(record)
                time.sleep(delay_between)

        return records

    def aggregate(
        self, records: list[LatencyRunRecord]
    ) -> dict[str, dict]:
        """Compute per-config statistics from a list of records.

        Returns dict keyed by config_name with latency percentiles,
        quality stats, cost stats.
        """
        import numpy as np

        by_config: dict[str, list[LatencyRunRecord]] = {}
        for r in records:
            by_config.setdefault(r.config_name, []).append(r)

        summary = {}
        for config_name, recs in by_config.items():
            valid = [r for r in recs if r.error is None]
            latencies = [r.total_latency_ms for r in valid]
            qualities = [r.quality_score for r in valid]
            costs = [r.estimated_cost_usd for r in valid]

            if not latencies:
                summary[config_name] = {"error": "no_valid_records"}
                continue

            lat_arr = np.array(latencies)
            summary[config_name] = {
                "n_cases": len(recs),
                "n_valid": len(valid),
                "mean_quality": round(float(np.mean(qualities)), 4) if qualities else 0.0,
                "p50_latency_ms":  int(np.percentile(lat_arr, 50)),
                "p95_latency_ms":  int(np.percentile(lat_arr, 95)),
                "p99_latency_ms":  int(np.percentile(lat_arr, 99)),
                "mean_latency_ms": int(np.mean(lat_arr)),
                "mean_cost_usd":   round(float(np.mean(costs)), 6) if costs else 0.0,
                "total_cost_usd":  round(float(np.sum(costs)), 4) if costs else 0.0,
                "quality_per_dollar": round(
                    float(np.mean(qualities)) / float(np.mean(costs)), 2
                ) if costs and np.mean(costs) > 0 else 0.0,
            }

        return summary
