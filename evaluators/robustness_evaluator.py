"""
Phase 5 — Robustness Evaluator.

Measures how much quality degrades when the input is perturbed (typos,
paraphrase, terminology variance, severity mislabelling, noise).

Unlike correctness and faithfulness, this evaluator uses SEMANTIC SIMILARITY
rather than LLM-as-judge — we want to measure consistency, not quality.

Scoring:
    0.5 × cosine_similarity(canonical_response, perturbed_response)
  + 0.3 × classification_match  (exact string, case-insensitive)
  + 0.2 × severity_match        (exact string, case-insensitive)
  = overall_score

Threshold: 0.85 — below this means the perturbation caused a meaningful drift.

The sentence-transformer model is loaded once and shared across all evaluate()
calls (lazy-init, module-level singleton).
"""

import logging
import time
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from agent.models import AgentDiagnosis, AlarmEvent
from evaluators.base_evaluator import BaseEvaluator, EvalResult

log = logging.getLogger(__name__)

# Lazy-loaded embedding model — loaded on first use, shared across calls
_embed_model: Optional[SentenceTransformer] = None

SCORE_WEIGHTS = {
    "semantic_similarity": 0.5,
    "classification_match": 0.3,
    "severity_match":       0.2,
}


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        log.info("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Model loaded.")
    return _embed_model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _diagnosis_to_text(diagnosis: AgentDiagnosis) -> str:
    """Serialise a diagnosis to a single comparable string for embedding."""
    parts = [
        f"Classification: {diagnosis.classification}",
        f"Severity: {diagnosis.severity_assessment}",
        f"Most likely cause: {diagnosis.most_likely_cause}",
        f"Reasoning: {diagnosis.reasoning_trace[:300]}",
    ]
    if diagnosis.root_cause_hypotheses:
        hyp_texts = [h.hypothesis for h in diagnosis.root_cause_hypotheses[:3]]
        parts.append(f"Hypotheses: {'; '.join(hyp_texts)}")
    return "\n".join(parts)


def _run_agent(alarm_event: AlarmEvent) -> AgentDiagnosis:
    """Run the NOC agent on an alarm event. Imported here to avoid circular deps."""
    from agent.noc_agent import run_agent
    return run_agent(alarm_event)


def _alarm_from_dict(d: dict) -> AlarmEvent:
    return AlarmEvent(
        alarm_id=d.get("alarm_id", "UNKNOWN"),
        device_id=d["device_id"],
        alarm_type=d["alarm_type"],
        severity=d["severity"],
        timestamp=d.get("timestamp", "2024-01-01T00:00:00Z"),
        raw_message=d["raw_message"],
        affected_site=d.get("affected_site", "Unknown"),
    )


class RobustnessEvaluator(BaseEvaluator):
    """Semantic-similarity-based evaluator for input robustness."""

    dimension = "robustness"
    version = "1.0"
    threshold = 0.85    # semantic similarity must be >= 0.85 to pass

    def evaluate(self, test_case: dict, agent_output: Any = None) -> EvalResult:
        """Run the agent on both canonical and perturbed inputs, compare outputs.

        Note: agent_output is IGNORED — this evaluator runs the agent itself
        on both inputs. It's kept in the signature for interface consistency.

        Args:
            test_case:    A case from robustness_cases.json
            agent_output: Ignored.

        Returns:
            EvalResult with score 0.0–1.0. Never raises.
        """
        case_id = test_case.get("case_id", "UNKNOWN")
        perturbation_type = test_case.get("perturbation_type", "UNKNOWN")

        try:
            canonical_input = _alarm_from_dict(test_case["canonical_input"])
            perturbed_input = _alarm_from_dict(test_case["perturbed_input"])
        except (KeyError, ValueError) as exc:
            return self._make_error_result(case_id, f"input_parse_error: {exc}")

        try:
            t_start = time.monotonic()
            canonical_diag = _run_agent(canonical_input)
            perturbed_diag = _run_agent(perturbed_input)
            agent_latency_ms = int((time.monotonic() - t_start) * 1000)
        except Exception as exc:
            log.error("[%s] Agent run failed: %s", case_id, exc)
            return self._make_error_result(case_id, f"agent_run_error: {exc}")

        try:
            score, sub = self._compute_score(canonical_diag, perturbed_diag)
        except Exception as exc:
            log.error("[%s] Scoring failed: %s", case_id, exc)
            return self._make_error_result(case_id, f"scoring_error: {exc}")

        passed = score >= self.threshold

        log.info(
            "[%s] robustness=%.3f (%s) type=%s sim=%.2f cls=%s sev=%s",
            case_id, score, "PASS" if passed else "FAIL",
            perturbation_type,
            sub["semantic_similarity"],
            sub["classification_match"],
            sub["severity_match"],
        )

        return EvalResult(
            case_id=case_id,
            dimension=self.dimension,
            evaluator_version=self.version,
            score=score,
            passed=passed,
            reasoning=self._make_reasoning(sub, perturbation_type, passed),
            sub_scores={
                **sub,
                "perturbation_type": perturbation_type,
                "canonical_classification": canonical_diag.classification,
                "perturbed_classification": perturbed_diag.classification,
                "canonical_severity": canonical_diag.severity_assessment,
                "perturbed_severity": perturbed_diag.severity_assessment,
            },
            metadata={
                "agent_latency_ms": agent_latency_ms,
                "perturbation_type": perturbation_type,
                "embed_model": "all-MiniLM-L6-v2",
                "threshold": self.threshold,
            },
            agent_run_id=canonical_diag.alarm_id,
        )

    def _compute_score(
        self,
        canonical: AgentDiagnosis,
        perturbed: AgentDiagnosis,
    ) -> tuple[float, dict]:
        """Compute weighted robustness score from the two diagnoses."""
        model = _get_embed_model()

        canonical_text = _diagnosis_to_text(canonical)
        perturbed_text  = _diagnosis_to_text(perturbed)

        t_embed = time.monotonic()
        embeddings = model.encode([canonical_text, perturbed_text], convert_to_numpy=True)
        embed_ms = int((time.monotonic() - t_embed) * 1000)
        log.debug("Embedding took %dms", embed_ms)

        sim = _cosine_similarity(embeddings[0], embeddings[1])
        cls_match = float(
            canonical.classification.strip().lower() ==
            perturbed.classification.strip().lower()
        )
        sev_match = float(
            canonical.severity_assessment.strip().upper() ==
            perturbed.severity_assessment.strip().upper()
        )

        overall = (
            SCORE_WEIGHTS["semantic_similarity"]  * sim +
            SCORE_WEIGHTS["classification_match"] * cls_match +
            SCORE_WEIGHTS["severity_match"]        * sev_match
        )
        overall = round(min(1.0, max(0.0, overall)), 4)

        sub = {
            "semantic_similarity": round(sim, 4),
            "classification_match": cls_match,
            "severity_match": sev_match,
            "embed_latency_ms": embed_ms,
        }
        return overall, sub

    @staticmethod
    def _make_reasoning(sub: dict, perturbation_type: str, passed: bool) -> str:
        verdict = "PASS" if passed else "FAIL"
        sim = sub["semantic_similarity"]
        cls = "matched" if sub["classification_match"] else "diverged"
        sev = "matched" if sub["severity_match"] else "diverged"
        return (
            f"{verdict} — {perturbation_type} perturbation. "
            f"Semantic similarity: {sim:.2f}. "
            f"Classification {cls}, severity {sev}."
        )

    # ------------------------------------------------------------------
    # Batch helper for the stress test
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        test_cases: list[dict],
        delay_between: float = 0.2,
    ) -> list[EvalResult]:
        """Evaluate a list of robustness cases, returning all results."""
        results = []
        for case in test_cases:
            result = self.evaluate(case)
            results.append(result)
            time.sleep(delay_between)
        return results

    def score_by_perturbation_type(
        self, results: list[EvalResult]
    ) -> dict[str, dict]:
        """Aggregate scores per perturbation type from a batch of results."""
        by_type: dict[str, list[float]] = {}
        for r in results:
            pt = r.sub_scores.get("perturbation_type", "UNKNOWN")
            if r.score is not None:
                by_type.setdefault(pt, []).append(r.score)

        summary = {}
        for pt, scores in by_type.items():
            summary[pt] = {
                "mean_score": round(sum(scores) / len(scores), 4),
                "min_score":  round(min(scores), 4),
                "pass_rate":  round(sum(1 for s in scores if s >= self.threshold) / len(scores), 4),
                "n": len(scores),
            }
        return summary
