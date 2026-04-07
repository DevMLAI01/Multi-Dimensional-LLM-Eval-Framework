# Multi-Dimensional LLM Eval Framework

A production-grade evaluation framework for a LangGraph telecom NOC (Network Operations Center) diagnostic agent. Measures agent quality across **5 dimensions** with automated CI/CD integration and a Streamlit dashboard.

---

## Overview

The framework evaluates a LangGraph agent that diagnoses telecom network alarms and recommends remediation actions. It uses the **LLM-as-judge** pattern: Claude Sonnet grades Claude Haiku outputs against a golden dataset of 201 hand-crafted cases.

```
alarm → [classifier] → [context fetcher] → [reasoner] → [recommender] → diagnosis + action
```

### Evaluation Dimensions

| Dimension | Weight | Threshold | What it measures |
|---|---|---|---|
| Correctness | 35% | 0.75 | Diagnostic accuracy vs. ground truth |
| Faithfulness | 25% | 0.80 | Groundedness to retrieved context (hallucination detection) |
| Robustness | 20% | 0.85 | Consistency under noisy / paraphrased inputs |
| Safety | 15% | 1.00 | Prompt injection / jailbreak resistance |
| Latency/Quality | 5% | 0.70 | Cost-quality tradeoff across model configs |

**Safety cap**: if safety pass rate < 100%, the overall score is capped at 0.60.

---

## Project Structure

```
agent/                     LangGraph NOC agent
  nodes/                   alarm_classifier, root_cause_reasoner, action_recommender
  prompts/                 Prompt YAMLs (versioned, SHA-256 hashed)
  noc_agent.py             StateGraph definition
data/
  golden_dataset/          201 hand-crafted eval cases (5 dimensions)
  synthetic/               Synthetic alarm event generator
evaluators/
  correctness_evaluator.py
  faithfulness_evaluator.py
  robustness_evaluator.py
  safety_evaluator.py
  latency_quality_evaluator.py
  statistical_significance.py
  tradeoff_report.py
eval_runner/
  runner.py                CLI — runs the full eval suite
  results_store.py         SQLite persistence
  scorer.py                Weighted overall score + safety cap
  regression_checker.py    CRITICAL/MAJOR/MINOR regression detection
  ab_test.py               Prompt A/B testing with paired t-test
  coverage_analyzer.py     Eval vs. production alarm-type coverage gaps
scripts/
  compare_prompts.py       Diff prompt hashes between two runs
dashboard/
  app.py                   Streamlit dashboard (4 tabs)
tests/                     pytest test suite (unit + integration)
.github/workflows/
  eval_ci.yml              GitHub Actions CI/CD pipeline
reports/                   Generated: eval_results.db, eval_summary_*.json, coverage_gaps.json
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key

```bash
# 1. Clone and install
git clone <repo>
cd "Multi-Dimensional LLM Eval Framework"
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Generate synthetic training data (one-time)
uv run python data/synthetic/generate_synthetic_data.py
```

### Run the Eval Suite

```bash
# Full eval across all dimensions
uv run python eval_runner/runner.py --run-id baseline

# Specific dimensions only
uv run python eval_runner/runner.py --run-id pr-42 --dimensions correctness faithfulness

# Compare against a baseline run
uv run python eval_runner/runner.py --run-id pr-42 --compare-to baseline
```

### Launch the Dashboard

```bash
uv run streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## CI/CD Integration

The GitHub Actions workflow (`.github/workflows/eval_ci.yml`) triggers on pull requests that change:
- `agent/prompts/**` — prompt YAML files
- `agent/nodes/**` — agent node logic
- `evaluators/**` — evaluator code
- `eval_runner/**` — runner / scorer code

**Pipeline stages:**

1. **Unit tests** — run with no API key; fast gate (~30s)
2. **Eval suite** — full 5-dimension eval against the golden dataset
3. **PR comment** — posts a score table with pass/fail status directly on the PR

---

## Prompt A/B Testing

Test whether a new prompt version is statistically significantly better:

```bash
uv run python eval_runner/ab_test.py \
    --prompt-a agent/prompts/reasoner_v1.yaml \
    --prompt-b agent/prompts/reasoner_v2.yaml \
    --dimension correctness \
    --cases 20
```

Output:
```
>>> Prompt B is significantly BETTER (p=0.023, delta=+0.082 / +10.3%). Safe to deploy.
```

Uses a **paired t-test** (α=0.05, minimum effect size=0.05) for significance.

---

## Prompt Version Tracking

Every eval run records SHA-256 hashes (12-char prefix) of all active prompt files. Compare two runs to see which prompts changed:

```bash
uv run python scripts/compare_prompts.py --run-a baseline --run-b pr-42
```

Output:
```
CHANGED (1):
  reasoner_v1.yaml
    abc123def456 → xyz789abc123
```

---

## Model Configuration

The latency/quality evaluator tests three configurations:

| Config | Classifier | Reasoner | Recommender |
|---|---|---|---|
| `haiku-all` | Haiku | Haiku | Haiku |
| `sonnet-all` | Sonnet | Sonnet | Sonnet |
| `hybrid` | Haiku | **Sonnet** | Haiku |

The `hybrid` config (default) balances cost and quality: Sonnet handles the complex reasoning step; Haiku handles cheaper classification and recommendation.

Override per-node model at runtime via environment variables:
```bash
NOC_REASONER_MODEL=claude-sonnet-4-6 uv run python eval_runner/runner.py --run-id test
```

---

## Running Tests

```bash
# Unit tests only (no API key required)
uv run pytest -m "not integration" -v

# Integration tests (requires ANTHROPIC_API_KEY)
uv run pytest -m integration -v

# All tests
uv run pytest -v
```

### Test counts by module

| Module | Unit | Integration |
|---|---|---|
| Agent | 9 | 3 |
| Correctness evaluator | 23 | 2 |
| Faithfulness evaluator | 35 | 2 |
| Robustness evaluator | 17 | 2 |
| Safety evaluator | 18 | 2 |
| Latency/quality evaluator | 24 | 4 |
| Eval runner | 29 | 3 |
| CI pipeline | 19 | 2 |
| Dashboard | 24 | 2 |
| **Total** | **198** | **22** |

---

## Architecture Decisions

See [docs/EVAL_DESIGN.md](docs/EVAL_DESIGN.md) for detailed design rationale.

Key decisions:
- **LLM-as-judge** over deterministic metrics: captures semantic correctness that string matching misses
- **Hybrid model config** by default: 3–5× cheaper than sonnet-all with <5% quality loss (validated by statistical significance testing)
- **Safety threshold = 1.0**: any safety failure blocks deployment regardless of other scores
- **SQLite** for results persistence: zero-dependency, portable, queryable via SQL and pandas
- **Paired t-test** for A/B testing: controls for case-difficulty variance better than independent samples t-test
