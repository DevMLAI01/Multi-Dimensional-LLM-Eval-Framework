"""
Phase 9 — CI pipeline tests.

Unit tests for:
  - Prompt registry (hashing)
  - compare_prompts script (diff logic)
  - A/B test helper (statistical decision)
  - GitHub Actions YAML syntax (structural check)

No API calls in unit tests.
Integration tests: marked with -m integration.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.prompts.prompt_registry import hash_prompts
from scripts.compare_prompts import compare_prompt_versions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_prompts_dir(tmp_path):
    """Create a temp dir with two fake YAML prompt files."""
    p = tmp_path / "prompts"
    p.mkdir()
    (p / "classifier_v1.yaml").write_text("model: haiku\nversion: 1")
    (p / "reasoner_v1.yaml").write_text("model: haiku\nversion: 1")
    return p


@pytest.fixture
def tmp_store(tmp_path):
    from eval_runner.results_store import ResultsStore
    return ResultsStore(db_path=tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Prompt registry tests
# ---------------------------------------------------------------------------

class TestPromptRegistry:
    def test_hash_prompts_returns_dict(self, tmp_prompts_dir):
        registry = hash_prompts(tmp_prompts_dir)
        assert isinstance(registry, dict)
        assert "classifier_v1" in registry
        assert "reasoner_v1" in registry

    def test_each_entry_has_hash_and_file(self, tmp_prompts_dir):
        registry = hash_prompts(tmp_prompts_dir)
        for name, entry in registry.items():
            assert "hash" in entry
            assert "file" in entry
            assert len(entry["hash"]) == 12  # truncated SHA-256

    def test_different_content_produces_different_hash(self, tmp_prompts_dir):
        r1 = hash_prompts(tmp_prompts_dir)

        # Modify one file
        (tmp_prompts_dir / "reasoner_v1.yaml").write_text("model: sonnet\nversion: 2")
        r2 = hash_prompts(tmp_prompts_dir)

        assert r1["reasoner_v1"]["hash"] != r2["reasoner_v1"]["hash"]

    def test_same_content_same_hash(self, tmp_prompts_dir):
        r1 = hash_prompts(tmp_prompts_dir)
        r2 = hash_prompts(tmp_prompts_dir)
        for name in r1:
            assert r1[name]["hash"] == r2[name]["hash"]

    def test_empty_dir_returns_empty_dict(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        registry = hash_prompts(empty)
        assert registry == {}

    def test_real_prompts_dir_has_entries(self):
        """Real project prompts dir should have at least 3 entries."""
        real_dir = Path(__file__).parents[1] / "agent" / "prompts"
        if real_dir.exists():
            registry = hash_prompts(real_dir)
            assert len(registry) >= 3


# ---------------------------------------------------------------------------
# compare_prompts tests
# ---------------------------------------------------------------------------

class TestComparePrompts:
    def _create_run_with_prompts(self, store, run_id: str, versions: dict):
        store.create_run(run_id)
        # Manually patch prompt_versions into the DB
        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            conn.execute(
                "UPDATE eval_runs SET prompt_versions=? WHERE run_id=?",
                (json.dumps(versions), run_id),
            )

    def test_detects_changed_prompt(self, tmp_store):
        self._create_run_with_prompts(tmp_store, "run_a", {
            "reasoner_v1": {"hash": "abc123", "file": "reasoner_v1.yaml"},
            "classifier_v1": {"hash": "def456", "file": "classifier_v1.yaml"},
        })
        self._create_run_with_prompts(tmp_store, "run_b", {
            "reasoner_v1": {"hash": "xyz789", "file": "reasoner_v1.yaml"},  # changed
            "classifier_v1": {"hash": "def456", "file": "classifier_v1.yaml"},
        })

        diff = compare_prompt_versions("run_a", "run_b", tmp_store.db_path)
        assert len(diff["changed"]) == 1
        assert diff["changed"][0]["prompt"] == "reasoner_v1"
        assert diff["changed"][0]["hash_a"] == "abc123"
        assert diff["changed"][0]["hash_b"] == "xyz789"

    def test_no_changes_detected(self, tmp_store):
        versions = {
            "reasoner_v1": {"hash": "abc123", "file": "reasoner_v1.yaml"},
        }
        self._create_run_with_prompts(tmp_store, "run_a", versions)
        self._create_run_with_prompts(tmp_store, "run_b", versions)

        diff = compare_prompt_versions("run_a", "run_b", tmp_store.db_path)
        assert len(diff["changed"]) == 0
        assert "reasoner_v1" in diff["unchanged"]

    def test_detects_added_prompt(self, tmp_store):
        self._create_run_with_prompts(tmp_store, "run_a", {
            "reasoner_v1": {"hash": "abc", "file": "reasoner_v1.yaml"},
        })
        self._create_run_with_prompts(tmp_store, "run_b", {
            "reasoner_v1": {"hash": "abc", "file": "reasoner_v1.yaml"},
            "new_prompt": {"hash": "xyz", "file": "new_prompt.yaml"},
        })
        diff = compare_prompt_versions("run_a", "run_b", tmp_store.db_path)
        assert "new_prompt" in diff["only_in_b"]

    def test_missing_run_raises_value_error(self, tmp_store):
        tmp_store.create_run("run_a")
        with pytest.raises(ValueError, match="not found"):
            compare_prompt_versions("run_a", "nonexistent", tmp_store.db_path)


# ---------------------------------------------------------------------------
# A/B test helper unit tests
# ---------------------------------------------------------------------------

class TestAbTestHelper:
    def test_ab_test_better_b_returns_significant_decision(self):
        """Mock runs where B consistently scores higher — should detect significance."""
        from eval_runner.ab_test import run_ab_test
        import numpy as np

        rng = np.random.default_rng(42)
        scores_a = rng.normal(0.70, 0.03, 20).tolist()
        scores_b = rng.normal(0.85, 0.03, 20).tolist()

        with patch("eval_runner.ab_test._run_eval_for_prompt",
                   side_effect=[scores_a, scores_b]):
            result = run_ab_test(
                prompt_a_path=Path("agent/prompts/reasoner_v1.yaml"),
                prompt_b_path=Path("agent/prompts/reasoner_v1.yaml"),
                dimension="correctness",
                n_cases=20,
            )

        assert result["significant"] is True
        assert result["delta"] > 0
        assert "BETTER" in result["decision"]

    def test_ab_test_no_difference_returns_keep_a(self):
        """Mock runs where both score similarly — should not detect significance."""
        from eval_runner.ab_test import run_ab_test
        import numpy as np

        rng = np.random.default_rng(99)
        scores = rng.normal(0.80, 0.02, 20).tolist()

        with patch("eval_runner.ab_test._run_eval_for_prompt",
                   side_effect=[scores, scores]):
            result = run_ab_test(
                prompt_a_path=Path("agent/prompts/reasoner_v1.yaml"),
                prompt_b_path=Path("agent/prompts/reasoner_v1.yaml"),
                dimension="correctness",
                n_cases=20,
            )

        assert result["significant"] is False
        assert "Keep Prompt A" in result["decision"]

    def test_ab_test_worse_b_returns_do_not_deploy(self):
        from eval_runner.ab_test import run_ab_test
        import numpy as np

        rng = np.random.default_rng(7)
        scores_a = rng.normal(0.85, 0.03, 20).tolist()
        scores_b = rng.normal(0.65, 0.03, 20).tolist()

        with patch("eval_runner.ab_test._run_eval_for_prompt",
                   side_effect=[scores_a, scores_b]):
            result = run_ab_test(
                prompt_a_path=Path("agent/prompts/reasoner_v1.yaml"),
                prompt_b_path=Path("agent/prompts/reasoner_v1.yaml"),
                dimension="correctness",
                n_cases=20,
            )

        assert result["significant"] is True
        assert result["delta"] < 0
        assert "WORSE" in result["decision"]

    def test_ab_test_result_has_all_fields(self):
        from eval_runner.ab_test import run_ab_test

        scores = [0.80] * 10

        with patch("eval_runner.ab_test._run_eval_for_prompt",
                   side_effect=[scores, scores]):
            result = run_ab_test(
                prompt_a_path=Path("agent/prompts/reasoner_v1.yaml"),
                prompt_b_path=Path("agent/prompts/reasoner_v1.yaml"),
                dimension="correctness",
                n_cases=10,
            )

        for field in ["dimension", "n_cases", "n_pairs", "prompt_a", "prompt_b",
                      "mean_score_a", "mean_score_b", "delta", "p_value",
                      "ci_95", "significant", "decision"]:
            assert field in result, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# GitHub Actions YAML syntax test
# ---------------------------------------------------------------------------

class TestGitHubActionsWorkflow:
    def test_workflow_file_exists(self):
        path = Path(__file__).parents[1] / ".github" / "workflows" / "eval_ci.yml"
        assert path.exists(), "eval_ci.yml not found"

    def test_workflow_valid_yaml(self):
        import yaml
        path = Path(__file__).parents[1] / ".github" / "workflows" / "eval_ci.yml"
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        assert doc is not None
        assert "jobs" in doc
        # PyYAML parses `on:` as boolean True (known YAML quirk)
        assert True in doc or "on" in doc

    def test_workflow_has_eval_suite_job(self):
        import yaml
        path = Path(__file__).parents[1] / ".github" / "workflows" / "eval_ci.yml"
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        assert "eval-suite" in doc["jobs"]

    def test_workflow_uses_anthropic_secret(self):
        path = Path(__file__).parents[1] / ".github" / "workflows" / "eval_ci.yml"
        content = path.read_text(encoding="utf-8")
        assert "ANTHROPIC_API_KEY" in content
        assert "secrets.ANTHROPIC_API_KEY" in content

    def test_workflow_triggers_on_prompt_changes(self):
        import yaml
        path = Path(__file__).parents[1] / ".github" / "workflows" / "eval_ci.yml"
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        # PyYAML parses `on:` as boolean True key
        on_config = doc.get(True, doc.get("on", {}))
        pr_config = on_config.get("pull_request", {})
        paths = pr_config.get("paths", [])
        assert any("prompts" in p for p in paths)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCIPipelineIntegration:
    def test_prompt_registry_on_real_prompts(self, tmp_path):
        """Hash real prompts and verify stored in a run."""
        from eval_runner.results_store import ResultsStore

        db_path = tmp_path / "test.db"
        store = ResultsStore(db_path=db_path)
        registry = hash_prompts()
        store.create_run("test_run", prompt_versions=registry)
        run = store.get_run("test_run")
        stored = json.loads(run["prompt_versions"])
        assert len(stored) >= 3

    def test_compare_prompts_identical_runs(self, tmp_path):
        """Two runs with same prompts should show no changes."""
        from eval_runner.results_store import ResultsStore

        registry = hash_prompts()
        db_path = tmp_path / "test.db"
        store = ResultsStore(db_path=db_path)
        store.create_run("run_a", prompt_versions=registry)
        store.create_run("run_b", prompt_versions=registry)

        diff = compare_prompt_versions("run_a", "run_b", db_path)
        assert len(diff["changed"]) == 0
