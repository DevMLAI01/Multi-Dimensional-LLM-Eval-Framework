"""
Phase 9 — Prompt Registry.

Hashes all prompt YAML files so eval runs can track which prompt version
was active. Stored in eval_runs.prompt_versions.
"""

import hashlib
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def hash_prompts(prompts_dir: Path = _PROMPTS_DIR) -> dict[str, dict]:
    """Hash all YAML prompt files and return a registry dict.

    Returns:
        {prompt_name: {"hash": str, "file": str}}
    """
    registry = {}
    for yaml_file in sorted(prompts_dir.glob("*.yaml")):
        content = yaml_file.read_bytes()
        sha = hashlib.sha256(content).hexdigest()[:12]
        registry[yaml_file.stem] = {
            "hash": sha,
            "file": yaml_file.name,
        }
    return registry
