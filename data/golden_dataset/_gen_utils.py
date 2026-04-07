"""
Shared utilities for golden dataset generators.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

ALARM_TYPES = [
    "LINK_DOWN", "HIGH_CPU", "PACKET_LOSS", "BGP_SESSION_DOWN",
    "INTERFACE_ERROR", "MEMORY_THRESHOLD", "POWER_SUPPLY_FAIL",
    "FAN_FAILURE", "OPTICAL_DEGRADATION", "SPANNING_TREE_CHANGE",
]
SEVERITIES = ["CRITICAL", "MAJOR", "MINOR", "WARNING"]

_SYNTHETIC_DIR = Path(__file__).parents[1] / "synthetic"
_GOLDEN_DIR = Path(__file__).parent

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def call_claude(
    prompt: str,
    model: str = SONNET,
    max_tokens: int = 4096,
    system: str | None = None,
) -> str:
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    resp = get_client().messages.create(**kwargs)
    return resp.content[0].text


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def parse_json(raw: str, context: str = "") -> list | dict:
    """Parse JSON from model output, with repair fallback."""
    cleaned = strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("JSON parse failed (%s): %s — attempting repair", context, exc)
        repair = strip_fences(
            call_claude(
                f"Fix this malformed/truncated JSON. Return ONLY valid JSON:\n\n{cleaned}",
                model=HAIKU,
                max_tokens=4096,
            )
        )
        return json.loads(repair)


def load_synthetic(filename: str) -> list[dict]:
    path = _SYNTHETIC_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_cases(cases: list[dict], filename: str) -> None:
    path = _GOLDEN_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    log.info("Saved %d cases → %s", len(cases), path)


def load_cases(filename: str) -> list[dict]:
    path = _GOLDEN_DIR / filename
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_device_ids(n: int = 20) -> list[str]:
    devices = load_synthetic("device_inventory.json")
    return [d["device_id"] for d in devices[:n]]


def get_sites(n: int = 10) -> list[str]:
    devices = load_synthetic("device_inventory.json")
    return list({d["site"] for d in devices})[:n]
