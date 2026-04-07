"""
Tool: search_runbooks

Fetches the runbook(s) matching an alarm_type from runbooks.json.
Returns the most relevant runbook for the given alarm type.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).parents[2] / "data" / "synthetic" / "runbooks.json"
_index: dict[str, list[dict]] | None = None


def _load() -> dict[str, list[dict]]:
    """Build alarm_type → list[runbook] index."""
    global _index
    if _index is None:
        if not _DATA_PATH.exists():
            log.warning(
                "runbooks.json not found at %s — returning empty runbook. "
                "Run data/synthetic/generate_synthetic_data.py first.",
                _DATA_PATH,
            )
            _index = {}
        else:
            with open(_DATA_PATH, encoding="utf-8") as f:
                records: list[dict] = json.load(f)
            _index = {}
            for r in records:
                alarm_type = r.get("alarm_type", "UNKNOWN")
                _index.setdefault(alarm_type, []).append(r)
            log.debug(
                "Loaded runbooks for %d alarm types from %s",
                len(_index),
                _DATA_PATH,
            )
    return _index


def search_runbooks(alarm_type: str) -> dict:
    """Return the primary runbook for alarm_type.

    Args:
        alarm_type: Alarm type string, e.g. 'LINK_DOWN'.

    Returns:
        The first matching runbook dict, or empty dict if no match.
        If multiple runbooks exist for the type, returns the first (primary scenario).
    """
    index = _load()
    entries = index.get(alarm_type, [])
    if not entries:
        log.debug("No runbook found for alarm_type '%s'.", alarm_type)
        return {}
    return entries[0]


def search_all_runbooks(alarm_type: str) -> list[dict]:
    """Return all runbooks for alarm_type (for richer context).

    Args:
        alarm_type: Alarm type string.

    Returns:
        List of matching runbook dicts, empty list if none found.
    """
    index = _load()
    return index.get(alarm_type, [])
