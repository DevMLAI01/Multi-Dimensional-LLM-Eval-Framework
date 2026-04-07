"""
Tool: query_alarm_history

Fetches recent alarms for a device from the synthetic alarm_history.json dataset.
Simulates what would be a real-time query against the data lake in production.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).parents[2] / "data" / "synthetic" / "alarm_history.json"
_cache: list[dict] | None = None


def _load() -> list[dict]:
    global _cache
    if _cache is None:
        if not _DATA_PATH.exists():
            log.warning(
                "alarm_history.json not found at %s — returning empty history. "
                "Run data/synthetic/generate_synthetic_data.py first.",
                _DATA_PATH,
            )
            _cache = []
        else:
            with open(_cache_path := _DATA_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
            log.debug("Loaded %d alarm history records from %s", len(_cache), _DATA_PATH)
    return _cache


def query_alarm_history(
    device_id: str,
    alarm_type: str | None = None,
    days_back: int = 30,
    max_results: int = 10,
) -> list[dict]:
    """Return recent alarms matching device_id (and optionally alarm_type).

    Args:
        device_id:   Device identifier to filter on.
        alarm_type:  If provided, filter to this alarm type only.
        days_back:   How many days of history to look back.
        max_results: Maximum number of records to return.

    Returns:
        List of alarm history dicts, sorted newest-first. Empty list if no match.
    """
    records = _load()
    if not records:
        return []

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days_back)

    results = []
    for record in records:
        if record.get("device_id") != device_id:
            continue
        if alarm_type and record.get("alarm_type") != alarm_type:
            continue
        # Parse timestamp — be lenient if format varies
        try:
            ts_str = record.get("timestamp", "")
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts < cutoff:
                continue
        except (ValueError, AttributeError):
            # Keep records with unparseable timestamps rather than dropping them
            pass
        results.append(record)

    # Sort newest first
    results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return results[:max_results]
