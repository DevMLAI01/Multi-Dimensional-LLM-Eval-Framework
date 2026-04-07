"""
Node: context_fetcher

Calls all three agent tools in parallel (using threads) to fetch:
  - alarm_history  (last 30 days for this device)
  - device_info    (inventory record)
  - runbook        (matching runbook for alarm type)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agent.tools.get_device_info import get_device_info
from agent.tools.query_alarm_history import query_alarm_history
from agent.tools.search_runbooks import search_runbooks

log = logging.getLogger(__name__)


def context_fetcher(state: dict[str, Any]) -> dict[str, Any]:
    """Fetch all contextual data for the alarm in parallel.

    Reads from state:
        alarm_event: AlarmEvent

    Writes to state:
        alarm_history: list[dict]
        device_info:   dict
        runbook:       dict
    """
    alarm = state["alarm_event"]

    tasks = {
        "alarm_history": lambda: query_alarm_history(
            device_id=alarm.device_id,
            alarm_type=alarm.alarm_type,
            days_back=30,
            max_results=10,
        ),
        "device_info": lambda: get_device_info(alarm.device_id),
        "runbook": lambda: search_runbooks(alarm.alarm_type),
    }

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_key = {executor.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
                log.debug("context_fetcher[%s]: fetched %s items", key, _count(results[key]))
            except Exception as exc:
                log.error("context_fetcher[%s] failed: %s", key, exc)
                results[key] = [] if key == "alarm_history" else {}

    return {
        "alarm_history": results.get("alarm_history", []),
        "device_info": results.get("device_info", {}),
        "runbook": results.get("runbook", {}),
    }


def _count(value: Any) -> str:
    if isinstance(value, list):
        return str(len(value))
    if isinstance(value, dict):
        return f"{len(value)} keys" if value else "empty"
    return str(value)
