"""
Tool: get_device_info

Fetches device inventory details for a device_id from device_inventory.json.
Simulates what would be a CMDB lookup in production.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).parents[2] / "data" / "synthetic" / "device_inventory.json"
_index: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _index
    if _index is None:
        if not _DATA_PATH.exists():
            log.warning(
                "device_inventory.json not found at %s — returning empty inventory. "
                "Run data/synthetic/generate_synthetic_data.py first.",
                _DATA_PATH,
            )
            _index = {}
        else:
            with open(_DATA_PATH, encoding="utf-8") as f:
                records: list[dict] = json.load(f)
            _index = {r["device_id"]: r for r in records}
            log.debug("Loaded %d device inventory records from %s", len(_index), _DATA_PATH)
    return _index


def get_device_info(device_id: str) -> dict:
    """Return the inventory record for device_id.

    Args:
        device_id: Device identifier, e.g. 'RTR-OSL-042'.

    Returns:
        Device inventory dict, or empty dict if device not found.
    """
    index = _load()
    result = index.get(device_id, {})
    if not result:
        log.debug("Device '%s' not found in inventory.", device_id)
    return result
