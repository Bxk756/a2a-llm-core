from __future__ import annotations

from typing import Any, Dict
import yaml


def _set_field(obj: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = obj
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_mapping(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def map_event(raw: Dict[str, Any], mapping: Dict[str, Any], event_type: str = "") -> Dict[str, Any]:
    """
    Converts tool-specific fields (e.g., SentinelOne) into the normalized A2A event schema.
    """
    out: Dict[str, Any] = {
        "host": {},
        "process": {},
        "file": {},
        "registry": {},
        "network": {},
        "meta": {
            "source": mapping.get("source", "unknown"),
            "event_type": event_type or raw.get("event_type", ""),
            "timestamp": raw.get("timestamp", ""),
        },
    }

    field_map = mapping.get("field_map", {})
    for src_field, dst_field in field_map.items():
        if src_field in raw and raw[src_field] is not None:
            _set_field(out, dst_field, raw[src_field])

    # Normalization / defaults
    # Ensure ports are ints if present
    for k in ("dst_port", "src_port"):
        v = out.get("network", {}).get(k)
        if v is not None and v != "":
            try:
                out["network"][k] = int(v)
            except (TypeError, ValueError):
                pass

    # Guess direction if not provided
    # (You can improve this later: if src_ip is private and dst_ip is public => outbound)
    if "direction" not in out["network"]:
        out["network"]["direction"] = raw.get("direction", "")

    return out
