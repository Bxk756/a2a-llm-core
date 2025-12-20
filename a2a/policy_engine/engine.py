from __future__ import annotations

import re
import ipaddress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ----------------------------
# Types
# ----------------------------
A2AEvent = Dict[str, Any]


@dataclass
class RuleMatch:
    rule_id: str
    severity: str
    points: int
    action: str
    reason: str


DEFAULT_THRESHOLDS = {
    "allow_max": 29,
    "log_max": 69,
    "escalate_max": 89,
    "block_min": 90,
}


# ----------------------------
# Utilities
# ----------------------------
def _get_field(event: A2AEvent, dotted: str) -> Any:
    cur: Any = event
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float, bool)):
        return str(v)
    return str(v)


def _contains_any(haystack: Any, needles: List[str]) -> bool:
    h = _as_str(haystack).lower()
    return any(n.lower() in h for n in needles)


def _any_of(value: Any, options: List[Any]) -> bool:
    if value is None:
        return False
    # compare case-insensitive for strings
    if isinstance(value, str):
        v = value.lower()
        return any(isinstance(o, str) and o.lower() == v for o in options)
    return value in options


def _equals(value: Any, target: Any) -> bool:
    if isinstance(value, str) and isinstance(target, str):
        return value.lower() == target.lower()
    return value == target


def _regex(value: Any, pattern: str) -> bool:
    v = _as_str(value)
    try:
        return re.search(pattern, v, flags=re.IGNORECASE) is not None
    except re.error:
        return False


def _in_cidr(ip: str, cidrs: List[str]) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip)
        for c in cidrs:
            if ip_obj in ipaddress.ip_network(c, strict=False):
                return True
        return False
    except ValueError:
        return False


def _not_in_cidr(ip: str, cidrs: List[str]) -> bool:
    if not ip:
        return True
    return not _in_cidr(ip, cidrs)


# ----------------------------
# Condition evaluation
# ----------------------------
def _eval_condition(event: A2AEvent, cond: Dict[str, Any]) -> bool:
    field = cond.get("field")
    if not field:
        return False

    val = _get_field(event, field)

    if "any_of" in cond:
        return _any_of(val, cond["any_of"])

    if "equals" in cond:
        return _equals(val, cond["equals"])

    if "contains_any" in cond:
        return _contains_any(val, cond["contains_any"])

    if "contains" in cond:
        return cond["contains"].lower() in _as_str(val).lower()

    if "regex" in cond:
        return _regex(val, cond["regex"])

    if "gt" in cond:
        try:
            return float(val) > float(cond["gt"])
        except (TypeError, ValueError):
            return False

    if "gte" in cond:
        try:
            return float(val) >= float(cond["gte"])
        except (TypeError, ValueError):
            return False

    if "lt" in cond:
        try:
            return float(val) < float(cond["lt"])
        except (TypeError, ValueError):
            return False

    if "lte" in cond:
        try:
            return float(val) <= float(cond["lte"])
        except (TypeError, ValueError):
            return False

    if "not_in_cidr" in cond:
        ip = _as_str(val)
        return _not_in_cidr(ip, cond["not_in_cidr"])

    if "in_cidr" in cond:
        ip = _as_str(val)
        return _in_cidr(ip, cond["in_cidr"])

    if "exists" in cond:
        exists = val is not None and _as_str(val) != ""
        return exists if bool(cond["exists"]) else not exists

    return False


def _eval_when(event: A2AEvent, when: Dict[str, Any]) -> bool:
    if not when:
        return False

    if "all" in when:
        return all(_eval_condition(event, c) for c in when["all"])

    if "any" in when:
        return any(_eval_condition(event, c) for c in when["any"])

    return False


# ----------------------------
# Policy loading
# ----------------------------
def load_policy(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ----------------------------
# Engine
# ----------------------------
class PolicyEngine:
    def __init__(self, policy: Dict[str, Any]):
        self.policy = policy or {}
        self.thresholds = (self.policy.get("decision_thresholds") or DEFAULT_THRESHOLDS).copy()
        self.rules = self.policy.get("rules") or []

    def evaluate(self, event: A2AEvent) -> Dict[str, Any]:
        matches: List[RuleMatch] = []
        score = 0

        for rule in self.rules:
            when = rule.get("when") or {}
            if _eval_when(event, when):
                pts = int(rule.get("points", 0))
                score += pts
                matches.append(
                    RuleMatch(
                        rule_id=str(rule.get("id", "UNKNOWN_RULE")),
                        severity=str(rule.get("severity", "low")),
                        points=pts,
                        action=str(rule.get("action", "LOG")),
                        reason=str(rule.get("reason", "")),
                    )
                )

        # Decide final action based on score, then allow "hard override" if any rule says BLOCK
        action = self._decide_action(score)

        # If any matched rule explicitly requests BLOCK, take it.
        if any(m.action.upper() == "BLOCK" for m in matches):
            action = "BLOCK"
        elif any(m.action.upper() == "ESCALATE" for m in matches) and action in ("ALLOW", "LOG"):
            action = "ESCALATE"

        findings = [m.rule_id for m in matches]

        return {
            "risk_score": score,
            "action": action,
            "findings": findings,
            "matches": [m.__dict__ for m in matches],
        }

    def _decide_action(self, score: int) -> str:
        t = self.thresholds
        if score >= int(t.get("block_min", 90)):
            return "BLOCK"
        if score <= int(t.get("allow_max", 29)):
            return "ALLOW"
        if score <= int(t.get("log_max", 69)):
            return "LOG"
        if score <= int(t.get("escalate_max", 89)):
            return "ESCALATE"
        return "ESCALATE"
