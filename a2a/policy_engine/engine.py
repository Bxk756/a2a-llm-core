from __future__ import annotations

import re
import ipaddress
from dataclasses import dataclass
from typing import Any, Dict, List

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
# Utility helpers
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
    return str(v)


def _any_of(value: Any, options: List[Any]) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        v = value.lower()
        return any(isinstance(o, str) and o.lower() == v for o in options)
    return value in options


def _contains_any(haystack: Any, needles: List[str]) -> bool:
    h = _as_str(haystack).lower()
    return any(n.lower() in h for n in needles)


def _equals(value: Any, target: Any) -> bool:
    if isinstance(value, str) and isinstance(target, str):
        return value.lower() == target.lower()
    return value == target


def _regex(value: Any, pattern: str) -> bool:
    try:
        return re.search(pattern, _as_str(value), flags=re.IGNORECASE) is not None
    except re.error:
        return False


def _in_cidr(ip: str, cidrs: List[str]) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip)
        return any(ip_obj in ipaddress.ip_network(c, strict=False) for c in cidrs)
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

    if "lt" in cond:
        try:
            return float(val) < float(cond["lt"])
        except (TypeError, ValueError):
            return False

    if "exists" in cond:
        exists = val is not None and _as_str(val) != ""
        return exists if cond["exists"] else not exists

    if "in_cidr" in cond:
        return _in_cidr(_as_str(val), cond["in_cidr"])

    if "not_in_cidr" in cond:
        return _not_in_cidr(_as_str(val), cond["not_in_cidr"])

    return False


def _eval_when(event: A2AEvent, when: Dict[str, Any]) -> bool:
    if "all" in when:
        return all(_eval_condition(event, c) for c in when["all"])
    if "any" in when:
        return any(_eval_condition(event, c) for c in when["any"])
    return False


# ----------------------------
# Engine
# ----------------------------
class PolicyEngine:
    def __init__(self, policy: Dict[str, Any]):
        self.policy = policy or {}
        self.rules = self.policy.get("rules", [])
        self.thresholds = {
            **DEFAULT_THRESHOLDS,
            **(self.policy.get("decision_thresholds") or {}),
        }

    def evaluate(self, event: A2AEvent) -> Dict[str, Any]:
        matches: List[RuleMatch] = []
        score = 0

        # Evaluate rules
        for rule in self.rules:
            when = rule.get("when", {})
            if _eval_when(event, when):
                pts = int(rule.get("points", 0))
                score += pts
                matches.append(
                    RuleMatch(
                        rule_id=rule.get("id", "UNKNOWN_RULE"),
                        severity=rule.get("severity", "low"),
                        points=pts,
                        action=rule.get("action", "LOG"),
                        reason=rule.get("reason", ""),
                    )
                )

        # Base action by score
        action = self._decide_action(score)

        # Hard overrides
        if any(m.action.upper() == "BLOCK" for m in matches):
            action = "BLOCK"
        elif any(m.action.upper() == "ESCALATE" for m in matches) and action in (
            "ALLOW",
            "LOG",
        ):
            action = "ESCALATE"

        # ----------------------------
        # AUDIT vs ENFORCE CONTROL
        # ----------------------------
        mode = self.policy.get("mode", "audit")

        final_action = action
        if mode == "audit":
            final_action = "LOG"

        findings = [m.rule_id for m in matches]

        return {
            "risk_score": score,
            "action": final_action,        # what happens now
            "enforcement_action": action,  # what WOULD happen in enforce
            "mode": mode,
            "findings": findings,
            "matches": [m.__dict__ for m in matches],
        }

    def _decide_action(self, score: int) -> str:
        t = self.thresholds
        if score >= t["block_min"]:
            return "BLOCK"
        if score <= t["allow_max"]:
            return "ALLOW"
        if score <= t["log_max"]:
            return "LOG"
        if score <= t["escalate_max"]:
            return "ESCALATE"
        return "ESCALATE"
