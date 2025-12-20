import yaml
from typing import Dict


class PolicyEngine:
    def __init__(self, policy_path: str):
        with open(policy_path, "r") as f:
            data = yaml.safe_load(f)
        self.policies = data.get("policies", [])

    def evaluate(self, text: str) -> Dict:
        text_lower = text.lower()

        for policy in self.policies:
            conditions = policy.get("match", {}).get("any", [])
            for condition in conditions:
                keyword = condition.get("contains", "").lower()
                if keyword and keyword in text_lower:
                    return {
                        "action": policy["action"],
                        "severity": policy["severity"],
                        "rule_id": policy["id"],
                        "notify": policy["notify"],
                    }

        # Safe default
        return {
            "action": "LOG",
            "severity": "info",
            "rule_id": "DEFAULT_LOG",
            "notify": False,
        }
