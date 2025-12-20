from typing import Dict, List
import yaml
import os


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_policy_with_packs(config_path: str, packs_dir: str) -> Dict:
    config = load_yaml(config_path)

    enabled = set(config.get("enabled_packs", []))
    rules: List[Dict] = []

    for fname in os.listdir(packs_dir):
        if not fname.endswith(".yaml"):
            continue

        pack = load_yaml(os.path.join(packs_dir, fname))
        pack_name = pack.get("pack")

        if pack_name in enabled:
            rules.extend(pack.get("rules", []))

    return {
        "version": config.get("version", 1),
        "mode": config.get("mode", "audit"),
        "decision_thresholds": config.get("decision_thresholds", {}),
        "rules": rules,
    }
