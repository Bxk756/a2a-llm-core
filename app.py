from a2a.policy_engine.loader import load_policy_with_packs

POLICY = load_policy_with_packs(
    "policies/config.yaml",
    "policies/packs"
)
ENGINE = PolicyEngine(POLICY)
