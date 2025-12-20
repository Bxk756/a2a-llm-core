from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

from a2a.policy_engine.mapper import load_mapping, map_event
from a2a.policy_engine.manager import PolicyManager

# ----------------------------
# App
# ----------------------------
app = FastAPI(
    title="A2A Shield Policy Engine",
    description="Hot-Reloading AI Security Policy Engine",
    version="1.1.0",
)

# ----------------------------
# Hot-Reloading Policy Manager
# ----------------------------
policy_manager = PolicyManager(
    config_path="policies/config.yaml",
    packs_dir="policies/packs",
    poll_interval=1.0,  # seconds
)

# ----------------------------
# Load mappings
# ----------------------------
SENTINELONE_MAPPING = load_mapping("mappings/sentinelone.yaml")


# ----------------------------
# Request model
# ----------------------------
class EvaluateRequest(BaseModel):
    source: str = "sentinelone"
    event_type: str = "process"
    raw: Dict[str, Any]


# ----------------------------
# Health / Status
# ----------------------------
@app.get("/health")
def health():
    engine = policy_manager.get_engine()
    return {
        "ok": True,
        "mode": engine.policy.get("mode"),
        "rules_loaded": len(engine.rules),
        "hot_reload": True,
        "policy_status": policy_manager.status(),
    }


# ----------------------------
# Evaluate endpoint
# ----------------------------
@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    if req.source.lower() != "sentinelone":
        return {
            "error": "Unsupported source",
            "supported_sources": ["sentinelone"],
        }

    engine = policy_manager.get_engine()

    event = map_event(
        raw=req.raw,
        mapping=SENTINELONE_MAPPING,
        event_type=req.event_type,
    )

    result = engine.evaluate(event)

    action_text = (
        f"MODE: {result['mode']}\n"
        f"RISK_SCORE: {result['risk_score']}\n"
        f"FINDINGS: {', '.join(result['findings']) if result['findings'] else 'none'}\n"
        f"ACTION : {result['action']}\n"
    )

    if result["mode"] == "audit":
        action_text += f"ENFORCEMENT_ACTION : {result['enforcement_action']}\n"

    return {
        "normalized_event": event,
        "decision": result,
        "action_text": action_text,
    }
