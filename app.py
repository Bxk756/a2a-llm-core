from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

from a2a.policy_engine.engine import PolicyEngine, load_policy
from a2a.policy_engine.mapper import load_mapping, map_event

app = FastAPI(title="A2A Policy Engine", version="1.0.0")

POLICY = load_policy("policies/base.yaml")
ENGINE = PolicyEngine(POLICY)

S1_MAPPING = load_mapping("mappings/sentinelone.yaml")

class EvaluateRequest(BaseModel):
    source: str = "sentinelone"   # for now we support sentinelone mapping
    event_type: str = "process"
    raw: Dict[str, Any]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    if req.source.lower() != "sentinelone":
        return {
            "error": "Unsupported source mapping",
            "supported": ["sentinelone"],
        }

    event = map_event(req.raw, S1_MAPPING, event_type=req.event_type)
    result = ENGINE.evaluate(event)

    # Return both JSON decision + action token text
    action_text = (
        f"RISK_SCORE: {result['risk_score']}\n"
        f"FINDINGS: {', '.join(result['findings']) if result['findings'] else 'none'}\n"
        f"ACTION : {result['action']}\n"
    )

    return {
        "normalized_event": event,
        "decision": result,
        "action_text": action_text,
    }
