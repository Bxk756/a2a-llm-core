import os
import hashlib
from datetime import datetime, timezone
from fastapi import FastAPI, Header, HTTPException, Request

from src.policy_engine import PolicyEngine
from src.model import load_model, generate_action
from src.audit_logger import AuditLogger

API_KEY = os.getenv("API_KEY", "CHANGE_ME_SUPER_SECRET_KEY")
AUDIT_LOG_DIR = os.getenv("AUDIT_LOG_DIR", "logs")
AUDIT_RETENTION_DAYS = int(os.getenv("AUDIT_RETENTION_DAYS", "30"))

app = FastAPI(title="A2A Security Agent")

print("Loading model checkpoint...")
model = load_model()
print("Model loaded successfully.")

policy_engine = PolicyEngine("policy.yaml")

# AES key must be set: AUDIT_AES_KEY_B64
audit = AuditLogger(AUDIT_LOG_DIR, AUDIT_RETENTION_DAYS, "AUDIT_AES_KEY_B64")


def _hash_api_key(value: str) -> str:
    if not value:
        return "none"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def verify_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/audit/verify/today")
def audit_verify_today():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return audit.verify_day(today)


@app.get("/audit/verify/{day}")
def audit_verify_day(day: str):
    return audit.verify_day(day)


@app.post("/agent/decide")
async def decide(
    body: dict,
    request: Request,
    x_api_key: str = Header(...)
):
    verify_key(x_api_key)

    prompt = body.get("prompt", "")
    policy_result = policy_engine.evaluate(prompt)
    ai_action = generate_action(model, prompt)

    final = {
        "action": policy_result["action"],
        "severity": policy_result["severity"],
        "rule_id": policy_result["rule_id"],
        "notify": policy_result["notify"],
        "ai_suggested": ai_action,
    }

    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    audit.log_event(
        event_type="agent.decide",
        actor={
            "type": "api_key",
            "api_key_fingerprint": _hash_api_key(x_api_key),
            "client_ip": client_host,
            "user_agent": user_agent,
        },
        request={
            "path": str(request.url.path),
            "method": request.method,
            "prompt": prompt,
        },
        decision={"final": final},
        meta={"service": "a2a-llm-core"},
    )

    return final
