import uuid
import io
import csv
import torch
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.tokenizer import WordTokenizer
from src.model import GPT, ModelConfig
from src.memory import AgentMemory

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cpu"
CKPT_PATH = "checkpoints/gpt.pt"
API_KEY = "CHANGE_ME_SUPER_SECRET_KEY"

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="A2A Shield Agent API",
    description="Security-focused AI agent with persistent memory, actions, and audit export",
    version="1.3.0",
)

# ============================================================
# SECURITY
# ============================================================

def require_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ============================================================
# LOAD MODEL (ONCE)
# ============================================================

print("Loading model checkpoint...")

ckpt = torch.load(CKPT_PATH, map_location="cpu")
model_cfg = ckpt["cfg"]["model_cfg"]

cfg = ModelConfig(**model_cfg)
model = GPT(cfg).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

tok = WordTokenizer("")
tok.stoi = ckpt["cfg"]["stoi"]
tok.itos = ckpt["cfg"]["itos"]
tok.vocab = list(tok.stoi.keys())

# Persistent SQLite memory
memory = AgentMemory(db_path="memory.db", max_turns=8)

print("Model loaded successfully.")

# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate(idx, max_new_tokens=80, temperature=0.6, top_k=20):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

    return idx

# ============================================================
# PARSING
# ============================================================

def extract_agent_b_response(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if "AGENT B :" in text:
        resp = text.split("AGENT B :", 1)[1].strip()
        return resp if resp else text
    return text


def extract_final_action(response: str) -> Optional[str]:
    """
    Deterministic action extraction:
    If multiple actions appear, choose the LAST one found.
    """
    if not response:
        return None

    last_action = None
    for a in ["ALLOW", "BLOCK", "LOG", "ESCALATE"]:
        if f"ACTION : {a}" in response:
            last_action = a
    return last_action

# ============================================================
# API MODELS
# ============================================================

class StartSessionResponse(BaseModel):
    session_id: str


class AgentRequest(BaseModel):
    session_id: Optional[str] = None
    prompt: str
    temperature: Optional[float] = 0.6
    top_k: Optional[int] = 20
    max_tokens: Optional[int] = 80


class AgentResponse(BaseModel):
    session_id: str
    response: str
    action: Optional[str]


class AuditSessionResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]


class AuditSessionsResponse(BaseModel):
    sessions: List[Dict[str, Any]]

# ============================================================
# ROUTES
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session/start", response_model=StartSessionResponse)
def start_session(dep=Depends(require_api_key)):
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.post("/session/reset", response_model=StartSessionResponse)
def reset_session(session_id: str, dep=Depends(require_api_key)):
    memory.reset(session_id)
    return {"session_id": session_id}


@app.post("/agent/decide", response_model=AgentResponse)
def agent_decide(req: AgentRequest, dep=Depends(require_api_key)):
    session_id = req.session_id or str(uuid.uuid4())

    # ---- Store AGENT A input ----
    memory.add(session_id, "AGENT A", req.prompt)

    # ---- Build context from SQLite ----
    context = memory.get_context(session_id)
    full_prompt = context + "\nAGENT B :"

    prompt_ids = torch.tensor(
        [tok.encode(full_prompt)],
        dtype=torch.long,
        device=DEVICE,
    )

    out_ids = generate(
        prompt_ids,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )

    decoded = tok.decode(out_ids[0].tolist())

    response = extract_agent_b_response(decoded)
    if not response:
        response = decoded.strip()

    # ---- Persist AGENT B output ----
    memory.add(session_id, "AGENT B", response)

    action = extract_final_action(response)

    return {
        "session_id": session_id,
        "response": response,
        "action": action,
    }

# ============================================================
# AUDIT ENDPOINTS
# ============================================================

@app.get("/audit/sessions", response_model=AuditSessionsResponse)
def audit_list_sessions(limit: int = 100, dep=Depends(require_api_key)):
    sessions = memory.list_sessions(limit=limit)
    return {"sessions": sessions}


@app.get("/audit/session/{session_id}", response_model=AuditSessionResponse)
def audit_get_session(session_id: str, dep=Depends(require_api_key)):
    messages = memory.fetch_messages(session_id=session_id, limit=None, ascending=True)
    return {"session_id": session_id, "messages": messages}


@app.get("/audit/session/{session_id}/export.csv")
def audit_export_csv(session_id: str, dep=Depends(require_api_key)):
    messages = memory.fetch_messages(session_id=session_id, limit=None, ascending=True)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "session_id", "role", "content", "ts"])
    for m in messages:
        writer.writerow([m["id"], m["session_id"], m["role"], m["content"], m["ts"]])

    output.seek(0)

    filename = f"a2a_audit_{session_id}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers=headers,
    )
