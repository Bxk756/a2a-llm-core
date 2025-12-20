import os
import argparse
import torch
import uuid

from src.tokenizer import WordTokenizer
from src.model import GPT, ModelConfig
from src.memory import AgentMemory


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    block_size,
    temperature=0.6,
    top_k=20,
):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)

    return idx


def main():
    parser = argparse.ArgumentParser("A2A Agent with Memory")

    parser.add_argument("--ckpt", type=str, default="checkpoints/gpt.pt")
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=20)

    args = parser.parse_args()

    session_id = args.session or str(uuid.uuid4())
    print(f"Session: {session_id}")

    device = "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location="cpu")

    model_cfg = ckpt["cfg"]["model_cfg"]
    cfg = ModelConfig(**model_cfg)

    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tok = WordTokenizer("")
    tok.stoi = ckpt["cfg"]["stoi"]
    tok.itos = ckpt["cfg"]["itos"]
    tok.vocab = list(tok.stoi.keys())

    memory = AgentMemory(max_turns=8)

    # ---- Add user input to memory ----
    memory.add(session_id, "AGENT A", args.prompt)

    # ---- Build context ----
    context = memory.get_context(session_id)
    full_prompt = context + "\nAGENT B :"

    prompt_ids = torch.tensor(
        [tok.encode(full_prompt)],
        dtype=torch.long,
        device=device,
    )

    out_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.tokens,
        block_size=cfg.block_size,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    output_text = tok.decode(out_ids[0].tolist())

    # Extract last AGENT B response
    response = output_text.split("AGENT B :")[-1].strip()

    memory.add(session_id, "AGENT B", response)

    print("\n--- AGENT RESPONSE ---\n")
    print(response)
    print("\n----------------------\n")


if __name__ == "__main__":
    main()
