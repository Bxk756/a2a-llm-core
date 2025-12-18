import os
import time
import math
import argparse
import random

import torch

from src.tokenizer import WordTokenizer
from src.model import GPT, ModelConfig


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_checkpoint(path, model, optimizer, step, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "cfg": cfg,
        },
        path,
    )


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt.get("step", 0), ckpt.get("cfg", {})


def batchify(encoded_ids, block_size, batch_size, device):
    if len(encoded_ids) < block_size + 1:
        raise ValueError("Not enough tokens for given block_size")

    ix = torch.randint(0, len(encoded_ids) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.tensor(encoded_ids[i : i + block_size], dtype=torch.long) for i in ix]
    )
    y = torch.stack(
        [torch.tensor(encoded_ids[i + 1 : i + 1 + block_size], dtype=torch.long) for i in ix]
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data_ids, block_size, batch_size, eval_iters, device):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = batchify(data_ids, block_size, batch_size, device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Train GPT from scratch (word-level)")

    # data
    parser.add_argument("--data_path", type=str, default="data/a2a_finetune.txt")
    parser.add_argument("--val_split", type=float, default=0.05)

    # model
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # logging / eval
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=25)

    # checkpoint
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_name", type=str, default="gpt.pt")
    parser.add_argument("--resume", action="store_true")

    # misc
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force_cpu", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.force_cpu)
    print(f"Device: {device}")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Missing {args.data_path}")

    text = read_text(args.data_path)
    print(f"Loaded text: {len(text)} characters")

    # ---- Tokenizer (WORD-LEVEL) ----
    tok = WordTokenizer(text)
    vocab_size = tok.vocab_size
    print(f"Vocab size: {vocab_size}")

    ids = tok.encode(text)

    split = int(len(ids) * (1.0 - args.val_split))
    train_ids = ids[:split]

    val_ids = ids[split:]
    if len(val_ids) < args.block_size + 1:
        val_ids = None
        print("Validation skipped: dataset too small for block_size")

    print(f"Tokens: train={len(train_ids)} val={(len(val_ids) if val_ids else 0)}")

    # ---- Model ----
    cfg = ModelConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    start_step = 0

    if args.resume and os.path.exists(ckpt_path):
        start_step, _ = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resumed from step {start_step}")

    t0 = time.time()
    running_loss = 0.0

    for step in range(start_step + 1, args.max_steps + 1):
        xb, yb = batchify(train_ids, args.block_size, args.batch_size, device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        running_loss += loss.item()

        if step % args.log_interval == 0:
            dt = time.time() - t0
            avg_loss = running_loss / args.log_interval
            running_loss = 0.0
            tok_per_sec = (args.batch_size * args.block_size * args.log_interval) / max(dt, 1e-9)
            print(f"step {step:4d} | loss {avg_loss:.4f} | tok/s {tok_per_sec:,.0f}")
            t0 = time.time()

        if step % args.eval_interval == 0 or step == args.max_steps:
            train_loss = estimate_loss(
                model, train_ids, args.block_size, args.batch_size, args.eval_iters, device
            )

            if val_ids is not None:
                val_loss = estimate_loss(
                    model, val_ids, args.block_size, args.batch_size, args.eval_iters, device
                )
                ppl = math.exp(min(20, val_loss))
                print(f"[eval] step {step} | train {train_loss:.4f} | val {val_loss:.4f} | ppl {ppl:.2f}")
            else:
                print(f"[eval] step {step} | train {train_loss:.4f} | val skipped")

            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                step,
                {
                    "stoi": tok.stoi,
                    "itos": tok.itos,
                    "model_cfg": cfg.__dict__,
                },
            )
            print(f"Saved checkpoint → {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
