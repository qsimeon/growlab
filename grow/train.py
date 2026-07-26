#!/usr/bin/env python3
"""Train a model along a parameter trajectory N(t), under a fixed FLOP budget.

The budget is FLOPs, not wall-clock and not steps. That is the whole point: to
compare trajectory shapes fairly you have to hold compute constant and let the
step count fall out of the shape. A model that is small early gets more steps
for the same spend; that is the mechanism under test, not a confound.

The trajectory is a parameter -- `--schedule "80:width:192,160:depth"` -- so the
search space is exposed rather than baked into a preset.

Run:
  uv run python grow/train.py --schedule "80:width:192,160:depth,240:width:256" \
      --flop-budget 1.2e15 --seed 0
Prints `OBJECTIVE <val_loss>` on the last line for the harness to parse.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from data import VOCAB_SIZE, TokenLoader, prepare
from model import GrowableGPT, apply_growth, count_params, flops_per_token

RUNS = Path(__file__).resolve().parent.parent / "runs"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_schedule(spec):
    """'80:width:192,160:depth' -> [(80, {...}), (160, {...})]"""
    if not spec:
        return []
    out = []
    for item in spec.split(","):
        parts = item.strip().split(":")
        step, action = int(parts[0]), parts[1]
        if action == "width":
            out.append((step, {"action": "width", "target_hidden": int(parts[2])}))
        elif action == "depth":
            out.append((step, {"action": "depth"}))
        else:
            raise ValueError(f"unknown action {action!r} in {item!r}")
    return sorted(out)


def wsd_lr(step, total, warmup, decay_fraction):
    """Warmup-stable-decay. Applied identically to every arm of the control."""
    if step < warmup:
        return step / max(1, warmup)
    decay_steps = max(1, int(total * decay_fraction))
    stable_end = total - decay_steps
    if step < stable_end:
        return 1.0
    return max(0.0, 1.0 - (step - stable_end) / decay_steps)


@torch.no_grad()
def evaluate(model, loader, device, n_batches=20):
    model.eval()
    total, n = 0.0, 0
    for x in loader.eval_batches(n_batches):
        x = x.to(device)
        _, loss = model(x, labels=x)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def estimate_total_steps(cfg, device):
    """Predict step count for the LR schedule without spending the budget.

    The LR schedule needs to know its own horizon up front, but the horizon
    depends on the trajectory. Simulate N(t) analytically -- no training -- and
    solve for how many steps the budget buys.
    """
    model = GrowableGPT(
        VOCAB_SIZE, cfg.start_dim, cfg.start_layers, cfg.start_dim // cfg.head_dim, cfg.head_dim
    ).to(device)
    sched = dict(parse_schedule(cfg.schedule))
    tokens = cfg.batch_size * cfg.seq_len
    spent, step = 0, 0
    while spent < cfg.flop_budget and step < 100_000:
        if step in sched:
            model = apply_growth(model, sched[step], device, cfg.new_init)
        spent += flops_per_token(model, cfg.seq_len) * tokens
        step += 1
    del model
    return step


def train(cfg):
    device = get_device()
    torch.manual_seed(cfg.seed)
    prepare(cfg.seq_len)

    total_steps = estimate_total_steps(cfg, device)
    print(f"[{cfg.name}] budget {cfg.flop_budget:.3e} FLOPs -> ~{total_steps} steps on {device}")

    torch.manual_seed(cfg.seed)
    model = GrowableGPT(
        VOCAB_SIZE, cfg.start_dim, cfg.start_layers, cfg.start_dim // cfg.head_dim, cfg.head_dim
    ).to(device)
    # Data order is seeded separately so every arm sees identical batches.
    train_loader = TokenLoader("train", cfg.batch_size, cfg.seq_len, seed=1234)
    val_loader = TokenLoader("val", cfg.batch_size, cfg.seq_len, seed=0)

    sched = dict(parse_schedule(cfg.schedule))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    RUNS.mkdir(exist_ok=True)
    tag = f"{cfg.name}_s{cfg.seed}"
    jsonl = open(RUNS / f"{tag}.jsonl", "w")

    tokens_per_step = cfg.batch_size * cfg.seq_len
    spent, step, t0 = 0, 0, time.time()
    growth_events, val_history = [], []
    last_growth_step = -10**9

    while spent < cfg.flop_budget:
        if step in sched:
            model = apply_growth(model, sched[step], device, cfg.new_init)
            # Optimizer state is stale after a shape change; rebuild it.
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            last_growth_step = step
            growth_events.append([step, count_params(model), model.arch()])
            print(f"  step {step}: grew -> {model.arch()} ({count_params(model):,} params)")

        # Base WSD, plus a short re-warmup after each growth event.
        scale = wsd_lr(step, total_steps, cfg.warmup, cfg.decay_fraction)
        since = step - last_growth_step
        if 0 <= since < cfg.warmup_after_growth:
            scale *= (since + 1) / cfg.warmup_after_growth
        for g in opt.param_groups:
            g["lr"] = cfg.lr * scale

        x = train_loader.batch().to(device)
        _, loss = model(x, labels=x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        spent += flops_per_token(model, cfg.seq_len) * tokens_per_step
        step += 1

        if step % cfg.log_every == 0 or step == 1:
            rec = {
                "step": step,
                "params": count_params(model),
                "arch": model.arch(),
                "train_loss": round(loss.item(), 5),
                "cum_flops": spent,
                "lr": round(cfg.lr * scale, 7),
            }
            if step % cfg.eval_every == 0:
                rec["val_loss"] = round(evaluate(model, val_loader, device), 5)
                val_history.append([step, rec["val_loss"]])
            jsonl.write(json.dumps(rec) + "\n")
            jsonl.flush()

    final_val = evaluate(model, val_loader, device, n_batches=50)
    result = {
        "name": cfg.name,
        "seed": cfg.seed,
        "final_val_loss": final_val,
        "final_params": count_params(model),
        "final_arch": model.arch(),
        "steps_completed": step,
        "flops_spent": spent,
        "flop_budget": cfg.flop_budget,
        "wall_time": time.time() - t0,
        "growth_events": growth_events,
        "val_history": val_history,
        "config": vars(cfg),
    }
    jsonl.close()
    (RUNS / f"{tag}.json").write_text(json.dumps(result, indent=2, default=str))
    print(
        f"[{cfg.name}] steps={step} params={count_params(model):,} "
        f"flops={spent:.3e} wall={result['wall_time']:.0f}s"
    )
    print(f"OBJECTIVE {final_val:.5f}")
    return result


def build_parser():
    p = argparse.ArgumentParser(description="Train along a parameter trajectory N(t).")
    p.add_argument("--name", default="run")
    p.add_argument("--schedule", default="", help='e.g. "80:width:192,160:depth"')
    p.add_argument("--flop-budget", type=float, default=1.2e15)
    p.add_argument("--start-dim", type=int, default=128)
    p.add_argument("--start-layers", type=int, default=3)
    p.add_argument("--head-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--warmup-after-growth", type=int, default=10)
    p.add_argument("--decay-fraction", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--new-init", choices=["scaled", "zero"], default="scaled")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=50)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
