#!/usr/bin/env python3
"""Corrected experiment: at fixed compute, does the SHAPE of N(t) matter?

Two design errors in the earlier control (docs/methodology.md) are fixed here:

  1. LR is swept PER ARM. Every configuration is reported at its own best LR,
     never at one borrowed from a different arm.
  2. The final architecture is NOT pinned. Flat baselines at three constant
     sizes locate the compute-optimal flat point empirically; grown arms end at
     two different endpoints so growth is judged both at a matched endpoint and
     against the best flat model overall.

Phase 1 sweeps LR at seed 0 with a small process pool -- val loss is unaffected
by contention, wall-clock is, so phase-1 timings are flagged `contended` and not
reported. Phase 2 runs the remaining seeds at each arm's best LR SERIALLY, so
the reported wall-clock is clean. Any run in runs/ whose stored config matches
the requested one exactly (every hyperparameter, not just the name) is reused.

  uv run python grow/sweep.py
  uv run python grow/sweep.py --one flat_2p7M --lr 1e-3 --seed 0
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from train import build_parser, train

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

BUDGET = float(os.environ.get("GROWLAB_BUDGET", 4e14))  # override only for smoke tests
NOISE = 0.05  # MPS kernel nondeterminism: re-running an identical config moves ~this much
SEEDS = 3
LR_GRID = [5e-4, 1e-3, 2e-3]
# One adaptive point if a config's optimum lands on an edge of the grid.
EDGE_EXTENSION = {5e-4: 2.5e-4, 2e-3: 4e-3}
MAX_PHASE2_RUNS = 12  # compute cap; tie-break LRs beyond it are dropped and recorded

# Growth events are placed at ~1/3 of the arm's expected step count in both
# grown arms, so the two trajectories spend a comparable FRACTION of training
# below their final size (grown_5p9M: 1000/2908 steps; grown_2p7M: 2000/~6000).
CONFIGS = {
    "flat_1p2M": dict(start_dim=128, start_layers=3, schedule=""),
    "flat_2p7M": dict(start_dim=192, start_layers=4, schedule=""),
    "flat_5p9M": dict(start_dim=256, start_layers=6, schedule=""),
    "grown_2p7M": dict(start_dim=128, start_layers=3, schedule="1000:width:192,2000:depth"),
    "grown_5p9M": dict(
        start_dim=128,
        start_layers=3,
        schedule="200:width:192,400:depth,600:width:256,800:depth,1000:depth",
    ),
}

# Hyperparameters that must match for a stored run to be reusable. `name`,
# `seed`, and the two logging knobs are excluded: they cannot change the result.
COMPARE_KEYS = [
    "schedule", "flop_budget", "start_dim", "start_layers", "head_dim", "lr",
    "weight_decay", "warmup", "warmup_after_growth", "decay_fraction",
    "batch_size", "seq_len", "new_init",
]


def make_cfg(config, lr, seed):
    cfg = build_parser().parse_args([])
    cfg.name = f"x_{config}_lr{lr:g}"
    cfg.seed = seed
    cfg.flop_budget = BUDGET
    cfg.lr = lr
    for k, v in CONFIGS[config].items():
        setattr(cfg, k, v)
    return cfg


def load_existing():
    """Every finished run in runs/, indexed by its exact hyperparameters."""
    out = []
    for path in sorted(RUNS.glob("*.json")):
        try:
            r = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "config" in r and "final_val_loss" in r:
            r["_path"] = path.name
            out.append(r)
    return out


def find_match(existing, cfg, seed):
    want = vars(cfg)
    for r in existing:
        c = r["config"]
        if r.get("seed") != seed:
            continue
        if all(str(c.get(k)) == str(want[k]) for k in COMPARE_KEYS):
            return r
    return None


def record(r, contended, source):
    return {
        "loss": round(r["final_val_loss"], 5),
        "seed": r["seed"],
        "steps": r["steps_completed"],
        "flops": r["flops_spent"],
        "wall_s": round(r["wall_time"], 1),
        "params": r["final_params"],
        "arch": r["final_arch"],
        "contended": contended,
        "source": source,
    }


CONTENDED = {}  # (config, lr, seed) -> did this run share the device with another


def run_pool(jobs, workers):
    """Run (config, lr, seed) jobs as subprocesses, `workers` at a time."""
    for j in jobs:
        CONTENDED[j] = workers > 1
    queue, live = list(jobs), []
    while queue or live:
        while queue and len(live) < workers:
            config, lr, seed = queue.pop(0)
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--one", config, "--lr", repr(lr), "--seed", str(seed)]
            print(f"  launch {config} lr={lr:g} seed={seed}", flush=True)
            live.append((subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE), config, lr, seed))
        time.sleep(2)
        for item in list(live):
            proc, config, lr, seed = item
            if proc.poll() is not None:
                live.remove(item)
                if proc.returncode != 0:
                    err = proc.stderr.read().decode()[-800:]
                    print(f"  FAILED {config} lr={lr:g} seed={seed}\n{err}", flush=True)
                else:
                    print(f"  done   {config} lr={lr:g} seed={seed}", flush=True)


def collect(config, lr, seed, existing, _=None):
    """Look up a finished run. Prior runs not launched here were serial."""
    m = find_match(existing, make_cfg(config, lr, seed), seed)
    if m is None:
        return None
    return record(m, CONTENDED.get((config, lr, seed), False), m["_path"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    t_start = time.time()

    # ---- phase 1: LR sweep at seed 0 --------------------------------------
    existing = load_existing()
    todo = []
    for config in CONFIGS:
        for lr in LR_GRID:
            if find_match(existing, make_cfg(config, lr, 0), 0) is None:
                todo.append((config, lr, 0))
    print(f"PHASE 1: LR sweep, {len(todo)} runs to do "
          f"({len(CONFIGS) * len(LR_GRID) - len(todo)} reused)", flush=True)
    run_pool(todo, args.workers)

    # ---- phase 1b: extend the grid where the optimum sits on an edge -------
    existing = load_existing()
    grids = {c: list(LR_GRID) for c in CONFIGS}

    def seed0(config):
        return {lr: collect(config, lr, 0, existing, True) for lr in grids[config]}

    ext = []
    for config in CONFIGS:
        pts = {lr: r["loss"] for lr, r in seed0(config).items() if r}
        edge = min(pts, key=pts.get)
        if edge in EDGE_EXTENSION:
            ext.append((config, EDGE_EXTENSION[edge], 0))
    ext = [e for e in ext if find_match(existing, make_cfg(*e), 0) is None]
    if ext:
        print(f"PHASE 1b: {len(ext)} edge extensions", flush=True)
        run_pool(ext, args.workers)
    for c, lr, _ in ext:
        grids[c] = sorted(set(grids[c]) | {lr})

    existing = load_existing()
    sweep, cands = {}, {}
    for config in CONFIGS:
        sweep[config] = {f"{lr:g}": r for lr, r in seed0(config).items()}
        pts = {lr: r["loss"] for lr, r in seed0(config).items() if r}
        order = sorted(pts, key=pts.get)
        # Picking the LR from a single seed is itself a coin-flip when two LRs
        # are within the noise floor, so carry the runner-up into phase 2 and
        # decide on the 3-seed mean instead.
        cands[config] = [lr for lr in order[:2] if pts[lr] <= pts[order[0]] + NOISE]
        print(f"  {config}: candidates {[f'{x:g}' for x in cands[config]]}  "
              + "  ".join(f"{k:g}={v:.3f}" for k, v in sorted(pts.items())), flush=True)

    # ---- phase 2: remaining seeds for every candidate LR, run serially -----
    existing = load_existing()
    todo, dropped = [], []
    for config in CONFIGS:
        for i, lr in enumerate(cands[config]):
            jobs = [(config, lr, s) for s in range(SEEDS)
                    if find_match(existing, make_cfg(config, lr, s), s) is None]
            if i and len(todo) + len(jobs) > MAX_PHASE2_RUNS:
                dropped.append([config, lr])  # tie-break we could not afford
                continue
            todo += jobs
    cands = {c: [lr for lr in v if [c, lr] not in dropped] for c, v in cands.items()}
    print(f"PHASE 2: {len(todo)} seed runs (serial, clean wall-clock)"
          + (f"; dropped tie-breaks {dropped}" if dropped else ""), flush=True)
    run_pool(todo, workers=1)

    # ---- summarise --------------------------------------------------------
    existing = load_existing()
    arms, best = {}, {}
    for config in CONFIGS:
        by_lr = {}
        for lr in cands[config]:
            runs = [collect(config, lr, s, existing, True) for s in range(SEEDS)]
            runs = [r for r in runs if r]
            if runs:
                by_lr[lr] = runs
        best[config] = min(by_lr, key=lambda lr: statistics.mean(r["loss"] for r in by_lr[lr]))
        runs = by_lr[best[config]]
        losses = [r["loss"] for r in runs]
        clean = [r["wall_s"] for r in runs if not r["contended"]]
        arms[config] = {
            "best_lr": best[config],
            "losses": losses,
            "mean": round(statistics.mean(losses), 4),
            "sd": round(statistics.stdev(losses), 4) if len(losses) > 1 else 0.0,
            "n_seeds": len(losses),
            "steps": runs[0]["steps"],
            "flops": runs[0]["flops"],
            "final_params": runs[0]["params"],
            "final_arch": runs[0]["arch"],
            "wall_s_serial": round(statistics.mean(clean), 1) if clean else None,
            "lrs_given_seeds": [f"{lr:g}" for lr in by_lr],
            "mean_by_lr": {f"{lr:g}": round(statistics.mean(r["loss"] for r in rs), 4)
                           for lr, rs in by_lr.items()},
            "runs": runs,
        }

    flats = {k: v for k, v in arms.items() if k.startswith("flat")}
    best_flat = min(flats, key=lambda k: flats[k]["mean"])
    comparisons = {}
    for grown in [k for k in arms if k.startswith("grown")]:
        matched = "flat_" + grown.split("_", 1)[1]
        for label, ref in (("matched_endpoint", matched), ("best_flat_overall", best_flat)):
            d = arms[ref]["mean"] - arms[grown]["mean"]  # >0 means grown is better
            comparisons[f"{grown}_vs_{ref}_{label}"] = {
                "grown_mean": arms[grown]["mean"],
                "flat_mean": arms[ref]["mean"],
                "delta_flat_minus_grown": round(d, 4),
                "exceeds_noise_floor": abs(d) > NOISE,
                "verdict": ("grown better" if d > NOISE else
                            "flat better" if d < -NOISE else "within noise"),
            }

    summary = {
        "question": "at fixed compute C, does the shape of N(t) matter?",
        "flop_budget": BUDGET,
        "seeds": SEEDS,
        "lr_grid": LR_GRID,
        "noise_floor": NOISE,
        "noise_note": (
            "re-running an identical config on MPS moves the val loss by ~0.05 from kernel "
            "nondeterminism alone; differences below that are not interpretable, and all "
            "numbers should be read at 2 significant figures"
        ),
        "wall_clock_note": (
            "phase-1 LR-sweep runs shared the device (contended=true) so their wall-clock is "
            "inflated; wall_s_serial is averaged over uncontended runs only"
        ),
        "lr_selection_rule": (
            "best LR by seed-0 loss; any LR within the noise floor of it at seed 0 also gets "
            "the full seed set and the arm is reported at whichever has the best 3-seed mean"
        ),
        "configs": CONFIGS,
        "sweep_seed0": sweep,
        "arms": arms,
        "best_flat": best_flat,
        "comparisons": comparisons,
        "total_driver_seconds": round(time.time() - t_start, 1),
    }
    (RUNS / "corrected_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 78)
    print(f"CORRECTED EXPERIMENT @ {BUDGET:.1e} FLOPs, {SEEDS} seeds, LR tuned per arm")
    print("=" * 78)
    print(f"{'config':<12}{'lr':>8}{'params':>10}{'mean':>8}{'sd':>7}{'steps':>8}{'wall_s':>9}")
    for k, a in arms.items():
        print(f"{k:<12}{a['best_lr']:>8.1e}{a['final_params']:>10,}{a['mean']:>8.3f}"
              f"{a['sd']:>7.3f}{a['steps']:>8}{str(a['wall_s_serial']):>9}")
    print(f"\nbest flat: {best_flat} ({arms[best_flat]['mean']:.3f})")
    for k, c in comparisons.items():
        print(f"  {k}: delta {c['delta_flat_minus_grown']:+.3f} -> {c['verdict']}")
    print("=" * 78)


if __name__ == "__main__":
    if "--one" in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument("--one")
        p.add_argument("--lr", type=float)
        p.add_argument("--seed", type=int)
        a = p.parse_args()
        train(make_cfg(a.one, a.lr, a.seed))
    else:
        main()
