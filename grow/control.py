#!/usr/bin/env python3
"""The control the original work never ran.

Arm A: flat at the final architecture.  Arm B: grown to the same final
architecture.  Identical LR schedule, identical FLOP budget, identical data
order, N seeds each.  The original comparison changed four things at once
(growth + WSD + 3.3x LR + step count) and ran one seed, so it could not
attribute its result to growth.  This changes exactly one thing.

  uv run python grow/control.py --seeds 3 --flop-budget 4e14
"""

import argparse
import json
import statistics
from pathlib import Path

from train import build_parser, train

RUNS = Path(__file__).resolve().parent.parent / "runs"

# Same endpoint for both arms: 6L 8H 256D. Arm B reaches it by step 1000.
FINAL = {"start_dim": 256, "start_layers": 6}
START = {"start_dim": 128, "start_layers": 3}
# Re-running an identical config on MPS varies by this much from kernel
# nondeterminism alone. Differences below it are not results.
SEED_NOISE = 0.05
SCHEDULE = "200:width:192,400:depth,600:width:256,800:depth,1000:depth"


def make_cfg(name, seed, budget, **over):
    cfg = build_parser().parse_args([])
    cfg.name, cfg.seed, cfg.flop_budget = name, seed, budget
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--flop-budget", type=float, default=4e14)
    p.add_argument("--new-init", choices=["scaled", "zero"], default="scaled")
    args = p.parse_args()

    arms = {
        "flat": dict(**FINAL, schedule=""),
        "grown": dict(**START, schedule=SCHEDULE, new_init=args.new_init),
    }

    results = {}
    for arm, over in arms.items():
        results[arm] = [
            train(make_cfg(arm, s, args.flop_budget, **over)) for s in range(args.seeds)
        ]

    summary = {"flop_budget": args.flop_budget, "seeds": args.seeds, "arms": {}}
    for arm, runs in results.items():
        losses = [r["final_val_loss"] for r in runs]
        summary["arms"][arm] = {
            "losses": losses,
            "mean": statistics.mean(losses),
            "stdev": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            "steps": [r["steps_completed"] for r in runs],
            "final_params": runs[0]["final_params"],
        }

    a, b = summary["arms"]["flat"], summary["arms"]["grown"]
    delta = a["mean"] - b["mean"]
    # Pooled SD for equal n. The earlier sqrt(sa^2+sb^2) was the SD of the
    # difference, mislabelled -- it disagreed with web/build_data.py by exactly
    # sqrt(2), so the repo shipped two effect sizes for one dataset.
    pooled = max(((a["stdev"] ** 2 + b["stdev"] ** 2) / 2) ** 0.5, 1e-9)
    summary["delta_flat_minus_grown"] = delta
    summary["cohens_d"] = delta / pooled
    # At n=3 a rank statement is the strongest honest claim: complete separation
    # is Mann-Whitney U=0, one-tailed p=0.05, the minimum attainable at 3v3.
    # Anything finer than SEED_NOISE is kernel nondeterminism, not signal.
    summary["complete_separation"] = min(a["losses"]) > max(b["losses"])
    summary["verdict"] = (
        "grown wins — every seed" if summary["complete_separation"]
        else "flat wins — every seed" if max(a["losses"]) < min(b["losses"])
        else f"overlapping (delta {delta:+.3f}, seed noise ~{SEED_NOISE})"
    )

    (RUNS / "control_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 68)
    print(f"CONTROL @ {args.flop_budget:.2e} FLOPs, {args.seeds} seeds, identical WSD")
    print("=" * 68)
    for arm in ("flat", "grown"):
        s = summary["arms"][arm]
        print(
            f"  {arm:6s} val_loss {s['mean']:.4f} +/- {s['stdev']:.4f}   "
            f"steps {s['steps']}   {s['final_params']:,} params"
        )
    print(f"\n  delta (flat - grown) = {delta:+.4f}  ({summary['effect_size_sigma']:+.1f} sigma)")
    print(f"  VERDICT: {summary['verdict']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
