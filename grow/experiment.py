#!/usr/bin/env python3
"""AutoLab entry point: search the space of parameter trajectories N(t).

THE SEARCH SPACE IS `SCHEDULE` AND `START`. Edit those.

The question is not "what learning rate" -- it is what *shape* the parameter
count should trace during training. Standard training fixes N(t) flat and never
justified it. Every arm here ends at the same architecture (6L 8H 256D) and
spends the same FLOP budget; only the path there differs. A model that is small
early buys more optimizer steps with the same compute, and the schedule decides
how that trade is made.

Constraints that make the number meaningful -- do not remove them:
  * The budget is FLOPs, never wall-clock and never steps.
  * Every run ends at the same final architecture as FLAT_REFERENCE.
  * The objective is the mean over >=3 seeds. Single-seed results at this scale
    are noise: seed sigma is ~0.01-0.05 and the effects under test are smaller.

Reference points measured by grow/control.py at this exact budget are in
runs/control_summary.json. Beat the `flat` arm's mean.
"""

import statistics

from train import build_parser, train

# --- the search space -------------------------------------------------------
# "step:action[:arg]" -- width takes a target hidden dim, depth appends a block.
SCHEDULE = "200:width:192,400:depth,600:width:256,800:depth,1000:depth"
START = {"start_dim": 128, "start_layers": 3}
# ----------------------------------------------------------------------------

FLOP_BUDGET = 4e14
SEEDS = 3
FINAL_ARCH = "6L 8H 256D"  # every trajectory must terminate here


def main():
    losses, runs = [], []
    for seed in range(SEEDS):
        cfg = build_parser().parse_args([])
        cfg.name, cfg.seed, cfg.flop_budget = "experiment", seed, FLOP_BUDGET
        cfg.schedule = SCHEDULE
        for k, v in START.items():
            setattr(cfg, k, v)
        r = train(cfg)
        if r["final_arch"] != FINAL_ARCH:
            raise SystemExit(
                f"trajectory ended at {r['final_arch']}, must end at {FINAL_ARCH} "
                "-- the comparison is only fair between equal final architectures"
            )
        losses.append(r["final_val_loss"])
        runs.append(r)

    mean = statistics.mean(losses)
    sd = statistics.stdev(losses) if len(losses) > 1 else 0.0
    print(f"\nschedule: {SCHEDULE}")
    print(f"start:    {START}")
    print(f"steps:    {[r['steps_completed'] for r in runs]}")
    print(f"seeds:    {[round(x, 4) for x in losses]}  mean {mean:.4f} +/- {sd:.4f}")
    print(f"OBJECTIVE {mean:.5f}")


if __name__ == "__main__":
    main()
