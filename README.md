# GrowLab

**A language model that grows its own parameters during training** — starting from a handful of neurons and earning capacity as it learns, instead of being instantiated at full size from step 0.

Training compute is the area under the parameter-count curve. Standard training holds that curve flat. Prior work here found a growth schedule that reaches the same final size and loss under roughly **half the area** — but with confounds that were never controlled for.

So this project does two things at once:

- **Runs the control** that prior work skipped — grown vs. flat, same LR schedule, same FLOP budget.
- **Searches the space** of growth trajectories with [Autolab](https://autolab.ai)'s autonomous research agent, live, hosted on [Maritime](https://maritime.sh).

The underlying question: scaling laws optimize parameters `N` and data `D` *assuming N is constant.* Nobody justified that. **What is the optimal trajectory N(t)?** The answer is a shape, not a number — and searching that space by hand is exactly why this stalled before.

Built at Sundai Hack #133. See `context/idea.md` for the science, `PLAN.md` for the build.
