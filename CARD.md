# Sundai card — paste into sundai.club/pitch

**Title:** GrowLab — a model that grows its parameters while it learns

**One-liner:**
Scaling laws tell you how big to make a model. They never asked what shape it
should be *over time*. We let an AI agent search that space — and it found
schedules that beat training at full size from step 0.

**The pitch (card body):**

Chinchilla asks: given compute *C*, choose *N* and *D*. It assumes the parameter
count *N* is **constant** for the whole run — and never argued for it. Training
compute is the *area under the parameter-count curve*, so we asked the obvious
next question:

> **Given compute *C*, choose the trajectory N(t).** The answer is a shape, not a number.

A model that starts small buys more optimizer steps for the same compute, because
early training learns cheap statistics that need almost no capacity.

**What we ran today** — the control the prior work skipped. Flat vs grown at
identical FLOP budget, identical LR schedule, identical data order, same final
architecture, 3 seeds each:

| | val loss | steps bought |
|---|---|---|
| flat (constant N) | 5.024 ± 0.532 | 2433 |
| **grown N(t)** | **4.111 ± 0.053** | **2908** |

1. **Growth wins with complete separation** — every grown seed beats every flat seed.
2. **Growth is 102× more stable across seeds.** At this learning rate, starting at
   full size is unstable and starting small isn't. Growth buys *learning-rate
   robustness*, acting as a curriculum — not just compute.

**Then we handed the search to an agent.** The space is brutal (when to grow ×
how much × width vs depth × how to init × LR around each event). Six hand-run
points is not a search. **Autolab**'s agent reproduced our hand-built schedule
exactly (4.111), then **beat it (4.076)** by shifting growth 50 steps earlier,
and is now bracketing the optimum — autonomously, while we built the frontend.
**Maritime** hosts the live dashboard so the demo doesn't depend on a laptop.

**We also audited our own headline.** The result we started from claimed "same
loss at 54% of the FLOPs." Decomposed, that was 60% fewer steps × 90% smaller on
average — growth was doing **9.7%** of the work, and every run used a single
seed. We found three more problems and fixed them before trusting anything:
FLOP-budgeting instead of wall-clock; a 4k vocab (at 50k vocab the embedding was
91% of params, so growth could only touch ~9% of compute); and the growth
operator itself, which zeroed new attention weights — reproducing the *softmax
barrier* inside the very direction meant to escape it.

**Links**
- Live dashboard: `<MARITIME_URL>`
- Code: https://github.com/qsimeon/growlab
- Agent's search: https://app.autolab.ai/projects/qsimeon/growlab

**Built with:** Autolab (autonomous research loop) · Maritime (hosting) · PyTorch/MPS

---

## Demo script (5 min, no slides)

1. **The question** (45s) — show the two N(t) shapes on the dashboard. "Compute is
   the area under this curve. Everyone trains the flat one. Nobody argued for it."
2. **The control** (90s) — scroll to the loss chart. Orange under blue the whole
   way. "Same FLOPs, same LR, same data, same final architecture. Every grown seed
   beats every flat seed, and it's 102× steadier."
3. **The honesty beat** (45s) — "We started from a 54%-FLOPs claim. We audited it:
   n=1, four things changed at once, and the vocab was eating 91% of the compute.
   Here's the version that survives."
4. **The agent** (90s) — scroll to the Autolab panel, live. "We gave the search to
   an agent. It reproduced our schedule, then beat it. It's still running."
5. **Close** (30s) — "The compute-optimal parameter trajectory is not flat. Now
   there's a harness that can find its shape."
