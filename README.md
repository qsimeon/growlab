# GrowLab

**A language model that grows its own parameters during training** — starting small and earning capacity as it learns, instead of being instantiated at full size from step 0.

> Scaling laws ask: given compute *C*, choose *N* and *D*. They assume the parameter count *N* is **constant** for the whole run — and never argued for it.
>
> **Given compute *C*, choose the trajectory N(t).** The answer is a shape, not a number.

Training compute is the area under the parameter-count curve. A model that starts small buys more optimizer steps for the same compute, because early training learns cheap statistics that need almost no capacity.

```
   N                                    N
   │                                    │
 5.9M├━━━━━━━━━━━━━━━━┓            5.9M │           ┏━━━━━━━━┓
   │████ FLAT ███████ ┃                 │      ┏━━━━┛████████┃
   │████████████████  ┃                 │ ┏━━━━┛██ GROWN ████┃
 1.2M│                ┃            1.2M ┛██████████████████  ┃
   └──────────────────────▶ t           └──────────────────────▶ t
      C = area under the curve             same C, different shape
```

## What we found today

Flat vs grown at **identical FLOP budget, identical data order, same final architecture, 3 seeds each**, each arm at its own best learning rate:

| arm | val loss | seeds | steps bought |
|---|---|---|---|
| flat (constant N) | 4.509 ± 0.015 | 4.525, 4.496, 4.506 | 2433 |
| **grown** | **4.111** ± 0.053 | 4.080, 4.172, 4.082 | **2908** |

**The claim that survives:** if you must reach a given model size, **growing into it beats starting there** — 0.40 nats, every grown seed beating every flat seed. Being small early buys 19% more steps for the same compute.

### Then we audited our own result, and two claims died

Our first measurement was a 0.91-nat gap and "102× more stable". Both were artifacts:

1. **The flat arm was mis-tuned.** It ran at LR=1e-3, inherited from the grown preset. At its own best LR (5e-4) its seed spread collapses from 0.532 to **0.015** — *more* stable than grown — and the gap halves to 0.40.
2. **The endpoint was ~10× too large.** Both arms were pinned to 5.9M params, while compute-optimal for a 4e14 budget is ≈0.6M. Lift that constraint and a plain flat **1.2M model scores 3.905**, beating the grown trajectory outright.

So growth wins **when the target size is fixed and larger than the budget wants** — it is not compute-optimal in general. Per-run nondeterminism on Apple Silicon is ~0.05, so treat the last two decimals as noise.

## An agent searches the space

The search space is brutal — when to grow × how much × width vs depth × how to init new weights × optimizer state × LR around each event. Six hand-run points is not a search. [Autolab](https://autolab.ai)'s agent proposes, runs and merges growth schedules autonomously against the objective. It reproduced our hand-built schedule exactly (4.111), then **beat it (4.076)** by shifting growth 50 steps earlier, and is now bracketing the optimum.

[Maritime](https://maritime.sh) hosts the public dashboard, so the demo doesn't depend on a laptop staying open.

```
  grow/ ──▶ autolab agent ──▶ runs/*.jsonl ──▶ web/ ──▶ maritime (public URL)
   │         proposes N(t)      trajectories    dashboard
   └─ FLOP-budgeted training loop, growth schedule as a parameter
```

## Three fixes that made the number mean anything

The inherited headline was *"matches baseline at 54% of FLOPs."* It decomposes as 60% fewer steps × 90% smaller on average — **growth was doing 9.7% of the work**, and every run used a single seed.

1. **Budget by FLOPs, not wall-clock.** Step count now falls out of the trajectory shape, which is the mechanism under test.
2. **4,096-token BPE.** The tied embed/lm_head matmul scales linearly in *d*; the body scales as *d²*. At V=50,304 and d=128 the embedding was **91% of params**, so growth could only ever touch ~9% of the budget. At V=4,096 the body is 80%.
3. **Fixed the growth operator.** The inherited `grow_width` zeroed every new dimension — so new attention heads got Q,K=0 → uniform softmax → zero gradient. That is the *softmax barrier* reproduced inside the direction meant to escape it. New weights now get real magnitude.

See [`context/provenance.md`](context/provenance.md) for the full lineage — why one-shot expansion died, and why growing a *mid-training* model escapes the trap that killed it.

## Run it

```bash
uv sync
uv run python grow/data.py                       # 4k BPE over WikiText-103
uv run python grow/control.py --seeds 3          # the control
uv run python web/build_data.py && open web/index.html
```

Train along an arbitrary trajectory:

```bash
uv run python grow/train.py --schedule "200:width:192,400:depth,600:width:256" \
    --flop-budget 4e14 --seed 0
```

Built at Sundai Hack #133. See `context/idea.md` for the science, `PLAN.md` for the build.
