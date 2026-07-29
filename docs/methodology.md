# Methodology — why the experiment is built this way

The two design decisions that determine whether any number here means anything:
what we hold the budget in, and which hyperparameters we fix versus tune.

---

## 1. Why the budget is FLOPs

The research question is *"given compute C, what is the best shape for N(t)?"*
Compute means floating-point operations. So the budget must be FLOPs, and the
step count must be allowed to fall out of the trajectory shape.

The alternatives are not neutral — each answers a different question:

```
  BUDGET BY STEPS       both arms run N steps
  ─────────────────     the grown arm is smaller for part of training,
                        so it spends strictly LESS compute.
                        → you have handicapped it, and the question
                          becomes "is a small model as good per step
                          as a big one?" — known, and not what we asked.

  BUDGET BY WALL-CLOCK  both arms run T seconds
  ─────────────────     how many steps that buys depends on how well
                        your hardware packs small matmuls.
                        → measures the GPU, not the models
                        → not reproducible across machines
                        (this is what the inherited script did)

  BUDGET BY FLOPs       both arms spend C operations              ✅
  ─────────────────     holds the resource actually being spent.
                        Step count falls out: 2433 flat vs 2908 grown.
                        That +19% IS the mechanism under test, not a confound.
```

Implementation: `grow/train.py` accumulates `flops_per_token(model) * tokens`
each step and stops when the budget is exhausted. `flops_per_token` in
`grow/model.py` is `6·N + 12·L·T·d`, where `6·N` covers every matmul weight
(including the tied embedding/LM-head projection, counted once) and `12·L·T·d`
is the parameter-free attention score/value term.

**Honest caveat that belongs in any write-up:** at matched FLOPs the grown arm
is **6–11% slower in wall-clock** (145–148s vs 132–140s), because small matmuls
underutilise the device. The compute saving does not translate into time saved
at this scale. Reporting FLOPs, steps and wall-clock together is more honest
than reporting only the one that flatters the result.

---

## 2. Why "just fix the learning rate" was wrong

The instinct — hold everything constant, vary one thing — is correct as a
default and is exactly what produced our false result.

**Fixing a variable only controls it when its optimum does not interact with
the treatment.** When the optimum moves with the treatment, fixing it does not
neutralise it; it silently picks a winner.

```
  optimal LR
      │
  1e-3├──●  small model (3L 128D)         tolerates a large LR
      │    ╲
  5e-4│     ╲──●  large model (6L 256D)   destabilises at 1e-3
      │
      └──────────────────▶ model size
```

We ran both arms at **LR = 1e-3**, inherited from the growing preset.

- The **grown** arm starts at 3L/128D, so 1e-3 is near its optimum.
- The **flat** arm is 6L/256D from step 0, where 1e-3 is roughly twice its
  stable limit.

So the "controlled" learning rate was a setting borrowed from one arm and
imposed on the other. The consequence, measured:

| flat arm LR | seeds | mean | σ |
|---|---|---|---|
| 1e-3 (what we used) | 4.464, 5.088, 5.521 | 5.024 | **0.532** |
| 5e-4 (its own best) | 4.525, 4.496, 4.506 | 4.509 | **0.015** |

The σ of 0.532 was never a property of flat training. It was one run
half-diverging at a learning rate that model could not take. Our claim that
"growth buys learning-rate robustness, 102× more stable" inverted the causality
— at 5e-4 the flat arm is roughly **3.6× more stable than the grown arm**.

**The rule for a claim of the form "X beats Y": tune each arm independently and
compare best to best.** Anything else measures "X beats Y at Y's bad settings."

### What to fix and what to tune

```
  ┌────────────────────────┬──────────────────────────────────────┐
  │ FIX — no interaction   │ TUNE PER ARM — optimum moves with    │
  │ with the treatment     │ the treatment                        │
  ├────────────────────────┼──────────────────────────────────────┤
  │ dataset and data order │ learning rate            ← bit us    │
  │ evaluation set         │ warmup length                        │
  │ compute budget         │ (possibly batch size)                │
  │ number of seeds        │                                      │
  │ final architecture *   │  * only when the claim is explicitly │
  │                        │    conditional on that endpoint      │
  └────────────────────────┴──────────────────────────────────────┘
```

---

## 3. Why the fixed endpoint was also a design choice, not a neutral control

Both arms were pinned to finish at 6L/8H/256D (5.9M parameters), enforced by
`FINAL_ARCH` in `grow/experiment.py`. That felt like good practice — same
finish line, fair race.

But for a 4×10¹⁴ FLOP budget, the compute-optimal size is around 0.6M
parameters. The mandated endpoint is roughly **10× too large**. So the control
measures something real but narrow: *if you are forced to finish at a size much
larger than your budget wants, spending time smaller helps.*

Lift the constraint and a plain flat 1.2M model scores **3.905**, beating the
grown 5.9M trajectory's 4.111 at the identical budget.

The endpoint constraint also made the winning configuration structurally
unreachable by the AutoLab agent — `experiment.py` raises `SystemExit` on any
trajectory that does not end at 5.9M.

---

## 4. What this implies for the next experiment

1. **Sweep LR per arm.** Report each arm at its own optimum. Never share a
   learning rate across conditions whose optimum differs.
2. **Do not pin the endpoint** — or, if pinning it, state the claim as
   conditional on that endpoint and say why it was chosen.
3. **Seeds above the noise floor.** Re-running an identical config on MPS
   yields ~0.05 of variation from kernel nondeterminism alone, which equals the
   entire σ we reported for the grown arm. Three seeds is not enough to
   distinguish effects smaller than that; report two significant figures.
4. **Report FLOPs, steps and wall-clock together.**
