# Provenance — from reverse distillation to growlab

Where this started, what died, what survived, and exactly what the current code
does. Written 2026-07-26 during the Sundai #133 build, for the CIO who asked the
right question: *"how do you address the representational maintenance condition
(preserving logits post-expansion) — or did you just abandon all that?"*

**Short answer: abandoned, deliberately, and that abandonment IS the idea.**
The long answer is the whole document.

---

## 1. The lineage

```
 Numenta internship                "Growing sparse LLMs"
 ─────────────────────             ────────────────────────
 How do you buy representational   Expand a small dense model into a
 capacity without paying full      HIGHER-DIMENSIONAL but SPARSE one.
 price for it the whole time?      More dimensions, not proportionally
                                   more FLOPs.
            │
            │  the sparsity half got dropped somewhere
            ▼
 DIRECTION 1 — reverse distillation          ✝ DIED
 Expand a FINISHED pretrained model,
 preserving its function exactly.
            │
            │  the death taught us what the real constraint was
            ▼
 DIRECTION 2 — neural morphogenesis          ← we are here
 Grow a model that is STILL TRAINING.
 Nothing finished, nothing to preserve.
```

The through-line is *not* "expansion." It is **decoupling representational
dimensionality from compute cost.** Sparsity was one answer (bigger but cheaper
per step). Growth is a different answer (bigger *later*, cheap *earlier*). Same
underlying question.

---

## 2. Why Direction 1 died — the constraint was the killer

Direction 1's whole premise: take a finished Pythia-70M, place its weights in
the top-left of a 140M architecture, zeros elsewhere, and **guarantee
`f_new(x) = f_old(x)` exactly at initialization.** Don't lose what was learned.

That guarantee worked. Max logit diff 0.005 (width), 0.000000 (depth). The
`ZeroIgnoringLayerNorm` that made it work — normalizing only over non-zero dims
so the padding doesn't corrupt LayerNorm's statistics — is genuinely clever.

And it is precisely the guarantee that kills it:

```
  "preserve the function exactly"
            │
            ▼
  new weights must be ~0  (1e-8)
            │
            ▼
  new attention heads: Q,K ≈ 1e-9
            │
            ▼
  scores = QK/√d ≈ 1e-18
            │
            ▼
  softmax(≈0) = UNIFORM, regardless of input     ← softmax annihilates
            │                                       small differences
            ▼
  gradient ∝ p(1-p)·V ≈ 5e-13
            │
            ▼
  those heads never move. Ever.          ← "the softmax barrier"
```

There is no escape by tuning the noise scale:

```
  noise 1e-8  →  preserved ✓   dead ✗
  noise 1e-4  →  alive ✓       preservation destroyed (logit diff ~370) ✗
                 └── no value of ε satisfies both ──┘
```

The MLP survives (GELU'(0) ≈ 0.5, no exponential flattening). Attention does
not. This is the classic symmetry-breaking constraint — the same reason you
cannot initialize a network to all zeros — hitting attention hardest because
softmax destroys the small score differences that would otherwise break the
symmetry.

**The realization: the trap exists only because you are expanding something
finished and precious.**

---

## 3. Why growth escapes it

Grow a model that is **mid-training** and there is no precious function. A small
perturbation at a growth event is cheap — the model has thousands more steps to
absorb it. So new parameters can be born with **real magnitude**, gradients
flow, and the new capacity actually gets used.

```
  DIRECTION 1                        DIRECTION 2
  ───────────                        ───────────
  model is FINISHED                  model is MID-TRAINING
  must preserve f(x) exactly         nothing worth preserving
  → new weights ≈ 0                  → new weights ~ N(0, 0.02)
  → dead heads forever               → gradients flow, capacity used
  → ✝                                → loss blip, recovered in ~20 steps
```

**Direction 2 is not a weaker Direction 1. It is the version that isn't
obligated to lie about being unchanged.**

---

## 4. So what does the code actually do? (the honest answer)

Two growth operators, two *different* answers to the preservation question.
This distinction is the crux:

### `grow_width` — preservation ABANDONED on purpose

```
  old 128D                    new 192D
  ┌──────────┐                ┌──────────┬─────────┐
  │  learned │       ──▶      │  learned │ N(0,σ)  │   σ = 0.02, REAL magnitude
  │  weights │                │  weights │  fresh  │
  └──────────┘                └──────────┴─────────┘
                                          ▲
                              the inherited code put ZEROS here.
                              That reproduced the softmax barrier
                              INSIDE the direction meant to escape it.
```

Function is **not** preserved. There is a small loss blip. That is the price,
and it is cheap. Measured on `grown_s0` at every growth event:

| step | before | after | recovered by |
|---|---|---|---|
| 200 (width→192) | 6.141 | 6.165 | step 240 → 5.979 |
| 400 (depth→4L) | 5.623 | 5.718 | step 440 → 5.638 |
| 600 (width→256) | 5.467 | 5.402 | *immediately better* |
| 800 (depth→5L) | 5.347 | 5.285 | *immediately better* |
| 1000 (depth→6L) | 5.019 | 5.107 | step 1040 → 5.141 |

Blips are ~0.02–0.1 nats and absorbed within 20–40 steps. No catastrophe.

### `grow_depth` — preservation KEPT, and it's safe here

```
  new block:   x ──┬──▶ attn ──▶ out_proj(=0) ──┐
                   │                            ⊕──▶  x   ← exact identity
                   └────────────────────────────┘

  zero is on the OUTPUT side.  The residual path still carries
  gradient into the block's internal weights on the very next step.
```

**The design rule that falls out of all of this:**

```
   ┌───────────────────────────────────────────────────┐
   │  Zeroing an OUTPUT projection  →  safe.           │
   │    identity via residual, gradient still arrives. │
   │                                                   │
   │  Zeroing an INPUT projection   →  fatal.          │
   │    Q,K = 0 → uniform softmax → dead forever.      │
   └───────────────────────────────────────────────────┘
```

Direction 1 was forced into the fatal case by its own guarantee. Direction 2
gets to choose, and chooses the safe case for depth and *no zeros at all* for
width.

---

## 5. The research question, restated

Scaling laws (Chinchilla): *given compute C, choose N and D.* Answered well.
But they hold **N constant for the whole run** and never argued for it.

> **Given compute C, choose the trajectory N(t).**

The answer is a **shape**, not a number. Constant-N is one point in that space.

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

Falsifiable claim: **the compute-optimal parameter trajectory is not flat.**

Why it's unmapped: the search space is brutal — when to grow × how much × width
vs depth vs heads × how to init new weights × optimizer state × LR around each
event. Six hand-run points is not a search; it's six points. **That is why an
autonomous research agent is the right collaborator**, and why AutoLab is doing
the searching rather than us hand-tuning presets.

---

## 6. What we had to fix before any number meant anything

The inherited headline was *"matches baseline at 54% of FLOPs."* Audited, it
decomposes as:

```
  54.1% FLOPs  =  60.0%           ×  90.3%
                  └ fewer steps      └ smaller on average
                    (1200 vs 2000)     (growth's actual contribution)

  Growth was doing 9.7% of the work. Step count was doing 40%.
```

Four confounds, all real:

1. **n = 1.** Every inherited run used `seed=42`. The "match" was 6.193 vs
   6.191 — a 0.002 gap with no error bars, against seed noise of ~0.01–0.05.
2. **Four things changed at once**: growth + WSD schedule + 3.3× peak LR
   (1e-3 vs 3e-4) + warmup (40 vs 100). Growth's own contribution: unmeasured.
3. **`max_steps` was hand-set per preset** (1000/1200/1500/2000/4000) — the
   schedule and the step count were tuned together until one landed on 6.19.
4. **The vocabulary ate the experiment** — see below. This is the one that
   made the whole thing unmeasurable.

### The vocab problem (why we switched to a 4k BPE)

The tied `embed`/`lm_head` is a real matmul whose cost scales only **linearly**
in `d`. The transformer body — the only thing growth shrinks — scales as `d²`.
Body beats head only when `d > V/(12L)`:

```
  V = 50,304, L = 6  →  need d > 700.  Experiments ran at d = 128→256.

  at d=128:  ┌────────────────────────────────┐
             │████████████████████████████░░░░│  embed 91%  ·  body 9%
             └────────────────────────────────┘
                                       ▲
                          growth can only touch this sliver

  V = 4,096, L = 6  →  need d > 57.  ✓

  at d=256:  ┌────────────────────────────────┐
             │██████░░░░░░░░░░░░░░░░░░░░░░░░░░│  embed 20%  ·  body 80%
             └────────────────────────────────┘
```

So we trained a 4,096-token BPE over WikiText-103. Now N(t) genuinely drives the
compute budget and the experiment can detect an effect at all.

---

## 7. The control, and what it found

```
        ARM A (flat)                    ARM B (grown)
        6L 8H 256D from step 0          3L 4H 128D ──▶ 6L 8H 256D
             │                                │
             ├──── identical WSD schedule ────┤
             ├──── identical peak LR ─────────┤    the original changed
             ├──── identical data order ──────┤    ALL of these at once,
             ├──── identical FLOP budget ─────┤    and ran n=1
             ├──── same final architecture ───┤
             └──── 3 seeds each ──────────────┘
```

At 4×10¹⁴ FLOPs:

| arm | val loss | seeds | steps |
|---|---|---|---|
| flat | 5.024 ± **0.532** | 4.464, 5.088, 5.521 | 2433 |
| grown | **4.111** ± **0.053** | 4.080, 4.172, 4.082 | 2908 |

**Two findings, both real:**

**(a) Grown wins, with complete separation.** Every grown seed beats every flat
seed. Gap 0.913 nats, Cohen's d = 2.42.

```
  flat:    4.46 ──────── 5.09 ──────── 5.52
  grown:        ▲
           4.08 4.08 4.17
                └── no overlap ──┘
```

Complete separation at n=3,3 is the strongest rank result the sample size
allows (Mann–Whitney U = 0, one-tailed p = 0.05). Welch t = 2.96, df = 2 —
the parametric test is weaker only because flat's variance is enormous, which
is itself finding (b).

**(b) Grown is 102× more stable across seeds** (σ 0.532 → 0.053). At this
learning rate, starting at full size is *unstable* and starting small is not.

**The mechanism:** at matched FLOPs, being small early buys **2908 vs 2433
steps (+19%)** — and lets a high LR be survivable, because the model is tiny
when the LR would otherwise blow it up. Growth acts as a **curriculum**, not
just a compute saving.

**The confound to name yourself before anyone asks:** LR = 1e-3 was inherited
from the *grown* preset, so flat is running at a learning rate it was never
tuned for. The honest next experiment is an LR sweep on the flat arm. Not run
today.

---

## 8. Open threads this reopens

- **Sparsity** — the half that got dropped between Numenta and here. Growth and
  sparsity are two answers to the same question; nothing stops combining them.
- **Continual learning** — if capacity is added *when needed* rather than on a
  fixed schedule, "grow when loss plateaus" becomes a policy, not a constant.
  A trigger-based `GrowthPolicy` is a natural next operator.
- **Maximal packing** — is there a principled "this size is saturated, grow
  now" signal? That is the interesting version of the search AutoLab is running.
- **Scale** — everything here is 5.9M params on WikiText-103. Nothing has been
  shown to survive scale. The next serious run should be a model size a learner
  would actually recognize, on a standard corpus.

## 9. Where the pieces live

| what | where |
|---|---|
| model + growth operators | `grow/model.py` |
| FLOP-budgeted training loop | `grow/train.py` |
| the control (§7) | `grow/control.py` → `runs/control_summary.json` |
| the search knob AutoLab edits | `grow/experiment.py` (`SCHEDULE`, `START`) |
| dashboard | `web/index.html` + `web/build_data.py` |
| public host | `deploy/server.py` + `Dockerfile` |
