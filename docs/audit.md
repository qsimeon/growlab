# Audit — every claim we made, re-tested

We spent the last hours of the build attacking our own result instead of
polishing it. This document records what we had claimed, what we did to check
it, and what was left standing. Two of the three headline claims did not
survive. The one that did survived in a considerably narrower form than we first
stated it.

The rule this document is written under: **a number stays in only if a specific
re-run supports it, and the caveat goes next to the number, not in a footnote.**

---

## 1. What we originally claimed

From `runs/control_summary.json`, three seeds per arm, 4×10¹⁴ FLOPs, identical
WSD schedule, identical data order, both arms ending at 6L/8H/256D:

| arm | val loss | seeds | steps |
|---|---|---|---|
| flat | 5.024 ± **0.532** | 4.4638, 5.0876, 5.5212 | 2433 |
| grown | **4.111** ± 0.053 | 4.0798, 4.1721, 4.0817 | 2908 |

Three claims came out of that table:

1. **A 0.913-nat gap** in favour of growth, with complete seed separation.
2. **Growth is "102× more stable"** across seeds (variance ratio 0.532²/0.053²).
3. **The compute-optimal parameter trajectory is not flat.**

---

## 2. Claim 1 — the gap. Halved, but it survives.

Both arms ran at **LR = 1e-3**, a value inherited from the *grown* preset. The
grown arm starts at 3L/128D, where 1e-3 is near optimal; the flat arm is
6L/256D from step 0, where 1e-3 is roughly twice its stable limit. So the
"controlled" learning rate was one arm's setting imposed on the other. We
re-ran the flat arm at its own best LR, 5e-4, three seeds:

```
  flat @ 1e-3    4.4638   5.0876   5.5212      mean 5.024   σ 0.532
  flat @ 5e-4    4.5248   4.4955   4.5057      mean 4.509   σ 0.015
  grown @ 1e-3   4.0798   4.1721   4.0817      mean 4.111   σ 0.053

                4.0        4.5         5.0         5.5
                 │          │           │           │
  grown          ●●  ●
  flat @ 5e-4               ●●●
  flat @ 1e-3               ●            ●           ●
                            └── that spread was one run half-diverging,
                                not a property of flat training
```

The honest comparison is best-versus-best: **4.509 vs 4.111, a gap of 0.397
nats** — 43% of what we first reported. Complete seed separation still holds:
the worst grown seed (4.1721) still beats the best flat seed (4.4955), with no
overlap. At n = 3,3 that is the strongest rank result the sample size permits
(Mann–Whitney U = 0, one-tailed p = 0.05).

**Verdict: the direction and the separation survive. The magnitude was inflated
by a factor of ~2.3 by a mis-tuned baseline.**

## 3. Claim 2 — "102× more stable". Inverted.

This one did not shrink; it reversed sign.

| | σ across 3 seeds |
|---|---|
| flat @ 1e-3 (what we reported) | 0.532 |
| flat @ 5e-4 (its own best LR) | **0.015** |
| grown @ 1e-3 | 0.053 |

At each arm's own learning rate the **flat arm is ~3.6× more stable than the
grown arm** (variance ratio ~12.6 the other way). Our claim that "growth buys
learning-rate robustness" was reading the causality backwards: what we measured
was one flat seed partially diverging at a learning rate that model could not
take.

**Verdict: dead, and it was dead in the most embarrassing way — it was an
artifact of our own experimental design, and we had presented it as the second
finding.**

## 4. Claim 3 — "the compute-optimal trajectory is not flat". Falsified as stated.

Both arms were pinned to end at 6L/8H/256D (5.9M parameters) by `FINAL_ARCH` in
`grow/experiment.py`. Same finish line, fair race — that was the reasoning. But
for a 4×10¹⁴ FLOP budget the compute-optimal model size is around 0.6M
parameters, so the mandated endpoint is roughly **10× larger than the budget
wants**.

Lift the constraint and run a plain flat model at a sensible size:

| run | arch | params | steps | val loss |
|---|---|---|---|---|
| grown (our result) | 6L 8H 256D | 5.90M | 2908 | 4.111 |
| flat, unconstrained | 3L 4H 128D | 1.18M | **11811** | **3.905** |

A constant-N 1.2M model, at the identical budget, beats our growth trajectory
outright — because at that size the same FLOPs buy 11,811 steps instead of
2,908.

```
  The claim we made:   "the compute-optimal N(t) is not flat"
  What we showed:      "conditional on an endpoint 10× larger than optimal,
                        a rising N(t) beats a flat N(t) at that endpoint"
```

Worse, the endpoint constraint made the actually-winning configuration
**structurally unreachable by the AutoLab agent**: `experiment.py` raises
`SystemExit` on any trajectory not ending at 5.9M, so the agent could never have
found it no matter how well it searched. We handed the agent a search space that
excluded the best known answer.

**Verdict: false as stated. The narrower conditional version is what the data
supports.**

---

## 5. Reproducibility — our decimals are mostly decoration

We re-ran one identical configuration (flat, seed 0, LR 1e-3, 4×10¹⁴ FLOPs) on
MPS and got **4.4161** where the recorded run had **4.4638**. Same code, same
seed, same data order, same step count.

```
  recorded          4.4638
  identical re-run  4.4161
                    ───────
  difference        0.0477     ← pure kernel nondeterminism on Apple Silicon
```

That ~0.05 noise floor is worth staring at, because:

- It is **the same size as the grown arm's entire reported σ** (0.053).
- It is **larger than the flat-at-5e-4 arm's σ** (0.015) — meaning the seed
  spread we report is *smaller* than the run-to-run variation of a single seed,
  which is internally inconsistent. Our σ understates true variability, so every
  effect size computed from it is optimistic.
- It means all our 4-decimal figures carry roughly **two significant figures of
  real signal**. `4.1112` should be read as `4.11`, and differences below ~0.05
  between individual runs should not be interpreted at all.

The 0.397-nat gap is ~8× the noise floor, so it survives this. The AutoLab
agent's improvement from 4.111 to 4.074 does not — and to its credit, the agent
said so itself when it stopped: *"remaining variation is seed noise."*

**Wall-clock is not a budget, illustrated for free:** the flat arm at 1e-3 and
the flat arm at 5e-4 do bit-for-bit identical amounts of work — same
architecture, same 2433 steps — and took 132–140s and 151–158s respectively.
A ~14% swing from ambient machine conditions alone.

---

## 6. What survived the audit intact

Not everything broke. These were checked specifically and are clean.

### FLOP accounting

`flops_per_token = 6·N + 12·L·T·d` in `grow/model.py`. The `6·N` term covers
every matmul weight forward and backward; the tied `embed`/`lm_head` tensor is
counted **once**, which is correct — it is one tensor participating in one real
matmul (the output projection), and the input-side lookup is a gather that
costs essentially nothing. The `12·L·T·d` term covers the parameter-free
attention score and value products, which a pure `6·N` count would miss
entirely.

The one residual imprecision points **against** the grown arm. `pos_embed`
(512×d parameters) is charged at `6·N` rates but is also just a lookup, so both
arms are overcharged for it — and the overcharge is relatively larger for a
small model:

```
  flat  (6L 256D):  phantom pos-embed cost / total  =  786K / 40.1M  = 1.96%
  grown (3L 128D):  phantom pos-embed cost / total  =  393K /  8.3M  = 4.75%
                                                                       ▲
                              the grown arm burns its budget ~2.4× faster
                              than physics requires, early on — i.e. it is
                              handicapped, not flattered
```

### `grow_width` weight transfer

Bit-exact. Old weights land in the top-left of every new tensor, and the QKV
copy reshapes to `(3, heads, head_dim, dim)` so old heads land in their correct
rows rather than being smeared across the q/k/v boundary. Weight tying survives
growth: `cp()` assigns to `param.data` of the shared `Parameter` object, so both
`embed.weight` and `lm_head.weight` — which are the same object — see the new
tensor. (Had it rebound `.weight` instead, the tie would have broken silently
and the parameter count would have jumped without anyone noticing.)

### Seeding

`torch.manual_seed(cfg.seed)` runs before model construction, so the three seeds
are three genuinely different initialisations, not three re-labels of one. The
data loader is seeded separately and identically (`seed=1234` train, `seed=0`
val) for every arm and every seed, so no arm ever gets luckier batches. Eval is
a fixed non-random slice, identical everywhere.

### The LR-horizon predictor

`estimate_total_steps` replays the growth schedule analytically — constructing
the model and summing `flops_per_token × tokens` with no training — to tell the
WSD schedule how long the run will be. It is exact for both arms: predicted
step counts match completed step counts, and all three seeds of each arm
completed exactly the same number of steps (2433 flat, 2908 grown), which is
what you would expect from a budget accounted deterministically from
architecture alone.

### The two asymmetries the flat arm never experiences

The grown arm gets an optimizer reset at each growth event (AdamW moments are
discarded because the tensor shapes changed) and a 10-step learning-rate
re-warmup afterwards. These are real asymmetries and they should be disclosed.
Their size:

```
  5 growth events × 10 re-warmup steps           =  50 of 2908 steps  (1.7%)
  integrated LR deficit: 5 × Σ(1 − k/10) for k=1..10
                                 = 5 × 4.5       ≈  22.5 step-equivalents
  as a fraction of the run's total integrated LR ≈  0.9%
```

Both **handicap** the grown arm: it loses optimizer state the flat arm keeps,
and it trains at a reduced learning rate for steps where the flat arm is at
full rate. Neither can explain a grown-arm win, which is the only thing that
matters for this audit.

---

## 7. A process failure worth recording

At commit `60763cd`, `web/index.html` set the verdict tile unconditionally:

```js
document.getElementById('verdict').innerHTML =
  `growth wins<span…> — every seed, and ${s.variance_ratio.toFixed(0)}× steadier</span>`;
```

It never read `c.verdict`. At the same moment, `runs/control_summary.json` —
computed by our own code, from our own data, and shipped in the very same
`data.json` the page was fetching — said:

```json
  "verdict": "no detectable difference"
```

The dashboard was, in the strict sense, not reporting the experiment. It was
reporting what we expected the experiment to say. This is the failure mode that
matters most in a demo, because a hardcoded conclusion is invisible: the page
looks exactly like a page that computed something. Current `HEAD` renders
`c.verdict` and only substitutes a headline when the computed separation
actually holds.

Two related inconsistencies found in the same pass, both still worth knowing:

- **`control.py` and `build_data.py` disagree about what "pooled σ" means.**
  `control.py` reports `effect_size_sigma = delta / sqrt(σ_a² + σ_b²)` = 1.709;
  `build_data.py` reports `cohens_d = delta / sqrt((σ_a² + σ_b²)/2)` = 2.417.
  Same data, two conventions, differing by exactly √2. Both numbers appear in
  `data.json`.
- **The shipped "no detectable difference" verdict was conservative for the
  wrong reason.** `control.py`'s 2σ rule failed only because flat's σ was
  enormous — and that σ was itself the LR artifact of §3. So the code was
  accidentally right while reasoning from a broken input, and the dashboard
  overrode it. Both errors, pointing in opposite directions, in the same commit.

---

## 8. The claim that stands

Stated as precisely as the evidence allows:

> **Growing into a fixed target size beats starting at that size.**
> At a 4×10¹⁴ FLOP budget, with both arms ending at 6L/8H/256D (5.9M
> parameters), identical data order, and each arm at its own best learning
> rate: grown 4.111 ± 0.053 versus flat 4.509 ± 0.015 — a **0.397-nat gap with
> complete seed separation** (worst grown seed 4.172 < best flat seed 4.496).
> The mechanism is that being small early buys 2908 steps instead of 2433 for
> the same compute, +19%.

And the conditions that claim is hostage to, which belong in the same breath:

- **It is conditional on that endpoint.** The 5.9M target is ~10× larger than
  compute-optimal for this budget. An unconstrained flat 1.2M model scores 3.905
  and beats the grown trajectory. Growth is not shown to be compute-optimal in
  general — only to be the better way to *arrive at an oversized model*.
- **Scale is untested.** 5.9M parameters, ~2900 steps, WikiText-103, one
  laptop.
- **Three seeds, against a ~0.05 nondeterminism floor.** Two significant
  figures, not four.
- **One learning rate per arm, chosen from a coarse comparison**, not a proper
  sweep.

That is a smaller claim than the one we started the day with. It is the one we
can defend.
