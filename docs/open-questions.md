# Open questions — ranked by how much the answer would change the story

Everything here is a question we could not answer with the evidence we have.
They are ordered by consequence: the top one decides whether there is a general
result at all, the bottom one decides whether anyone would use it today. Each
entry says what we currently know, what experiment would settle it, and what
changes if the answer goes each way.

```
   how much it moves the story
   ▲
   │  Q1  does growth ever beat the COMPUTE-OPTIMAL flat model?     ← decides
   │      (or only oversized endpoints?)                              whether
   │                                                                  there is
   │  Q2  does any of it survive at real scale?                       a general
   │                                                                  claim
   │  Q3  should growth be TRIGGERED, not scheduled?
   │
   │  Q4  should the model choose width vs depth for itself?
   │
   │  Q5  what about sparsity — the dropped half of the lineage?
   │
   │  Q6  the compute saving is not a time saving. Does that matter?
   └──────────────────────────────────────────────────────────────▶
```

---

## Q1. Does growth ever beat the compute-optimal flat model, or only oversized endpoints?

**What we know.** Our result is conditional on a target size that is roughly
10× larger than compute-optimal for the budget. Both arms were pinned to 5.9M
parameters; compute-optimal at 4×10¹⁴ FLOPs is around 0.6M. Within that
constraint, growth wins by 0.397 nats with complete seed separation. Remove the
constraint and an unconstrained flat 1.2M model scores 3.905, beating our grown
5.9M trajectory's 4.111.

```
    val loss (lower better), all at 4e14 FLOPs

    3.9 ┤ ● flat 1.2M (unconstrained, 11811 steps)   ← the actual winner
    4.0 ┤
    4.1 ┤        ● grown 1.2M→5.9M (2908 steps)
    4.2 ┤
    4.3 ┤
    4.4 ┤
    4.5 ┤        ● flat 5.9M @ best LR (2433 steps)
        └────────────────────────────────────────
          unconstrained │ pinned to a 5.9M endpoint

    Inside the box on the right, growth wins cleanly.
    The box is where we chose to look.
```

So the honest state is: growth is the better way to *arrive at an oversized
model*, and we have no evidence that it beats the frontier.

**What would settle it.** Sweep the endpoint. For a fixed budget C, run flat
models across a range of N to locate the compute-optimal size empirically, then
run growth trajectories that *terminate at that optimal size* — and, separately,
trajectories that end above it, at it, and below it. The question is whether the
grown curve ever dips below the flat frontier or merely approaches it from
above.

There is a sharper version worth running: **grow past the compute-optimal
size.** If early training genuinely does not need capacity, the best trajectory
might spend its first phase well *below* optimal and its last phase well above,
with the time-average landing near the Chinchilla point. That is a different
hypothesis from "growth saves compute," and it is the one most likely to be
true.

**What changes.** If growth beats the flat frontier, the falsifiable claim from
`context/idea.md` is alive and the constant-N assumption in scaling laws is
genuinely leaving something on the table. If it never does, the result reduces
to a practical technique — *if you are forced to ship a model of size N, grow
into it* — which is useful, unsurprising, and not a scaling-law result.

---

## Q2. Does the advantage survive at real scale?

**What we know.** Nothing about scale. Everything here is 5.9M parameters,
~2900 steps, a 4,096-token vocabulary, WikiText-103, one laptop. That is small
enough that the entire experiment fits in the region where the loss curve is
still steep and capacity is nobody's bottleneck — which is exactly the regime
where "start smaller" wins most easily and least interestingly.

Chinchilla makes a specific prediction about where growth should fail. At
roughly 20 tokens per parameter, a grown model reaches its final size having
seen far fewer than 20·N tokens *at* that size:

```
  flat 5.9M:   sees 2433 × 4096 ≈ 10.0M tokens, all of them at 5.9M params
  grown:       sees 2908 × 4096 ≈ 11.9M tokens, but only the last ~1900 steps
               (≈ 7.8M tokens) are at the full 5.9M

  So the grown model is structurally undertrained FOR WHAT IT IS.
  The open question is whether time spent smaller substitutes for those
  missing tokens — and whether that substitution rate holds as N grows.
```

**What would settle it.** Repeat the control at two or three model sizes an
order of magnitude apart, at Chinchilla-appropriate token counts, and check
whether the gap widens, holds, or closes. A gap that shrinks with scale is the
expected null result and should be stated as such if that is what happens.

**What changes.** Everything about how seriously the result should be taken.
None of the current numbers should be extrapolated.

---

## Q3. Should growth be triggered adaptively rather than run on a timetable?

**What we know.** Our schedule is a hardcoded list of `(step, action)` pairs —
`200:width:192,400:depth,600:width:256,800:depth,1000:depth`. Those step numbers
have no justification beyond "they worked." The AutoLab agent's most productive
move was to shift them 50 steps earlier, which tells us the timing matters and
that we did not know where the right timing was.

But "step 200" is not a *reason* to grow. The underlying intuition in
`context/idea.md` is that a model should grow **when its current size stops
being the thing limiting it** — when the loss curve flattens at fixed capacity.
That is a condition, not a clock.

```
  SCHEDULED (what we do)              TRIGGERED (what the idea implies)

  loss                                 loss
   │  ╲                                 │  ╲
   │   ╲   ▲ grow at step 200           │   ╲
   │    ╲__│___                         │    ╲___
   │        ╲  ▲ grow at 400            │        ╲___  ← plateau detected
   │         ╲_│_                       │            ▲ grow HERE, whenever
   │             ╲                      │            │ that happens
   └──────────────────▶ step            └──────────────────▶ step

  timing is a hyperparameter            timing is a measurement
  that must be searched                 the run makes for itself
```

**What would settle it.** Implement a trigger-based policy alongside
`ScheduledGrowthPolicy` — grow when the slope of validation loss over a window
falls below a threshold — and compare it against the best *searched* fixed
schedule at matched FLOPs. The trigger version is only interesting if it matches
or beats a schedule that was tuned with hindsight, because the trigger has to
work without hindsight.

The harder and more interesting form of the question: **is there a principled
saturation signal?** Something like "this width is now the binding constraint" —
measurable from gradient statistics, or the rank/spectrum of activations, or
attention-head redundancy — rather than a proxy like loss slope.

**What changes.** A trigger converts growth from a hyperparameter you have to
search into a **policy** that works out of the box, and it connects directly to
continual learning: capacity added when needed rather than when scheduled. It
also makes the AutoLab search space much smaller and better shaped, since the
agent would search over trigger sensitivities rather than over explicit step
numbers.

---

## Q4. Should the model choose width versus depth by marginal return?

**What we know.** Growth kind and ordering are currently our choice, not the
model's. The AutoLab agent explored this axis across 8 experiments:

| what it tried | objective | outcome |
|---|---|---|
| baseline (our hand-built schedule) | 4.111 | merged |
| delayed growth (grow later, longer small) | 4.142 | discarded |
| width-first ordering | 4.090 | discarded |
| aggressive front-loading (start 4L/192D) | 4.220 | discarded |
| timing shifted 50 steps earlier | 4.076 | merged |
| extend warmup: start 2L/128D | 4.079 | discarded |
| growth completes ~step 800 (even spacing) | **4.074** | merged, best |
| finer-grained: 7 interleaved events | 4.112 | discarded |

Two readings, and the second is more robust than the first:

1. **Interleaved width/depth beat width-first**, 4.074 vs 4.090.
2. **Growth *timing* mattered more than growth *granularity*.** Moving the
   schedule earlier improved things (4.111 → 4.076); making it finer-grained did
   not (7 events → 4.112, no better than the 5-event baseline). Meanwhile
   over-correcting on timing hurt badly in both directions — delaying growth
   cost 0.031 and front-loading it cost 0.109.

**The caveat that has to travel with this table:** the per-run nondeterminism
floor on this hardware is ~0.05 nats (see `audit.md` §5). Every difference in
that table except the front-loaded run (4.220) is at or under the noise floor.
The ranking is a hypothesis generator, not a result. The agent itself stopped
for exactly this reason, reporting *"remaining variation is seed noise."*

**What would settle it.** Give the growth policy a choice at each event and a
criterion for making it — for example, estimate the marginal loss improvement
per FLOP of adding width versus adding depth from a short probe, and take the
better one. Then compare against the best fixed ordering. Run enough seeds to
get under 0.05.

**What changes.** It moves one more decision from the researcher into the
system, and it produces something more interesting than a schedule: a *shape
policy* that could differ across architectures and datasets rather than a magic
list of step numbers.

---

## Q5. What happened to sparsity — the dropped half of the lineage?

**What we know.** The original Numenta question was **growth plus sparsity**:
expand a small dense model into a higher-dimensional but *sparse* one, so you
get more representational dimensions without proportionally more FLOPs. Somewhere
between there and here, the sparsity half fell out and only the expansion half
survived.

That matters because the two are answers to the *same* question — how to buy
representational capacity without paying full price for it the whole time — and
they are not mutually exclusive:

```
   capacity ↑ without FLOPs ↑

        SPARSITY                    GROWTH
        bigger, but cheaper         cheap early, bigger later
        per step                    (this repo)
             │                           │
             └───────────┬───────────────┘
                         ▼
              nothing prevents both:
              grow into a sparse, higher-dimensional model
              → the original Numenta formulation, which
                nobody has actually run
```

**What would settle it.** Add a sparse growth operator: at a width event, expand
to a larger dimension but mask a fraction of the new weights, so the parameter
count rises faster than the FLOP count. Compare at matched FLOPs against dense
growth. The interesting metric is loss per FLOP, not loss per parameter.

**What changes.** Probably not the current claim, but it reopens the actual
research programme this project descends from, and it changes the shape of the
search space: N(t) becomes two curves — dimensions over time and density over
time — that trade against each other under one FLOP budget.

---

## Q6. The compute saving is not a time saving. Does that matter?

**What we know.** At matched FLOPs the grown arm is **6–11% slower in
wall-clock** — 145–148s versus 132–140s — because small matmuls underutilise the
device. The FLOPs are genuinely saved; the seconds are not.

```
  FLOPs spent          equal by construction (that is the experiment)
  optimizer steps      grown +19%     (2908 vs 2433)   ← the mechanism
  wall-clock           grown +6–11%   (slower!)        ← the practical cost

  A 3L/128D matmul does not keep the hardware busy. You save operations
  and spend them again on poor utilisation.
```

This is a scale artifact, not a refutation: the smaller the model, the worse
the utilisation, and our models are extremely small. At sizes where every matmul
saturates the device, the FLOP saving should convert into a time saving. But we
have not shown that, and reporting only FLOPs would be selecting the metric that
flatters the result.

**What would settle it.** Report FLOPs, steps and wall-clock together at every
scale tested, and find the size at which the wall-clock curve crosses. There is
also an engineering answer worth trying — larger batch sizes while the model is
small, so early steps use the device properly — but note that changing batch
size mid-run changes the optimisation problem too, so it is a confound as much
as a fix.

**What changes.** Nothing about the science; everything about adoption. A
technique that saves 40% of your FLOPs and costs you 10% more wall-clock is a
result, not a product. This is the first question a practitioner will ask, and
we should have the number ready rather than discovering it in the Q&A.
