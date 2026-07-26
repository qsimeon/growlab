# The Idea

## In one line

**Training compute is the area under the parameter-count curve.** Standard training holds N constant — a rectangle. If N grows during training and ends at the same size, you reach the same endpoint under less area.

Why it might work: early training learns cheap statistics (token frequency, basic syntax) needing almost no capacity. Paying for full width from step 0 buys capacity before you can use it.

## The research question, stated properly

Scaling laws ask: *given compute C, choose N and D.* They carry an unexamined assumption — **N is constant.** Nobody argued for it; it's just how models are built.

> **Reframed: given C, choose the trajectory N(t).**

The answer is a *shape*, not a number. Constant-N is one point in that space.

**Falsifiable claim: the compute-optimal N(t) is not flat.**

This doesn't contradict Chinchilla — Chinchilla is correct *within the space of constant-N runs* and silent outside it. If growth wins, the claim is "the constant-N frontier isn't the global frontier." Chinchilla is the best fixed gear ratio; we're asking whether a gearbox beats it.

Chinchilla also predicts **where growth should fail**: ~20 tokens/param means a grown model reaches final size having seen far fewer than 20N tokens *at* that size — structurally undertrained for what it is. The real question is whether time spent smaller substitutes for those missing tokens.

## Why the search space needs an agent

when to grow × how much × width vs. depth vs. heads × new-weight init × optimizer state × LR schedule around each event.

Hand-searching this is what stalled the project before — six RALPH iterations is six points, not a search. **This is why the demo and the paper are the same artifact.**

## Provenance (read `for_writing/slides/*.pdf` in reverse_distillation)

Numenta internship → "Growing sparse LLMs" → reverse distillation → neural morphogenesis.

The original Numenta question was **sparsity + growth**: expand a small dense pretrained model into a *higher-dimensional but sparse* one — more dimensions, not proportionally more FLOPs. The sparsity half got dropped along the way. That matters because it shows the deep question was never "expansion" — it was **how to buy representational capacity without paying full price the whole time.** Growth is a different answer to that same question (bigger over *time* instead of bigger but *sparse*). Don't reintroduce sparsity today; just know that's the real lineage.

## Direction 1 is dead (don't revive it)

Take a *finished* pretrained model, expand its matrices (originals top-left, ~0 elsewhere), preserve the function exactly, then train. Preservation works beautifully (`ZeroIgnoringLayerNorm` fixes LayerNorm corrupting stats over zero-padded dims; max logit diff 0.005). Then:

```
must preserve a precious function
 → new weights ≈ 1e-8  →  new heads' Q,K ≈ 1e-9
 → scores ≈ 1e-18  →  softmax(≈0) = uniform regardless of input
 → gradient ≈ 5e-13  →  those heads never move. Ever.
```

The **softmax barrier**. MLP survives (GELU'(0)≈0.5, no exponential flattening); attention doesn't. No noise scale escapes: 1e-8 preserves but is dead, 1e-4 lives but destroys preservation. It's classic symmetry-breaking, hitting attention hardest.

**The trap is the obligation to preserve exactly** — which only exists because you're expanding something finished.

## Why gradual growth escapes

Grow a model that's *still mid-training* and nothing is precious. A loss spike at a growth event is cheap. So new parameters get **real magnitude**, gradients flow, capacity gets used.

**One subtlety worth internalizing** — new depth layers zero their **output** projection (identity via residual), and that is safe, because `dL/dW_out = upstream_grad × block_activationsᵀ` and those activations aren't zero. `W_out` moves off zero on step one. *Zero output = muted but fully wired. Zero input = disconnected, and softmax welds it shut.* That distinction is the whole reason Direction 2 is technically sound.

## The honest state of the evidence

Repo headline: `rapid_growth_wsd_compact` matches baseline val_loss 6.19 at **54% of FLOPs**. **Treat as a motivating observation, not a finding.** Three confounds, worst first:

| Confound | Damage |
|---|---|
| **LR schedule** — WSD was introduced *with* growth; baseline never got WSD | **Total.** Gain may be entirely "WSD beats the old schedule," which is known. The repo's own `research_insights.md` concedes this. |
| **Barely-trained** — loss 6.19 ≈ perplexity 490 | Severe. That's the steep early curve where capacity is nobody's bottleneck and "start smaller" wins trivially. |
| **Wall-clock ≠ FLOPs** — 5-min budget gave small models free optimizer steps | Least bad. Measurement bug, fixable in an afternoon. |

Also: 17.6M params, single seed, nothing shown to survive scale.

Prior results, for reference (WikiText, L40S, 5-min budgets, 17.6M final params; `baseline_large` = 6.19):

| Preset | Val loss | % baseline FLOPs |
|---|---|---|
| front_loaded_growth | 5.74 | 93% |
| rapid_growth | 6.39 | 68% |
| rapid_growth_wsd | **6.14** | 68% |
| **rapid_growth_wsd_compact** | 6.19 | **54%** |
| depth_only_wsd | 6.28 | 49% |

Growth primitives already implemented in `autoresearch_grow/train.py`: `grow_width`, `grow_depth`, `grow_heads`, `grow_embedding`, plus `ScheduledGrowthPolicy` (explicit `(step, action)` list — this is the searchable interface).

## The one experiment that must run first

| | Arm A | Arm B |
|---|---|---|
| Trajectory | flat at final N | grown |
| LR schedule | WSD | WSD ← **same** |
| Budget | fixed **FLOPs** | fixed **FLOPs** ← not wall-clock |
| Seeds | ≥2 | ≥2 |

Cheap and decisive. Survives → real result. Doesn't → you learned it in an hour, the framing above is still unexplored, and the demo is unaffected.

**The demo must never depend on the result going one way.** It's "watch an agent search growth-schedule space" — true either way. If the agent finds nothing better than the seed, that means the seed is locally optimal, which is a legible thing to say out loud.
