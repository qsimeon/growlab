# Provenance — where the idea came from and what had to die first

This project did not start from "let's grow a model." It arrived at growth by
failing at something else, and the failure is the most useful thing we know.
A previous direction — expanding an already-trained model while preserving its
function exactly — was mathematically successful and practically dead. Working
out *why* it was dead produced the constraint that the current design is built
around. So this document is the argument, not just the history: the current code
looks the way it does because of a specific gradient-flow failure that we can
write down.

---

## 1. The lineage

```
  Numenta — "growing sparse LLMs"
  ────────────────────────────────────────────────────────────
  Underlying question: how do you buy representational capacity
  without paying full price for it the whole time?
  Their answer: expand a small dense model into a HIGHER-dimensional
  but SPARSE one. More dimensions, not proportionally more FLOPs.
            │
            │  the sparsity half fell out along the way
            ▼
  DIRECTION 1 — reverse distillation                       ✝ DIED
  ────────────────────────────────────────────────────────────
  Expand a FINISHED pretrained model into a larger architecture,
  guaranteeing f_new(x) = f_old(x) at initialization.
            │
            │  its death identified the real constraint
            ▼
  DIRECTION 2 — neural morphogenesis                    ← this repo
  ────────────────────────────────────────────────────────────
  Grow a model that is STILL TRAINING. Nothing is finished,
  so nothing has to be preserved.
```

The through-line is not "expansion." It is **decoupling representational
dimensionality from compute cost**. Sparsity answers that by making a model
bigger but cheaper *per step*. Growth answers it by making the model bigger
*later* and cheaper *earlier*. Two different answers, one question — which is
why the sparsity thread is a live open question rather than a dead end (see
`open-questions.md`).

---

## 2. Why Direction 1 died: the softmax barrier

Direction 1's premise was appealing. Take a finished Pythia-70M, write its
weights into the top-left corner of a 140M-parameter architecture, fill the rest
with something negligible, and **guarantee that the expanded model computes
exactly the same function as the original at step 0.** Nothing learned is lost;
training simply continues in a bigger space.

The guarantee worked. Max logit difference was 0.005 for width expansion and
0.000000 for depth. Making it work even required a genuinely nice piece of
engineering — a `ZeroIgnoringLayerNorm` that normalises only over the non-zero
dimensions, so the zero-padding does not corrupt LayerNorm's mean and variance.

And the guarantee is exactly what killed it. Follow the chain:

```
  REQUIREMENT: preserve the function exactly
        │
        ▼
  new weights must be ≈ 0        (in practice ε = 1e-8)
        │
        ▼
  a new attention head therefore has  Q, K ≈ 1e-9
        │
        ▼
  scores = QKᵀ/√d  ≈  1e-18
        │
        ▼
  softmax(≈0 everywhere) = UNIFORM,  regardless of the input
        │                            ← softmax annihilates differences
        │                              this small; every token attends
        │                              equally to every token, always
        ▼
  gradient ∝ p(1−p)·V  ≈  5e-13
        │
        ▼
  the head never moves.  Not slowly — never.
```

The failure is specific to attention. The MLP survives the same treatment
because `GELU'(0) ≈ 0.5`, so a zero-ish MLP still passes gradient at ordinary
magnitude. Attention does not, because the softmax is exponential: it takes
scores that differ by 1e-18 and returns a distribution that is uniform to within
floating-point noise. The information that would break the symmetry is destroyed
before the gradient is ever computed.

This is the classic symmetry-breaking problem — the same reason you cannot
initialise a network to all zeros — but it lands hardest on attention, and it is
the reason we call it *the softmax barrier* rather than just "small init."

The obvious escape is to tune the noise scale. It does not exist:

```
  ε = 1e-8   →   preservation ✓         head is dead ✗
  ε = 1e-4   →   head is alive ✓        preservation destroyed
                                        (logit diff ≈ 370) ✗
                 └────── no ε satisfies both ──────┘
```

The two requirements are not in tension over a narrow range that careful tuning
could thread. They are mutually exclusive by construction: "preserved" *means*
"the new weights contribute nothing," and "alive" *means* "the new weights
contribute something."

**The realisation that ends Direction 1:** the trap exists only because you are
expanding something *finished and precious*. The preservation constraint is not
a design choice inside the method — it is the entire premise of the method, and
it is the thing doing the killing.

---

## 3. Why growing a mid-training model escapes the trap

Grow a model that is still training and there is no precious function to
protect. Its current function is a way-station; it is going to be overwritten by
gradient descent anyway. A perturbation at a growth event costs a small loss
blip and thousands of remaining steps in which to absorb it.

So the constraint lifts, and with it the ~0 initialisation:

```
  DIRECTION 1                          DIRECTION 2
  ──────────────────────────────       ──────────────────────────────
  model is FINISHED                    model is MID-TRAINING
  must preserve f(x) exactly           nothing worth preserving
       ↓                                    ↓
  new weights ≈ 1e-8                   new weights ~ N(0, 0.02)
       ↓                                    ↓
  scores ≈ 1e-18 → uniform softmax     ordinary scores, ordinary softmax
       ↓                                    ↓
  gradient ≈ 5e-13 → dead forever      ordinary gradients, capacity used
       ↓                                    ↓
  ✝                                    loss blip of ~0.02–0.1 nats,
                                       absorbed within 20–40 steps
```

Direction 2 is not a weakened Direction 1 that gave up on rigour. It is the
version that is **not obligated to pretend nothing changed**. Preservation was
never the goal; it was a proxy for "don't destroy what was learned," and
mid-training there is a cheaper way to not destroy it: keep the old weights
where they are and let the optimiser reconcile the rest.

Measured on `grown_s0`, at each of the five growth events:

| step | event | val loss before | after | recovered by |
|---|---|---|---|---|
| 200 | width → 192 | 6.141 | 6.165 | step 240 → 5.979 |
| 400 | depth → 4L | 5.623 | 5.718 | step 440 → 5.638 |
| 600 | width → 256 | 5.467 | 5.402 | *immediately better* |
| 800 | depth → 5L | 5.347 | 5.285 | *immediately better* |
| 1000 | depth → 6L | 5.019 | 5.107 | step 1040 → 5.141 |

Blips are small, sometimes negative, and always gone within tens of steps
against a ~2900-step run. This is the empirical content of "nothing precious to
preserve."

---

## 4. The design rule that falls out

The softmax barrier is really a statement about *where* in a block you are
allowed to put a zero. Compare the two cases.

**Zeroing an OUTPUT projection is safe.**

```
              ┌──────────────────────────────┐
   x ─────────┤  ln → attn → out_proj (= 0)  ├────┐
   │          └──────────────────────────────┘    ⊕──▶ x     ← exact identity
   └───────────────── residual ────────────────────┘

   The block contributes nothing to the forward pass, but its internal
   weights are fully wired to the loss:

        dL/dW_out  =  upstream_grad  ×  block_activationsᵀ
                          ▲                    ▲
                  arrives intact via     NOT zero — attention ran
                  the residual path      normally on real inputs

   So W_out is non-zero after step one, and the block starts contributing.
   Muted, not disconnected.
```

**Zeroing an INPUT projection is fatal.**

```
   x ──▶ Q = W_q x = 0,  K = W_k x = 0
              │
              ▼
        scores = 0  →  softmax uniform  →  output independent of input
              │
              ▼
        dL/dW_q ∝ (how much the loss changes when scores change) ≈ 0
              │
              ▼
        W_q never leaves zero. The head is disconnected, and softmax
        welds it shut.
```

```
   ┌────────────────────────────────────────────────────────┐
   │  Zero an OUTPUT projection  →  SAFE.                   │
   │     identity through the residual, gradient still      │
   │     arrives on the very next step.                     │
   │                                                        │
   │  Zero an INPUT projection   →  FATAL.                  │
   │     Q,K = 0 → uniform softmax → no gradient → dead.    │
   └────────────────────────────────────────────────────────┘
```

Direction 1 was forced into the fatal case by its own guarantee: preserving the
function of a *widened* model requires the new input dimensions to be inert,
which means zeroing the input side. Direction 2 gets to choose, and it chooses
differently for each operator.

---

## 5. How the two operators in `grow/model.py` differ

The crux, and the thing most likely to be misread: **the two growth operators
take opposite positions on preservation, on purpose.**

### `grow_depth` — preservation KEPT, because here it is free

Appending a block deep-copies the last block and zeros `attn.out_proj` and
`down` — both output-side. The new block is an exact identity at insertion, so
the function is preserved bit-for-bit, and by the argument above it is still
receiving gradient. Preservation costs nothing here, so we take it.

### `grow_width` — preservation deliberately ABANDONED

```
   old 128D                       new 192D
   ┌────────────┐                 ┌────────────┬────────────┐
   │  learned   │      ──▶        │  learned   │  N(0,0.02) │
   │  weights   │                 │  weights   │   fresh    │
   └────────────┘                 └────────────┴────────────┘
                                                     ▲
                             the inherited code put ZEROS here — which
                             reproduced the softmax barrier INSIDE the
                             direction that was supposed to escape it
```

The inherited `grow_width` allocated a zero tensor and copied the old weights
into the corner, so every new width dimension began at exactly 0. New attention
heads got `Q, K = 0`, uniform softmax, no gradient — Direction 1's death
mechanism, faithfully re-implemented inside Direction 2. We changed the fill to
`randn × 0.02`. Function preservation is lost; a ~0.02–0.1 nat blip is the
price; the new heads are alive. `--new-init zero` still exists so the old
behaviour can be ablated rather than merely asserted to be wrong.

Two details that matter and are easy to get wrong:

- **QKV rows are `[3, heads, head_dim]`-major.** A naive top-left copy smears
  the old q/k/v blocks across the wrong rows. `grow_width` reshapes to
  `(3, heads, head_dim, dim)` and copies each of q, k, v into its own slice, so
  old heads land where they belong.
- **New LayerNorm dimensions initialise to (gain 1, bias 0), not (0, 0).**
  Zeroing the gain leaves the new dimensions contributing nothing to the output
  while *still* shifting LayerNorm's mean and variance — the worst of both
  worlds: neither preserving the old function nor using the new capacity.

---

## 6. What the whole lineage was actually asking

Restated in the form the current experiment tests:

Scaling laws (Chinchilla and successors) ask *given compute C, choose N and D.*
That question is answered well. But every such law holds **N constant for the
entire run**, and no one ever argued for that — it is simply how models are
built.

> **Given compute C, choose the trajectory N(t).**

The answer is a *shape*, not a number, and constant-N is a single point in that
space.

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

This does not contradict Chinchilla. Chinchilla is correct *within the space of
constant-N runs* and silent outside it. The claim under test is that the
constant-N frontier is not the global frontier — Chinchilla is the best fixed
gear ratio, and we are asking whether a gearbox beats it.

Chinchilla also predicts where growth ought to **fail**, which is worth stating
because it is the sharpest available criticism: at roughly 20 tokens per
parameter, a grown model arrives at its final size having seen far fewer than
20·N tokens *at that size*. It is structurally undertrained for what it has
become. The open question is whether time spent smaller substitutes for those
missing tokens. See `audit.md` for how much of our evidence on that survives
scrutiny, and `open-questions.md` for what would settle it.
