# PLAN

Handoff from the CIO planning session. Decisions here are locked — don't re-derive.

## Pitch

*A language model that grows its own parameters during training — start tiny, earn capacity, reach full size for roughly half the compute. Autolab's agent searches growth-schedule space live; Maritime puts it on a URL.*

## Architecture

```
grow/train.py                  autolab                      web/
┌──────────────┐  run cmd    ┌──────────────┐  log/API   ┌──────────────┐
│ FLOP-budgeted│────────────▶│ proposes N(t)│───────────▶│ growth chart │
│ growth loop  │◀────────────│ runs, merges │            │ + best-so-far│
│ prints loss  │  on YOUR    └──────────────┘            └──────────────┘
│ JSONL out    │  attached                                      ▲
└──────────────┘  compute                            served by Maritime
```

Three components, not four. **Autolab replaces the hand-rolled agent loop** — that was the plan before we knew Autolab was a real, installable product.

1. **`grow/`** — port `rapid_growth_wsd_compact` out of `reverse_distillation/autoresearch_grow/train.py`, stripped of RALPH scaffolding. Must: budget by **FLOPs** (≈`6·N(t)·tokens_per_step`, summed with N updated at growth events), log `{step, params, loss, cum_flops}` per step to JSONL, print **one scalar objective** at the end, take the growth schedule as a **parameter**.
2. **Autolab** — `login` → `install` → `init` (objective: minimize val loss at fixed FLOPs) → `serve` (attach compute) → `start`. Commands in `context/stack.md`.
3. **`web/`** — growth-trajectory chart (params + loss vs. step), current best. Two sources, same rendering: **static precomputed** from `autoresearch_grow/results/*.json` (must work, build first), then **live** from Autolab, served via Maritime.

## Build order

| # | Step | Checkpoint |
|---|---|---|
| 0 | Read `context/idea.md` | You understand why N(t) is the question |
| 1 | Port `grow/` — FLOP-budgeted, JSONL out, scalar objective, schedule as param | A trajectory file + a script Autolab can call |
| 2 | **Run the control**: grown vs. flat, same WSD, same FLOPs, ≥2 seeds | You know if the 54% survives — either answer is fine |
| 3 | Build `web/` against the *static* trajectory | Demo looks good with zero network deps ← **your safety net** |
| 4 | `autolab login/install/init/serve/start` | `autolab status` shows a live project with a run |
| 5 | Deploy to Maritime (code `SUNDAI`) | Public URL responds |
| 6 | Wire `web/` live-with-fallback · submit card at `sundai.club/pitch` before 8pm | Demo-ready |

Steps 1→3 are sequential. Start `autolab login` early in parallel — it needs a browser.

## Cut list (top down)

1. Live Autolab run + Maritime deploy → ship static playground, describe the live piece.
2. Custom status endpoint → demo Autolab's own dashboard (`autolab open`).
3. Extra policy search → don't hand-tune under time pressure.
4. **Never cut:** the growth visualization. It's the whole story.

## Out of scope

Direction 1 / softmax barrier · scaling past 17.6M params · the 14M→12B iterative chain · a slide deck for 8pm.

**Fallback only:** if Autolab is genuinely blocked, `parameter_golf/intel/autoresearch/` has design notes for a hand-rolled loop. Try `autolab init` + `serve` first — it's almost certainly faster.
