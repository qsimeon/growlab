# CLAUDE.md

## Operating contract

Quilee is **Chief Ideation Officer**: he drives design, Claude builds. Two modes, don't blur them:

1. **Ideation** — propose, question, tighten scope. No app code.
2. **Build** — after design lock, ship end-to-end without re-litigating settled decisions. A working demo, not a scaffold.

Target: **Sundai Hack #133**, today, <10hrs, live demo at 8pm.

## What we're building

**GrowLab** — a language model that grows its own parameter count during training, plus an autonomous agent searching the space of growth schedules, running live.

The real claim: scaling laws optimize N and D *assuming N is constant* — nobody justified that. We ask what the optimal **trajectory N(t)** is. Read `context/idea.md`.

Three pieces:

| | |
|---|---|
| **The science** | `reverse_distillation/autoresearch_grow/` — growth primitives + a reported 54%-FLOPs result that has **three known confounds**. Motivating observation, not a finding. |
| **The harness** | [Autolab](https://autolab.ai) CLI — a real autonomous-research product. Point it at a repo + run command + objective; it proposes, runs, and merges experiments. Use the actual tool. |
| **The hosting** | [Maritime](https://maritime.sh) — makes the demo reachable on a URL. Not the training compute. |

## Read first

1. **`context/idea.md`** — the science from first principles. Supersedes `STATUS.md` in the reverse_distillation repo.
2. `PLAN.md` — architecture + build order.
3. `context/stack.md` — Autolab/Maritime commands, event logistics.

Read-only sources: `/Users/quileesimeon/reverse_distillation` (`autoresearch_grow/train.py`, `results/*.json`, `for_writing/slides/*.pdf`) · `/Users/quileesimeon/parameter_golf/intel/autoresearch/` (agent-design notes, fallback only).

## Non-negotiables

- **Budget by FLOPs, not wall-clock.** The inherited script uses wall-clock, which silently hands smaller models extra optimizer steps. This is the single most important line of code in the project.
- **Growth schedule is a parameter**, not a hardcoded preset — a list of `(step, action)` per `ScheduledGrowthPolicy`. The agent needs a space to search.
- **Run the control before quoting the 54%**: grown vs. flat, *same* WSD schedule both arms, *same* FLOP budget, ≥2 seeds.
- **The demo must not depend on growth winning.** It's "watch an agent search growth-schedule space" — true and interesting either way. Never build a demo that only works if the result goes our way.
- **Don't touch Direction 1 / the softmax barrier.** Real open research problem, not a 10-hour problem.
- **Never block on external approval.** Maritime is private beta; Autolab needs a browser login. Always have a fully local path working first.
- **Keep the model tiny.** Whatever machine you attach to Autolab is the compute — assume laptop CPU, not L40S.
- **No slides.** The event says so explicitly. Build a live screen worth sharing.

## Layout

```
CLAUDE.md · PLAN.md · README.md
context/       idea.md, stack.md          research + platform grounding
grow/          growth engine, FLOP-budgeted, JSONL trajectory out
.autolab/      created by `autolab init` — don't hand-edit
web/           demo frontend (static playground + live view)
deploy/        maritime config
```
