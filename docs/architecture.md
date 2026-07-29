# Architecture — how the pieces fit

There are three machines in this project and they do very different jobs. A
cloud control node decides *what experiment to run*. A laptop *runs it*. A
micro-VM *shows the result to strangers*. Nothing about the science depends on
the third one, and nothing about the demo depends on the first one. That
separation is deliberate: it is what makes the demo survivable when a login
expires, a laptop lid closes, or an agent decides it is finished.

---

## 1. The whole system on one page

```
        ┌──────────────────────────────────────────────────────────────┐
        │  AUTOLAB CONTROL NODE — app.autolab.ai                       │
        │  An LLM agent (claude-opus-4-8) holding the objective:       │
        │  "minimise mean val loss at a FIXED 4e14 FLOP budget by      │
        │   choosing the growth trajectory N(t)."                      │
        │  It proposes → dispatches → reads one number → decides next. │
        └──────┬─────────────────────────────────────────▲─────────────┘
               │                                         │
   writes a SCHEDULE into                       reads back ONE SCALAR
   grow/experiment.py, then                     the last stdout line:
   dispatches the run command                     OBJECTIVE 4.07450
               │                                         │
               ▼                                         │
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  YOUR LAPTOP — attached with `autolab serve`. THIS IS THE ONLY COMPUTE.  │
 │                                                                          │
 │   grow/data.py ──▶ data/*.npy ──▶ grow/train.py ──▶ runs/<tag>.jsonl    │
 │   4k BPE over        flat uint16    FLOP-budgeted     one line per 10    │
 │   WikiText-103       token stream   loop, grows       steps: params,     │
 │                                     N(t) mid-run      loss, cum_flops    │
 │                                          │                   │           │
 │                                          └──▶ runs/<tag>.json            │
 │                                               final scalars  │           │
 │                                                              ▼           │
 │                                            web/build_data.py ──▶         │
 │                                            merges runs/ + `autolab log`  │
 │                                                              │           │
 └──────────────────────────────────────────────────────────────┼──────────┘
                                                                ▼
                                                        web/data.json
                                                    (the ONLY shared artifact)
                                                                │
                                     baked into the image at    │
                                     build time; optionally     │
                                     refreshed via POST /ingest │
                                                                ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  MARITIME MICRO-VM — built from ./Dockerfile                 │
        │  python:3.11-slim + deploy/server.py on :8080                │
        │  GET /            → web/index.html   (the live dashboard)    │
        │  GET /slides.html → web/slides.html  (the deck)              │
        │  GET /health      → {"ok":true, "last_push":…}               │
        │  POST /ingest     → overwrite data.json (token-gated)        │
        │                                                              │
        │  NO training · NO GPU · NO LLM · NO connection to AutoLab    │
        └──────────────────────────────────────────────────────────────┘
```

The important structural fact: **AutoLab and Maritime never talk to each
other.** They are not two halves of a pipeline. They are two independent
consumers of the same repository, and the only thing that crosses between them
is a single JSON file that the laptop writes. If Maritime is down the science
continues; if AutoLab is finished or logged out the dashboard still renders,
because `build_data.py` degrades every section to `null` rather than failing.

---

## 2. The data path, stage by stage

### `grow/data.py` — turn WikiText-103 into a flat token array

Trains a **4,096-token byte-level BPE** over the first 300k documents of
WikiText-103, encodes train and validation splits, appends `<eos>` between
documents, and packs everything into one flat `uint16` array truncated to a
multiple of `seq_len`. Output: `data/bpe4096.json`, `data/train.npy`,
`data/val.npy`. The step is idempotent — it returns immediately if the arrays
already exist.

The small vocabulary is not a shortcut, it is the thing that makes the
experiment measurable at all. The tied `embed`/`lm_head` matmul costs
`2·V·d` per token and so scales **linearly** in `d`; the transformer body — the
only part growth shrinks — scales as `d²`. Body FLOPs only dominate when
`d > V/(12L)`. At `V=50,304, L=6` that threshold is `d > 700`, far above the
128→256 range we train in, so at `V=50k` the embedding is ~91% of the budget
and growth can only ever touch the remaining sliver. At `V=4,096, L=6` the
threshold drops to `d > 57`, and the body carries ~80% of the cost. N(t) now
actually drives the compute bill.

`TokenLoader` samples batches from a `numpy` generator seeded at **1234 for
train and 0 for val, independently of the model seed**, so every arm of every
comparison sees the identical stream of batches in the identical order. Model
init varies with `--seed`; data never does. Evaluation uses `eval_batches`, a
fixed non-random slice from the head of the val array, so the eval set is
literally the same tokens for every run ever made.

### `grow/train.py` — the FLOP-budgeted loop

This is the core of the project, and its shape is dictated by one decision: the
budget is **FLOPs**, so the step count is an *output* rather than an input.

```
  spent = 0
  while spent < flop_budget:                  ← the loop condition IS the budget
      for action in schedule.get(step, ()):   ← several actions may share a step,
          model = apply_growth(model, …)        applied in written order
          opt   = AdamW(model.parameters())   ← optimizer state rebuilt (stale shapes)
      lr = wsd(step, total_steps) × post_growth_rewarmup
      … forward / backward / clip(1.0) / step …
      spent += flops_per_token(model, seq_len) × (batch_size × seq_len)
      step  += 1
```

Two subtleties worth knowing before reading the code:

**The LR schedule needs a horizon it cannot know.** Warmup-stable-decay has to
be told how many total steps there will be, but the step count depends on the
trajectory, which depends on the schedule. `estimate_total_steps` resolves this
by constructing the model, replaying the growth schedule **analytically with no
training at all**, accumulating `flops_per_token × tokens` until the budget is
exhausted, and returning the step count. It is exact for both arms because the
FLOP formula is a pure function of the architecture — no measurement, no
hardware dependence.

**The schedule string is the agent's search surface, so it fails loudly.**
`parse_schedule` returns a dict keyed by step with a *list* of actions, because
several events may legitimately land on the same step and must apply in written
order. `grow_width` raises on a target that is not a multiple of `head_dim` or
that does not exceed the current width, rather than silently rounding or
no-op'ing — an agent that writes a schedule must not be able to run a different
one without being told.

**Growth returns a new object.** `grow_width` builds a whole new `GrowableGPT`
and copies old weights into the top-left of every tensor, so `model` is
rebound, not mutated. `grow_depth` appends to the existing `ModuleList` and
mutates in place. Either way the parameter tensors are new or newly-shaped, so
the AdamW state (exp_avg, exp_avg_sq) no longer matches and is discarded.

Outputs, per run tagged `<name>_s<seed>`:

| file | contents |
|---|---|
| `runs/<tag>.jsonl` | one record every 10 steps: `step, params, arch, train_loss, cum_flops, lr`, plus `val_loss` every 50 steps. Flushed on write, so the dashboard can read it mid-run. |
| `runs/<tag>.json` | the final record: `final_val_loss` (50 eval batches, not 20), `steps_completed`, `flops_spent`, `wall_time`, `growth_events`, `val_history`, and the full config. |
| stdout | last line is `OBJECTIVE <val_loss>`, which is the entire contract with AutoLab. |

### `grow/model.py` — the architecture and the FLOP formula

A minimal pre-norm GPT: tied embedding/LM head, 4× MLP, `head_dim` held fixed
at 32 so widening adds *heads* rather than fattening existing ones.

```
  flops_per_token = 6·N  +  12·L·T·d
                    │        └── attention score & value products: parameter-free,
                    │            therefore invisible to a 6·N count
                    └── every matmul weight, forward+backward, counting the tied
                        embed/lm_head tensor ONCE (it is one tensor, one matmul
                        on the output side; the input-side lookup is a gather
                        and costs nothing)
```

Also here: `grow_width` (widen every layer, new weights at real magnitude) and
`grow_depth` (append a block whose output projections are zeroed, so it starts
as an exact identity through the residual). Why those two operators differ in
their attitude to function preservation is the subject of `provenance.md`.

### `grow/control.py` and `grow/experiment.py` — two entry points, two purposes

`control.py` is the human-run A/B: it runs `flat` (start at 6L/8H/256D,
no schedule) and `grown` (start at 3L/4H/128D, five growth events) for N seeds
each at the same budget, and writes `runs/control_summary.json` with per-arm
means, standard deviations, an effect size and a verdict string.

`experiment.py` is the **AutoLab-facing** entry point, and it is deliberately
thin. Two module-level constants — `SCHEDULE` and `START` — *are* the search
space. The agent edits those lines and nothing else. The file then runs three
seeds, asserts the trajectory ended at `FINAL_ARCH = "6L 8H 256D"` (raising
`SystemExit` if not), and prints the mean as `OBJECTIVE`.

### `web/build_data.py` — the join point

The only file that reads from both worlds. It walks `runs/` for per-arm
trajectories and final losses, shells out to `autolab log --limit 15` and
`autolab status` with `COLUMNS=220` (the CLI wraps its table to the terminal
width and becomes unparseable at 80), regex-parses the experiment table, and
adds proper two-sample statistics on top of `control_summary.json` — Welch's t,
Cohen's d and the variance ratio — reporting them **alongside** control.py's
own pooled-sigma verdict rather than replacing it. Writes `web/data.json`.
`web/monitor.sh` just re-runs it every 15 seconds.

### `web/index.html` — the dashboard

One self-contained file: no build step, no framework, no dependencies. It
fetches `data.json`, renders the verdict tile, a validation-loss-vs-FLOPs
chart, the N(t) parameter-trajectory chart, and the AutoLab experiment log,
with a table toggle behind every chart. Every rendered number traces to a field
in `data.json` — see `audit.md` for what happened the one time that was not
true.

---

## 3. The two platforms, and what each does *not* do

### AutoLab is the researcher

AutoLab supplies the **agent**, not the compute. Its control node holds the
objective, the constraints and the experiment history; your machine, attached
with `autolab serve`, is the execution node and the only silicon involved.
Registered at `autolab init`:

```
  objective    Minimize mean validation loss at a FIXED FLOP budget (4e14) by
               choosing the parameter growth trajectory N(t). Edit SCHEDULE and
               START in grow/experiment.py. Lower OBJECTIVE is better.
  constraints  1) budget is FLOPs, never wall-clock, never steps
               2) every trajectory must end at exactly 6L 8H 256D
               3) objective must be the mean over >=3 seeds
               4) do not modify evaluation, data pipeline, or flops_per_token
               5) the knob is the SHAPE of N(t) — tuning LR is off-objective
  prep         uv sync && uv run python grow/data.py
  run          uv run python grow/experiment.py
```

The loop it then executes, per experiment:

```
   agent proposes a hypothesis + a schedule
        │
        ▼
   writes SCHEDULE/START into grow/experiment.py, commits it (an experiment
   is a commit — the project is a git-like object)
        │
        ▼
   queues the job → laptop picks it up over `autolab serve`
        │
        ▼
   laptop runs 3 seeds × ~2900 steps ≈ 7 minutes, prints OBJECTIVE
        │
        ▼
   agent reads that ONE number, marks the experiment merged or discarded,
   and writes the next hypothesis
```

It ran **8 experiments autonomously** and then stopped itself with status
`Goal reached — N(t)-shape search converged: best trajectory found
(OBJECTIVE 4.0745 vs baseline 4.1108); remaining variation is seed noise.`
An agent that declines to keep optimising below the noise floor is doing the
right thing, and it is worth saying out loud.

**What AutoLab does NOT do:** it does not provide GPUs or any compute. It does
not see the JSONL trajectories, the loss curves, the growth events or the
dashboard — its entire view of a run is the single float on the last stdout
line. It has no knowledge that Maritime exists. And it cannot reach the
configuration that actually wins, because constraint (2) is enforced by
`SystemExit` in `experiment.py`: any trajectory not ending at 5.9M parameters
is rejected before its loss is ever reported.

### Maritime is the front door

Maritime runs serverless micro-VMs on bare metal. `maritime deploy` builds the
repo's `Dockerfile` — a `python:3.11-slim` image containing exactly
`deploy/server.py` and the `web/` directory — and serves it on a public URL.
`server.py` is ~60 lines of `http.server`: static files out of `web/`, a
`/health` endpoint, `Cache-Control: no-store` on everything so a refresh really
refreshes, and a token-gated `POST /ingest` that overwrites `data.json` after
validating that the body parses as JSON.

The ingest endpoint exists so the laptop can push fresh snapshots without the
container holding any credentials, a GPU, or a copy of the training code. Note
honestly: no pusher is wired up in this repo, and `/ingest` is disabled unless
`INGEST_TOKEN` is set, so what the public URL currently serves is the
`data.json` snapshot baked into the image at build time.

**What Maritime does NOT do:** no training, no GPU, no LLM, no AutoLab
credentials, no knowledge of the experiment. It is reachability, and that is
the whole job. It is the reason the demo does not depend on a laptop staying
open in a room with unreliable wifi.

---

## 4. File-by-file map

| path | what it is for |
|---|---|
| `grow/data.py` | Trains the 4,096-token BPE over WikiText-103 and packs both splits into flat `uint16` arrays. Also holds `TokenLoader`, whose seeding is independent of the model seed. |
| `grow/model.py` | `GrowableGPT` plus the two growth operators (`grow_width`, `grow_depth`) and `flops_per_token` = `6·N + 12·L·T·d`. |
| `grow/train.py` | The FLOP-budgeted training loop. Parses `--schedule`, predicts the step horizon analytically, applies growth, rebuilds the optimizer, writes JSONL + JSON, prints `OBJECTIVE`. |
| `grow/control.py` | The flat-vs-grown A/B the original work never ran. Writes `runs/control_summary.json`. |
| `grow/experiment.py` | AutoLab's entry point. `SCHEDULE` and `START` are the search space; `FINAL_ARCH` is the constraint; prints the 3-seed mean as `OBJECTIVE`. |
| `data/bpe4096.json` | The trained tokenizer. |
| `data/train.npy`, `data/val.npy` | Packed token streams. Regenerable from `grow/data.py`, deliberately not in git. |
| `runs/*.jsonl` | Per-run training trajectory, one record per 10 steps, flushed live. |
| `runs/*.json` | Per-run final result: loss, steps, FLOPs, wall time, growth events, full config. |
| `runs/control_summary.json` | The two-arm summary: per-arm losses, mean, stdev, delta, effect size, verdict. |
| `web/build_data.py` | Merges `runs/` + live `autolab log`/`status` into `web/data.json`, adding Welch t / Cohen's d / variance ratio. |
| `web/data.json` | The single artifact the dashboard consumes and the only coupling between the science and the hosting. |
| `web/index.html` | The dashboard: verdict, loss-vs-FLOPs, N(t) trajectory, AutoLab experiment log. Zero dependencies. |
| `web/slides.html` | The deck, served from the same VM (`growlab-slides.pdf` is the export). |
| `web/monitor.sh` | `while true; build_data.py; sleep 15` — keeps `data.json` fresh during a run. |
| `deploy/server.py` | The static server plus `/health` and token-gated `POST /ingest`. |
| `Dockerfile` | `python:3.11-slim` + `server.py` + `web/`. The whole Maritime image. |
| `.autolab/config.json` | Project name, control-node host, base commit. Created by `autolab init`; not hand-edited. |
| `context/idea.md` | The science from first principles: why N(t) is the question. |
| `context/provenance.md` | Long-form lineage, distilled in `docs/provenance.md`. |
| `context/stack.md` | AutoLab/Maritime command reference and event logistics. |
| `docs/methodology.md` | Why the budget is FLOPs; why "just fix the LR" was the error that produced a false result. |
| `docs/architecture.md` | This file. |
| `docs/provenance.md` | Why one-shot expansion died, why mid-training growth escapes it. |
| `docs/audit.md` | Every claim, re-tested. What survived and what did not. |
| `docs/open-questions.md` | What we still do not know, ranked by consequence. |
| `README.md` | Public front page. |
| `CARD.md`, `PITCH.md` | Sundai submission card and the four-minute pitch script. |
| `PLAN.md` | The locked build plan from the design session. |
