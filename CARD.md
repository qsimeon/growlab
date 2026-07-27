# Sundai project card — paste into sundai.club/pitch

**Title:** GrowLab — an AI agent researching how models should grow

**One-liner:**
We handed a real research question to an autonomous agent, let it run its own
experiments on our hardware, published the results live — and then spent the
last hour trying to break our own finding. It half broke.

---

## The card body

**The question.** Scaling laws tell you how many parameters to buy for a given
compute budget. They assume you buy them all on day one and keep them for the
whole run. Nobody argued for that — it is just how models are built. Training
cost is the area under the parameter-count curve, so we asked what happens if
the model is allowed to grow while it learns.

**The system — this is the part we actually built.**

- **Autolab** ran the science. Its research agent invented a growth schedule,
  dispatched it to our laptop as an execution node, read back a single score,
  and decided what to try next. It ran **8 full experiments autonomously**.
  After the first one, no human chose what to try. It reproduced our
  hand-written schedule exactly, then beat it, then stopped and told us the
  remaining differences were below the noise floor.
- **Maritime** made it public. An always-on machine serves the live results
  page and the slide deck, so the demo exists whether or not our laptop is open.

Two agents, two jobs: one does the research, the other publishes it.

**What the experiment actually compares.** A baseline trained at full size from
step 0, against a model growing 1.2M → 5.9M parameters. Same compute budget
counted in operations, same data in the same order (WikiText-103), same final
architecture, 3 models per side.

| | val loss | seed spread |
|---|---|---|
| baseline, constant size | 4.509 | ± 0.015 |
| **grown** | **4.111** | ± 0.053 |

**The narrow claim that survives:** if you must reach a given model size,
**growing into it beats starting there** — 0.40 nats, with every growing model
beating every baseline. Being small early buys 19% more training steps for the
same compute.

**Then we tried to break it, and partly succeeded.** We ran an adversarial
review of our own repo. Two things we had claimed did not survive:

1. We first measured a 0.91-nat gap and a "102× more stable" result. Both were
   artifacts of running the **baseline at a learning rate tuned for the growing
   model**. Given its own sensible learning rate, the baseline is *more* stable
   than the growing model, and the gap halves to 0.40.
2. We had claimed the compute-optimal parameter trajectory is not flat. It is
   not that simple: our experiment pinned both arms to a 5.9M endpoint that is
   roughly **10× larger than compute-optimal for this budget**. Lift that
   constraint and a plain 1.2M constant-size model scores **3.905**, beating our
   growth trajectory outright.

So the honest result is conditional: **growth wins when the target size is
fixed and larger than the budget wants. It does not make growth compute-optimal
in general.** We would rather report that than the bigger number.

**Caveats we will not hide:** small models (5.9M), short runs, 3 seeds, and a
per-run nondeterminism floor of ~0.05 on Apple Silicon — which means our
four-decimal numbers carry about two significant figures of real signal.

**Next:** trigger growth when progress stalls instead of on a fixed timetable;
let the model choose wider vs deeper for itself; and repeat all of it at a size
people actually train at.

---

## Links

- **Live dashboard** — https://api.maritime.sh/a/961c500c-7530-4c1a-b8cd-d276b7bec384/
- **Slides** — https://api.maritime.sh/a/961c500c-7530-4c1a-b8cd-d276b7bec384/slides.html
- **The agent's search** — https://app.autolab.ai/projects/qsimeon/growlab
- **Code** — https://github.com/qsimeon/growlab

**Built with:** Autolab (autonomous research loop) · Maritime (hosting) ·
PyTorch on Apple Silicon
