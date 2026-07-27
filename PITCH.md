# Four-minute pitch script

Deck: `web/slides.html` (arrow keys) · PDF: `web/growlab-slides.pdf`
Have a second tab open on the live dashboard in case someone asks.

Timings are a guide, not a straitjacket. The two slides worth protecting are
**4 (the platforms)** and **6 (the result)**.

---

### 1 · Title — 15s

> Hi, we're GrowLab. Every language model you have ever used was built at its
> final size on day one, and then trained. We asked a simple question: what if
> the model is allowed to grow while it learns? And then we handed that question
> to an AI agent and let it do the research.

*Advance.*

---

### 2 · The motivation — 20s

> Scaling laws are how the field decides how big to make a model. They work.
> But they quietly assume you buy all your parameters up front and keep them for
> the entire run. Nobody ever argued for that. It is just how it is done.
>
> So the question is not how big. The question is what shape.

*Advance.*

---

### 3 · The idea — 30s

> Here is the intuition. Compute per step scales with the size of the model, so
> the total training bill is the area under this curve.
>
> On the left is how it works today — a rectangle. You pay full price on every
> single step. On the right is what we do. Same model at the finish line, much
> less area underneath.
>
> And it should work, because a model in its first few hundred steps is learning
> word frequencies and basic grammar. You do not need a large model for that.
> Paying for one is buying capacity before there is anything to use it for.

*Advance.*

---

### 4 · The two platforms — 40s **(this is the hack — do not rush it)**

> This is where the two sponsor tools do real work, and they do completely
> different jobs.
>
> On the left, **Autolab** runs the science. Its research agent invents a growth
> plan, ships it to our machine, our machine trains the models and returns a
> single number, and the agent decides what to try next. It ran eight full
> experiments on its own. After the first one, no human chose what to try.
>
> On the right, **Maritime** puts it in front of you. It keeps an always-on
> machine serving the live results page. Our laptop can close and the results
> stay up. That is the difference between a demo and a folder on someone's
> laptop.
>
> One agent does the research. The other makes it public.

*Advance.*

---

### 5 · The experiment — 30s

> Here is exactly what we compare, because this only means something if it is
> controlled.
>
> The baseline is the same model trained at full size from step zero. Ours grows
> from 1.2 million parameters to 5.9 million. Both get the same compute budget,
> counted in actual operations. Same data, same order, WikiText-103. Same
> architecture at the finish line. Three models per side, so we can tell a real
> effect from luck.
>
> And here is the mechanism: because ours is small early, the same budget buys it
> nineteen per cent more training steps.

*Advance.*

---

### 6 · The result — 30s

> The growing model is ahead from the very beginning and never gives the lead
> back.
>
> It reaches the baseline's *final* quality using about sixty per cent of the
> compute — and then keeps improving with what is left over.
>
> Final scores: 5.02 for the baseline, 4.11 for the growing model. Lower is
> better.

*Advance.*

---

### 7 · The part that surprised us — 25s

> This is the bit we did not expect. Every dot here is one independently trained
> model.
>
> The three growing models land on top of each other. The three baselines are
> scattered across a full point of loss. Every growing model beat every baseline.
>
> Starting small seems to protect the model through the fragile early phase,
> when a strong learning rate can otherwise destabilise it. Growth is behaving
> like a curriculum, not just a saving.

*Advance.*

---

### 8 · What the agent found — 30s

> Then we let the agent search on its own. Every point here is one complete
> experiment it designed, ran and scored.
>
> It reproduced our hand-written plan exactly — that is the first point — and
> then beat it.
>
> What it learned is that *when* a model grows matters much more than how finely
> it grows. Growing too early or too late were both worse. Splitting growth into
> more, smaller steps changed nothing. What won was finishing growth about a
> third of the way in, and alternating between wider and deeper instead of doing
> all the widening first.
>
> Then it stopped and told us the remaining differences were smaller than the
> noise between runs. It knew when to quit.

*Advance.*

---

### 9 · Where this goes — 20s

> Three things next. Let the model choose when to grow, triggered by progress
> stalling rather than a fixed timetable. Let it choose wider or deeper for
> itself. And prove it at a scale people actually train at — these are small
> models trained briefly, and we are not going to pretend otherwise.

*Advance.*

---

### 10 · Close — 15s

> The best size for a model is not a number. It is a shape.
>
> All three links are live right now. The agent is still searching, and the
> results page updates as it goes. Thank you.

---

## If asked

**"Is this just a learning-rate effect?"**
Honest answer: partly possible, and we say so. Both arms got the identical
learning-rate schedule, but that schedule was originally chosen for the growing
setup. A learning-rate sweep on the baseline is the first thing we would run
next.

**"Three seeds isn't many."**
Correct. Three per side is enough to see complete separation — every growing
model beat every baseline — but not enough to put a tight number on the size of
the effect. It is a strong signal, not a settled fact.

**"How big are these models?"**
Small: 5.9 million parameters at the end, on WikiText-103, a few thousand steps.
We are explicit that nothing here has been shown to survive at scale.

**"What did Maritime actually run?"**
The public results page, on an always-on machine, deployed from our repo. It
does not train anything — that runs on our hardware, dispatched by Autolab.

**"What did Autolab actually do?"**
It chose every experiment after the first, wrote the configuration, queued it to
our machine, read the score back, and decided whether to keep or discard it.
Eight experiments, and it beat the plan we wrote by hand.
