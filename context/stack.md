# The Stack — Autolab, Maritime, and today's event

## Autolab — the research harness

⚠️ **Two unrelated products share the name.** `autolab.moe` = academic benchmark/leaderboard, **not relevant**. `autolab.ai` / `autolab.sh` (CLI `autolab`, docs.autolab.ai) = the real product and today's sponsor. Use the latter.

An LLM agent drives the research loop: proposes hypotheses, writes experiment code, schedules jobs, analyzes results, merges improvements. Git-like model: **project** = objective + baseline, **experiment** = commit.

```
you ── autolab CLI ──▶ Control Node (app.autolab.ai) ──▶ Execution Nodes
                       agent + queue + dashboard          YOUR machines
```

**Autolab supplies no compute.** You attach your own with `autolab serve`. Whatever you attach (probably a laptop CPU) is what experiments run on — size the model accordingly.

```bash
curl -fsSL https://app.autolab.ai/install.sh | sh
autolab login                 # browser step — do this EARLY
autolab install               # installs a skill so Claude Code can drive Autolab itself
autolab init                  # asks: name, run command, objective
autolab start                 # agent goes live
autolab serve --project you/growlab    # attach this machine, or the queue just waits
autolab status | autolab log | autolab open
```

Non-interactive: `autolab init -y --name growlab --objective "minimize val loss at fixed FLOPs" --run "python grow/train.py" --start`

Queue your own: `autolab submit -m "msg"` (with code) or `autolab submit --nocode -m "idea"` (agent writes it).

**Read `docs.autolab.ai/guides/coding-agents/` before building any custom experiment runner** — `autolab install` may make it unnecessary.

## Maritime — hosting

Serverless micro-VMs built for agents. Each gets a dedicated VM on bare metal, SSH/root, sleeps when idle, from $1/mo. **Free credits today: code `SUNDAI`.** Private beta — verify access early.

```bash
npm i -g maritime
maritime init <template>   # "coding agent" / "data agent" templates are closest
maritime deploy
maritime guide --json      # machine-readable contract, built for AI agents to drive
```

Also has a TypeScript SDK and REST API (maritime.sh/docs/api). **No GPU promised** — Maritime's job here is *reachability*, not training compute. Key is already in `~/.secrets/`.

## Today's event — Sundai Hack #133

"AI agents with maritime." Boston ↔ SF, ~10am–10pm.

1. **Kickoff:** name + short idea paragraph into `tinyurl.com/maritime133`.
2. **8:00 PM:** submit project card at `sundai.club/pitch`. Voting closes 8:15.
3. **Top 5** get 5 min: Zoom, share screen, **demo**.

> **"Demo your Sundai Card at 8pm. NO SLIDES."**

Don't build a deck. Build a live screen worth sharing.

**Context hygiene** (from the event's own deck, worth following): keep agent-facing docs ~100 lines or less, progressive disclosure over inlining, stateless sessions.
