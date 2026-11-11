#!/usr/bin/env python3
"""Regenerate web/data.json from runs/ and the live AutoLab project.

Safe to run on a loop while experiments are in flight -- every section degrades
to null rather than failing, so the dashboard renders whatever exists yet.
"""

import json
import math
import os
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS, OUT = ROOT / "runs", Path(__file__).resolve().parent / "data.json"
# Each arm runs at its own best peak LR. The first headline ran flat at the grown
# arm's 1e-3, about twice its stable limit, and overstated the gap (docs/audit.md).
ARMS = {
    "flat": ("vflat_lr5e4", "Flat N(t), LR 5e-4"),
    "grown": ("grown", "Grown N(t), LR 1e-3"),
}


def read_jsonl(path):
    if not path.exists():
        return []
    recs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # last line may be mid-write
    return recs


def arm_series(arm):
    """Trajectory from seed 0 (representative) + final losses across all seeds."""
    prefix, label = ARMS[arm]
    curve = read_jsonl(RUNS / f"{prefix}_s0.jsonl")
    finals, steps = [], []
    for p in sorted(RUNS.glob(f"{prefix}_s*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        finals.append(r["final_val_loss"])
        steps.append(r["steps_completed"])
    return {
        "label": label,
        "loss": [
            {"flops": r["cum_flops"], "step": r["step"], "value": r["val_loss"]}
            for r in curve
            if "val_loss" in r
        ],
        "params": [
            {"flops": r["cum_flops"], "step": r["step"], "value": r["params"], "arch": r["arch"]}
            for r in curve
        ],
        "final_losses": finals,
        "mean": statistics.mean(finals) if finals else None,
        "stdev": statistics.stdev(finals) if len(finals) > 1 else 0.0,
        "steps": steps,
        "n_seeds": len(finals),
    }


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
EXP_RE = re.compile(
    r"^\s*([0-9a-f]{8})\s+(\w+)\s+(—|[0-9a-f]{8})\s+(?:objective\s+([\d.]+))?\s*(.*?)\s{2,}"
    r"(\d{4}-\d{2}-\d{2}T[\d:]+)\s*$"
)


def autolab_state(previous=None):
    def run(*args):
        try:
            # Wide terminal: the table wraps its columns to COLUMNS and becomes unparseable.
            r = subprocess.run(
                ["autolab", *args],
                capture_output=True,
                text=True,
                timeout=25,
                cwd=ROOT,
                env={**os.environ, "COLUMNS": "220"},
            )
            # The CLI colours its table whenever FORCE_COLOR is inherited (agent
            # shells set it), and the escapes make every row miss EXP_RE.
            return ANSI_RE.sub("", r.stdout) if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    raw = run("log", "--limit", "15") or ""
    experiments = []
    for line in raw.splitlines():
        m = EXP_RE.match(line)
        if m:
            eid, status, commit, metric, title, created = m.groups()
            experiments.append(
                {
                    "id": eid,
                    "status": status,
                    "metric": float(metric) if metric else None,
                    "title": title.strip(),
                    "created": created,
                }
            )
    if not experiments and previous and previous.get("experiments"):
        # The AutoLab project no longer lists its runs; keep the last snapshot.
        return {**previous, "live": False}
    scored = [e["metric"] for e in experiments if e["metric"] is not None]
    status = run("status") or ""
    return {
        "project": "qsimeon/growlab",
        "url": "https://app.autolab.ai/projects/qsimeon/growlab",
        "experiments": experiments,
        "best": min(scored) if scored else None,
        "n_done": len(scored),
        "live": "Active" in status,
    }


def add_stats(control, series):
    """Proper two-sample statistics.

    control.py's shipped verdict compares the mean gap to a *pooled standard
    deviation*, which is an effect-size ruler, not a significance test. Both are
    reported here rather than swapping one for the other after seeing the data.
    """
    a, b = series["flat"], series["grown"]
    if not (a["mean"] and b["mean"] and a["n_seeds"] > 1 and b["n_seeds"] > 1):
        return control

    sea = a["stdev"] / math.sqrt(a["n_seeds"])
    seb = b["stdev"] / math.sqrt(b["n_seeds"])
    pooled = math.sqrt((a["stdev"] ** 2 + b["stdev"] ** 2) / 2)
    delta = a["mean"] - b["mean"]
    va, vb = sea**2, seb**2
    if max(b["final_losses"]) < min(a["final_losses"]):
        verdict = "grown wins"
    elif max(a["final_losses"]) < min(b["final_losses"]):
        verdict = "flat wins"
    else:
        verdict = "no clear winner"
    control = dict(control or {})
    control["verdict"] = verdict
    control["stats"] = {
        "delta": delta,
        "welch_t": delta / math.sqrt(va + vb),
        "welch_df": (va + vb) ** 2
        / (va**2 / (a["n_seeds"] - 1) + vb**2 / (b["n_seeds"] - 1)),
        "cohens_d": delta / pooled,
        "variance_ratio": (a["stdev"] / b["stdev"]) ** 2 if b["stdev"] else None,
        "sd_flat": a["stdev"],
        "sd_grown": b["stdev"],
    }
    return control


def main():
    summary_path = RUNS / "control_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    # Keep only the design facts: the summary's gap and verdict are the shared-LR run.
    control = {k: summary[k] for k in ("flop_budget", "seeds")} if summary else None
    previous = json.loads(OUT.read_text()).get("autolab") if OUT.exists() else None
    series = {arm: arm_series(arm) for arm in ARMS}
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "control": add_stats(control, series),
        "series": series,
        "autolab": autolab_state(previous),
    }
    OUT.write_text(json.dumps(data, indent=1))
    n = {a: data["series"][a]["n_seeds"] for a in ARMS}
    print(f"wrote {OUT} · seeds {n} · control={'yes' if control else 'pending'}")


if __name__ == "__main__":
    main()
