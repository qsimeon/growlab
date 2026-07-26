#!/usr/bin/env python3
"""Regenerate web/data.json from runs/ and the live AutoLab project.

Safe to run on a loop while experiments are in flight -- every section degrades
to null rather than failing, so the dashboard renders whatever exists yet.
"""

import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS, OUT = ROOT / "runs", Path(__file__).resolve().parent / "data.json"
ARMS = {"flat": "Flat N(t)", "grown": "Grown N(t)"}


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
    curve = read_jsonl(RUNS / f"{arm}_s0.jsonl")
    finals, steps = [], []
    for p in sorted(RUNS.glob(f"{arm}_s*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        finals.append(r["final_val_loss"])
        steps.append(r["steps_completed"])
    return {
        "label": ARMS[arm],
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


def autolab_state():
    def run(*args):
        try:
            r = subprocess.run(
                ["autolab", *args], capture_output=True, text=True, timeout=25, cwd=ROOT
            )
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    log = run("log", "--limit", "15")
    status = run("status")
    return {
        "project": "qsimeon/growlab",
        "url": "https://app.autolab.ai/projects/qsimeon/growlab",
        "log": log,
        "status_raw": status,
        "live": status is not None and "Active" in (status or ""),
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
    import math

    sea = a["stdev"] / math.sqrt(a["n_seeds"])
    seb = b["stdev"] / math.sqrt(b["n_seeds"])
    pooled = math.sqrt((a["stdev"] ** 2 + b["stdev"] ** 2) / 2)
    delta = a["mean"] - b["mean"]
    control = dict(control or {})
    control["stats"] = {
        "delta": delta,
        "welch_t": delta / math.sqrt(sea**2 + seb**2),
        "cohens_d": delta / pooled,
        "variance_ratio": (a["stdev"] / b["stdev"]) ** 2 if b["stdev"] else None,
        "sd_flat": a["stdev"],
        "sd_grown": b["stdev"],
    }
    return control


def main():
    summary_path = RUNS / "control_summary.json"
    control = json.loads(summary_path.read_text()) if summary_path.exists() else None
    series = {arm: arm_series(arm) for arm in ARMS}
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "control": add_stats(control, series),
        "series": series,
        "autolab": autolab_state(),
    }
    OUT.write_text(json.dumps(data, indent=1))
    n = {a: data["series"][a]["n_seeds"] for a in ARMS}
    print(f"wrote {OUT} · seeds {n} · control={'yes' if control else 'pending'}")


if __name__ == "__main__":
    main()
