#!/usr/bin/env python3
"""Rank the bounded 353M tuning grid and held-out confirmation seeds."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

VAL_RE = re.compile(
    r"step:(\d+)/(\d+) val_loss:([0-9.eE+-]+) train_time:([0-9]+)ms"
)
STATUS_RE = re.compile(r"finished with status=(\d+)")
STEP_RE = re.compile(r"step:(\d+)/(\d+) .*?step_avg:([0-9.eE+-]+)ms")


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def parse(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    text = (root / row["tag"] / "run.out").read_text(errors="ignore")
    vals = [(int(s), float(v), int(t) / 1000.0) for s, _, v, t in VAL_RE.findall(text)]
    if not vals:
        raise RuntimeError(f"no validation values for {row['tag']}")
    statuses = [int(x) for x in STATUS_RE.findall(text)]
    status = statuses[-1] if statuses else -1
    step_times = [float(x) for _, _, x in STEP_RE.findall(text)]
    curve = {step: (val, sec) for step, val, sec in vals}
    last_step = max(curve)
    return {
        "row": row,
        "tag": row["tag"],
        "family": row["family"],
        "seed": int(row["seed"]),
        "curve": curve,
        "last_step": last_step,
        "last_val": curve[last_step][0],
        "last_time_s": curve[last_step][1],
        "step_avg_ms": step_times[-1] if step_times else float("nan"),
        "status": status,
    }


def mean_sd(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0


def fmt_interval(xs: list[float]) -> str:
    mean, sd = mean_sd(xs)
    if len(xs) == 3:
        half = 4.303 * sd / math.sqrt(3)
        return f"{mean:+.6f} [{mean-half:+.6f}, {mean+half:+.6f}]"
    return f"{mean:+.6f}"


def threshold_time(item: dict[str, Any], target: float) -> float | None:
    hits = [(s, sec) for s, (v, sec) in item["curve"].items() if v <= target]
    return min(hits, default=(0, None), key=lambda x: x[0])[1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning-root", type=Path, required=True)
    ap.add_argument("--tuning-config", type=Path, required=True)
    ap.add_argument("--confirm-root", type=Path, required=True)
    ap.add_argument("--confirm-config", type=Path, required=True)
    ap.add_argument("--selection-dir", type=Path, required=True)
    args = ap.parse_args()

    tuning = [parse(args.tuning_root, r) for r in rows(args.tuning_config)]
    confirm = [parse(args.confirm_root, r) for r in rows(args.confirm_config)]
    selection = json.loads((args.selection_dir / "selected_scale353m_configs.json").read_text())

    common_tune = min(x["last_step"] for x in tuning)
    print(f"Common tuning comparison step: {common_tune}")
    print("rank\ttag\tfamily\tval\tstep_ms\tstatus")
    ordered = sorted(tuning, key=lambda x: (x["curve"][common_tune][0], x["tag"]))
    for rank, item in enumerate(ordered, 1):
        print(
            f"{rank}\t{item['tag']}\t{item['family']}\t"
            f"{item['curve'][common_tune][0]:.6f}\t{item['step_avg_ms']:.2f}\t{item['status']}"
        )

    print("\nSelected on tuning seed 3:")
    for fam in ("nm_tune", "spectral_tune"):
        d = selection[fam]
        print(f"{fam:15s} {d['tag']:28s} val={d['final_val']:.6f}")

    by_family: dict[str, list[dict[str, Any]]] = {}
    for item in confirm:
        by_family.setdefault(item["family"], []).append(item)
    if set(by_family) != {"nm", "spectral"}:
        raise RuntimeError(f"unexpected confirmation families: {sorted(by_family)}")
    for items in by_family.values():
        items.sort(key=lambda x: x["seed"])
    nm = by_family["nm"]
    sp = by_family["spectral"]
    if [x["seed"] for x in nm] != [x["seed"] for x in sp]:
        raise RuntimeError("confirmation seeds are not paired")

    print("\nHeld-out confirmation (seeds 4,5,6):")
    for family, items in (("Newton-Muon", nm), ("Spectral transition", sp)):
        vals = [x["last_val"] for x in items]
        mean, sd = mean_sd(vals)
        print(f"{family:20s} mean={mean:.6f} sd={sd:.6f} values=" + ",".join(f"{x:.6f}" for x in vals))

    diffs = [s["last_val"] - n["last_val"] for s, n in zip(sp, nm)]
    print("\nPaired spectral-minus-NM final differences:")
    for s, d in zip(sp, diffs):
        print(f"seed={s['seed']} diff={d:+.6f}")
    print("mean and descriptive 95% t-interval:", fmt_interval(diffs))

    common_steps = sorted(set.intersection(*(set(x["curve"]) for x in confirm)))
    early_steps = [s for s in common_steps if 100 <= s <= 1500]
    early_diffs: list[float] = []
    print("\nPaired early average differences (spectral-minus-NM, steps 100-1500):")
    for s_item, n_item in zip(sp, nm):
        vals = [s_item["curve"][step][0] - n_item["curve"][step][0] for step in early_steps]
        avg = statistics.mean(vals)
        early_diffs.append(avg)
        print(f"seed={s_item['seed']} avg_diff={avg:+.6f}")
    print("mean and descriptive 95% t-interval:", fmt_interval(early_diffs))

    print("\nMean wall-clock seconds to validation targets:")
    print("target\tNewton-Muon\tSpectral\tSpectral_change_pct")
    for target in (4.0, 3.8, 3.6, 3.5):
        n_times = [threshold_time(x, target) for x in nm]
        s_times = [threshold_time(x, target) for x in sp]
        if any(x is None for x in n_times + s_times):
            print(f"{target:.1f}\tMISSING\tMISSING\tMISSING")
            continue
        n_mean = statistics.mean(x for x in n_times if x is not None)
        s_mean = statistics.mean(x for x in s_times if x is not None)
        pct = 100.0 * (s_mean / n_mean - 1.0)
        print(f"{target:.1f}\t{n_mean:.1f}\t{s_mean:.1f}\t{pct:+.2f}%")


if __name__ == "__main__":
    main()
