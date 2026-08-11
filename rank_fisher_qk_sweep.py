#!/usr/bin/env python3
"""Rank Fisher-QK and Newton-Muon runs at a common validation step."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


def tb_last(run_dir: Path, tag: str):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None
    events = sorted(run_dir.rglob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    if not events:
        return None
    try:
        ea = EventAccumulator(str(events[-1].parent), size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in set(ea.Tags().get("scalars", [])):
            return None
        values = ea.Scalars(tag)
        return float(values[-1].value) if values else None
    except Exception:
        return None


def parse(root: Path):
    rows = []
    for run_out in sorted(root.glob("*/run.out")):
        text = run_out.read_text(errors="ignore")
        vals = {
            int(step): float(loss)
            for step, loss in re.findall(r"step:(\d+)/6200 val_loss:([0-9.]+)", text)
        }
        trains = re.findall(
            r"step:(\d+)/6200 train_loss:([0-9.]+) train_time:([0-9.]+)ms", text
        )
        def one(pattern, default=""):
            m = re.search(pattern, text, re.M)
            return m.group(1) if m else default
        status = re.findall(r"finished with status=(\d+)", text)
        error = bool(re.search(
            r"Traceback|RuntimeError|AssertionError|out of memory|CUDA error|Killed|ValueError",
            text, re.I,
        ))
        row = {
            "tag": run_out.parent.name,
            "mode": one(r"^MODE=(\S+)", "unknown"),
            "coeff": one(r"^COEFF=(\S+)", ""),
            "cg": one(r"^CG_ITERS=(\S+)", ""),
            "damp": one(r"^DAMP_REL=(\S+)", ""),
            "scale_mode": one(r"^SCALE_MODE=(\S+)", ""),
            "outer": one(r"^OUTER_SCALE=(\S+)", ""),
            "qk_lr": one(r"^QK_LR=(\S+)", ""),
            "vals": vals,
            "max_val_step": max(vals) if vals else -1,
            "min_val": min(vals.values()) if vals else math.inf,
            "last_train_step": int(trains[-1][0]) if trains else -1,
            "last_train": float(trains[-1][1]) if trains else math.nan,
            "train_time_ms": float(trains[-1][2]) if trains else math.nan,
            "status": status[-1] if status else "",
            "error": error,
        }
        if row["train_time_ms"] > 0 and row["last_train_step"] >= 0:
            row["steps_per_hour"] = row["last_train_step"] * 3_600_000.0 / row["train_time_ms"]
        else:
            row["steps_per_hour"] = math.nan
        for tag, key in [
            ("fisher_qk/cg_final_residual_mean", "cg_resid"),
            ("fisher_qk/qk_update_rms_mean", "qk_rms"),
            ("fisher_qk/curvature_capture_seconds", "capture_s"),
            ("fisher_qk/optimizer_seconds", "optimizer_s"),
            ("fisher_qk/descent_fallback_layers", "fallback"),
        ]:
            row[key] = tb_last(run_out.parent, tag)
        rows.append(row)
    return rows


def fmt(x, digits=4):
    if x is None or not math.isfinite(float(x)):
        return ""
    return f"{float(x):.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--top", type=int, default=0)
    args = parser.parse_args()
    rows = parse(args.root)
    good = [r for r in rows if not r["error"] and r["max_val_step"] >= 100]
    if not good:
        raise SystemExit("No successful runs with validation data found")
    common_step = 100 * (min(r["max_val_step"] for r in good) // 100)
    for r in good:
        r["common_val"] = r["vals"].get(common_step, math.inf)
        last_step = r["max_val_step"]
        r["last_val"] = r["vals"].get(last_step, math.inf)
    good.sort(key=lambda r: r["common_val"])
    shown = good[:args.top] if args.top > 0 else good

    print(f"Common validation comparison step: {common_step}\n")
    print("\t".join([
        "rank", "tag", "mode", "coeff", "cg", "damp", "scale", "outer",
        "qk_lr", "common_val", "last_val", "last_val_step", "min_val",
        "last_train", "train_step", "steps_per_hour", "cg_resid", "qk_rms",
        "capture_s", "optimizer_s", "fallback", "status",
    ]))
    for i, r in enumerate(shown, 1):
        print("\t".join(map(str, [
            i, r["tag"], r["mode"], r["coeff"], r["cg"], r["damp"],
            r["scale_mode"], r["outer"], r["qk_lr"], fmt(r["common_val"]),
            fmt(r["last_val"]), r["max_val_step"], fmt(r["min_val"]),
            fmt(r["last_train"]), r["last_train_step"], fmt(r["steps_per_hour"], 1),
            fmt(r["cg_resid"], 4), fmt(r["qk_rms"], 7), fmt(r["capture_s"], 4),
            fmt(r["optimizer_s"], 4), fmt(r["fallback"], 1), r["status"],
        ])))

    print("\nBest configuration by mode:")
    for mode in sorted({r["mode"] for r in good}):
        best = min((r for r in good if r["mode"] == mode), key=lambda r: r["common_val"])
        print(f"{mode:14s} common_val={best['common_val']:.4f} tag={best['tag']}")

    fisher = [r for r in good if r["mode"] == "fisher_qk"]
    controls = [r for r in good if r["mode"] == "newton_muon"]
    if fisher and controls:
        bf = min(fisher, key=lambda r: r["common_val"])
        bn = min(controls, key=lambda r: r["common_val"])
        gap = bf["common_val"] - bn["common_val"]
        print(
            f"\nBest Fisher-QK minus best Newton-Muon at common step: {gap:+.4f} "
            "(negative favors Fisher-QK)"
        )

    bad = [r for r in rows if r not in good]
    if bad:
        print("\nRuns requiring inspection:")
        for r in bad:
            print(r["tag"], "error=", r["error"], "max_val_step=", r["max_val_step"], "status=", r["status"])


if __name__ == "__main__":
    main()
