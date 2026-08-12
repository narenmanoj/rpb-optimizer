#!/usr/bin/env python3
"""Rank the focused Fisher-QK momentum/scale refinement sweep."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any


def tb_last(run_dir: Path, tag: str) -> float | None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception:
        return None

    events = sorted(
        run_dir.rglob("events.out.tfevents*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not events:
        return None

    try:
        accumulator = EventAccumulator(
            str(events[-1].parent),
            size_guidance={"scalars": 0},
        )
        accumulator.Reload()
        if tag not in set(accumulator.Tags().get("scalars", [])):
            return None
        values = accumulator.Scalars(tag)
        return float(values[-1].value) if values else None
    except Exception:
        return None


def one(text: str, pattern: str, default: str = "") -> str:
    match = re.search(pattern, text, re.M)
    return match.group(1) if match else default


def parse(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for run_out in sorted(root.glob("*/run.out")):
        text = run_out.read_text(errors="ignore")

        val_entries = [
            {
                "step": int(step),
                "loss": float(loss),
                "time_ms": float(time_ms),
            }
            for step, loss, time_ms in re.findall(
                r"step:(\d+)/6200 val_loss:([0-9.]+) "
                r"train_time:([0-9.]+)ms",
                text,
            )
        ]

        train_entries = re.findall(
            r"step:(\d+)/6200 train_loss:([0-9.]+) "
            r"train_time:([0-9.]+)ms",
            text,
        )

        status = re.findall(r"finished with status=(\d+)", text)
        error = bool(
            re.search(
                r"Traceback|RuntimeError|AssertionError|out of memory|"
                r"CUDA error|Killed|ValueError",
                text,
                re.I,
            )
        )

        vals = {entry["step"]: entry["loss"] for entry in val_entries}
        max_val_step = max(vals) if vals else -1
        max_val_time_ms = max((entry["time_ms"] for entry in val_entries), default=-1.0)

        row: dict[str, Any] = {
            "tag": run_out.parent.name,
            "mode": one(text, r"^MODE=(\S+)", "unknown"),
            "coeff": one(text, r"^COEFF=(\S+)", ""),
            "cg": one(text, r"^CG_ITERS=(\S+)", ""),
            "damp": one(text, r"^DAMP_REL=(\S+)", ""),
            "scale_mode": one(text, r"^SCALE_MODE=(\S+)", ""),
            "outer": one(text, r"^OUTER_SCALE=(\S+)", ""),
            "qk_lr": one(text, r"^QK_LR=(\S+)", ""),
            "momentum_mode": one(text, r"^MOMENTUM_MODE=(\S+)", ""),
            "momentum": one(text, r"^MOMENTUM=(\S+)", ""),
            "qkv_control_lr": one(text, r"^QKV_CONTROL_LR=(\S+)", ""),
            "qkv_control_momentum": one(
                text,
                r"^QKV_CONTROL_MOMENTUM=(\S+)",
                "",
            ),
            "val_entries": val_entries,
            "vals": vals,
            "max_val_step": max_val_step,
            "max_val_time_ms": max_val_time_ms,
            "min_val": min(vals.values()) if vals else math.inf,
            "last_train_step": int(train_entries[-1][0]) if train_entries else -1,
            "last_train": float(train_entries[-1][1]) if train_entries else math.nan,
            "train_time_ms": float(train_entries[-1][2]) if train_entries else math.nan,
            "status": status[-1] if status else "",
            "error": error,
        }

        if row["train_time_ms"] > 0 and row["last_train_step"] >= 0:
            row["steps_per_hour"] = (
                row["last_train_step"] * 3_600_000.0 / row["train_time_ms"]
            )
        else:
            row["steps_per_hour"] = math.nan

        for tag, key in [
            ("fisher_qk/cg_final_residual_mean", "cg_resid"),
            ("fisher_qk/qk_update_rms_mean", "qk_rms"),
            ("fisher_qk/current_gradient_dot_mean", "current_dot"),
            ("fisher_qk/score_osc_max_mean", "score_osc"),
            ("fisher_qk/bilinear_ratio_mean", "bilinear_ratio"),
            ("fisher_qk/curvature_capture_seconds", "capture_s"),
            ("fisher_qk/optimizer_seconds", "optimizer_s"),
            ("fisher_qk/descent_fallback_layers", "fallback"),
        ]:
            row[key] = tb_last(run_out.parent, tag)

        rows.append(row)

    return rows


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def select_at_time(row: dict[str, Any], common_time_ms: float) -> dict[str, float] | None:
    eligible = [
        entry
        for entry in row["val_entries"]
        if entry["time_ms"] <= common_time_ms + 1e-6
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda entry: entry["time_ms"])


def print_table(rows: list[dict[str, Any]], *, top: int = 0) -> None:
    shown = rows[:top] if top > 0 else rows

    print(
        "\t".join(
            [
                "rank",
                "tag",
                "mode",
                "cg",
                "mom_mode",
                "momentum",
                "qk_lr",
                "common_val",
                "last_val",
                "last_val_step",
                "time_val",
                "time_val_step",
                "steps_per_hour",
                "cg_resid",
                "qk_rms",
                "current_dot",
                "score_osc",
                "bilinear",
                "capture_s",
                "optimizer_s",
                "fallback",
                "status",
            ]
        )
    )

    for rank, row in enumerate(shown, 1):
        print(
            "\t".join(
                map(
                    str,
                    [
                        rank,
                        row["tag"],
                        row["mode"],
                        row["cg"],
                        row["momentum_mode"],
                        row["momentum"],
                        row["qk_lr"],
                        fmt(row["common_val"]),
                        fmt(row["last_val"]),
                        row["max_val_step"],
                        fmt(row.get("time_val")),
                        row.get("time_val_step", ""),
                        fmt(row["steps_per_hour"], 1),
                        fmt(row["cg_resid"], 4),
                        fmt(row["qk_rms"], 7),
                        fmt(row["current_dot"], 7),
                        fmt(row["score_osc"], 4),
                        fmt(row["bilinear_ratio"], 4),
                        fmt(row["capture_s"], 4),
                        fmt(row["optimizer_s"], 4),
                        fmt(row["fallback"], 1),
                        row["status"],
                    ],
                )
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--top", type=int, default=0)
    args = parser.parse_args()

    rows = parse(args.root)
    good = [row for row in rows if not row["error"] and row["max_val_step"] >= 100]
    if not good:
        raise SystemExit("No successful runs with validation data found")

    common_step = 100 * (min(row["max_val_step"] for row in good) // 100)
    common_time_ms = min(row["max_val_time_ms"] for row in good)

    for row in good:
        row["common_val"] = row["vals"].get(common_step, math.inf)
        row["last_val"] = row["vals"].get(row["max_val_step"], math.inf)
        time_entry = select_at_time(row, common_time_ms)
        if time_entry is None:
            row["time_val"] = math.inf
            row["time_val_step"] = -1
        else:
            row["time_val"] = time_entry["loss"]
            row["time_val_step"] = time_entry["step"]

    good.sort(key=lambda row: row["common_val"])

    print(f"Common validation comparison step: {common_step}")
    print(f"Common wall-clock cutoff: {common_time_ms / 3_600_000.0:.3f} hours\n")
    print_table(good, top=args.top)

    print("\nBest configuration by optimizer mode at common step:")
    for mode in sorted({row["mode"] for row in good}):
        best = min(
            (row for row in good if row["mode"] == mode),
            key=lambda row: row["common_val"],
        )
        print(
            f"{mode:14s} common_val={best['common_val']:.4f} "
            f"tag={best['tag']}"
        )

    fisher_rows = [row for row in good if row["mode"] == "fisher_qk"]
    control_rows = [row for row in good if row["mode"] == "newton_muon"]

    if fisher_rows:
        print("\nBest Fisher-QK configuration by momentum mode:")
        for mode in ["none", "rhs", "direction"]:
            candidates = [
                row for row in fisher_rows if row["momentum_mode"] == mode
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda row: row["common_val"])
            print(
                f"{mode:10s} common_val={best['common_val']:.4f} "
                f"tag={best['tag']}"
            )

    if fisher_rows and control_rows:
        best_fisher = min(fisher_rows, key=lambda row: row["common_val"])
        best_control = min(control_rows, key=lambda row: row["common_val"])
        step_gap = best_fisher["common_val"] - best_control["common_val"]

        best_fisher_time = min(fisher_rows, key=lambda row: row["time_val"])
        best_control_time = min(control_rows, key=lambda row: row["time_val"])
        time_gap = best_fisher_time["time_val"] - best_control_time["time_val"]

        print(
            "\nBest Fisher-QK minus best Newton-Muon at common step: "
            f"{step_gap:+.4f} (negative favors Fisher-QK)"
        )
        print(
            "Best Fisher-QK minus best Newton-Muon at common wall time: "
            f"{time_gap:+.4f} (negative favors Fisher-QK)"
        )
        print(
            f"  Fisher wall-time winner: {best_fisher_time['tag']} "
            f"loss={best_fisher_time['time_val']:.4f} "
            f"step={best_fisher_time['time_val_step']}"
        )
        print(
            f"  Newton-Muon wall-time winner: {best_control_time['tag']} "
            f"loss={best_control_time['time_val']:.4f} "
            f"step={best_control_time['time_val_step']}"
        )

    print("\nMatched CG3-vs-CG1 comparisons at qk_lr=0.00022:")
    for momentum in ["0.85", "0.9", "0.925"]:
        cg3 = next(
            (
                row
                for row in fisher_rows
                if row["cg"] == "3"
                and row["momentum_mode"] == "rhs"
                and row["momentum"] == momentum
                and row["qk_lr"] == "0.00022"
            ),
            None,
        )
        cg1 = next(
            (
                row
                for row in fisher_rows
                if row["cg"] == "1"
                and row["momentum_mode"] == "rhs"
                and row["momentum"] == momentum
                and row["qk_lr"] == "0.00022"
            ),
            None,
        )
        if cg3 is None or cg1 is None:
            continue
        print(
            f"momentum={momentum}: CG3={cg3['common_val']:.4f}, "
            f"CG1={cg1['common_val']:.4f}, "
            f"CG3-CG1={cg3['common_val'] - cg1['common_val']:+.4f}"
        )

    print("\nCG3 RHS-momentum sweep at qk_lr=0.00022:")
    for momentum in ["0.85", "0.9", "0.925"]:
        candidate = next(
            (
                row
                for row in fisher_rows
                if row["cg"] == "3"
                and row["momentum_mode"] == "rhs"
                and row["momentum"] == momentum
                and row["qk_lr"] == "0.00022"
            ),
            None,
        )
        if candidate is not None:
            print(
                f"momentum={momentum}: common_val={candidate['common_val']:.4f} "
                f"tag={candidate['tag']}"
            )

    print("\nCG3 RMS refinement at RHS momentum=0.90:")
    for qk_lr in ["0.00018", "0.00022", "0.00026"]:
        candidate = next(
            (
                row
                for row in fisher_rows
                if row["cg"] == "3"
                and row["momentum_mode"] == "rhs"
                and row["momentum"] == "0.9"
                and row["qk_lr"] == qk_lr
            ),
            None,
        )
        if candidate is not None:
            print(
                f"qk_lr={qk_lr}: common_val={candidate['common_val']:.4f} "
                f"tag={candidate['tag']}"
            )

    bad = [row for row in rows if row not in good]
    if bad:
        print("\nRuns requiring inspection:")
        for row in bad:
            print(
                row["tag"],
                "error=", row["error"],
                "max_val_step=", row["max_val_step"],
                "status=", row["status"],
            )


if __name__ == "__main__":
    main()
