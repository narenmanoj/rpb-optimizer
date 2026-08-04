#!/usr/bin/env python3
"""Rank joint hybrid / Newton-Muon / Muon sweeps at a common validation step."""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

ERROR_RE = re.compile(
    r"Traceback|RuntimeError|AssertionError|out of memory|CUDA error|Killed",
    re.IGNORECASE,
)

FIELDS = {
    "mode": r"\bMODE=([A-Za-z0-9_]+)",
    "base_lr": r"\bBASE_LR=([0-9.]+)",
    "eta": r"\bETA=([0-9.]+)",
    "rowsign": r"\bROWSIGN_POWER=([0-9.]+)",
    "spectral": r"\bSPECTRAL_BLEND=([0-9.]+)",
    "precond": r"\bPRECOND_BLEND=([0-9.]+)",
    "rpb_momentum": r"\bRPB_MOMENTUM=([0-9.]+)",
    "nor": r"\bNOR_ENABLE=([0-9]+)",
    "nor_beta2": r"\bNOR_BETA2=([0-9.]+)",
    "matrix_momentum": r"\bMATRIX_MOMENTUM=([0-9.]+)",
    "control_lr": r"\bQKV_CONTROL_LR=([0-9.]+)",
    "control_momentum": r"\bQKV_CONTROL_MOMENTUM=([0-9.]+)",
}


def first(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--min-step", type=int, default=1000)
    parser.add_argument("--top", type=int, default=0, help="0 prints every successful run")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for run_out in sorted(args.root.glob("*/run.out")):
        text = run_out.read_text(errors="ignore")
        vals = {
            int(step): float(value)
            for step, value in re.findall(
                r"step:(\d+)/6200 val_loss:([0-9.]+)", text
            )
        }
        trains = re.findall(r"step:(\d+)/6200 train_loss:([0-9.]+)", text)
        statuses = re.findall(r"finished with status=(\d+)", text)

        row: dict[str, object] = {
            "tag": run_out.parent.name,
            "vals": vals,
            "max_val_step": max(vals) if vals else -1,
            "min_val": min(vals.values()) if vals else float("inf"),
            "last_train": trains[-1][1] if trains else "",
            "last_train_step": trains[-1][0] if trains else "",
            "status": statuses[-1] if statuses else "",
            "error": bool(ERROR_RE.search(text)),
        }
        for name, pattern in FIELDS.items():
            row[name] = first(pattern, text)
        rows.append(row)

    good = [
        row for row in rows
        if not row["error"] and int(row["max_val_step"]) >= args.min_step
    ]
    if not good:
        raise SystemExit("No successful runs with enough validation data found.")

    common_step = 100 * (
        min(int(row["max_val_step"]) for row in good) // 100
    )
    for row in good:
        vals = row["vals"]
        assert isinstance(vals, dict)
        row["common_val"] = vals.get(common_step, float("inf"))

    good.sort(key=lambda row: float(row["common_val"]))

    print(f"Common validation comparison step: {common_step}\n")
    print(
        "rank\ttag\tmode\tbase_lr\teta\trowsign\tspectral\tprecond\t"
        "rpb_mom\tnor\tnor_beta2\tmatrix_mom\tcontrol_lr\tcontrol_mom\t"
        "common_val\tlast_val\tlast_val_step\tmin_val\tlast_train\t"
        "train_step\tstatus"
    )

    shown = good if args.top <= 0 else good[: args.top]
    for rank, row in enumerate(shown, 1):
        vals = row["vals"]
        assert isinstance(vals, dict)
        last_step = int(row["max_val_step"])
        last_val = vals[last_step]
        print(
            f"{rank}\t{row['tag']}\t{row['mode']}\t{row['base_lr']}\t"
            f"{row['eta']}\t{row['rowsign']}\t{row['spectral']}\t"
            f"{row['precond']}\t{row['rpb_momentum']}\t{row['nor']}\t"
            f"{row['nor_beta2']}\t{row['matrix_momentum']}\t"
            f"{row['control_lr']}\t{row['control_momentum']}\t"
            f"{float(row['common_val']):.4f}\t{last_val:.4f}\t{last_step}\t"
            f"{float(row['min_val']):.4f}\t{row['last_train']}\t"
            f"{row['last_train_step']}\t{row['status']}"
        )

    print("\nBest configuration by mode:")
    by_mode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in good:
        by_mode[str(row["mode"])].append(row)
    for mode in sorted(by_mode):
        row = min(by_mode[mode], key=lambda item: float(item["common_val"]))
        print(
            f"{mode:14s} common_val={float(row['common_val']):.4f} "
            f"tag={row['tag']}"
        )

    best_hybrid = min(
        (row for row in good if row["mode"] == "hybrid"),
        key=lambda item: float(item["common_val"]),
        default=None,
    )
    best_nm = min(
        (row for row in good if row["mode"] == "newton_muon"),
        key=lambda item: float(item["common_val"]),
        default=None,
    )
    if best_hybrid is not None and best_nm is not None:
        gap = float(best_hybrid["common_val"]) - float(best_nm["common_val"])
        print(
            "\nBest hybrid minus best Newton-Muon at common step: "
            f"{gap:+.4f} (negative favors hybrid)"
        )

    bad = [row for row in rows if row not in good]
    if bad:
        print("\nRuns requiring inspection:")
        for row in bad:
            print(
                row["tag"],
                "mode=", row["mode"],
                "error=", row["error"],
                "max_val_step=", row["max_val_step"],
                "status=", row["status"],
            )


if __name__ == "__main__":
    main()
