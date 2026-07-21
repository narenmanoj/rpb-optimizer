#!/usr/bin/env python3
"""Rank timeout-capped RPB hybrid sweeps at a common validation step."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

FIELDS = {
    "eta": r"\bETA=([0-9.]+)",
    "rowsign": r"\bROWSIGN_POWER=([0-9.]+)",
    "spectral": r"\bSPECTRAL_BLEND=([0-9.]+)",
    "precond": r"\bPRECOND_BLEND=([0-9.]+)",
    "nor": r"\bNOR_ENABLE=([0-9]+)",
    "nor_beta2": r"\bNOR_BETA2=([0-9.]+)",
}

ERROR_RE = re.compile(
    r"Traceback|RuntimeError|AssertionError|out of memory|CUDA error|Killed",
    re.IGNORECASE,
)


def first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Sweep output root containing */run.out")
    parser.add_argument("--min-step", type=int, default=1000)
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
            row[name] = first_match(pattern, text)
        rows.append(row)

    good = [
        row
        for row in rows
        if not row["error"] and int(row["max_val_step"]) >= args.min_step
    ]
    if not good:
        raise SystemExit("No successful runs with enough validation data found.")

    common_step = min(int(row["max_val_step"]) for row in good)
    common_step = 100 * (common_step // 100)

    for row in good:
        vals = row["vals"]
        assert isinstance(vals, dict)
        row["common_val"] = vals.get(common_step, float("inf"))

    good.sort(key=lambda row: float(row["common_val"]))

    print(f"Common validation comparison step: {common_step}\n")
    print(
        "rank\ttag\teta\trowsign\tspectral\tprecond\tnor\tnor_beta2\t"
        "common_val\tlast_val\tlast_val_step\tmin_val\t"
        "last_train\ttrain_step\tstatus"
    )

    for rank, row in enumerate(good, 1):
        vals = row["vals"]
        assert isinstance(vals, dict)
        last_step = int(row["max_val_step"])
        last_val = vals[last_step]
        print(
            f"{rank}\t{row['tag']}\t{row['eta']}\t{row['rowsign']}\t"
            f"{row['spectral']}\t{row['precond']}\t{row['nor']}\t"
            f"{row['nor_beta2']}\t{float(row['common_val']):.4f}\t"
            f"{last_val:.4f}\t{last_step}\t{float(row['min_val']):.4f}\t"
            f"{row['last_train']}\t{row['last_train_step']}\t{row['status']}"
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
