#!/usr/bin/env python3
"""Rank the RPB-to-Newton-Muon bridge sweep at a common validation step."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ERROR_RE = re.compile(
    r"Traceback|RuntimeError|AssertionError|out of memory|CUDA error|Killed",
    re.IGNORECASE,
)

FIELDS = {
    "mode": r"\bMODE=([A-Za-z0-9_]+)",
    "base_lr": r"\bBASE_LR=([0-9.]+)",
    "qkv_lr": r"\bQKV_LR=([0-9.]+)",
    "momentum": r"\bMOMENTUM=([0-9.]+)",
    "bridge": r"\bBRIDGE_BLEND=([0-9.]+)",
    "rowsign": r"\bROWSIGN_POWER=([0-9.]+)",
    "radius": r"\bRADIUS_BLEND=([0-9.]+)",
    "headnorm": r"\bHEADNORM_BLEND=([0-9.]+)",
}


def first(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--min-step", type=int, default=1000)
    parser.add_argument("--top", type=int, default=0)
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

    common_step = 100 * (min(int(row["max_val_step"]) for row in good) // 100)
    for row in good:
        vals = row["vals"]
        assert isinstance(vals, dict)
        row["common_val"] = vals.get(common_step, float("inf"))

    good.sort(key=lambda row: float(row["common_val"]))

    print(f"Common validation comparison step: {common_step}\n")
    print(
        "rank\ttag\tmode\tbase_lr\tqkv_lr\tmomentum\tbridge\trowsign\t"
        "radius\theadnorm\tcommon_val\tlast_val\tlast_val_step\tmin_val\t"
        "last_train\ttrain_step\tstatus"
    )

    shown = good if args.top <= 0 else good[: args.top]
    for rank, row in enumerate(shown, 1):
        vals = row["vals"]
        assert isinstance(vals, dict)
        last_step = int(row["max_val_step"])
        print(
            f"{rank}\t{row['tag']}\t{row['mode']}\t{row['base_lr']}\t"
            f"{row['qkv_lr']}\t{row['momentum']}\t{row['bridge']}\t"
            f"{row['rowsign']}\t{row['radius']}\t{row['headnorm']}\t"
            f"{float(row['common_val']):.4f}\t{vals[last_step]:.4f}\t"
            f"{last_step}\t{float(row['min_val']):.4f}\t{row['last_train']}\t"
            f"{row['last_train_step']}\t{row['status']}"
        )

    exact = [row for row in good if row["mode"] == "newton_muon"]
    bridge = [row for row in good if row["mode"] == "bridge"]
    positive_bridge = [
        row for row in bridge if float(str(row["bridge"] or "0")) > 0.0
    ]
    alpha0 = [
        row for row in bridge if float(str(row["bridge"] or "0")) == 0.0
    ]
    internal = [row for row in bridge if "internal_recovery" in str(row["tag"])]

    def best(group: list[dict[str, object]]):
        return min(group, key=lambda item: float(item["common_val"])) if group else None

    best_exact = best(exact)
    best_bridge = best(bridge)
    best_positive = best(positive_bridge)
    best_alpha0 = best(alpha0)
    best_internal = best(internal)

    print("\nKey configurations:")
    for label, row in (
        ("best exact Newton-Muon", best_exact),
        ("best bridge (all alpha)", best_bridge),
        ("best bridge with alpha>0", best_positive),
        ("bridge alpha=0 recovery", best_alpha0),
        ("bridge internal recovery", best_internal),
    ):
        if row is None:
            print(f"{label:30s} MISSING")
        else:
            print(
                f"{label:30s} common_val={float(row['common_val']):.4f} "
                f"tag={row['tag']}"
            )

    if best_positive is not None and best_exact is not None:
        gap = float(best_positive["common_val"]) - float(best_exact["common_val"])
        print(
            "\nBest alpha>0 bridge minus exact Newton-Muon: "
            f"{gap:+.4f} (negative favors the RPB contribution)"
        )

    if best_alpha0 is not None and best_exact is not None:
        gap = float(best_alpha0["common_val"]) - float(best_exact["common_val"])
        print(
            "Bridge alpha=0 recovery minus exact Newton-Muon: "
            f"{gap:+.4f} (should be close to zero)"
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
