#!/usr/bin/env python3
"""Generate a deterministic 64-task RPB-to-Newton-Muon bridge sweep.

The table contains 56 bridge configurations and 8 exact Newton-Muon controls.
The bridge has a literal exact-control endpoint:

  QKV_OPT_MODE=bridge, RPB_NM_BRIDGE_BLEND=0

uses the raw QKV gradient, full input-Gram inverse before momentum, a separate
Q/K/V matrix-sign map, and the standard Muon/Newton-Muon update scale.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

COLUMNS = [
    "tag",
    "mode",
    "base_lr",
    "qkv_lr",
    "momentum",
    "bridge_blend",
    "rowsign",
    "radius_blend",
    "headnorm_blend",
]

QKV_LR = [0.00032, 0.00036, 0.00040, 0.00044, 0.00048]
MOMENTUM = [0.90, 0.95, 0.97]
BRIDGE = [0.125, 0.25, 0.50, 0.75, 1.00]
ROWSIGN = [0.00, 0.25, 0.50, 0.70, 0.85]
RADIUS = [0.00, 0.50, 1.00]
HEADNORM = [0.00, 0.50, 1.00]


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def bridge_row(
    qkv_lr: float,
    momentum: float,
    bridge_blend: float,
    rowsign: float,
    radius_blend: float,
    headnorm_blend: float,
    suffix: str = "",
) -> dict[str, object]:
    tag = (
        f"br_a{token(bridge_blend)}_p{token(rowsign)}_"
        f"rb{token(radius_blend)}_hb{token(headnorm_blend)}_"
        f"lr{token(qkv_lr)}_m{token(momentum)}{suffix}"
    )
    return {
        "tag": tag,
        "mode": "bridge",
        "base_lr": 0.004,
        "qkv_lr": qkv_lr,
        "momentum": momentum,
        "bridge_blend": bridge_blend,
        "rowsign": rowsign,
        "radius_blend": radius_blend,
        "headnorm_blend": headnorm_blend,
    }


def control_row(qkv_lr: float, momentum: float) -> dict[str, object]:
    return {
        "tag": f"ctrl_nm_lr{token(qkv_lr)}_m{token(momentum)}",
        "mode": "newton_muon",
        "base_lr": 0.004,
        "qkv_lr": qkv_lr,
        "momentum": momentum,
        "bridge_blend": 0.0,
        "rowsign": 0.0,
        "radius_blend": 0.0,
        "headnorm_blend": 0.0,
    }


def discretize(u: float, choices: list[float]) -> float:
    index = min(int(float(u) * len(choices)), len(choices) - 1)
    return float(choices[index])


def key(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row[column] for column in COLUMNS[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("rpb_nm_bridge_configs.tsv"),
    )
    args = parser.parse_args()

    # Explicit anchors make the interpolation and recovery tests easy to read.
    rows: list[dict[str, object]] = [
        # The bridge implementation should recover the exact control at alpha=0.
        bridge_row(0.00040, 0.95, 0.00, 0.85, 1.00, 1.00, "_exact_recovery"),
        # Trace a simple path away from Newton-Muon using the best joint-sweep RPB geometry.
        bridge_row(0.00040, 0.95, 0.125, 0.85, 1.00, 1.00, "_path"),
        bridge_row(0.00040, 0.95, 0.25, 0.85, 1.00, 1.00, "_path"),
        bridge_row(0.00040, 0.95, 0.50, 0.85, 1.00, 1.00, "_path"),
        bridge_row(0.00040, 0.95, 0.75, 0.85, 1.00, 1.00, "_path"),
        bridge_row(0.00040, 0.95, 1.00, 0.85, 1.00, 1.00, "_path"),
        # With p=0 and both headwise scale patterns removed, the RPB candidate itself
        # is collinear with the raw gradient; alpha=1 is an internal recovery check.
        bridge_row(0.00040, 0.95, 1.00, 0.00, 0.00, 0.00, "_internal_recovery"),
        bridge_row(0.00040, 0.95, 1.00, 0.50, 0.50, 0.50, "_mid_geometry"),
    ]

    seen = {key(row) for row in rows}
    engine = torch.quasirandom.SobolEngine(
        dimension=6,
        scramble=True,
        seed=20260805,
    )

    for point in engine.draw(1024).tolist():
        row = bridge_row(
            discretize(point[0], QKV_LR),
            discretize(point[1], MOMENTUM),
            discretize(point[2], BRIDGE),
            discretize(point[3], ROWSIGN),
            discretize(point[4], RADIUS),
            discretize(point[5], HEADNORM),
        )
        row_key = key(row)
        if row_key in seen:
            continue
        seen.add(row_key)
        rows.append(row)
        if len(rows) == 56:
            break

    if len(rows) != 56:
        raise RuntimeError(f"Generated only {len(rows)} unique bridge rows")

    # Exact Newton-Muon controls around the stable learning-rate/momentum region.
    rows.extend(
        [
            control_row(0.00032, 0.95),
            control_row(0.00036, 0.95),
            control_row(0.00040, 0.95),
            control_row(0.00044, 0.95),
            control_row(0.00048, 0.95),
            control_row(0.00036, 0.97),
            control_row(0.00040, 0.97),
            control_row(0.00044, 0.97),
        ]
    )

    if len(rows) != 64:
        raise AssertionError(len(rows))
    tags = [str(row["tag"]) for row in rows]
    if len(tags) != len(set(tags)):
        raise RuntimeError("Duplicate tags generated")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        handle.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[column]) for column in COLUMNS) + "\n")

    print(f"Wrote {args.output} with {len(rows)} configurations")
    print("bridge=56 newton_muon=8")


if __name__ == "__main__":
    main()
