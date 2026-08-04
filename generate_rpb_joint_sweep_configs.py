#!/usr/bin/env python3
"""Generate a deterministic 64-task joint RPB/control sweep configuration.

The file contains 56 hybrid RPB configurations (8 explicit anchors plus 48
unique Sobol-discretized points) and 8 exact-control configurations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

COLUMNS = [
    "tag",
    "mode",
    "base_lr",
    "eta",
    "rowsign",
    "spectral",
    "precond",
    "rpb_momentum",
    "nor_enable",
    "nor_beta2",
    "matrix_momentum",
    "qkv_control_lr",
    "qkv_control_momentum",
]

ETA = [25.0, 30.0, 35.0, 40.0, 45.0]
ROWSIGN = [0.85, 0.90, 0.95, 1.00]
SPECTRAL = [0.75, 0.875, 1.00]
PRECOND = [0.00, 0.125, 0.25, 0.50, 1.00]
MOMENTUM = [0.85, 0.90, 0.95]
NOR_ENABLE = [0, 1]
NOR_BETA2 = [0.90, 0.95, 0.99]


def token(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def hybrid_row(
    eta: float,
    rowsign: float,
    spectral: float,
    precond: float,
    momentum: float,
    nor_enable: int,
    nor_beta2: float,
    suffix: str = "",
) -> dict[str, object]:
    tag = (
        f"hyb_e{token(eta)}_p{token(rowsign)}_sb{token(spectral)}_"
        f"pb{token(precond)}_m{token(momentum)}_"
        f"nor{nor_enable}b{token(nor_beta2)}{suffix}"
    )
    return {
        "tag": tag,
        "mode": "hybrid",
        "base_lr": 0.004,
        "eta": eta,
        "rowsign": rowsign,
        "spectral": spectral,
        "precond": precond,
        "rpb_momentum": momentum,
        "nor_enable": nor_enable,
        "nor_beta2": nor_beta2,
        "matrix_momentum": 0.95,
        "qkv_control_lr": 0.0004,
        "qkv_control_momentum": 0.95,
    }


def control_row(
    mode: str,
    base_lr: float,
    matrix_momentum: float,
) -> dict[str, object]:
    qkv_lr = 0.1 * base_lr
    return {
        "tag": (
            f"ctrl_{mode}_lr{token(base_lr)}_m{token(matrix_momentum)}"
        ),
        "mode": mode,
        "base_lr": base_lr,
        "eta": 0.0,
        "rowsign": 1.0,
        "spectral": 0.0,
        "precond": 1.0 if mode == "newton_muon" else 0.0,
        "rpb_momentum": 0.0,
        "nor_enable": 0,
        "nor_beta2": 0.95,
        "matrix_momentum": matrix_momentum,
        "qkv_control_lr": qkv_lr,
        "qkv_control_momentum": matrix_momentum,
    }


def discretize(u: float, choices: list[object]) -> object:
    idx = min(int(float(u) * len(choices)), len(choices) - 1)
    return choices[idx]


def key(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row[c] for c in COLUMNS[2:]) if row["mode"] == "hybrid" else (
        row["mode"], row["base_lr"], row["matrix_momentum"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("rpb_joint_sweep_configs.tsv"),
    )
    args = parser.parse_args()

    # Explicit anchors preserve the major conclusions and endpoints from Stages 1-3.
    rows: list[dict[str, object]] = [
        hybrid_row(35, 0.90, 1.00, 0.25, 0.90, 1, 0.90, "_anchor"),
        hybrid_row(35, 0.90, 1.00, 0.25, 0.90, 0, 0.90, "_anchor"),
        hybrid_row(35, 0.90, 1.00, 0.00, 0.90, 0, 0.90, "_anchor"),
        hybrid_row(35, 0.90, 1.00, 1.00, 0.90, 0, 0.90, "_anchor"),
        hybrid_row(35, 0.90, 0.75, 0.25, 0.90, 0, 0.90, "_anchor"),
        hybrid_row(35, 1.00, 1.00, 0.25, 0.90, 0, 0.90, "_anchor"),
        hybrid_row(25, 0.90, 0.00, 1.00, 0.90, 0, 0.90, "_rpbcontrol"),
        hybrid_row(35, 0.90, 1.00, 0.25, 0.90, 1, 0.99, "_anchor"),
    ]

    seen = {key(row) for row in rows}
    engine = torch.quasirandom.SobolEngine(
        dimension=7, scramble=True, seed=20260803
    )

    # Draw more points than needed because discretization can create duplicates.
    for point in engine.draw(512).tolist():
        row = hybrid_row(
            float(discretize(point[0], ETA)),
            float(discretize(point[1], ROWSIGN)),
            float(discretize(point[2], SPECTRAL)),
            float(discretize(point[3], PRECOND)),
            float(discretize(point[4], MOMENTUM)),
            int(discretize(point[5], NOR_ENABLE)),
            float(discretize(point[6], NOR_BETA2)),
        )
        k = key(row)
        if k in seen:
            continue
        seen.add(k)
        rows.append(row)
        if len(rows) == 56:
            break

    if len(rows) != 56:
        raise RuntimeError(f"Generated only {len(rows)} unique hybrid configs")

    # Exact controls. Newton-Muon uses the repository's known 0.004 neighborhood;
    # Muon includes its historical 0.0036 neighborhood. The momentum-0.90 anchors
    # test a nearby dynamics choice without expanding the control grid excessively.
    rows.extend(
        [
            control_row("newton_muon", 0.0036, 0.95),
            control_row("newton_muon", 0.0040, 0.95),
            control_row("newton_muon", 0.0044, 0.95),
            control_row("newton_muon", 0.0040, 0.90),
            control_row("muon", 0.0032, 0.95),
            control_row("muon", 0.0036, 0.95),
            control_row("muon", 0.0040, 0.95),
            control_row("muon", 0.0036, 0.90),
        ]
    )

    if len(rows) != 64:
        raise AssertionError(len(rows))
    tags = [str(row["tag"]) for row in rows]
    if len(tags) != len(set(tags)):
        raise RuntimeError("Duplicate tags generated")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")

    print(f"Wrote {args.output} with {len(rows)} configurations")
    print("hybrid=56 newton_muon=4 muon=4")


if __name__ == "__main__":
    main()
