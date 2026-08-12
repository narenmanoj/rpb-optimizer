#!/usr/bin/env python3
"""Generate the focused Fisher-QK momentum sweep configuration table."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

FIELDS = [
    "index",
    "tag",
    "mode",
    "coeff",
    "cg_iters",
    "damp_rel",
    "scale_mode",
    "outer_scale",
    "qk_lr",
    "curv_batch",
    "curv_refresh",
    "momentum_mode",
    "momentum",
    "post_transform",
    "beta",
    "v_lr",
    "v_momentum",
    "qkv_control_lr",
    "qkv_control_momentum",
]


def row(
    tag: str,
    *,
    mode: str = "fisher_qk",
    coeff: str = "unit",
    cg_iters: int = 3,
    damp_rel: float = 0.1,
    scale_mode: str = "rms",
    outer_scale: float = 0.5,
    qk_lr: float = 0.00022,
    curv_batch: int = 1,
    curv_refresh: int = 1,
    momentum_mode: str = "none",
    momentum: float = 0.0,
    post_transform: str = "none",
    beta: float = 0.0,
    v_lr: float = 0.0004,
    v_momentum: float = 0.95,
    qkv_control_lr: float = 0.0004,
    qkv_control_momentum: float = 0.95,
) -> dict[str, object]:
    return {
        "tag": tag,
        "mode": mode,
        "coeff": coeff,
        "cg_iters": cg_iters,
        "damp_rel": damp_rel,
        "scale_mode": scale_mode,
        "outer_scale": outer_scale,
        "qk_lr": qk_lr,
        "curv_batch": curv_batch,
        "curv_refresh": curv_refresh,
        "momentum_mode": momentum_mode,
        "momentum": momentum,
        "post_transform": post_transform,
        "beta": beta,
        "v_lr": v_lr,
        "v_momentum": v_momentum,
        "qkv_control_lr": qkv_control_lr,
        "qkv_control_momentum": qkv_control_momentum,
    }


CONFIGS = [
    # Same-harness Newton-Muon controls.
    row(
        "ctrl_nm_lr0p0004_m0p95",
        mode="newton_muon",
        qkv_control_lr=0.0004,
        qkv_control_momentum=0.95,
    ),
    row(
        "ctrl_nm_lr0p00044_m0p97",
        mode="newton_muon",
        qkv_control_lr=0.00044,
        qkv_control_momentum=0.97,
    ),

    # No-momentum Fisher baselines at the two useful RMS scales.
    row(
        "fish_unit_cg3_nomom_rms0p00022",
        cg_iters=3,
        momentum_mode="none",
        momentum=0.0,
        qk_lr=0.00022,
    ),
    row(
        "fish_unit_cg3_nomom_rms0p00033",
        cg_iters=3,
        momentum_mode="none",
        momentum=0.0,
        qk_lr=0.00033,
    ),

    # Momentum on the Fisher system right-hand side.
    row(
        "fish_unit_cg3_rhs_m0p90_rms0p00022",
        cg_iters=3,
        momentum_mode="rhs",
        momentum=0.90,
        qk_lr=0.00022,
    ),
    row(
        "fish_unit_cg3_rhs_m0p90_rms0p00033",
        cg_iters=3,
        momentum_mode="rhs",
        momentum=0.90,
        qk_lr=0.00033,
    ),
    row(
        "fish_unit_cg3_rhs_m0p95_rms0p00022",
        cg_iters=3,
        momentum_mode="rhs",
        momentum=0.95,
        qk_lr=0.00022,
    ),
    row(
        "fish_unit_cg3_rhs_m0p95_rms0p00033",
        cg_iters=3,
        momentum_mode="rhs",
        momentum=0.95,
        qk_lr=0.00033,
    ),

    # Momentum after the Fisher solve, before RMS normalization.
    row(
        "fish_unit_cg3_direction_m0p95_rms0p00022",
        cg_iters=3,
        momentum_mode="direction",
        momentum=0.95,
        qk_lr=0.00022,
    ),
    row(
        "fish_unit_cg3_direction_m0p95_rms0p00033",
        cg_iters=3,
        momentum_mode="direction",
        momentum=0.95,
        qk_lr=0.00033,
    ),

    # CG1 + RHS momentum controls: momentum-smoothed raw-gradient-like directions.
    row(
        "fish_unit_cg1_rhs_m0p95_rms0p00022",
        cg_iters=1,
        momentum_mode="rhs",
        momentum=0.95,
        qk_lr=0.00022,
    ),
    row(
        "fish_unit_cg1_rhs_m0p95_rms0p00033",
        cg_iters=1,
        momentum_mode="rhs",
        momentum=0.95,
        qk_lr=0.00033,
    ),
]


def main() -> None:
    output = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "fisher_qk_momentum_sweep_configs.tsv"
    )
    rows = [{"index": index, **config} for index, config in enumerate(CONFIGS)]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for item in rows:
        key = str(item["mode"])
        counts[key] = counts.get(key, 0) + 1

    print(f"Wrote {output} with {len(rows)} configurations")
    print(" ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
