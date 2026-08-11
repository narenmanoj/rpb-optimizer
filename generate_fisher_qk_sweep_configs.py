#!/usr/bin/env python3
"""Generate the fixed first production sweep for joint Fisher-QK training."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

FIELDS = [
    "index", "tag", "mode", "coeff", "cg_iters", "damp_rel",
    "scale_mode", "outer_scale", "qk_lr", "curv_batch", "curv_refresh",
    "momentum_mode", "momentum", "post_transform", "beta",
    "v_lr", "v_momentum", "qkv_control_lr", "qkv_control_momentum",
]


def row(tag: str, *, mode: str = "fisher_qk", coeff: str = "projected",
        cg_iters: int = 3, damp_rel: float = 0.1, scale_mode: str = "native",
        outer_scale: float = 0.5, qk_lr: float = 0.00011,
        curv_batch: int = 1, curv_refresh: int = 1,
        momentum_mode: str = "none", momentum: float = 0.0,
        post_transform: str = "none", beta: float = 0.0,
        v_lr: float = 0.0004, v_momentum: float = 0.95,
        qkv_control_lr: float = 0.0004,
        qkv_control_momentum: float = 0.95) -> dict[str, object]:
    return dict(
        tag=tag, mode=mode, coeff=coeff, cg_iters=cg_iters,
        damp_rel=damp_rel, scale_mode=scale_mode,
        outer_scale=outer_scale, qk_lr=qk_lr,
        curv_batch=curv_batch, curv_refresh=curv_refresh,
        momentum_mode=momentum_mode, momentum=momentum,
        post_transform=post_transform, beta=beta,
        v_lr=v_lr, v_momentum=v_momentum,
        qkv_control_lr=qkv_control_lr,
        qkv_control_momentum=qkv_control_momentum,
    )


CONFIGS = [
    row("ctrl_nm_lr0p0004_m0p95", mode="newton_muon",
        qkv_control_lr=0.0004, qkv_control_momentum=0.95),
    row("ctrl_nm_lr0p00044_m0p97", mode="newton_muon",
        qkv_control_lr=0.00044, qkv_control_momentum=0.97),

    row("fish_proj_cg3_dr0p1_native_os0p5",
        coeff="projected", cg_iters=3, damp_rel=0.1,
        scale_mode="native", outer_scale=0.5),
    row("fish_proj_cg3_dr0p1_native_os1",
        coeff="projected", cg_iters=3, damp_rel=0.1,
        scale_mode="native", outer_scale=1.0),
    row("fish_proj_cg3_dr1_native_os1",
        coeff="projected", cg_iters=3, damp_rel=1.0,
        scale_mode="native", outer_scale=1.0),
    row("fish_unit_cg3_dr0p1_native_os0p5",
        coeff="unit", cg_iters=3, damp_rel=0.1,
        scale_mode="native", outer_scale=0.5),

    row("fish_proj_cg3_dr0p1_rms_lr0p00011",
        coeff="projected", cg_iters=3, damp_rel=0.1,
        scale_mode="rms", qk_lr=0.00011),
    row("fish_proj_cg3_dr0p1_rms_lr0p00022",
        coeff="projected", cg_iters=3, damp_rel=0.1,
        scale_mode="rms", qk_lr=0.00022),
    row("fish_proj_cg3_dr0p1_rms_lr0p00033",
        coeff="projected", cg_iters=3, damp_rel=0.1,
        scale_mode="rms", qk_lr=0.00033),
    row("fish_proj_cg3_dr1_rms_lr0p00011",
        coeff="projected", cg_iters=3, damp_rel=1.0,
        scale_mode="rms", qk_lr=0.00011),
    row("fish_unit_cg3_dr0p1_rms_lr0p00011",
        coeff="unit", cg_iters=3, damp_rel=0.1,
        scale_mode="rms", qk_lr=0.00011),
    row("fish_unit_cg1_dr0p1_rms_lr0p00011",
        coeff="unit", cg_iters=1, damp_rel=0.1,
        scale_mode="rms", qk_lr=0.00011),
]


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "fisher_qk_sweep_configs.tsv")
    rows = []
    for index, config in enumerate(CONFIGS):
        rows.append({"index": index, **config})
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for item in rows:
        counts[str(item["mode"])] = counts.get(str(item["mode"]), 0) + 1
    print(f"Wrote {output} with {len(rows)} configurations")
    print(" ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
