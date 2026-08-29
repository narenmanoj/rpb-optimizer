#!/usr/bin/env python3
"""Generate the bounded, symmetric 353M Newton-Muon/Fisher tuning grid."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

MODEL = dict(
    iterations=7500,
    warmdown_iters=2177,
    model_n_layer=24,
    model_n_head=16,
    model_n_embd=1024,
    model_vocab_size=50257,
    batch_size=512,
    device_batch_size=32,
    sequence_length=1024,
    val_every=100,
    diag_every=200,
    expected_gpu="H200",
    timeout="1800m",
    smoke=False,
    seed=3,
)


def common_env() -> dict[str, Any]:
    return {
        "CYCLEA_ADAMW_BETA1": 0.9,
        "CYCLEA_ADAMW_BETA2": 0.95,
        "CYCLEA_ADAMW_LR": 0.001,
        "CYCLEA_ADAMW_WEIGHT_DECAY": 0.0,
        "CYCLEA_BACKBONE_MODE": "newton_muon",
        "CYCLEA_HEAD_LR": 0.004,
        "CYCLEA_MATRIX_LR": 0.0004,
        "CYCLEA_MATRIX_MOMENTUM": 0.95,
        "CYCLEA_QKV_LR": 0.00044,
        "CYCLEA_QKV_MOMENTUM": 0.97,
        "CYCLEA_SYSTEM_MODE": "newton_muon",
        "CYCLEA_V_ADAMW_BETA1": 0.9,
        "CYCLEA_V_ADAMW_BETA2": 0.95,
        "CYCLEA_V_ADAMW_WEIGHT_DECAY": 0.1,
        "CYCLEA_V_LR": 0.00044,
        "CYCLEA_V_MODE": "newton_muon",
        "CYCLEA_V_MOMENTUM": 0.97,
        "FISHER_BETA": 0.0,
        "FISHER_BLEND_SCALE_MODE": "fisher",
        "FISHER_CG_ITERS": 3,
        "FISHER_CG_TOL": 0.0,
        "FISHER_COEFF_FLOOR": 0.001,
        "FISHER_COEFF_MODE": "unit",
        "FISHER_COEFF_NORMALIZE": "median",
        "FISHER_CURV_BATCH": 1,
        "FISHER_CURV_PRECISION": "bf16",
        "FISHER_CURV_REFRESH": 4,
        "FISHER_CURV_REFRESH_LATE": 4,
        "FISHER_CURV_REFRESH_SWITCH_STEP": -1,
        "FISHER_DAMP_FLOOR": 1e-8,
        "FISHER_DAMP_REL": 0.1,
        "FISHER_LAYER_MASK": "",
        "FISHER_LAYER_POLICY": "all_fisher",
        "FISHER_MOMENTUM": 0.9,
        "FISHER_MOMENTUM_MODE": "rhs",
        "FISHER_NM_BLEND_END": 0.0,
        "FISHER_NM_BLEND_SCHEDULE": "constant",
        "FISHER_NM_BLEND_SCHEDULE_END": 0,
        "FISHER_NM_BLEND_SCHEDULE_START": 0,
        "FISHER_NM_BLEND_START": 0.0,
        "FISHER_NM_LR": 0.00044,
        "FISHER_NM_MOMENTUM": 0.97,
        "FISHER_NM_NESTEROV": 1,
        "FISHER_NM_SHADOW": 0,
        "FISHER_OUTER_SCALE": 0.5,
        "FISHER_POST_TRANSFORM": "none",
        "FISHER_QK_LR": 0.00022,
        "FISHER_QK_LR_END": 0.00022,
        "FISHER_QK_LR_SCHEDULE": "constant",
        "FISHER_QK_LR_SCHEDULE_END": 0,
        "FISHER_QK_LR_SCHEDULE_START": 0,
        "FISHER_SCALE_MODE": "rms",
        "FISHER_SCORE_OSC_CAP": 0.0,
        "FISHER_SPECTRAL_BLEND": 0.0,
        "QKV_CONTROL_STEPS": 5,
        "RPB_PRECOND_EWMA": 0.95,
        "RPB_PRECOND_INIT_DIAG": 0.001,
        "RPB_PRECOND_REFRESH": 32,
        "RPB_RIDGE_MULT": 0.2,
    }


def make_row(tag: str, family: str, description: str, env: dict[str, Any]) -> dict[str, Any]:
    row = dict(MODEL)
    row.update(tag=tag, family=family, description=description, env=env)
    return row


def nm_row(tag: str, matrix_lr: float, matrix_mom: float, qkv_lr: float, qkv_mom: float,
           description: str) -> dict[str, Any]:
    env = common_env()
    env.update({
        "CYCLEA_SYSTEM_MODE": "newton_muon",
        "CYCLEA_BACKBONE_MODE": "newton_muon",
        "CYCLEA_MATRIX_LR": matrix_lr,
        "CYCLEA_MATRIX_MOMENTUM": matrix_mom,
        "CYCLEA_QKV_LR": qkv_lr,
        "CYCLEA_QKV_MOMENTUM": qkv_mom,
    })
    return make_row(tag, "nm_tune", description, env)


def spectral_env(*, qk_lr: float = 0.00022, spectral_blend: float = 0.75,
                 blend_start: int = 500, blend_end: int = 1000,
                 scale_start: int = 1000, scale_end: int = 1500,
                 nm_lr: float = 0.00044, nm_mom: float = 0.97,
                 matrix_lr: float = 0.0004, matrix_mom: float = 0.95,
                 v_lr: float = 0.00044, v_mom: float = 0.97) -> dict[str, Any]:
    env = common_env()
    env.update({
        "CYCLEA_SYSTEM_MODE": "fisher",
        "CYCLEA_BACKBONE_MODE": "newton_muon",
        "CYCLEA_MATRIX_LR": matrix_lr,
        "CYCLEA_MATRIX_MOMENTUM": matrix_mom,
        "CYCLEA_QKV_LR": nm_lr,
        "CYCLEA_QKV_MOMENTUM": nm_mom,
        "CYCLEA_V_MODE": "newton_muon",
        "CYCLEA_V_LR": v_lr,
        "CYCLEA_V_MOMENTUM": v_mom,
        "FISHER_QK_LR": qk_lr,
        "FISHER_QK_LR_END": nm_lr,
        "FISHER_QK_LR_SCHEDULE": "linear",
        "FISHER_QK_LR_SCHEDULE_START": scale_start,
        "FISHER_QK_LR_SCHEDULE_END": scale_end,
        "FISHER_SPECTRAL_BLEND": spectral_blend,
        "FISHER_NM_BLEND_START": 0.0,
        "FISHER_NM_BLEND_END": 1.0,
        "FISHER_NM_BLEND_SCHEDULE": "linear",
        "FISHER_NM_BLEND_SCHEDULE_START": blend_start,
        "FISHER_NM_BLEND_SCHEDULE_END": blend_end,
        "FISHER_NM_LR": nm_lr,
        "FISHER_NM_MOMENTUM": nm_mom,
        "FISHER_NM_NESTEROV": 1,
        "FISHER_NM_SHADOW": 1,
    })
    return env


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Eight Newton-Muon configurations: the same bounded family used at 124M.
    rows.extend([
        nm_row("nm_hist_q44m97", 0.0004, 0.95, 0.00044, 0.97,
               "historical split endpoint"),
        nm_row("nm_u36_m95", 0.00036, 0.95, 0.00036, 0.95,
               "uniform matrix/QKV LR 0.00036, momentum 0.95"),
        nm_row("nm_u40_m95", 0.0004, 0.95, 0.0004, 0.95,
               "uniform matrix/QKV LR 0.00040, momentum 0.95"),
        nm_row("nm_u44_m95", 0.00044, 0.95, 0.00044, 0.95,
               "uniform matrix/QKV LR 0.00044, momentum 0.95"),
        nm_row("nm_u48_m95", 0.00048, 0.95, 0.00048, 0.95,
               "uniform matrix/QKV LR 0.00048, momentum 0.95"),
        nm_row("nm_u36_m97", 0.00036, 0.97, 0.00036, 0.97,
               "uniform matrix/QKV LR 0.00036, momentum 0.97"),
        nm_row("nm_u40_m97", 0.0004, 0.97, 0.0004, 0.97,
               "uniform matrix/QKV LR 0.00040, momentum 0.97"),
        nm_row("nm_u44_m97", 0.00044, 0.97, 0.00044, 0.97,
               "uniform matrix/QKV LR 0.00044, momentum 0.97"),
    ])

    # Eight spectral-transition configurations with an equal tuning budget.
    rows.extend([
        make_row("spec_center", "spectral_tune",
                 "frozen Cycle-C spectral transition",
                 spectral_env()),
        make_row("spec_qk18", "spectral_tune",
                 "lower Fisher Q/K RMS 0.00018",
                 spectral_env(qk_lr=0.00018)),
        make_row("spec_qk26", "spectral_tune",
                 "higher Fisher Q/K RMS 0.00026",
                 spectral_env(qk_lr=0.00026)),
        make_row("spec_a50", "spectral_tune",
                 "spectral blend 0.50",
                 spectral_env(spectral_blend=0.50)),
        make_row("spec_a100", "spectral_tune",
                 "full spectral blend 1.00",
                 spectral_env(spectral_blend=1.00)),
        make_row("spec_early_400_900", "spectral_tune",
                 "earlier direction handoff 400-900 and scale ramp 900-1400",
                 spectral_env(blend_start=400, blend_end=900,
                              scale_start=900, scale_end=1400)),
        make_row("spec_late_700_1400", "spectral_tune",
                 "later direction handoff 700-1400 and scale ramp 1400-1900",
                 spectral_env(blend_start=700, blend_end=1400,
                              scale_start=1400, scale_end=1900)),
        make_row("spec_uniform_nm40_m95", "spectral_tune",
                 "spectral warmup into uniform Newton-Muon LR 0.00040 momentum 0.95",
                 spectral_env(nm_lr=0.0004, nm_mom=0.95,
                              matrix_lr=0.0004, matrix_mom=0.95,
                              v_lr=0.0004, v_mom=0.95)),
    ])

    for idx, row in enumerate(rows):
        row["index"] = idx
    return rows


def write(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "scale353m_tuning_configs.jsonl"
    tsv = out_dir / "scale353m_tuning_configs.tsv"
    jsonl.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    fields = [
        "index", "tag", "family", "seed", "matrix_lr", "matrix_momentum",
        "qkv_lr", "qkv_momentum", "fisher_qk_lr", "spectral_blend",
        "blend_start", "blend_end", "scale_start", "scale_end",
        "late_nm_lr", "late_nm_momentum", "description",
    ]
    with tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            e = r["env"]
            w.writerow({
                "index": r["index"],
                "tag": r["tag"],
                "family": r["family"],
                "seed": r["seed"],
                "matrix_lr": e["CYCLEA_MATRIX_LR"],
                "matrix_momentum": e["CYCLEA_MATRIX_MOMENTUM"],
                "qkv_lr": e["CYCLEA_QKV_LR"],
                "qkv_momentum": e["CYCLEA_QKV_MOMENTUM"],
                "fisher_qk_lr": e["FISHER_QK_LR"],
                "spectral_blend": e["FISHER_SPECTRAL_BLEND"],
                "blend_start": e["FISHER_NM_BLEND_SCHEDULE_START"],
                "blend_end": e["FISHER_NM_BLEND_SCHEDULE_END"],
                "scale_start": e["FISHER_QK_LR_SCHEDULE_START"],
                "scale_end": e["FISHER_QK_LR_SCHEDULE_END"],
                "late_nm_lr": e["FISHER_NM_LR"],
                "late_nm_momentum": e["FISHER_NM_MOMENTUM"],
                "description": r["description"],
            })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path, nargs="?", default=Path("."))
    args = ap.parse_args()
    rows = build_rows()
    write(rows, args.out_dir)
    print(f"Wrote scale353m_tuning_configs: {len(rows)} configurations")
    print("nm_tune=8 spectral_tune=8 tuning_seed=3")


if __name__ == "__main__":
    main()
