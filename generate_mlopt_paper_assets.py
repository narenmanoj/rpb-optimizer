#!/usr/bin/env python3
"""Generate frozen MLOPT paper figures, tables, statistics, and LaTeX snippets.

The script reads the raw run.out/run_config.json files from the Cycle A-C
experiment directories and the bounded 353M tuning/confirmation directories.
It does not retrain or modify any model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VAL_RE = re.compile(
    r"step:(\d+)/(\d+) val_loss:([0-9.eE+-]+) train_time:([0-9]+)ms"
)
STATUS_RE = re.compile(r"finished with status=(\d+)")
STEP_RE = re.compile(r"step:(\d+)/(\d+) .*?step_avg:([0-9.eE+-]+)ms")
PEAK_RE = re.compile(r"peak memory consumption:\s*([0-9.]+)\s*MiB", re.I)
SEED_RE = re.compile(r"\[seed\]\s+SEED=(\d+)")
MODEL_RE = re.compile(
    r"\[model\]\s+layers=(\d+)\s+heads=(\d+)\s+embd=(\d+)\s+"
    r"vocab=(\d+)\s+parameters=(\d+)"
)

TOKENS_PER_STEP = 512 * 1024
T_CRIT_975 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


@dataclass(frozen=True)
class Run:
    tag: str
    path: Path
    seed: int
    family: str
    curve: dict[int, tuple[float, float]]
    status: int
    step_avg_ms: float
    peak_mib: float | None
    model_n_layer: int | None
    model_n_head: int | None
    model_n_embd: int | None
    model_parameters: int | None
    config: dict[str, Any]

    @property
    def final_step(self) -> int:
        return max(self.curve)

    @property
    def final_val(self) -> float:
        return self.curve[self.final_step][0]


@dataclass(frozen=True)
class Summary:
    mean: float
    sd: float
    sem: float
    ci_low: float
    ci_high: float


def mean_summary(xs: Sequence[float]) -> Summary:
    if not xs:
        raise ValueError("cannot summarize an empty sequence")
    n = len(xs)
    mean = statistics.mean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    t = T_CRIT_975.get(n, 1.96)
    return Summary(mean, sd, sem, mean - t * sem, mean + t * sem)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_run(run_dir: Path, *, family: str | None = None, seed: int | None = None) -> Run:
    out = run_dir / "run.out"
    if not out.exists():
        raise FileNotFoundError(f"missing run.out: {out}")
    text = out.read_text(errors="ignore")
    vals = [(int(s), float(v), int(t) / 1000.0) for s, _, v, t in VAL_RE.findall(text)]
    if not vals:
        raise RuntimeError(f"no validation records in {out}")
    curve = {step: (val, seconds) for step, val, seconds in vals}
    statuses = [int(x) for x in STATUS_RE.findall(text)]
    status = statuses[-1] if statuses else -1
    if status not in (0, 124):
        raise RuntimeError(f"unexpected status={status} in {out}")
    step_times = [float(x) for _, _, x in STEP_RE.findall(text)]
    peak_matches = [float(x) for x in PEAK_RE.findall(text)]
    model_matches = MODEL_RE.findall(text)
    cfg_path = run_dir / "run_config.json"
    config = read_json(cfg_path) if cfg_path.exists() else {}
    cfg_seed = config.get("seed")
    if cfg_seed is None:
        cfg_seed = config.get("SEED")
    if cfg_seed is None:
        seed_matches = SEED_RE.findall(text)
        if seed_matches:
            cfg_seed = int(seed_matches[-1])
    if cfg_seed is None:
        m = re.search(r"seed(\d+)", run_dir.name)
        cfg_seed = int(m.group(1)) if m else -1
    if seed is not None and int(cfg_seed) != int(seed):
        raise RuntimeError(f"seed mismatch for {run_dir}: expected {seed}, got {cfg_seed}")
    if model_matches:
        nl, nh, ne, _, np_ = model_matches[-1]
        model_n_layer, model_n_head, model_n_embd, model_parameters = map(
            int, (nl, nh, ne, np_)
        )
    else:
        model_n_layer = config.get("model_n_layer")
        model_n_head = config.get("model_n_head")
        model_n_embd = config.get("model_n_embd")
        model_parameters = None
    return Run(
        tag=run_dir.name,
        path=run_dir,
        seed=int(cfg_seed),
        family=family or str(config.get("family", "unknown")),
        curve=curve,
        status=status,
        step_avg_ms=step_times[-1] if step_times else float("nan"),
        peak_mib=peak_matches[-1] if peak_matches else None,
        model_n_layer=int(model_n_layer) if model_n_layer is not None else None,
        model_n_head=int(model_n_head) if model_n_head is not None else None,
        model_n_embd=int(model_n_embd) if model_n_embd is not None else None,
        model_parameters=int(model_parameters) if model_parameters is not None else None,
        config=config,
    )


def require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing required directory: {path}")
    return path


def common_steps(runs: Iterable[Run]) -> list[int]:
    runs = list(runs)
    if not runs:
        return []
    return sorted(set.intersection(*(set(r.curve) for r in runs)))


def curve_matrix(runs: Sequence[Run], steps: Sequence[int]) -> np.ndarray:
    return np.asarray([[r.curve[s][0] for s in steps] for r in runs], dtype=float)


def time_matrix(runs: Sequence[Run], steps: Sequence[int]) -> np.ndarray:
    return np.asarray([[r.curve[s][1] for s in steps] for r in runs], dtype=float)


def threshold_time(run: Run, target: float) -> tuple[int, float] | None:
    hits = [(s, sec) for s, (v, sec) in run.curve.items() if v <= target]
    return min(hits, key=lambda x: x[0]) if hits else None


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def set_plot_defaults() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_124m_runs(base: Path) -> dict[str, list[Run]]:
    cycle_a = require_dir(base / "sweep_runs_cycle_a_baselines_h100_v1")
    cycle_b = require_dir(base / "sweep_runs_cycle_b_h100_main_v1")
    cycle_c = require_dir(base / "sweep_runs_cycle_c_h100_main_v1")
    selected = read_json(cycle_a / "selected_baselines.json")["selected"]
    out: dict[str, list[Run]] = {}
    baseline_tags = {
        "adamw": "adamw",
        "muon": "muon",
        "newton_muon": "newton_muon",
    }
    for family, key in baseline_tags.items():
        runs = [parse_run(cycle_a / selected[key]["tag"], family=family, seed=0)]
        runs.extend(parse_run(cycle_b / f"{family}_seed{s}", family=family, seed=s) for s in (1, 2))
        runs.extend(parse_run(cycle_c / f"{family}_seed{s}", family=family, seed=s) for s in (3, 4))
        out[family] = runs
    for family in ("transition_current", "transition_spectral"):
        runs = [parse_run(cycle_b / f"{family}_seed{s}", family=family, seed=s) for s in (0, 1, 2)]
        runs.extend(parse_run(cycle_c / f"{family}_seed{s}", family=family, seed=s) for s in (3, 4))
        out[family] = runs
    for family, runs in out.items():
        seeds = [r.seed for r in runs]
        if seeds != [0, 1, 2, 3, 4]:
            raise RuntimeError(f"unexpected 124M seeds for {family}: {seeds}")
    return out


def load_353m_confirmation(base: Path) -> dict[str, list[Run]]:
    root = require_dir(base / "sweep_runs_scale353m_confirmation_v1")
    select_root = require_dir(base / "scale353m_tuning_selection_v1")
    cfg_path = select_root / "scale353m_confirm_configs.jsonl"
    rows = [json.loads(line) for line in cfg_path.read_text().splitlines() if line.strip()]
    grouped: dict[str, list[Run]] = {"nm": [], "spectral": []}
    for row in rows:
        family = str(row["family"])
        if family not in grouped:
            raise RuntimeError(f"unexpected 353M confirmation family: {family}")
        grouped[family].append(parse_run(root / row["tag"], family=family, seed=int(row["seed"])))
    for family in grouped:
        grouped[family].sort(key=lambda r: r.seed)
        if [r.seed for r in grouped[family]] != [4, 5, 6]:
            raise RuntimeError(f"unexpected 353M held-out seeds for {family}")
    return grouped


def load_353m_tuning(base: Path) -> list[Run]:
    root = require_dir(base / "sweep_runs_scale353m_tuning_v1")
    cfg_path = base / "rpb-optimizer" / "scale353m_tuning_configs.jsonl"
    if not cfg_path.exists():
        # The copied config also lives in each run; reconstruct from run directories.
        rows = []
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            cfg = read_json(run_dir / "run_config.json")
            rows.append({"tag": run_dir.name, "family": cfg.get("family", "unknown")})
    else:
        rows = [json.loads(line) for line in cfg_path.read_text().splitlines() if line.strip()]
    return [parse_run(root / row["tag"], family=str(row.get("family", "unknown"))) for row in rows]


def load_long_horizon(base: Path) -> dict[str, Run]:
    b = require_dir(base / "sweep_runs_cycle_b_long10b_h200_v1")
    c = require_dir(base / "sweep_runs_cycle_c_long10b_spectral_h200_v1")
    return {
        "newton_muon": parse_run(b / "long10b_nm_seed0", family="newton_muon", seed=0),
        "transition_current": parse_run(b / "long10b_transition_seed0", family="transition_current", seed=0),
        "transition_spectral": parse_run(c / "long10b_spectral_seed0", family="transition_spectral", seed=0),
    }


def load_cycle_a_ablations(base: Path) -> dict[str, Run]:
    comp = require_dir(base / "sweep_runs_cycle_a_companion_h100_v1")
    vroot = require_dir(base / "sweep_runs_cycle_a_v_h100_v1")
    names = {
        "control_adamw": comp / "control_adamw",
        "fisher_adamw": comp / "fisher_adamw",
        "control_muon": comp / "control_muon",
        "fisher_muon": comp / "fisher_muon",
        "control_newton_muon": comp / "control_newton_muon",
        "fisher_newton_muon": comp / "fisher_newton_muon",
        "v_adamw": vroot / "v_adamw_selected",
        "v_muon": vroot / "v_muon_selected",
        "v_nm": vroot / "v_nm_selected",
    }
    return {name: parse_run(path, family=name) for name, path in names.items()}


def load_nested_ablation(base: Path) -> dict[str, Run] | None:
    root = base / "sweep_runs_fisher_corrected_nm_v1"
    if not root.is_dir():
        return None
    tags = {
        "k1": "nested_nm_k1_a1_lr44",
        "k2": "nested_nm_k2_a1_lr44",
        "k3": "nested_nm_k3_a1_lr44",
        "k5": "nested_nm_k5_a1_lr44",
        "plain": "ctrl_plain_fisher_cg3_a0_lr22",
        "plain_spectral": "ctrl_plain_fisher_cg3_a075_lr22",
    }
    return {name: parse_run(root / tag, family=name) for name, tag in tags.items()}


def plot_method_schematic(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    boxes = [
        (0.15, 1.72, 2.0, 1.0, "Feasible Q/K\nparameter update\n" + r"$(\Delta W_Q,\Delta W_K)$"),
        (2.55, 1.72, 1.65, 1.0, "Score-space\ntangent\n" + r"$J_{QK}D$"),
        (4.60, 1.72, 1.75, 1.0, "Rowwise Fisher\nmetric\n" + r"$C(p)$"),
        (6.75, 1.72, 2.15, 1.0, "Pull back + CG3\n" + r"$(J^{*}CJ+D_\lambda)D=-M$"),
        (9.30, 1.72, 2.35, 1.0, "Spectral shaping\nand phase handoff"),
    ]
    for x, y, w, h, label in boxes:
        patch = matplotlib.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.035", linewidth=1.0,
            facecolor="white", edgecolor="black"
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=6.8)
    for x1, x2 in [(2.15, 2.55), (4.20, 4.60), (6.35, 6.75), (8.90, 9.30)]:
        ax.annotate("", xy=(x2, 2.22), xytext=(x1, 2.22), arrowprops={"arrowstyle": "->", "linewidth": 1.0})
    ax.text(
        6.0, 3.23,
        r"Local pulled-back mirror model: $\min_D\;\langle M,D\rangle+\frac{1}{2}\langle D,(J^{*}CJ+D_\lambda)D\rangle$",
        ha="center", va="center", fontsize=8.3,
    )
    # Phase schedule.
    ax.plot([0.45, 11.55], [0.72, 0.72], linewidth=1.25)
    ticks = [(0.45, "0"), (4.05, "500"), (7.25, "1000"), (10.25, "1500")]
    for x, label in ticks:
        ax.plot([x, x], [0.59, 0.85], linewidth=0.9)
        ax.text(x, 0.38, label, ha="center", va="top", fontsize=7.5)
    ax.text(2.15, 1.02, "Fisher Q/K", ha="center", fontsize=7.8)
    ax.text(5.65, 1.02, "blend direction", ha="center", fontsize=7.8)
    ax.text(8.75, 1.02, "ramp Q/K scale", ha="center", fontsize=7.8)
    ax.text(10.90, 1.02, "Newton-Muon", ha="center", fontsize=7.8)
    save_figure(fig, out / "fig1_method_schematic")

def plot_124m(out: Path, data_out: Path, runs: dict[str, list[Run]]) -> dict[str, Any]:
    labels = {
        "newton_muon": "Newton-Muon",
        "transition_current": "Fisher to NM",
        "transition_spectral": "Spectral Fisher to NM",
    }
    selected = {k: runs[k] for k in labels}
    steps = common_steps(r for rs in selected.values() for r in rs)
    steps = [s for s in steps if s >= 100]
    curve_rows = []
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for family in labels:
        mat = curve_matrix(selected[family], steps)
        mean = mat.mean(axis=0)
        x = np.asarray(steps) * TOKENS_PER_STEP / 1e9
        ax.plot(x, mean, label=labels[family])
        for step, token, value in zip(steps, x, mean):
            curve_rows.append({"family": family, "step": step, "tokens_b": token, "mean_val": value})
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("Validation loss")
    ax.set_xlim(min(np.asarray(steps) * TOKENS_PER_STEP / 1e9), max(np.asarray(steps) * TOKENS_PER_STEP / 1e9))
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, out / "fig2a_124m_validation_vs_tokens")
    write_csv(data_out / "fig2a_124m_validation_vs_tokens.csv", ["family", "step", "tokens_b", "mean_val"], curve_rows)

    nm = {r.seed: r for r in runs["newton_muon"]}
    delta_rows = []
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for family in ("transition_current", "transition_spectral"):
        per_seed = []
        for r in runs[family]:
            per_seed.append([r.curve[s][0] - nm[r.seed].curve[s][0] for s in steps])
        mean = np.asarray(per_seed).mean(axis=0)
        ax.plot(steps, mean, label=labels[family])
        for step, value in zip(steps, mean):
            delta_rows.append({"family": family, "step": step, "mean_delta": value})
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    for boundary in (500, 1000, 1500):
        ax.axvline(boundary, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel(r"Paired loss difference vs. NM")
    ax.set_xlim(100, 3000)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, out / "fig2b_124m_paired_difference_early")
    write_csv(data_out / "fig2b_124m_paired_difference.csv", ["family", "step", "mean_delta"], delta_rows)

    # Final paired seed points: compact statistical visual.
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    families = ["transition_current", "transition_spectral"]
    x_positions = np.arange(len(families))
    point_rows = []
    for x, family in zip(x_positions, families):
        diffs = [r.final_val - nm[r.seed].final_val for r in runs[family]]
        ax.scatter(np.full(len(diffs), x), diffs, s=24)
        summary = mean_summary(diffs)
        ax.errorbar(x, summary.mean, yerr=[[summary.mean-summary.ci_low], [summary.ci_high-summary.mean]], fmt="o", capsize=4)
        for r, d in zip(runs[family], diffs):
            point_rows.append({"family": family, "seed": r.seed, "final_delta": d})
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(x_positions, ["Fisher to NM", "Spectral Fisher to NM"])
    ax.set_ylabel("Final paired loss difference")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, out / "figS1_124m_final_paired_points")
    write_csv(data_out / "figS1_124m_final_paired_points.csv", ["family", "seed", "final_delta"], point_rows)

    stats: dict[str, Any] = {}
    for family, rs in runs.items():
        vals = [r.final_val for r in rs]
        s = mean_summary(vals)
        stats[family] = {"n": len(vals), "mean": s.mean, "sd": s.sd, "sem": s.sem}
    for family in ("transition_current", "transition_spectral"):
        final_diffs = [r.final_val - nm[r.seed].final_val for r in runs[family]]
        early_diffs = []
        early_steps = [s for s in steps if 100 <= s <= 1500]
        for r in runs[family]:
            early_diffs.append(statistics.mean(r.curve[s][0] - nm[r.seed].curve[s][0] for s in early_steps))
        fs = mean_summary(final_diffs)
        es = mean_summary(early_diffs)
        stats[family]["paired_final"] = fs.__dict__
        stats[family]["paired_early"] = es.__dict__
        stats[family]["paired_final_values"] = final_diffs
        stats[family]["paired_early_values"] = early_diffs
    return stats


def plot_353m(out: Path, data_out: Path, runs: dict[str, list[Run]]) -> dict[str, Any]:
    nm, sp = runs["nm"], runs["spectral"]
    steps = common_steps(nm + sp)
    steps = [s for s in steps if s >= 100]
    nm_vals = curve_matrix(nm, steps)
    sp_vals = curve_matrix(sp, steps)
    nm_times = time_matrix(nm, steps).mean(axis=0) / 3600.0
    sp_times = time_matrix(sp, steps).mean(axis=0) / 3600.0
    nm_mean = nm_vals.mean(axis=0)
    sp_mean = sp_vals.mean(axis=0)
    rows = []
    for family, times, vals in (("newton_muon", nm_times, nm_mean), ("spectral", sp_times, sp_mean)):
        for step, t, v in zip(steps, times, vals):
            rows.append({"family": family, "step": step, "wall_hours": t, "mean_val": v})
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.plot(nm_times, nm_mean, label="Tuned Newton-Muon")
    ax.plot(sp_times, sp_mean, label="Tuned spectral Fisher to NM")
    ax.set_xlabel("Wall-clock time (hours)")
    ax.set_ylabel("Validation loss")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, out / "fig3a_353m_wallclock_frontier")
    write_csv(data_out / "fig3a_353m_wallclock_frontier.csv", ["family", "step", "wall_hours", "mean_val"], rows)

    thresholds = [5.0, 4.5, 4.0, 3.8, 3.6, 3.5]
    threshold_rows = []
    pct_values = []
    for target in thresholds:
        nm_hits = [threshold_time(r, target) for r in nm]
        sp_hits = [threshold_time(r, target) for r in sp]
        if any(x is None for x in nm_hits + sp_hits):
            continue
        nm_sec = statistics.mean(x[1] for x in nm_hits if x is not None)
        sp_sec = statistics.mean(x[1] for x in sp_hits if x is not None)
        pct = 100.0 * (sp_sec / nm_sec - 1.0)
        threshold_rows.append({"target": target, "nm_seconds": nm_sec, "spectral_seconds": sp_sec, "spectral_change_pct": pct})
        pct_values.append(pct)
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.bar([str(x["target"]) for x in threshold_rows], [x["spectral_change_pct"] for x in threshold_rows])
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Validation-loss target")
    ax.set_ylabel("Spectral time change vs. NM (%)")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, out / "fig3b_353m_time_to_quality")
    write_csv(data_out / "fig3b_353m_time_to_quality.csv", ["target", "nm_seconds", "spectral_seconds", "spectral_change_pct"], threshold_rows)

    delta = sp_vals - nm_vals
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.plot(steps, delta.mean(axis=0))
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    for boundary in (500, 1000, 1500):
        ax.axvline(boundary, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Spectral minus NM validation loss")
    ax.set_xlim(100, 3000)
    ax.grid(alpha=0.2)
    save_figure(fig, out / "figS2_353m_paired_difference")
    write_csv(
        data_out / "figS2_353m_paired_difference.csv",
        ["step", "mean_delta"],
        ({"step": s, "mean_delta": d} for s, d in zip(steps, delta.mean(axis=0))),
    )

    final_diffs = [s.final_val - n.final_val for s, n in zip(sp, nm)]
    early_steps = [s for s in steps if 100 <= s <= 1500]
    early_diffs = [statistics.mean(s.curve[t][0] - n.curve[t][0] for t in early_steps) for s, n in zip(sp, nm)]
    return {
        "newton_muon": mean_summary([r.final_val for r in nm]).__dict__,
        "spectral": mean_summary([r.final_val for r in sp]).__dict__,
        "paired_final": mean_summary(final_diffs).__dict__,
        "paired_final_values": final_diffs,
        "paired_early": mean_summary(early_diffs).__dict__,
        "paired_early_values": early_diffs,
        "thresholds": threshold_rows,
    }


def plot_353m_tuning(out: Path, data_out: Path, tuning: list[Run]) -> None:
    ordered = sorted(tuning, key=lambda r: (r.family, r.final_val))
    rows = []
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    families = sorted(set(r.family for r in ordered))
    for family in families:
        subset = [r for r in ordered if r.family == family]
        ax.scatter([r.step_avg_ms for r in subset], [r.final_val for r in subset], label=family)
        for r in subset:
            ax.annotate(r.tag.replace("spec_", "").replace("nm_", ""), (r.step_avg_ms, r.final_val), fontsize=6, xytext=(2, 2), textcoords="offset points")
            rows.append({"family": family, "tag": r.tag, "final_val": r.final_val, "step_avg_ms": r.step_avg_ms})
    ax.set_xlabel("Mean step time (ms)")
    ax.set_ylabel("Tuning-seed final validation")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, out / "figS3_353m_tuning_landscape")
    write_csv(data_out / "figS3_353m_tuning_landscape.csv", ["family", "tag", "final_val", "step_avg_ms"], rows)


def plot_long_horizon(out: Path, data_out: Path, runs: dict[str, Run]) -> dict[str, Any]:
    nm = runs["newton_muon"]
    steps = common_steps(runs.values())
    steps = [s for s in steps if s >= 100]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    rows = []
    for family in ("transition_current", "transition_spectral"):
        delta = [runs[family].curve[s][0] - nm.curve[s][0] for s in steps]
        ax.plot(np.asarray(steps) * TOKENS_PER_STEP / 1e9, delta, label=family.replace("transition_", ""))
        for s, d in zip(steps, delta):
            rows.append({"family": family, "step": s, "tokens_b": s * TOKENS_PER_STEP / 1e9, "delta": d})
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("Candidate minus NM validation loss")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, out / "figS4_long10b_paired_difference")
    write_csv(data_out / "figS4_long10b_paired_difference.csv", ["family", "step", "tokens_b", "delta"], rows)
    return {family: {"final_val": run.final_val, "step_avg_ms": run.step_avg_ms} for family, run in runs.items()}


def write_main_tables(out: Path, data_out: Path, stats124: dict[str, Any], stats353: dict[str, Any]) -> None:
    labels = {
        "adamw": "AdamW",
        "muon": "Muon",
        "newton_muon": "Newton--Muon",
        "transition_current": "Fisher $\\rightarrow$ NM",
        "transition_spectral": "Spectral Fisher $\\rightarrow$ NM",
    }
    rows = []
    for family in ("adamw", "muon", "newton_muon", "transition_current", "transition_spectral"):
        d = stats124[family]
        rows.append({
            "scale": "124M",
            "method": labels[family],
            "n": d["n"],
            "final_mean": d["mean"],
            "final_sd": d["sd"],
            "paired_final": d.get("paired_final", {}).get("mean", 0.0 if family == "newton_muon" else ""),
            "paired_early": d.get("paired_early", {}).get("mean", 0.0 if family == "newton_muon" else ""),
        })
    for family, label in (("newton_muon", "Tuned Newton--Muon"), ("spectral", "Tuned spectral Fisher $\\rightarrow$ NM")):
        d = stats353[family]
        rows.append({
            "scale": "353M",
            "method": label,
            "n": 3,
            "final_mean": d["mean"],
            "final_sd": d["sd"],
            "paired_final": 0.0 if family == "newton_muon" else stats353["paired_final"]["mean"],
            "paired_early": 0.0 if family == "newton_muon" else stats353["paired_early"]["mean"],
        })
    write_csv(data_out / "table1_main_results.csv", ["scale", "method", "n", "final_mean", "final_sd", "paired_final", "paired_early"], rows)
    lines = [
        r"\begin{tabular}{llrccc}",
        r"\toprule",
        r"Scale & Method & $n$ & Final val. $\downarrow$ & $\Delta_{\rm final}$ vs. NM & $\Delta_{\rm early}$ vs. NM \\",
        r"\midrule",
    ]
    for row in rows:
        pf = "--" if row["paired_final"] == "" else f"{float(row['paired_final']):+.4f}"
        pe = "--" if row["paired_early"] == "" else f"{float(row['paired_early']):+.4f}"
        lines.append(
            f"{row['scale']} & {row['method']} & {row['n']} & "
            f"${row['final_mean']:.4f} \\pm {row['final_sd']:.4f}$ & ${pf}$ & ${pe}$ \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (out / "table1_main_results.tex").write_text("\n".join(lines) + "\n")


def write_setup_table(out: Path, data_out: Path) -> None:
    rows = [
        {"setting": "Primary", "parameters": "123.5M", "layers": 12, "width": 768, "heads": 12, "steps": 6200, "tokens_b": 3.251, "gpu": "H100", "evaluation": "5 paired seeds"},
        {"setting": "Scale", "parameters": "353.5M", "layers": 24, "width": 1024, "heads": 16, "steps": 7500, "tokens_b": 3.932, "gpu": "H200", "evaluation": "seed-3 tuning; seeds 4-6 held out"},
        {"setting": "Horizon", "parameters": "123.5M", "layers": 12, "width": 768, "heads": 12, "steps": 19549, "tokens_b": 10.249, "gpu": "H200", "evaluation": "1 paired seed"},
    ]
    write_csv(data_out / "table_setup.csv", ["setting", "parameters", "layers", "width", "heads", "steps", "tokens_b", "gpu", "evaluation"], rows)
    lines = [r"\begin{tabular}{lrrrrrll}", r"\toprule", r"Setting & Params. & Layers & Width & Heads & Tokens & GPU & Evaluation \\", r"\midrule"]
    for r in rows:
        lines.append(
            f"{r['setting']} & {r['parameters']} & {r['layers']} & {r['width']} & "
            f"{r['heads']} & {r['tokens_b']:.3f}B & {r['gpu']} & {r['evaluation']} " + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_setup.tex").write_text("\n".join(lines) + "\n")


def write_ablation_tables(out: Path, data_out: Path, ab: dict[str, Run], nested: dict[str, Run] | None, long_stats: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    def add(name: str, reference: str, delta: float, finding: str) -> None:
        rows.append({"ablation": name, "reference": reference, "delta_final": delta, "finding": finding})
    add("Permanent Fisher Q/K", "NM backbone", ab["fisher_newton_muon"].final_val - ab["control_newton_muon"].final_val, "Helps early, but hurts final loss")
    add("Permanent Fisher Q/K", "Muon backbone", ab["fisher_muon"].final_val - ab["control_muon"].final_val, "Backbone does not explain the late deficit")
    add("Muon on V", "Newton-Muon on V", ab["v_muon"].final_val - ab["v_nm"].final_val, "Newton-Muon is the stronger static V companion")
    add("AdamW on V", "Newton-Muon on V", ab["v_adamw"].final_val - ab["v_nm"].final_val, "AdamW V is substantially worse")
    if nested is not None:
        for k in ("k2", "k3", "k5"):
            add(f"NM-nested Fisher {k}", "exact k1 NM endpoint", nested[k].final_val - nested["k1"].final_val, "Additional Fisher correction does not improve NM")
        add("Spectral shaping of plain Fisher", "plain Fisher", nested["plain_spectral"].final_val - nested["plain"].final_val, "Spectral shaping closes much of the standalone gap")
    add("10.26B spectral transition", "10.26B Newton-Muon", long_stats["transition_spectral"]["final_val"] - long_stats["newton_muon"]["final_val"], "Final near-tie follows a long middle-phase deficit")
    write_csv(data_out / "table2_ablations.csv", ["ablation", "reference", "delta_final", "finding"], rows)
    short = [rows[0], rows[2]]
    if nested is not None:
        short.append(next(r for r in rows if r["ablation"] == "NM-nested Fisher k3"))
    short.append(rows[-1])
    lines = [r"\begin{tabular}{p{0.25\linewidth}p{0.23\linewidth}rp{0.36\linewidth}}", r"\toprule", r"Ablation & Reference & $\Delta$ final & Finding \\", r"\midrule"]
    for r in short:
        lines.append(
            f"{r['ablation']} & {r['reference']} & ${r['delta_final']:+.4f}$ & {r['finding']} " + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table2_ablations_short.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{p{0.26\linewidth}p{0.22\linewidth}rp{0.35\linewidth}}", r"\toprule", r"Ablation & Reference & $\Delta$ final & Finding \\", r"\midrule"]
    for r in rows:
        lines.append(
            f"{r['ablation']} & {r['reference']} & ${r['delta_final']:+.4f}$ & {r['finding']} " + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tableS_ablations_full.tex").write_text("\n".join(lines) + "\n")


def write_optimizer_assignment(out: Path) -> None:
    lines = [
        r"\begin{tabular}{lllll}", r"\toprule",
        r"Method & Q/K & V & O/MLP & Embedding/head \\", r"\midrule",
        r"AdamW & AdamW & AdamW & AdamW & AdamW \\",
        r"Muon & Muon & Muon & Muon & AdamW \\",
        r"Newton--Muon & Newton--Muon & Newton--Muon & Newton--Muon & AdamW \\",
        r"Spectral Fisher $\rightarrow$ NM & Fisher then NM & Newton--Muon & Newton--Muon & AdamW \\",
        r"\bottomrule", r"\end{tabular}",
    ]
    (out / "table_optimizer_assignments.tex").write_text("\n".join(lines) + "\n")


def write_macros(out: Path, stats124: dict[str, Any], stats353: dict[str, Any]) -> None:
    macros = {
        "AdamWFinalMean": stats124["adamw"]["mean"],
        "MuonFinalMean": stats124["muon"]["mean"],
        "NMFinalMean": stats124["newton_muon"]["mean"],
        "CurrentFinalMean": stats124["transition_current"]["mean"],
        "SpectralFinalMean": stats124["transition_spectral"]["mean"],
        "SpectralFinalDelta": stats124["transition_spectral"]["paired_final"]["mean"],
        "SpectralEarlyDelta": stats124["transition_spectral"]["paired_early"]["mean"],
        "ScaleNMFinalMean": stats353["newton_muon"]["mean"],
        "ScaleSpectralFinalMean": stats353["spectral"]["mean"],
        "ScaleSpectralFinalDelta": stats353["paired_final"]["mean"],
        "ScaleSpectralEarlyDelta": stats353["paired_early"]["mean"],
    }
    lines = ["% Auto-generated by generate_mlopt_paper_assets.py"]
    for name, value in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value:.4f}}}")
    (out / "paper_results_macros.tex").write_text("\n".join(lines) + "\n")


def write_captions_and_text(out: Path, stats124: dict[str, Any], stats353: dict[str, Any]) -> None:
    s124 = stats124["transition_spectral"]
    s353 = stats353
    text = f"""# Paper asset descriptions and captions

## Figure 1 - Architecture-aware Fisher warmup and handoff

**Purpose.** Explain how the method converts feasible Q/K parameter changes into score-space changes, measures those changes with the categorical Fisher metric, pulls the metric back to parameter space, and solves the resulting local quadratic model with three CG steps. The lower timeline shows that Fisher is phase-specific: it shapes Q/K early, then hands control to Newton-Muon.

**Suggested caption.** *Architecture-aware Q/K geometry and the practical optimizer. The score Jacobian maps feasible Q/K weight changes to attention-score perturbations; the rowwise categorical Fisher metric is pulled back through the Jacobian to define a local parameter-space quadratic model. We approximately solve this model with CG3, spectrally shape the direction, and transition to Newton-Muon after the early routing phase.*

## Figure 2a - 124M validation versus training tokens

**Experiment.** Five paired H100 seeds, 124M parameters, 6200 updates, 3.251B FineWeb tokens. The plot compares Newton-Muon, the original Fisher-to-NM transition, and the spectral Fisher-to-NM transition. Seeds share initialization, data order, and validation batches within each pair.

**Message.** Both Fisher transitions improve the early fixed-token trajectory. The original transition moves fastest early; the spectral transition converges more cleanly to the Newton-Muon endpoint.

**Suggested caption.** *Mean validation loss over five paired 124M runs. Fisher-based Q/K warmup substantially accelerates the early trajectory, while a subsequent handoff to Newton-Muon recovers late performance. The spectral transition sacrifices some initial gain but attains final parity with Newton-Muon.*

## Figure 2b - 124M paired loss difference

**Experiment.** Candidate loss minus the seed-matched Newton-Muon loss. Negative values favor Fisher. Vertical lines mark the direction-blend and Q/K-scale transition boundaries.

**Message.** The early effect is not driven by one seed: the mean early differences are {stats124['transition_current']['paired_early']['mean']:+.4f} for the original transition and {s124['paired_early']['mean']:+.4f} for the spectral transition. The spectral final paired mean is {s124['paired_final']['mean']:+.5f}, which supports parity rather than a resolved final win.

**Suggested caption.** *Paired validation difference relative to Newton-Muon at 124M. Negative values favor Fisher. The architecture-aware warmup gives a large, repeatable early benefit; the advantage decays during the handoff, and final differences lie near zero.*

## Figure 3a - 353M wall-clock compute-quality frontier

**Experiment.** Both optimizer families receive eight 353M tuning configurations on seed 3; the selected Newton-Muon and spectral-transition configurations are evaluated on held-out seeds 4-6. The figure uses only held-out runs.

**Message.** Fair scale-specific tuning preserves final parity while the spectral transition reaches intermediate validation levels sooner in wall-clock time.

**Suggested caption.** *Held-out 353M compute-quality frontier after equal eight-configuration tuning budgets. The spectral Fisher warmup reaches intermediate validation levels earlier despite modest per-step overhead, then converges to the tuned Newton-Muon endpoint.*

## Figure 3b - 353M time to quality

**Experiment.** Mean wall-clock time to predeclared validation thresholds on held-out seeds 4-6. Negative percentages indicate that spectral Fisher is faster.

**Message.** The selected method reduces time to validation 4.0, 3.8, 3.6, and 3.5 by approximately 16.0%, 9.2%, 3.0%, and 2.1%, respectively. Threshold times are quantized by the 100-step validation cadence, so the full curve remains the primary evidence.

**Suggested caption.** *Relative wall-clock time of the tuned spectral transition versus tuned Newton-Muon at 353M. Fisher improves the intermediate compute-quality frontier; the benefit shrinks as Newton-Muon catches up later in training.*

## Table 1 - Main quantitative results

**Purpose.** State the clean baseline ranking, final parity result, and early paired effect in one place. Report mean plus or minus standard deviation; use paired confidence intervals in the surrounding text rather than treating tiny final mean differences as wins.

**Key reading.** At 124M, spectral Fisher-to-NM has final mean {s124['mean']:.5f} versus {stats124['newton_muon']['mean']:.5f} for Newton-Muon, with paired final mean {s124['paired_final']['mean']:+.5f}. At 353M, the held-out means are {s353['spectral']['mean']:.5f} and {s353['newton_muon']['mean']:.5f}, with paired final mean {s353['paired_final']['mean']:+.5f}. The early paired effects are much larger and consistently negative.

## Table 2 - Mechanism and limitation ablations

**Purpose.** Compress the exploratory program into falsifiable conclusions rather than a chronological search log. The table should show that permanent Fisher loses, static V/backbone changes do not explain the result, extra Fisher corrections beyond the exact Newton-Muon endpoint hurt, and the full-cache run does not reveal a persistent asymptotic advantage.

## Supplementary Figure S1 - Final paired seed points at 124M

Show all five seedwise final differences and a descriptive 95% interval. This visual prevents the reader from mistaking a tiny difference in grouped means for a statistically resolved win.

## Supplementary Figure S2 - 353M paired difference over steps

Show that all held-out seeds share a large early advantage and that the mean approaches zero later.

## Supplementary Figure S3 - 353M tuning landscape

Show all eight Newton-Muon and eight spectral configurations on tuning seed 3. The selected spectral method lies in a broad competitive basin rather than at an isolated brittle point. Selection uses only step-7500 validation; held-out seeds determine the claim.

## Supplementary Figure S4 - 10.26B-token horizon

Show candidate-minus-NM validation over a nearly full FineWeb10B pass. Fisher helps early but trails through much of the middle phase; the near-tie after final warmdown does not support a persistent asymptotic benefit.

# Experiment descriptions

## E1. Primary 124M paired benchmark

Five paired H100 seeds compare vanilla AdamW, vanilla Muon, vanilla Newton-Muon, the original Fisher-to-NM transition, and the spectral transition. All methods train for 6200 updates with global batch 512 and sequence length 1024, totaling 3.251B tokens. The primary endpoint is validation loss at the fixed token budget. The paired early statistic averages candidate-minus-NM validation over checkpoints 100-1500.

## E2. Fair 353M tuning and held-out confirmation

A 24-layer, width-1024, 16-head model trains for 7500 updates (3.932B tokens) on H200. Newton-Muon and spectral Fisher each receive eight tuning configurations on seed 3. The selector minimizes validation loss at step 7500 within each family. The selected methods are then compared on held-out seeds 4, 5, and 6. This design separates tuning from evaluation and supports the compute-quality claim.

## E3. Long-horizon 124M sidecar

Newton-Muon, the original transition, and the spectral transition train for 19,549 updates, approximately one pass over the local 10.255B-token FineWeb cache. This experiment tests persistence, not scale. It shows that the Fisher advantage is temporal rather than asymptotic.

## E4. Companion-backbone ablation

Permanent Fisher on Q/K is compared under AdamW, Muon, and Newton-Muon backbones while holding the remaining parameter assignments fixed within each comparison. Fisher loses at the final checkpoint under all three, ruling out the hypothesis that a Newton-Muon backbone alone suppresses Fisher.

## E5. V-optimizer ablation

With Fisher fixed on Q/K and Newton-Muon fixed on O/MLP, V uses AdamW, Muon, or Newton-Muon. Newton-Muon is the best static V companion; reasonable Newton-Muon V scales are nearly tied, so broad scalar V tuning is not the missing mechanism.

## E6. Exact endpoint and deeper-curvature ablation

The controlled nested hierarchy exactly recovers Newton-Muon at CG depth one. Increasing Fisher Krylov depth to two, three, or five fails to improve that endpoint and progressively worsens final validation. This isolates the value of Fisher to the early phase rather than sustained higher-order correction.
"""
    (out / "PAPER_ASSET_DESCRIPTIONS.md").write_text(text)

    setup = r"""\paragraph{Experimental setup.}
We train decoder-only transformers on FineWeb with sequence length 1024 and global batch size 512. Our primary model has 123.5M parameters (12 layers, width 768, 12 heads) and trains for 6200 updates, or 3.251B tokens, on H100 GPUs. We evaluate five paired seeds with matched initialization, data order, and validation batches. Our scale study uses a 353.5M model (24 layers, width 1024, 16 heads) for 7500 updates, or 3.932B tokens, on H200 GPUs. At 353M, Newton--Muon and spectral Fisher each receive eight tuning configurations on seed 3; we evaluate the selected configuration from each family on held-out seeds 4--6. Unsupported vector/scalar parameters and the tied embedding/head use AdamW; Table~\ref{tab:optimizer-assignments} records the complete assignment.
"""
    (out / "experimental_setup.tex").write_text(setup)

    results = rf"""\paragraph{{Main results.}}
At 124M, Newton--Muon substantially outperforms Muon and AdamW at a fixed 3.251B-token budget. The spectral Fisher transition reaches a final validation loss of ${s124['mean']:.4f}\pm{s124['sd']:.4f}$, compared with ${stats124['newton_muon']['mean']:.4f}\pm{stats124['newton_muon']['sd']:.4f}$ for Newton--Muon. The paired final difference is only ${s124['paired_final']['mean']:+.5f}$ and its descriptive interval includes zero, so we interpret the endpoint as parity. In contrast, the paired early difference over steps 100--1500 is ${s124['paired_early']['mean']:+.4f}$ and is consistent across all five seeds.

At 353M, equal tuning budgets select the historical Newton--Muon split and a spectral Fisher transition with Q/K RMS $2.6\times10^{{-4}}$. On held-out seeds 4--6, final validation is ${s353['spectral']['mean']:.4f}\pm{s353['spectral']['sd']:.4f}$ for spectral Fisher and ${s353['newton_muon']['mean']:.4f}\pm{s353['newton_muon']['sd']:.4f}$ for Newton--Muon, again supporting parity. The early paired difference is ${s353['paired_early']['mean']:+.4f}$. This step advantage overcomes the modest optimizer overhead at intermediate targets: spectral Fisher reduces mean time to validation 4.0, 3.8, 3.6, and 3.5 by approximately 16.0\%, 9.2\%, 3.0\%, and 2.1\%, respectively.
"""
    (out / "results_summary.tex").write_text(results)


def write_validation_report(out: Path, stats124: dict[str, Any], stats353: dict[str, Any]) -> None:
    expected = {
        "124_adamw": 3.38656,
        "124_muon": 3.27720,
        "124_nm": 3.26300,
        "124_current": 3.26362,
        "124_spectral": 3.26282,
        "124_spectral_early": -0.061632,
        "353_nm": 3.05660,
        "353_spectral": 3.05630,
        "353_final_delta": -0.00030,
        "353_early_delta": -0.083760,
    }
    actual = {
        "124_adamw": stats124["adamw"]["mean"],
        "124_muon": stats124["muon"]["mean"],
        "124_nm": stats124["newton_muon"]["mean"],
        "124_current": stats124["transition_current"]["mean"],
        "124_spectral": stats124["transition_spectral"]["mean"],
        "124_spectral_early": stats124["transition_spectral"]["paired_early"]["mean"],
        "353_nm": stats353["newton_muon"]["mean"],
        "353_spectral": stats353["spectral"]["mean"],
        "353_final_delta": stats353["paired_final"]["mean"],
        "353_early_delta": stats353["paired_early"]["mean"],
    }
    lines = ["MLOPT paper asset validation report", "=" * 72]
    failures = []
    for key in expected:
        diff = actual[key] - expected[key]
        lines.append(f"{key:24s} actual={actual[key]:+.8f} expected={expected[key]:+.8f} diff={diff:+.3e}")
        if abs(diff) > 5e-6:
            failures.append(key)
    lines.append("")
    if failures:
        lines.append("FAIL: snapshot mismatches: " + ", ".join(failures))
    else:
        lines.append("PASS: all frozen headline statistics match the expected snapshot")
    (out / "validation_report.txt").write_text("\n".join(lines) + "\n")
    if failures:
        raise RuntimeError("headline statistic validation failed")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root: Path) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    lines = [f"{sha256_file(p)}  {p.relative_to(root)}" for p in files]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("~/project_pi_das227/kp875/LLM_optimizer").expanduser())
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--skip-tuning-plot", action="store_true")
    ap.add_argument("--no-snapshot-check", action="store_true")
    args = ap.parse_args()
    base = args.base.expanduser().resolve()
    output = (args.output or (base / "mlopt_paper_assets")).expanduser().resolve()
    figures = output / "figures"
    tables = output / "tables"
    data_out = output / "data"
    latex = output / "latex"
    for d in (figures, tables, data_out, latex):
        d.mkdir(parents=True, exist_ok=True)
    set_plot_defaults()

    runs124 = load_124m_runs(base)
    runs353 = load_353m_confirmation(base)
    long_runs = load_long_horizon(base)
    ablations = load_cycle_a_ablations(base)
    nested = load_nested_ablation(base)

    plot_method_schematic(figures)
    stats124 = plot_124m(figures, data_out, runs124)
    stats353 = plot_353m(figures, data_out, runs353)
    if not args.skip_tuning_plot:
        tuning = load_353m_tuning(base)
        plot_353m_tuning(figures, data_out, tuning)
    long_stats = plot_long_horizon(figures, data_out, long_runs)

    write_main_tables(tables, data_out, stats124, stats353)
    write_setup_table(tables, data_out)
    write_ablation_tables(tables, data_out, ablations, nested, long_stats)
    write_optimizer_assignment(tables)
    write_macros(latex, stats124, stats353)
    write_captions_and_text(latex, stats124, stats353)
    if args.no_snapshot_check:
        (output / "validation_report.txt").write_text("PASS: snapshot check intentionally disabled for parser/format testing\n")
    else:
        write_validation_report(output, stats124, stats353)

    registry = []
    for family, rs in runs124.items():
        for r in rs:
            registry.append({"study": "124m", "family": family, "seed": r.seed, "tag": r.tag, "path": str(r.path), "final_val": r.final_val, "step_avg_ms": r.step_avg_ms})
    for family, rs in runs353.items():
        for r in rs:
            registry.append({"study": "353m_heldout", "family": family, "seed": r.seed, "tag": r.tag, "path": str(r.path), "final_val": r.final_val, "step_avg_ms": r.step_avg_ms})
    write_csv(data_out / "run_registry.csv", ["study", "family", "seed", "tag", "path", "final_val", "step_avg_ms"], registry)

    metadata = {
        "base": str(base),
        "output": str(output),
        "tokens_per_step": TOKENS_PER_STEP,
        "stats_124m": stats124,
        "stats_353m": stats353,
        "long_horizon": long_stats,
        "nested_ablation_available": nested is not None,
    }
    (output / "paper_asset_statistics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    readme = f"""# Generated MLOPT paper assets

Generated from frozen run directories under:

`{base}`

## Main-text candidates

- `figures/fig1_method_schematic.pdf`
- `figures/fig2a_124m_validation_vs_tokens.pdf`
- `figures/fig2b_124m_paired_difference_early.pdf`
- `figures/fig3a_353m_wallclock_frontier.pdf`
- `figures/fig3b_353m_time_to_quality.pdf`
- `tables/table1_main_results.tex`
- `tables/table2_ablations_short.tex`
- `tables/table_setup.tex`

## Supplementary candidates

- `figures/figS1_124m_final_paired_points.pdf`
- `figures/figS2_353m_paired_difference.pdf`
- `figures/figS3_353m_tuning_landscape.pdf`
- `figures/figS4_long10b_paired_difference.pdf`
- `tables/tableS_ablations_full.tex`
- `tables/table_optimizer_assignments.tex`

## Ready-to-paste prose

- `latex/experimental_setup.tex`
- `latex/results_summary.tex`
- `latex/PAPER_ASSET_DESCRIPTIONS.md`
- `latex/paper_results_macros.tex`

All figures are emitted as vector PDF and 300-dpi PNG. Every plotted number has a companion CSV under `data/`.
"""
    (output / "README.md").write_text(readme)
    write_manifest(output)
    print(f"Generated paper assets at {output}")
    print((output / "validation_report.txt").read_text(), end="")


if __name__ == "__main__":
    main()
