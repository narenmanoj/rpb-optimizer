#!/usr/bin/env python3
"""Run one frozen 353M tuning configuration with reproducibility metadata and smoke validation."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ERR = re.compile(
    r"traceback|runtimeerror|valueerror|assertionerror|cuda error|out of memory|killed|nan detected",
    re.I,
)


def read_row(path: Path, index: int) -> dict[str, Any]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not 0 <= index < len(rows):
        raise SystemExit(f"index {index} outside [0,{len(rows)-1}]")
    row = rows[index]
    if int(row.get("index", index)) != index:
        raise SystemExit(f"config index mismatch for {index}: {row.get('index')}")
    return row


def stream(cmd: list[str], cwd: Path, env: dict[str, str], log) -> int:
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
        log.flush()
    return proc.wait()


def scalar_values(run_dir: Path, tag: str) -> list[float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return []
    events = sorted(run_dir.rglob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    if not events:
        return []
    ea = EventAccumulator(str(events[-1].parent), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [float(v.value) for v in ea.Scalars(tag)]


def git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def validate_smoke(row: dict[str, Any], run_dir: Path, text: str, status: int) -> None:
    failures: list[str] = []
    final = f"step:{row['iterations']}/{row['iterations']} val_loss"
    if status != 0:
        failures.append(f"training status={status}")
    if final not in text:
        failures.append(f"missing final validation marker {final!r}")
    if ERR.search(text):
        failures.append("runtime/nonfinite error pattern detected")
    if row["expected_gpu"].lower() not in text.lower():
        failures.append(f"expected GPU {row['expected_gpu']} not found in log")
    model_line = (
        f"layers={row['model_n_layer']} heads={row['model_n_head']} "
        f"embd={row['model_n_embd']}"
    )
    if model_line not in text:
        failures.append(f"model configuration not found: {model_line}")
    match = re.search(r"\[model\].*parameters=(\d+)", text)
    if match is None:
        failures.append("model parameter count not found")
    elif int(row["model_n_layer"]) >= 24 and int(match.group(1)) < 300_000_000:
        failures.append(f"353M smoke instantiated only {int(match.group(1))} parameters")

    if row["env"].get("CYCLEA_SYSTEM_MODE") == "fisher":
        layers = scalar_values(run_dir, "fisher_qk/layers_updated")
        dots = scalar_values(run_dir, "fisher_qk/current_gradient_dot_mean")
        fallbacks = scalar_values(run_dir, "fisher_qk/descent_fallback_layers")
        expected = float(row["model_n_layer"])
        if not layers or abs(max(layers) - expected) > 1e-6:
            failures.append(f"Fisher layers_updated never reached {expected:g}")
        if not dots or not all(math.isfinite(x) for x in dots) or max(dots) >= 0:
            failures.append("missing or non-descent Fisher current-gradient diagnostic")
        if not fallbacks:
            failures.append("missing Fisher fallback diagnostic")
        else:
            nonzero = [x for x in fallbacks if x > 0]
            max_allowed = max(2, math.ceil(int(row["model_n_layer"]) / 6))
            if len(nonzero) > 1 or max(nonzero, default=0) > max_allowed:
                failures.append(
                    f"persistent/excessive fallback: {len(nonzero)}/{len(fallbacks)} points, "
                    f"max_layers={max(nonzero, default=0):g}, allowed={max_allowed}"
                )
            elif nonzero:
                print(
                    f"WARN {row['tag']}: sparse fallback at {len(nonzero)}/{len(fallbacks)} "
                    f"diagnostic points, max_layers={max(nonzero):g}"
                )
        if "spectral" in row["family"]:
            blends = scalar_values(run_dir, "fisher_qk/nm_blend_mean")
            if not blends or max(blends) < 0.9:
                failures.append("spectral transition smoke did not reach Newton-Muon blend")

    if failures:
        print("SMOKE VALIDATION FAILURES")
        for item in failures:
            print("-", item)
        raise SystemExit(1)
    print(f"PASS: smoke validated for {row['tag']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--index", type=int, required=True)
    args = ap.parse_args()

    row = read_row(args.config, args.index)
    tag = row["tag"]
    run_dir = args.output_root / tag
    run_out = run_dir / "run.out"
    if run_out.exists() and any(x in run_out.read_text(errors="ignore") for x in
                                ("finished with status=0", "finished with status=124")):
        print(f"Run already completed: {tag}")
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    frozen = dict(row)
    frozen["git_sha_at_launch"] = git_sha(args.repo)
    (run_dir / "run_config.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")

    for name in ("data", "triton_kernels.py", "diagnostics.py",
                 "attention_geometry_core.py", "train_gpt_scale353m_tune.py"):
        target = args.repo / name
        if not target.exists():
            raise SystemExit(f"missing required repo path {target}")
        (run_dir / name).symlink_to(target, target_is_directory=target.is_dir())

    env = os.environ.copy()
    env.update({k: str(v) for k, v in row.get("env", {}).items()})
    env.update({
        "NUM_ITERATIONS": str(row["iterations"]),
        "WARMDOWN_ITERS": str(row["warmdown_iters"]),
        "BATCH_SIZE": str(row["batch_size"]),
        "DEVICE_BATCH_SIZE": str(row["device_batch_size"]),
        "SEQUENCE_LENGTH": str(row["sequence_length"]),
        "MODEL_N_LAYER": str(row["model_n_layer"]),
        "MODEL_N_HEAD": str(row["model_n_head"]),
        "MODEL_N_EMBD": str(row["model_n_embd"]),
        "MODEL_VOCAB_SIZE": str(row["model_vocab_size"]),
        "VAL_LOSS_EVERY": str(row["val_every"]),
        "SAVE_EVERY": "0",
        "DIAG_EVERY": str(row["diag_every"]),
        "FISHER_DIAG_EVERY": str(row["diag_every"]),
        "SEED": str(row["seed"]),
        "PYTHONHASHSEED": str(row["seed"]),
        "LEARNING_RATE": "0.004",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })

    status = 1
    with run_out.open("w") as log:
        for line in (
            "CYCLE_C_RUN=" + json.dumps(frozen, sort_keys=True) + "\n",
            f"GIT_SHA={frozen['git_sha_at_launch']}\n",
        ):
            sys.stdout.write(line)
            log.write(line)
        for key in sorted(k for k in env if k.startswith(
            ("CYCLEA_", "FISHER_", "MODEL_", "BATCH_", "DEVICE_", "SEQUENCE_",
             "WARM", "NUM_", "SEED"))):
            line = f"{key}={env[key]}\n"
            sys.stdout.write(line)
            log.write(line)
        log.flush()
        stream(["nvidia-smi"], run_dir, env, log)
        compat = (
            "import torch; print('torch',torch.__version__); print('cuda',torch.version.cuda); "
            "print('gpu',torch.cuda.get_device_name(0)); print('arch',torch.cuda.get_arch_list())"
        )
        stream([sys.executable, "-c", compat], run_dir, env, log)
        status = stream([
            "timeout", "--signal=TERM", "--kill-after=30s", str(row["timeout"]),
            sys.executable, "-u", "train_gpt_scale353m_tune.py",
        ], run_dir, env, log)
        line = f"TAG={tag} finished with status={status}\n"
        sys.stdout.write(line)
        log.write(line)

    text = run_out.read_text(errors="ignore")
    if bool(row.get("smoke")):
        validate_smoke(row, run_dir, text, status)
    elif status not in (0, 124):
        raise SystemExit(status)


if __name__ == "__main__":
    main()
