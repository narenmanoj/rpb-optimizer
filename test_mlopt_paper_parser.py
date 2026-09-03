#!/usr/bin/env python3
from __future__ import annotations
import tempfile
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("assets", HERE / "generate_mlopt_paper_assets.py")
assert SPEC and SPEC.loader
assets = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assets
SPEC.loader.exec_module(assets)

with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "run"
    root.mkdir()
    (root / "run_config.json").write_text('{"seed": 7, "family": "test"}\n')
    (root / "run.out").write_text(
        "[model] layers=12 heads=12 embd=768 vocab=50257 parameters=123532032\n"
        "step:0/100 val_loss:10.0 train_time:1ms step_avg:nanms\n"
        "step:100/100 val_loss:3.5 train_time:123456ms step_avg:1200.0ms\n"
        "peak memory consumption: 40000 MiB\n"
        "TAG=test finished with status=0\n"
    )
    run = assets.parse_run(root, family="test", seed=7)
    assert run.final_step == 100
    assert abs(run.final_val - 3.5) < 1e-12
    assert abs(run.curve[100][1] - 123.456) < 1e-12
    assert run.model_parameters == 123532032
    assert run.peak_mib == 40000
print("PASS")
