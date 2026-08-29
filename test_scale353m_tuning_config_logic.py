#!/usr/bin/env python3
"""Sanity checks for the bounded 353M tuning grid."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent
rows = [json.loads(x) for x in (root / "scale353m_tuning_configs.jsonl").read_text().splitlines() if x.strip()]

assert len(rows) == 16
assert [r["index"] for r in rows] == list(range(16))
assert len({r["tag"] for r in rows}) == 16
assert Counter(r["family"] for r in rows) == {"nm_tune": 8, "spectral_tune": 8}

for row in rows:
    assert row["seed"] == 3
    assert row["iterations"] == 7500
    assert row["model_n_layer"] == 24
    assert row["model_n_head"] == 16
    assert row["model_n_embd"] == 1024
    assert row["batch_size"] == 512
    assert row["device_batch_size"] == 32
    assert row["expected_gpu"] == "H200"
    assert row["timeout"] == "1800m"

for row in rows[:8]:
    assert row["env"]["CYCLEA_SYSTEM_MODE"] == "newton_muon"
for row in rows[8:]:
    env = row["env"]
    assert env["CYCLEA_SYSTEM_MODE"] == "fisher"
    assert env["FISHER_NM_BLEND_END"] == 1.0
    assert env["FISHER_NM_SHADOW"] == 1
    assert env["FISHER_CG_ITERS"] == 3
    assert env["FISHER_CURV_REFRESH"] == 4

assert {r["env"]["FISHER_QK_LR"] for r in rows[8:]} >= {0.00018, 0.00022, 0.00026}
assert {r["env"]["FISHER_SPECTRAL_BLEND"] for r in rows[8:]} >= {0.5, 0.75, 1.0}
assert any(r["env"]["FISHER_NM_BLEND_SCHEDULE_START"] == 400 for r in rows[8:])
assert any(r["env"]["FISHER_NM_BLEND_SCHEDULE_START"] == 700 for r in rows[8:])

print("PASS")
