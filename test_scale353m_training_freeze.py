#!/usr/bin/env python3
"""Verify that the 353M tuning study uses the frozen Cycle-C implementation."""
from __future__ import annotations

import hashlib
from pathlib import Path

EXPECTED = {
    "train_gpt_scale353m_tune.py": "632dd7881fab5a2b0f18ee8ace88c558f7ee07bd93c7a86e80acf8e68cca8c7e",
    "attention_geometry_core.py": "4e811c209506d27ee8ca1c23034e452b4c2b2742105b89693bfd6a0016018cb2",
}

for name, expected in EXPECTED.items():
    path = Path(__file__).resolve().parent / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (name, actual, expected)
    print(name, actual)

print("PASS: 353M tuning uses the frozen Cycle-C training and geometry code")
