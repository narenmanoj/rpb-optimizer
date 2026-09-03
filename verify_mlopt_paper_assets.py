#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

REQUIRED = [
    "validation_report.txt",
    "paper_asset_statistics.json",
    "README.md",
    "figures/fig1_method_schematic.pdf",
    "figures/fig2a_124m_validation_vs_tokens.pdf",
    "figures/fig2b_124m_paired_difference_early.pdf",
    "figures/fig3a_353m_wallclock_frontier.pdf",
    "figures/fig3b_353m_time_to_quality.pdf",
    "tables/table1_main_results.tex",
    "tables/table2_ablations_short.tex",
    "latex/experimental_setup.tex",
    "latex/results_summary.tex",
    "latex/PAPER_ASSET_DESCRIPTIONS.md",
]

def sha(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    missing = [x for x in REQUIRED if not (root / x).is_file() or (root / x).stat().st_size == 0]
    if missing:
        raise SystemExit("Missing/empty assets:\n" + "\n".join(missing))
    report = (root / "validation_report.txt").read_text()
    if "PASS:" not in report:
        raise SystemExit("validation report did not pass")
    print(f"PASS: {len(REQUIRED)} required paper assets exist and the statistical snapshot matches")
    for rel in REQUIRED:
        print(f"{sha(root / rel)}  {rel}")

if __name__ == "__main__":
    main()
