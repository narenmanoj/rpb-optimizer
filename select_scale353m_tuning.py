#!/usr/bin/env python3
"""Select the best seed-3 NM and spectral configs and create held-out confirmations."""
from __future__ import annotations

import argparse
import csv
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

FINAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.eE+-]+)")
STATUS_RE = re.compile(r"finished with status=(\d+)")
STEP_RE = re.compile(r"step:(\d+)/(\d+) .*?step_avg:([0-9.eE+-]+)ms")


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_run(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = root / row["tag"] / "run.out"
    if not path.exists():
        raise RuntimeError(f"missing run.out for {row['tag']}: {path}")
    text = path.read_text(errors="ignore")
    status_matches = STATUS_RE.findall(text)
    status = int(status_matches[-1]) if status_matches else -1
    finals = [(int(a), int(b), float(c)) for a, b, c in FINAL_RE.findall(text)]
    if not finals:
        raise RuntimeError(f"no validation records for {row['tag']}")
    final_step, total, final_val = finals[-1]
    if final_step != int(row["iterations"]) or total != int(row["iterations"]):
        raise RuntimeError(
            f"{row['tag']} stopped at {final_step}/{total}; expected {row['iterations']}"
        )
    if status not in (0, 124):
        raise RuntimeError(f"{row['tag']} has unexpected status={status}")
    step_times = [float(c) for _, _, c in STEP_RE.findall(text)]
    return {
        "tag": row["tag"],
        "family": row["family"],
        "final_val": final_val,
        "status": status,
        "step_avg_ms": step_times[-1] if step_times else float("nan"),
        "row": row,
    }


def confirmation_row(base: dict[str, Any], family: str, seed: int, index: int) -> dict[str, Any]:
    row = deepcopy(base)
    source_tag = base["tag"]
    row.update({
        "index": index,
        "seed": seed,
        "family": family,
        "tag": f"confirm_{family}_seed{seed}",
        "description": f"held-out seed {seed} confirmation of tuning winner {source_tag}",
        "selected_from_tag": source_tag,
        "selected_on_seed": int(base["seed"]),
    })
    return row


def write_configs(rows: list[dict[str, Any]], out_jsonl: Path, out_tsv: Path) -> None:
    out_jsonl.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    fields = [
        "index", "tag", "family", "seed", "selected_from_tag",
        "selected_on_seed", "description",
    ]
    with out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = load_rows(args.config)
    parsed = [parse_run(args.run_root, row) for row in rows]
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in parsed:
        groups.setdefault(item["family"], []).append(item)
    required = {"nm_tune", "spectral_tune"}
    if set(groups) != required:
        raise SystemExit(f"expected families {sorted(required)}, found {sorted(groups)}")

    winners = {family: min(items, key=lambda x: (x["final_val"], x["tag"]))
               for family, items in groups.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        family: {
            "tag": item["tag"],
            "final_val": item["final_val"],
            "step_avg_ms": item["step_avg_ms"],
            "seed": item["row"]["seed"],
            "config": item["row"],
        }
        for family, item in winners.items()
    }
    (args.output_dir / "selected_scale353m_configs.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "353M bounded tuning selection",
        "selection metric: validation loss at step 7500 on seed 3",
        "",
    ]
    for family in ("nm_tune", "spectral_tune"):
        items = sorted(groups[family], key=lambda x: (x["final_val"], x["tag"]))
        lines.append(f"[{family}]")
        for rank, item in enumerate(items, 1):
            lines.append(
                f"{rank:2d}  {item['tag']:28s}  val={item['final_val']:.6f}  "
                f"step_ms={item['step_avg_ms']:.2f}"
            )
        lines.append("")
    lines.append(
        f"selected NM:       {winners['nm_tune']['tag']} "
        f"({winners['nm_tune']['final_val']:.6f})"
    )
    lines.append(
        f"selected spectral: {winners['spectral_tune']['tag']} "
        f"({winners['spectral_tune']['final_val']:.6f})"
    )
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "scale353m_tuning_selection.txt").write_text(summary)
    print(summary, end="")

    confirm: list[dict[str, Any]] = []
    idx = 0
    for seed in (4, 5, 6):
        confirm.append(confirmation_row(winners["nm_tune"]["row"], "nm", seed, idx))
        idx += 1
        confirm.append(
            confirmation_row(winners["spectral_tune"]["row"], "spectral", seed, idx)
        )
        idx += 1
    write_configs(
        confirm,
        args.output_dir / "scale353m_confirm_configs.jsonl",
        args.output_dir / "scale353m_confirm_configs.tsv",
    )
    print(f"Wrote {len(confirm)} held-out confirmation configurations (seeds 4,5,6)")


if __name__ == "__main__":
    main()
