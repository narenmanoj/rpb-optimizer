#!/usr/bin/env python3
"""Summarize v2 one-step attention-geometry audit JSON files.

For every candidate and scale protocol, the script reports three distinct views:

1. the trial selected by construction-batch (A) reduction;
2. held-out batch-B transfer at that same A-selected scale;
3. the best A-reduction trial whose local model predicts positive reduction.

Batch B never selects a scale. The script also summarizes the native local-model
sanity check recorded before any protocol-specific rescaling.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def as_float(value: Any, default: float = float("nan")) -> float:
    return float(value) if finite(value) else default


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if finite(value)]
    return mean(vals) if vals else float("nan")


def safe_std(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if finite(value)]
    return pstdev(vals) if len(vals) > 1 else (0.0 if vals else float("nan"))


def protocol_items(candidate: dict[str, Any]):
    """Yield (protocol, payload) with a v1 fallback."""
    protocols = candidate.get("protocols")
    if isinstance(protocols, dict) and protocols:
        yield from protocols.items()
        return

    # v1 compatibility. These trials were all Newton-Muon-norm matched.
    yield "matched_v1", {
        "direction": candidate.get("direction", {}),
        "trials": candidate.get("trials", []),
    }


def select_records(payload: dict[str, Any]):
    actual_records: list[dict[str, Any]] = []
    trust_records: list[dict[str, Any]] = []

    for candidate in payload.get("candidates", []):
        solver = candidate.get("solver", {})
        solve_seconds = solver.get("solve_seconds", float("nan"))
        native_model = candidate.get("native_model", {})

        for protocol, pdata in protocol_items(candidate):
            direction = pdata.get("direction", candidate.get("direction", {}))
            valid = [
                trial
                for trial in pdata.get("trials", [])
                if finite(trial.get("reduction_a"))
                and finite(trial.get("reduction_b"))
            ]
            if not valid:
                continue

            def make_record(trial: dict[str, Any], selection: str):
                return {
                    "name": candidate["name"],
                    "scope": candidate.get("scope", ""),
                    "protocol": protocol,
                    "selection": selection,
                    "scale": float(trial["scale_factor"]),
                    "boundary": bool(trial.get("scale_at_boundary", False)),
                    "red_a": float(trial["reduction_a"]),
                    "red_b": float(trial["reduction_b"]),
                    "pred": as_float(trial.get("predicted_reduction")),
                    "trust": as_float(trial.get("trust_ratio_a")),
                    "update_norm": as_float(trial.get("update_norm")),
                    "linear_a": as_float(trial.get("linear_gain_a")),
                    "linear_b": as_float(trial.get("linear_gain_b")),
                    "score_osc": as_float(
                        trial.get(
                            "exact_score_osc_max",
                            direction.get("score_osc_max", float("nan")),
                        )
                    ),
                    "bilinear": as_float(
                        trial.get(
                            "bilinear_to_linear_ratio_scaled",
                            direction.get("bilinear_ratio", float("nan")),
                        )
                    ),
                    "cos_nm": as_float(direction.get("cosine_to_nm_qk")),
                    "solve_seconds": (
                        float(solve_seconds) if finite(solve_seconds) else float("nan")
                    ),
                    "native_sanity_pass": native_model.get("sanity_pass"),
                    "native_pred": as_float(native_model.get("predicted_reduction")),
                }

            best_actual = max(valid, key=lambda trial: float(trial["reduction_a"]))
            actual_records.append(make_record(best_actual, "actual_A"))

            trust_valid = [
                trial
                for trial in valid
                if finite(trial.get("predicted_reduction"))
                and float(trial["predicted_reduction"]) > 0
            ]
            if trust_valid:
                best_trust = max(
                    trust_valid, key=lambda trial: float(trial["reduction_a"])
                )
                trust_records.append(make_record(best_trust, "positive_prediction"))

    return actual_records, trust_records


def print_table(title: str, rows: list[dict[str, Any]], top: int, sort_key: str):
    rows = sorted(rows, key=lambda row: row[sort_key], reverse=True)
    print(title)
    print(
        "rank\tcandidate\tprotocol\tscope\tscale\tboundary\tred_A\tred_B\t"
        "pred\ttrust_A\tupdate_norm\tcos_NM\tsolve_s"
    )
    for rank, row in enumerate(rows[:top], 1):
        print(
            f"{rank}\t{row['name']}\t{row['protocol']}\t{row['scope']}\t"
            f"{row['scale']:.6g}\t{int(row['boundary'])}\t"
            f"{row['red_a']:.8g}\t{row['red_b']:.8g}\t"
            f"{row['pred']:.8g}\t{row['trust']:.8g}\t"
            f"{row['update_norm']:.8g}\t{row['cos_nm']:.6g}\t"
            f"{row['solve_seconds']:.6g}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    files = sorted(args.root.rglob("audit_step*_layer*_seed*.json"))
    if not files:
        raise SystemExit(f"No audit JSON files found under {args.root}")

    aggregate_actual: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    aggregate_trust: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    sanity: dict[str, list[dict[str, Any]]] = defaultdict(list)

    print(f"Found {len(files)} audit files under {args.root}\n")

    for path in files:
        payload = json.loads(path.read_text())
        actual, trust = select_records(payload)

        for row in actual:
            aggregate_actual[(row["name"], row["protocol"])].append(
                {**row, "step": int(payload["step"]), "layer": int(payload["layer"])}
            )
        for row in trust:
            aggregate_trust[(row["name"], row["protocol"])].append(
                {**row, "step": int(payload["step"]), "layer": int(payload["layer"])}
            )
        for candidate in payload.get("candidates", []):
            model = candidate.get("native_model", {})
            if finite(model.get("predicted_reduction")):
                sanity[candidate["name"]].append(
                    {
                        "pass": model.get("sanity_pass"),
                        "pred": float(model["predicted_reduction"]),
                        "objective": float(model.get("model_objective", float("nan"))),
                    }
                )

        print("=" * 132)
        print(
            f"{path.name}: version={payload.get('audit_version', 1)} "
            f"step={payload['step']} layer={payload['layer']} "
            f"baseline_A={payload['baseline_loss_a']:.6f} "
            f"baseline_B={payload['baseline_loss_b']:.6f} "
            f"elapsed={payload['audit_elapsed_seconds']:.2f}s"
        )
        print()
        print_table(
            "Best actual batch-A trial (batch B at the same A-selected scale)",
            actual,
            args.top,
            "red_a",
        )
        print_table(
            "Held-out batch-B transfer at the batch-A-selected scale",
            actual,
            args.top,
            "red_b",
        )
        print_table(
            "Best trial with positive local-model prediction",
            trust,
            args.top,
            "red_a",
        )

    def aggregate_rows(groups: dict[tuple[str, str], list[dict[str, Any]]]):
        output = []
        for (name, protocol), observations in groups.items():
            output.append(
                {
                    "name": name,
                    "protocol": protocol,
                    "n": len(observations),
                    "mean_b": safe_mean(row["red_b"] for row in observations),
                    "std_b": safe_std(row["red_b"] for row in observations),
                    "mean_a": safe_mean(row["red_a"] for row in observations),
                    "positive_b": sum(row["red_b"] > 0 for row in observations),
                    "mean_scale": safe_mean(row["scale"] for row in observations),
                    "boundary_count": sum(row["boundary"] for row in observations),
                    "mean_pred": safe_mean(row["pred"] for row in observations),
                    "mean_trust": safe_mean(row["trust"] for row in observations),
                    "mean_cos_nm": safe_mean(row["cos_nm"] for row in observations),
                    "mean_solve": safe_mean(row["solve_seconds"] for row in observations),
                }
            )
        return output

    actual_aggregate = aggregate_rows(aggregate_actual)
    trust_aggregate = aggregate_rows(aggregate_trust)

    print("=" * 132)
    print("Aggregate: held-out batch-B transfer at each file's batch-A-selected scale")
    print(
        "rank\tcandidate\tprotocol\tn\tmean_red_B\tstd_red_B\tpositive_B\t"
        "mean_red_A\tmean_scale\tboundary\tmean_cos_NM\tmean_solve_s"
    )
    for rank, row in enumerate(
        sorted(actual_aggregate, key=lambda item: item["mean_b"], reverse=True)[: args.top],
        1,
    ):
        print(
            f"{rank}\t{row['name']}\t{row['protocol']}\t{row['n']}\t"
            f"{row['mean_b']:.8g}\t{row['std_b']:.8g}\t"
            f"{row['positive_b']}/{row['n']}\t{row['mean_a']:.8g}\t"
            f"{row['mean_scale']:.6g}\t{row['boundary_count']}/{row['n']}\t"
            f"{row['mean_cos_nm']:.6g}\t{row['mean_solve']:.6g}"
        )
    print()

    print("=" * 132)
    print("Aggregate: best positive-prediction trial in each file")
    print(
        "rank\tcandidate\tprotocol\tn\tmean_red_B\tstd_red_B\tpositive_B\t"
        "mean_red_A\tmean_scale\tboundary\tmean_pred\tmean_trust_A"
    )
    for rank, row in enumerate(
        sorted(trust_aggregate, key=lambda item: item["mean_b"], reverse=True)[: args.top],
        1,
    ):
        print(
            f"{rank}\t{row['name']}\t{row['protocol']}\t{row['n']}\t"
            f"{row['mean_b']:.8g}\t{row['std_b']:.8g}\t"
            f"{row['positive_b']}/{row['n']}\t{row['mean_a']:.8g}\t"
            f"{row['mean_scale']:.6g}\t{row['boundary_count']}/{row['n']}\t"
            f"{row['mean_pred']:.8g}\t{row['mean_trust']:.8g}"
        )
    print()

    print("=" * 132)
    print("Native local-model sanity")
    print("candidate\tn\tpasses\tmean_predicted_reduction\tmean_model_objective")
    sanity_rows = []
    for name, observations in sanity.items():
        sanity_rows.append(
            {
                "name": name,
                "n": len(observations),
                "passes": sum(item["pass"] is True for item in observations),
                "mean_pred": safe_mean(item["pred"] for item in observations),
                "mean_objective": safe_mean(item["objective"] for item in observations),
            }
        )
    sanity_rows.sort(key=lambda row: (row["passes"] / max(row["n"], 1), row["mean_pred"]), reverse=True)
    for row in sanity_rows:
        print(
            f"{row['name']}\t{row['n']}\t{row['passes']}/{row['n']}\t"
            f"{row['mean_pred']:.8g}\t{row['mean_objective']:.8g}"
        )


if __name__ == "__main__":
    main()
