#!/usr/bin/env python3
"""Evaluate a causal Guide-route prior for reranking saved GTL top-k outputs."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from audit_route_verify_run import load_graph, shortest_distance


def expected_viewpoint(previous_dialog, current_step):
    if previous_dialog is None:
        return None
    path = previous_dialog.get("answer_seen_path")
    if not path:
        return None
    elapsed = max(0, current_step - int(previous_dialog["nav_idx"]))
    return path[min(elapsed, len(path) - 1)]


def select_candidate(detail, expected, graph, weight, top_k):
    candidates = detail.get("localization_top_viewpoints", [])[:top_k]
    probabilities = detail.get("localization_top_probabilities", [])[:top_k]
    if expected is None or not candidates or len(candidates) != len(probabilities):
        return detail.get("localized_viewpoint")
    costs = [
        -math.log(max(float(probability), 1e-12))
        + weight * shortest_distance(graph, candidate, expected)
        for candidate, probability in zip(candidates, probabilities)
    ]
    return candidates[min(range(len(costs)), key=costs.__getitem__)]


def analyze(trajectories, connectivity_dir, weights, top_k):
    graphs = {}
    rows = []
    for trajectory in trajectories:
        scan = trajectory["scan"]
        graph = graphs.setdefault(scan, load_graph(connectivity_dir, scan))
        previous_dialog = None
        for detail in trajectory.get("navigation_detail", []):
            if not detail.get("ask"):
                continue
            candidates = detail.get("localization_top_viewpoints", [])
            if not candidates:
                continue
            expected = expected_viewpoint(previous_dialog, detail["nav_idx"])
            actual = detail["gt_viewpoint"]
            row = {
                "instr_id": str(trajectory["instr_id"]),
                "scan": scan,
                "step": int(detail["nav_idx"]),
                "actual": actual,
                "raw": candidates[0],
                "topk_hit": actual in candidates[:top_k],
                "has_prior": expected is not None,
                "expected": expected,
                "prior_exact": expected == actual,
                "selected": {},
            }
            for weight in weights:
                row["selected"][str(weight)] = select_candidate(
                    detail, expected, graph, weight, top_k
                )
            rows.append(row)
            previous_dialog = detail

    def accuracy(predicate, subset):
        selected = [row for row in rows if subset(row)]
        return (
            100 * sum(predicate(row) for row in selected) / len(selected)
            if selected
            else 0.0
        )

    summary = {
        "dialogs": len(rows),
        "dialogs_with_prior": sum(row["has_prior"] for row in rows),
        "raw_top1_percent": accuracy(
            lambda row: row["raw"] == row["actual"], lambda row: True
        ),
        "raw_top1_with_prior_percent": accuracy(
            lambda row: row["raw"] == row["actual"], lambda row: row["has_prior"]
        ),
        "topk_oracle_percent": accuracy(
            lambda row: row["topk_hit"], lambda row: True
        ),
        "topk_oracle_with_prior_percent": accuracy(
            lambda row: row["topk_hit"], lambda row: row["has_prior"]
        ),
        "prior_exact_percent": accuracy(
            lambda row: row["prior_exact"], lambda row: row["has_prior"]
        ),
        "rerank": {},
    }
    for weight in weights:
        key = str(weight)
        summary["rerank"][key] = {
            "all_percent": accuracy(
                lambda row: row["selected"][key] == row["actual"],
                lambda row: True,
            ),
            "with_prior_percent": accuracy(
                lambda row: row["selected"][key] == row["actual"],
                lambda row: row["has_prior"],
            ),
        }
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument(
        "--connectivity-dir",
        type=Path,
        default=Path("external/RAINbow/dataset/connectivity"),
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    trajectories = json.loads(args.trajectories.read_text())
    summary, rows = analyze(
        trajectories, args.connectivity_dir, args.weights, args.top_k
    )
    output = {
        "trajectories": str(args.trajectories),
        "top_k": args.top_k,
        "summary": summary,
        "rows": rows,
    }
    rendered = json.dumps(output, indent=2) + "\n"
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
