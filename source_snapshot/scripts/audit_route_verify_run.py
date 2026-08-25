#!/usr/bin/env python3
"""Audit causal Route-and-Verify trajectories without trusting aggregate logs."""

import argparse
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path


GOAL_CONFIRMATION = (
    "You are already at the target location. Stop here now; "
    "do not move anywhere else."
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(path):
    result = []
    for item in path:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def load_graph(connectivity_dir, scan):
    records = json.loads(
        (connectivity_dir / f"{scan}_connectivity.json").read_text()
    )
    included = {
        index: record
        for index, record in enumerate(records)
        if record["included"]
    }
    graph = {record["image_id"]: [] for record in included.values()}
    for index, record in included.items():
        source = record["image_id"]
        source_position = (record["pose"][3], record["pose"][7], record["pose"][11])
        for neighbor_index, connected in enumerate(record["unobstructed"]):
            if not connected or neighbor_index not in included:
                continue
            target = included[neighbor_index]
            target_position = (
                target["pose"][3],
                target["pose"][7],
                target["pose"][11],
            )
            graph[source].append(
                (target["image_id"], math.dist(source_position, target_position))
            )
    return graph


def shortest_distance(graph, source, target):
    if source == target:
        return 0.0
    distances = {source: 0.0}
    queue = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == target:
            return distance
        for neighbor, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    raise ValueError(f"No path from {source} to {target}")


def dialog_efficiency(dtc, annotation):
    dtc_gt = len(annotation["dialog"])
    nsc_gt = len(annotation["nav_trajectory"])
    denominator = nsc_gt - dtc_gt
    if denominator <= 0:
        return 1.0 if dtc <= dtc_gt else 0.0
    return 1.0 - min(max(dtc - dtc_gt, 0) / denominator, 1.0)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def audit(trajectories, annotations, connectivity_dir):
    annotations_by_id = {str(item["instr_id"]): item for item in annotations}
    trajectory_ids = [str(item["instr_id"]) for item in trajectories]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("Duplicate trajectory instruction IDs")
    if set(trajectory_ids) != set(annotations_by_id):
        missing = set(annotations_by_id) - set(trajectory_ids)
        extra = set(trajectory_ids) - set(annotations_by_id)
        raise ValueError(f"Instruction mismatch: missing={missing}, extra={extra}")

    graphs = {}
    rows = []
    for trajectory in trajectories:
        annotation = annotations_by_id[str(trajectory["instr_id"])]
        scan = annotation["scan"]
        graph = graphs.setdefault(scan, load_graph(connectivity_dir, scan))
        goals = set(annotation["end_panos"])
        path = flatten(trajectory["path"])
        if not path:
            raise ValueError(f"{trajectory['instr_id']}: empty path")
        details = trajectory.get("navigation_detail", [])
        dialogs = [detail for detail in details if detail.get("ask")]
        confirmations = [
            detail
            for detail in dialogs
            if detail.get("answer") == GOAL_CONFIRMATION
        ]
        true_confirmations = [
            detail for detail in confirmations if detail["gt_viewpoint"] in goals
        ]
        false_confirmations = [
            detail for detail in confirmations if detail["gt_viewpoint"] not in goals
        ]
        localization_distances = []
        exact_localizations = 0
        for detail in dialogs:
            localized = detail.get("localized_viewpoint")
            actual = detail.get("gt_viewpoint")
            if localized is None or actual is None:
                continue
            exact_localizations += localized == actual
            localization_distances.append(
                shortest_distance(graph, actual, localized)
            )

        success = path[-1] in goals
        dtc = len(dialogs)
        efficiency = dialog_efficiency(dtc, annotation)
        proposed_stops = sum(
            bool(detail.get("proposed_stop_before_dialog")) for detail in details
        )
        rejected_stops = sum(bool(detail.get("stop_rejected")) for detail in details)
        max_length_reached = len(details) >= 50
        rows.append(
            {
                "instr_id": trajectory["instr_id"],
                "scan": scan,
                "success": int(success),
                "score": efficiency * success,
                "dtc": dtc,
                "steps": max(0, len(path) - 1),
                "path_nodes": len(path),
                "proposed_stops": proposed_stops,
                "rejected_stops": rejected_stops,
                "confirmations": len(confirmations),
                "true_confirmations": len(true_confirmations),
                "false_confirmations": len(false_confirmations),
                "exact_localizations": exact_localizations,
                "localized_dialogs": len(localization_distances),
                "mean_localization_distance": mean(localization_distances),
                "max_localization_distance": max(localization_distances, default=0.0),
                "max_length_reached": int(max_length_reached),
                "stop_viewpoint": path[-1],
            }
        )

    localized_dialogs = sum(row["localized_dialogs"] for row in rows)
    summary = {
        "episodes": len(rows),
        "score": mean([row["score"] for row in rows]),
        "score_percent": 100 * mean([row["score"] for row in rows]),
        "sr_percent": 100 * mean([row["success"] for row in rows]),
        "mean_dtc": mean([row["dtc"] for row in rows]),
        "mean_steps": mean([row["steps"] for row in rows]),
        "proposed_stops": sum(row["proposed_stops"] for row in rows),
        "rejected_stops": sum(row["rejected_stops"] for row in rows),
        "confirmations": sum(row["confirmations"] for row in rows),
        "true_confirmations": sum(row["true_confirmations"] for row in rows),
        "false_confirmations": sum(row["false_confirmations"] for row in rows),
        "localized_dialogs": localized_dialogs,
        "exact_localization_percent": (
            100 * sum(row["exact_localizations"] for row in rows) / localized_dialogs
            if localized_dialogs
            else 0.0
        ),
        "mean_localization_distance": (
            sum(
                row["mean_localization_distance"] * row["localized_dialogs"]
                for row in rows
            )
            / localized_dialogs
            if localized_dialogs
            else 0.0
        ),
        "max_length_reached": sum(row["max_length_reached"] for row in rows),
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--connectivity-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    trajectories = json.loads(args.trajectories.read_text())
    annotations = json.loads(args.annotations.read_text())
    summary, rows = audit(
        trajectories, annotations, args.connectivity_dir
    )
    summary["trajectories"] = str(args.trajectories)
    summary["trajectories_sha256"] = sha256(args.trajectories)
    summary["annotations"] = str(args.annotations)
    summary["annotations_sha256"] = sha256(args.annotations)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "route_verify_audit.json").write_text(
        json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n"
    )
    with (args.out_dir / "route_verify_per_episode.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
