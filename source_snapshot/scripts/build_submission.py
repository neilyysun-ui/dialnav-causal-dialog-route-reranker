#!/usr/bin/env python3
"""Combine split outputs into a validated DialNav submission."""

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_FIELDS = {"instr_id", "scan", "target", "path", "dialog"}
REQUIRED_DIALOG_FIELDS = {
    "nav_idx",
    "question",
    "answer",
    "localized_viewpoint",
    "viewpoint",
}


def load(path):
    with path.open() as handle:
        return json.load(handle)


def load_graph(connectivity_dir, scan):
    nodes = load(connectivity_dir / f"{scan}_connectivity.json")
    included = {node["image_id"]: index for index, node in enumerate(nodes) if node["included"]}
    adjacency = {node_id: set() for node_id in included}
    for node_id, index in included.items():
        adjacency[node_id] = {
            nodes[neighbor]["image_id"]
            for neighbor, connected in enumerate(nodes[index]["unobstructed"])
            if connected and nodes[neighbor]["included"]
        }
    return adjacency


def normalize_target(target):
    normalized = " ".join(str(target).strip().lower().split())
    if normalized.startswith("target :"):
        normalized = normalized[len("target :"):].strip()
    elif normalized.startswith("target:"):
        normalized = normalized[len("target:"):].strip()
    return normalized


def validate_split(split, records, ground_truth, connectivity_dir, graph_cache):
    expected = {str(record["instr_id"]) for record in ground_truth}
    observed = [str(record["instr_id"]) for record in records]
    if len(observed) != len(set(observed)):
        raise ValueError(f"{split}: duplicate instruction IDs")
    if set(observed) != expected:
        raise ValueError(
            f"{split}: coverage mismatch, missing={len(expected - set(observed))}, "
            f"extra={len(set(observed) - expected)}"
        )
    ground_truth_by_id = {str(record["instr_id"]): record for record in ground_truth}
    for record in records:
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(f"{split}/{record.get('instr_id')}: missing fields {sorted(missing)}")
        if not isinstance(record["path"], list) or not record["path"]:
            raise ValueError(f"{split}/{record['instr_id']}: path must be non-empty")
        expected_record = ground_truth_by_id[str(record["instr_id"])]
        if record["scan"] != expected_record["scan"]:
            raise ValueError(f"{split}/{record['instr_id']}: scan does not match ground truth")
        if normalize_target(record["target"]) != normalize_target(expected_record["target"]):
            raise ValueError(f"{split}/{record['instr_id']}: target does not match ground truth")
        expected_start = expected_record.get("start_pano")
        if expected_start is not None and record["path"][0] != expected_start:
            raise ValueError(
                f"{split}/{record['instr_id']}: path starts at {record['path'][0]}, "
                f"expected {expected_start}"
            )
        if record["scan"] not in graph_cache:
            graph_cache[record["scan"]] = load_graph(connectivity_dir, record["scan"])
        graph = graph_cache[record["scan"]]
        for viewpoint in record["path"]:
            if viewpoint not in graph:
                raise ValueError(f"{split}/{record['instr_id']}: unknown viewpoint {viewpoint}")
        for source, target in zip(record["path"], record["path"][1:]):
            if source != target and target not in graph[source]:
                raise ValueError(
                    f"{split}/{record['instr_id']}: non-adjacent path edge {source} -> {target}"
                )
        if not isinstance(record["dialog"], list):
            raise ValueError(f"{split}/{record['instr_id']}: dialog must be a list")
        for turn in record["dialog"]:
            missing = REQUIRED_DIALOG_FIELDS - set(turn)
            if missing:
                raise ValueError(
                    f"{split}/{record['instr_id']}: dialog turn missing fields {sorted(missing)}"
                )
            for field in ("localized_viewpoint", "viewpoint"):
                if turn[field] not in graph:
                    raise ValueError(
                        f"{split}/{record['instr_id']}: unknown dialog {field} {turn[field]}"
                    )
        nav_indices = [turn["nav_idx"] for turn in record["dialog"]]
        if nav_indices != sorted(nav_indices) or len(nav_indices) != len(set(nav_indices)):
            raise ValueError(f"{split}/{record['instr_id']}: invalid dialog nav_idx order")
    by_id = {str(record["instr_id"]): record for record in records}
    return [by_id[str(record["instr_id"])] for record in ground_truth]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen", type=Path, required=True)
    parser.add_argument("--unseen", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--connectivity-dir", type=Path, required=True)
    parser.add_argument(
        "--test-template",
        type=Path,
        help="Organizer-provided submission containing the held-out test records.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    seen_data = load(args.seen)
    unseen_data = load(args.unseen)
    seen_records = seen_data.get("val_seen")
    unseen_records = unseen_data.get("val_unseen")
    if seen_records is None or unseen_records is None:
        raise ValueError("Input submissions do not contain the requested split")

    graph_cache = {}
    result = {
        "val_seen": validate_split(
            "val_seen",
            seen_records,
            load(args.split_dir / "val_seen.json"),
            args.connectivity_dir,
            graph_cache,
        ),
        "val_unseen": validate_split(
            "val_unseen",
            unseen_records,
            load(args.split_dir / "val_unseen.json"),
            args.connectivity_dir,
            graph_cache,
        ),
    }
    if args.test_template:
        test_records = load(args.test_template).get("test")
        if test_records is None:
            raise ValueError("Test template does not contain the test split")
        result["test"] = validate_split(
            "test",
            test_records,
            test_records,
            args.connectivity_dir,
            graph_cache,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    hash_path = args.out.with_suffix(args.out.suffix + ".sha256")
    hash_path.write_text(f"{digest}  {args.out.name}\n")
    counts = ", ".join(f"{split}={len(records)}" for split, records in result.items())
    print(f"Validated {counts}")
    print(f"Wrote {args.out}")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
