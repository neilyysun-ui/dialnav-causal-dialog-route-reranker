#!/usr/bin/env python3
"""Verify the frozen alternative submission without reading evaluator labels."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "submission.json"
EXPECTED_SHA256 = "52c191922ff14423efe5b290c0a9e1aa0fa0e094c06227145db3a421fbbf5efc"
EXPECTED_COUNTS = {"val_seen": 91, "val_unseen": 241, "test": 285}
REQUIRED = {"instr_id", "scan", "target", "path", "dialog"}


def main():
    assert hashlib.sha256(SUBMISSION.read_bytes()).hexdigest() == EXPECTED_SHA256
    data = json.loads(SUBMISSION.read_text())
    assert {split: len(data[split]) for split in EXPECTED_COUNTS} == EXPECTED_COUNTS
    for split, records in data.items():
        ids = [str(record["instr_id"]) for record in records]
        assert len(ids) == len(set(ids)), f"duplicate ID in {split}"
        for record in records:
            assert REQUIRED.issubset(record)
            assert record["path"]
            nav_indices = [turn["nav_idx"] for turn in record["dialog"]]
            assert nav_indices == sorted(set(nav_indices))
    print("submission-integrity checks passed")


if __name__ == "__main__":
    main()
