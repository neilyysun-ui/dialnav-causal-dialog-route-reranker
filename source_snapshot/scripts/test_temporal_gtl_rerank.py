#!/usr/bin/env python3
"""Unit checks for the causal Guide-route localization prior."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_temporal_gtl_rerank import expected_viewpoint, select_candidate  # noqa: E402


def main():
    assert expected_viewpoint(None, 0) is None
    previous = {"nav_idx": 2, "answer_seen_path": ["a", "b", "c"]}
    assert expected_viewpoint(previous, 2) == "a"
    assert expected_viewpoint(previous, 4) == "c"
    assert expected_viewpoint(previous, 8) == "c"
    graph = {"x": [("y", 1.0)], "y": [("x", 1.0)]}
    detail = {
        "localized_viewpoint": "x",
        "localization_top_viewpoints": ["x", "y"],
        "localization_top_probabilities": [0.6, 0.4],
    }
    assert select_candidate(detail, "y", graph, weight=0.0, top_k=2) == "x"
    assert select_candidate(detail, "y", graph, weight=1.0, top_k=2) == "y"
    print("temporal GTL rerank checks passed")


if __name__ == "__main__":
    main()
