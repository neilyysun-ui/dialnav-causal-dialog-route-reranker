#!/usr/bin/env python3
"""Unit checks for online temporal GTL localization reranking."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "RAINbow" / "holistic"))

from holistic_utils.temporal_localization import (  # noqa: E402
    TemporalLocalizationReranker,
)


def main():
    distances = {
        "scan": {
            "a": {"a": 0.0, "b": 1.0, "c": 2.0},
            "b": {"a": 1.0, "b": 0.0, "c": 1.0},
            "c": {"a": 2.0, "b": 1.0, "c": 0.0},
        }
    }
    metadata = {
        "top_viewpoints": ["a", "b"],
        "top_probabilities": [0.6, 0.4],
    }
    reranker = TemporalLocalizationReranker(
        batch_size=2, shortest_distances=distances, weight=1.0, top_k=2
    )
    selected, expected, probability = reranker.rerank(
        0, "scan", "a", metadata, 0
    )
    assert (selected, expected, probability) == ("a", None, 0.6)

    reranker.update(2, [0], [["a", "b", "c"], []])
    assert reranker.expected_viewpoint(0, 2) == "a"
    assert reranker.expected_viewpoint(0, 4) == "c"
    assert reranker.expected_viewpoint(0, 8) == "c"
    selected, expected, probability = reranker.rerank(
        0, "scan", "a", metadata, 3
    )
    assert (selected, expected, probability) == ("b", "b", 0.4)

    reranker.update(5, [0], None)
    assert reranker.expected_viewpoint(0, 6) is None
    assert reranker.expected_viewpoint(1, 6) is None

    try:
        TemporalLocalizationReranker(1, distances, weight=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative weights must fail")
    print("temporal localization checks passed")


if __name__ == "__main__":
    main()
