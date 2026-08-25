import math


class TemporalLocalizationReranker:
    """Rerank GTL top-k predictions using only the previous Guide route."""

    def __init__(self, batch_size, shortest_distances, weight=0.5, top_k=5):
        if weight < 0:
            raise ValueError("temporal localization weight must be non-negative")
        if top_k <= 0:
            raise ValueError("temporal localization top-k must be positive")
        self.shortest_distances = shortest_distances
        self.weight = weight
        self.top_k = top_k
        self.previous_routes = [None] * batch_size

    def expected_viewpoint(self, index, current_step):
        previous = self.previous_routes[index]
        if previous is None or not previous["path"]:
            return None
        elapsed = max(0, current_step - previous["step"])
        return previous["path"][min(elapsed, len(previous["path"]) - 1)]

    def rerank(self, index, scan, raw_viewpoint, metadata, current_step):
        expected = self.expected_viewpoint(index, current_step)
        candidates = metadata.get("top_viewpoints", [])[: self.top_k]
        probabilities = metadata.get("top_probabilities", [])[: self.top_k]
        raw_probability = (
            float(probabilities[0]) if probabilities else None
        )
        if (
            expected is None
            or not candidates
            or len(candidates) != len(probabilities)
        ):
            return raw_viewpoint, expected, raw_probability

        scan_distances = self.shortest_distances.get(scan, {})
        if expected not in scan_distances:
            return raw_viewpoint, expected, raw_probability

        costs = []
        for candidate, probability in zip(candidates, probabilities):
            candidate_distances = scan_distances.get(candidate, {})
            if expected not in candidate_distances:
                costs.append(float("inf"))
                continue
            costs.append(
                -math.log(max(float(probability), 1e-12))
                + self.weight * candidate_distances[expected]
            )
        if not costs or all(math.isinf(cost) for cost in costs):
            return raw_viewpoint, expected, raw_probability
        selected_index = min(range(len(costs)), key=costs.__getitem__)
        return (
            candidates[selected_index],
            expected,
            float(probabilities[selected_index]),
        )

    def update(self, step, indices, answer_seen_paths):
        for index in indices:
            path = []
            if answer_seen_paths is not None and index < len(answer_seen_paths):
                path = answer_seen_paths[index] or []
            self.previous_routes[index] = {
                "step": int(step),
                "path": list(path),
            }
