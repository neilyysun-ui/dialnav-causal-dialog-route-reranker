from interface.WTA import WTA


class StatefulConfidenceWtaModule(WTA):
    """Confidence WTA with per-episode memory for redundant-dialog control."""

    def __init__(
        self,
        threshold=0.7,
        cooldown_steps=0,
        refractory_threshold=None,
        refractory_steps=0,
        decay=0.0,
        floor=0.0,
        force_initial_ask=False,
        max_asks=None,
    ):
        self.threshold = threshold
        self.cooldown_steps = cooldown_steps
        self.refractory_threshold = refractory_threshold
        self.refractory_steps = refractory_steps
        self.decay = decay
        self.floor = floor
        self.force_initial_ask = force_initial_ask
        self.max_asks = max_asks
        self.last_ask_steps = None
        self.ask_counts = None

    def _reset(self, batch_size):
        self.last_ask_steps = [None] * batch_size
        self.ask_counts = [0] * batch_size

    def wta(self, t, prob, nav_outs):
        batch_size = len(prob)
        if t == 0 or self.last_ask_steps is None or len(self.last_ask_steps) != batch_size:
            self._reset(batch_size)

        max_probs = prob.max(1).values.detach().cpu().tolist()
        decisions = []
        for index, max_prob in enumerate(max_probs):
            last_ask = self.last_ask_steps[index]
            elapsed = None if last_ask is None else t - last_ask
            threshold = max(self.floor, self.threshold - self.decay * self.ask_counts[index])

            in_cooldown = elapsed is not None and elapsed <= self.cooldown_steps
            in_refractory = elapsed is not None and elapsed <= self.refractory_steps
            if in_refractory and self.refractory_threshold is not None:
                threshold = min(threshold, self.refractory_threshold)

            within_budget = (
                self.max_asks is None or self.ask_counts[index] < self.max_asks
            )
            ask = within_budget and (
                (self.force_initial_ask and t == 0)
                or (not in_cooldown and max_prob < threshold)
            )
            decisions.append(ask)
            if ask:
                self.last_ask_steps[index] = t
                self.ask_counts[index] += 1
        return decisions
