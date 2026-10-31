"""Baseline policies. Interface: reset(), act(obs) -> server index."""
import numpy as np


class RoundRobin:
    name = "round_robin"

    def __init__(self, n_servers, **_):
        self.n, self.i = n_servers, -1

    def reset(self):
        self.i = -1

    def act(self, obs):
        self.i = (self.i + 1) % self.n
        return self.i


class WeightedRoundRobin:
    """Smooth weighted round robin; weights = server capacities (needs prior knowledge)."""
    name = "weighted_round_robin"

    def __init__(self, n_servers, capacities=None, **_):
        self.n = n_servers
        self.w = np.asarray(capacities if capacities is not None else np.ones(n_servers), float)
        self.cur = np.zeros(n_servers)

    def reset(self):
        self.cur[:] = 0

    def act(self, obs):
        self.cur += self.w
        i = int(np.argmax(self.cur))
        self.cur[i] -= self.w.sum()
        return i


class LeastConnections:
    """Pick the server with the fewest active connections (random tie-break)."""
    name = "least_connections"

    def __init__(self, n_servers, seed=0, **_):
        self.n, self.rng = n_servers, np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, obs):
        conns = np.asarray(obs[self.n:2 * self.n])
        best = np.flatnonzero(conns == conns.min())
        return int(self.rng.choice(best))


class RandomPolicy:
    name = "random"

    def __init__(self, n_servers, seed=0, **_):
        self.n, self.rng = n_servers, np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, obs):
        return int(self.rng.integers(self.n))


BASELINES = {"round_robin": RoundRobin, "weighted_round_robin": WeightedRoundRobin,
             "least_connections": LeastConnections, "random": RandomPolicy}
