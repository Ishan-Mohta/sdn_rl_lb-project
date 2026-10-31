"""Shared rollout / evaluation helpers."""
import numpy as np


def rollout(env, policy, seed=None):
    """Run one episode with `policy` (needs .act(obs); optional .reset())."""
    if hasattr(policy, "reset"):
        policy.reset()
    obs, _ = env.reset(seed=seed)
    ret, lats, lvs = 0.0, [], []
    while True:
        obs, r, term, trunc, info = env.step(policy.act(obs))
        ret += r
        lats.append(info["latency"])
        lvs.append(info["load_var"])
        if term or trunc:
            break
    return {"reward": ret,
            "avg_latency": float(np.mean(lats)),
            "p95_latency": float(np.percentile(lats, 95)),
            "load_var": float(np.mean(lvs))}


def evaluate(env, policy, n_episodes=30, seed0=10_000):
    """Mean metrics over n_episodes with fixed seeds (same traffic for every method)."""
    res = [rollout(env, policy, seed=seed0 + i) for i in range(n_episodes)]
    return {k: float(np.mean([r[k] for r in res])) for k in res[0]}, res


class GreedyAgentPolicy:
    """Wraps a trained DQNAgent as a greedy policy."""
    def __init__(self, agent):
        self.agent = agent

    def act(self, obs):
        return self.agent.act(obs, greedy=True)
