"""Integration module for Person A's Ryu controller.

    from select_server import Selector, build_state
    sel = Selector("results/models/dueling_seed0.pt")
    state = build_state(loads, conns, latencies, request_size)   # raw metrics in
    server_idx = sel.select_server(state)

State vector ORDER (must match envs/lb_env.py):
    [load_0..load_{n-1}, conns_0..conns_{n-1}, lat_0..lat_{n-1}, request_size]
Everything is normalised to [0, 1] with the same constants used in training.
"""
import numpy as np

from envs.lb_env import EnvConfig

_C = EnvConfig()


def build_state(drain_time, conns, latency, request_size=None, cfg: EnvConfig = _C):
    """Convert RAW per-server metrics into the normalised state the agent was trained on.

    drain_time : estimated seconds needed to clear each server's backlog (or a CPU/queue proxy)
    conns      : active connection count per server
    latency    : moving average of recent response time per server (same unit as drain_time)
    request_size : size of the flow being routed (omit if the model was trained with --no-size)
    """
    s = np.concatenate([
        np.clip(np.asarray(drain_time, float) / cfg.drain_ref, 0, 1),
        np.clip(np.asarray(conns, float) / cfg.max_conns, 0, 1),
        np.clip(np.asarray(latency, float) / cfg.lat_ref, 0, 1),
    ])
    if request_size is not None:
        s = np.append(s, min(1.0, request_size / (5 * cfg.mean_size)))
    return s.astype(np.float32)


class Selector:
    def __init__(self, model_path, device="cpu"):
        from agents.dqn_agent import DQNAgent   # imported lazily so Ryu env only needs torch here
        self.agent = DQNAgent.load(model_path, device=device)

    def select_server(self, state) -> int:
        return self.agent.act(state, greedy=True)
