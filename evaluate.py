"""Evaluate baselines and trained agents on identical traffic (same episode seeds).

    python evaluate.py                    # uses results/models/*.pt if present
    python evaluate.py --baselines-only   # no PyTorch needed
Writes <out>/eval_summary.csv  (mean/std over all evaluated episodes; agents pooled over seeds).
"""
import argparse
import csv
import glob
import os

import numpy as np

from baselines import BASELINES
from envs.lb_env import EnvConfig, LBEnv
from utils import GreedyAgentPolicy, rollout

METRICS = ["avg_latency", "p95_latency", "load_var", "reward"]


def summarize(rows):
    out = {}
    for m in METRICS:
        vals = [r[m] for r in rows]
        out[f"{m}_mean"], out[f"{m}_std"] = float(np.mean(vals)), float(np.std(vals))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed0", type=int, default=10_000)
    p.add_argument("--baselines-only", action="store_true")
    args = p.parse_args()

    model_files = {} if args.baselines_only else {
        algo: sorted(glob.glob(os.path.join(args.out, "models", f"{algo}_seed*.pt")))
        for algo in ("dqn", "dueling")}

    # use the exact env config the agents were trained with
    env_cfg = EnvConfig()
    first = next((f for fs in model_files.values() for f in fs), None)
    if first:
        import torch
        extra = torch.load(first, map_location="cpu", weights_only=False)["extra"]
        cfg = dict(extra["env_cfg"]); cfg["capacities"] = tuple(cfg["capacities"])
        env_cfg = EnvConfig(**cfg)
    env = LBEnv(env_cfg)
    seeds = [args.seed0 + i for i in range(args.episodes)]

    results = {}
    for name, cls in BASELINES.items():
        pol = cls(env.n, seed=0, capacities=env_cfg.capacities)
        results[name] = summarize([rollout(env, pol, seed=s) for s in seeds])

    for algo, files in model_files.items():
        if not files:
            continue
        from agents.dqn_agent import DQNAgent
        rows = []
        for f in files:
            pol = GreedyAgentPolicy(DQNAgent.load(f))
            rows += [rollout(env, pol, seed=s) for s in seeds]
        results[algo] = summarize(rows)
        print(f"{algo}: pooled {len(files)} seeds x {len(seeds)} episodes")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "eval_summary.csv")
    cols = ["method"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for name, r in results.items():
            w.writerow({"method": name, **r})

    print(f"\n{'method':22s} {'latency':>9s} {'p95':>8s} {'load_var':>9s} {'reward':>9s}")
    for name, r in results.items():
        print(f"{name:22s} {r['avg_latency_mean']:9.3f} {r['p95_latency_mean']:8.2f} "
              f"{r['load_var_mean']:9.4f} {r['reward_mean']:9.1f}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
