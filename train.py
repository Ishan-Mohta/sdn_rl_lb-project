"""Train DQN and/or Dueling DQN on the simulated load-balancing environment.

Examples
    python train.py --smoke                       # 1-minute sanity check
    python train.py --algo both --seeds 3 --episodes 300
    python train.py --algo dueling --alpha 1 --beta 5 --out results_beta5
"""
import argparse
import csv
import os
import random
from dataclasses import asdict

import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from envs.lb_env import EnvConfig, LBEnv
from utils import GreedyAgentPolicy, evaluate

LOG_FIELDS = ["episode", "train_reward", "train_avg_latency", "train_load_var", "loss", "epsilon",
              "eval_reward", "eval_avg_latency", "eval_load_var"]


def train_one(algo, seed, args, env_cfg):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env, eval_env = LBEnv(env_cfg), LBEnv(env_cfg)
    s_dim, a_dim = env.observation_space.shape[0], env.action_space.n

    eps_decay = args.eps_decay_steps or int(0.4 * args.episodes * env_cfg.episode_len)
    agent = DQNAgent(s_dim, a_dim, dueling=(algo == "dueling"), hidden=args.hidden, lr=args.lr,
                     gamma=args.gamma, batch_size=args.batch_size, buffer_size=args.buffer_size,
                     target_update=args.target_update, eps_decay_steps=eps_decay,
                     learn_start=args.learn_start, seed=seed)
    n_params = sum(p.numel() for p in agent.q.parameters())
    print(f"[{algo} seed={seed}] params={n_params}  eps_decay_steps={eps_decay}")

    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
    log_path = os.path.join(args.out, "logs", f"{algo}_seed{seed}.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=seed if ep == 0 else None)
            ep_r, lats, lvs, losses = 0.0, [], [], []
            while True:
                a = agent.act(obs)
                obs2, r, term, trunc, info = env.step(a)
                # time-limit truncation is NOT a real terminal state -> keep bootstrapping
                agent.remember(obs, a, r, obs2, term)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                ep_r += r; lats.append(info["latency"]); lvs.append(info["load_var"])
                obs = obs2
                if term or trunc:
                    break

            row = {"episode": ep, "train_reward": ep_r, "train_avg_latency": np.mean(lats),
                   "train_load_var": np.mean(lvs), "loss": np.mean(losses) if losses else "",
                   "epsilon": agent.epsilon, "eval_reward": "", "eval_avg_latency": "", "eval_load_var": ""}
            if (ep + 1) % args.eval_every == 0 or ep == args.episodes - 1:
                m, _ = evaluate(eval_env, GreedyAgentPolicy(agent), n_episodes=args.eval_episodes)
                row.update(eval_reward=m["reward"], eval_avg_latency=m["avg_latency"], eval_load_var=m["load_var"])
                print(f"  ep {ep+1:4d}  train_R={ep_r:9.1f}  eval_R={m['reward']:8.1f}  "
                      f"eval_lat={m['avg_latency']:.3f}  eps={agent.epsilon:.2f}")
            writer.writerow(row)

    model_path = os.path.join(args.out, "models", f"{algo}_seed{seed}.pt")
    agent.save(model_path, extra={"env_cfg": asdict(env_cfg), "algo": algo, "seed": seed})
    print(f"  saved {model_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["dqn", "dueling", "both"], default="both")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--out", default="results")
    # agent hyper-parameters
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--target-update", type=int, default=500)
    p.add_argument("--learn-start", type=int, default=1_000)
    p.add_argument("--eps-decay-steps", type=int, default=0, help="0 = 40%% of total training steps")
    # environment / reward
    p.add_argument("--alpha", type=float, default=1.0, help="latency weight in reward")
    p.add_argument("--beta", type=float, default=1.0, help="load-variance weight in reward")
    p.add_argument("--no-size", action="store_true", help="hide request size from the state")
    p.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline works")
    args = p.parse_args()

    if args.smoke:
        args.episodes, args.seeds, args.eval_every, args.eval_episodes = 6, 1, 3, 2
        args.learn_start, args.out = 200, "results_smoke"

    env_cfg = EnvConfig(alpha=args.alpha, beta=args.beta, use_request_size=not args.no_size)
    algos = ["dqn", "dueling"] if args.algo == "both" else [args.algo]
    for algo in algos:
        for seed in range(args.seeds):
            train_one(algo, seed, args, env_cfg)


if __name__ == "__main__":
    main()
