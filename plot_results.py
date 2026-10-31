"""Make paper figures from training logs and eval_summary.csv.

    python plot_results.py            # reads ./results, writes ./results/plots/*.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"dqn": "tab:blue", "dueling": "tab:red"}
LABELS = {"dqn": "DQN", "dueling": "Dueling DQN", "round_robin": "Round Robin",
          "weighted_round_robin": "Weighted RR", "least_connections": "Least Conn.", "random": "Random"}


def smooth(x, k=10):
    return pd.Series(x).rolling(k, min_periods=1).mean().to_numpy()


def curves(out, col, ylabel, fname, only_eval=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    for algo in ("dqn", "dueling"):
        runs = []
        for f in sorted(glob.glob(os.path.join(out, "logs", f"{algo}_seed*.csv"))):
            df = pd.read_csv(f)
            if only_eval:
                df = df.dropna(subset=[col])
                runs.append(pd.Series(df[col].to_numpy(), index=df["episode"].to_numpy()))
            else:
                runs.append(pd.Series(smooth(df[col].to_numpy()), index=df["episode"].to_numpy()))
        if not runs:
            continue
        m = pd.concat(runs, axis=1)
        mean, std = m.mean(axis=1), m.std(axis=1).fillna(0)
        ax.plot(mean.index, mean.values, color=COLORS[algo], label=f"{LABELS[algo]} (n={len(runs)})")
        ax.fill_between(mean.index, mean - std, mean + std, color=COLORS[algo], alpha=0.2)
    ax.set_xlabel("Episode"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "plots", fname), dpi=200); plt.close(fig)


def bars(out, metric, ylabel, fname):
    df = pd.read_csv(os.path.join(out, "eval_summary.csv"))
    order = [m for m in ["random", "round_robin", "weighted_round_robin", "least_connections", "dqn", "dueling"]
             if m in set(df["method"])]
    df = df.set_index("method").loc[order]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [COLORS.get(m, "tab:gray") for m in order]
    ax.bar([LABELS[m] for m in order], df[f"{metric}_mean"], yerr=df[f"{metric}_std"], color=colors,
           capsize=3, alpha=0.85)
    ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "plots", fname), dpi=200); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", default="results"); args = p.parse_args()
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
    if glob.glob(os.path.join(args.out, "logs", "*.csv")):
        curves(args.out, "train_reward", "Training reward (smoothed, \u03b5-greedy)", "learning_curve_train.png")
        curves(args.out, "eval_reward", "Greedy evaluation reward", "learning_curve_eval.png", only_eval=True)
        curves(args.out, "eval_avg_latency", "Greedy evaluation avg latency", "learning_curve_latency.png", only_eval=True)
    if os.path.exists(os.path.join(args.out, "eval_summary.csv")):
        bars(args.out, "avg_latency", "Average latency (time units)", "bar_latency.png")
        bars(args.out, "p95_latency", "95th-percentile latency (time units)", "bar_p95_latency.png")
        bars(args.out, "load_var", "Load variance across servers", "bar_load_variance.png")
    print("plots written to", os.path.join(args.out, "plots"))


if __name__ == "__main__":
    main()
