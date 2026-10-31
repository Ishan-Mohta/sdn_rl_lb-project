# SDN Load Balancing with DQN and Dueling DQN — Person B (ML side)

## Project layout
```
envs/lb_env.py        Simulated load-balancing environment (state / action / reward)
baselines.py          Round Robin, Weighted RR, Least Connections, Random
agents/networks.py    QNet (DQN) and DuelingQNet
agents/dqn_agent.py   One agent class; dueling=True/False is the only difference
agents/replay_buffer.py
train.py              Train DQN / Dueling DQN over several seeds, logs CSV, saves models
evaluate.py           Compare everything on identical traffic -> eval_summary.csv
plot_results.py       Learning curves + bar charts for the paper
select_server.py      select_server(state) for Person A's Ryu controller
```

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Your workflow (in order)

**Phase 1 - Verify the pipeline (10 min)**
```bash
python evaluate.py --baselines-only      # baseline numbers, no training needed
python train.py --smoke                  # tiny DQN + Dueling run; must finish without errors
```
Reference baseline results (50 episodes): Round Robin ~12.6 latency, Weighted RR ~1.22,
Least Connections ~1.08, Random ~13.5. A good agent should get below Least Connections.

**Phase 2 - Agree on the interface with Person A**
Send them the state contract below. Everything depends on it.

**Phase 3 - Train**
```bash
python train.py --algo both --seeds 3 --episodes 300
python evaluate.py
python plot_results.py
```
Check `results/plots/`. If the agents are clearly worse than Least Connections, go to Phase 4.

**Phase 4 - Tune (change one thing at a time, keep a table of runs)**
| What | Flag | Try |
|---|---|---|
| learning rate | `--lr` | 1e-3, 5e-4, 1e-4 |
| discount | `--gamma` | 0.9, 0.95, 0.99 (rewards are dense, lower often learns faster) |
| network size | `--hidden` | 64, 128, 256 |
| target sync | `--target-update` | 200, 500, 1000 |
| episodes | `--episodes` | 300, 600, 1000 |
| reward weights | `--alpha --beta` | beta in 1, 2, 5 (use a separate `--out` folder per setting) |

Final runs: 5 seeds per method.

**Phase 5 - Integrate (after Person A has Mininet + Ryu working)**
Person A calls `Selector("results/models/dueling_seed0.pt").select_server(build_state(...))`.
If the agent behaves worse on Mininet than in simulation, collect Mininet data and fine-tune briefly.

**Phase 6 - Paper sections:** RL/DQN/Dueling background, method, training details, results analysis.

## State / action / reward contract (share with Person A)
State, 10 values in [0,1], this order:
`[load_0..2, conns_0..2, lat_0..2, request_size]`

| Feature | Meaning | Normalisation |
|---|---|---|
| load_i | time to clear server i's backlog | / 10, clip to 1 |
| conns_i | active connections on server i | / 20, clip to 1 |
| lat_i | moving average of recent response time | / 10, clip to 1 |
| request_size | size of the flow being routed | / 5 x mean size, clip to 1 |

Action: server index 0..2.  Reward: `-(alpha * latency/10 + beta * load_variance)`.
If Person A cannot measure flow size, train with `--no-size` (state becomes 9 values).
Use `build_state()` in `select_server.py` to do the normalisation, so both sides use identical constants.

## Simulator parameters (report these in the paper)
3 servers with speeds 1, 2, 4; 500 requests per episode; exponential request sizes (mean 1);
bursty arrivals from a 2-state Markov chain (mean inter-arrival 0.22 normal, 0.10 burst).
All of it is in `EnvConfig`.

## Notes on honest reporting
- Learning curves of `train_reward` include epsilon-greedy exploration; the `eval_reward` curves are greedy.
- Evaluation uses identical episode seeds for every method, so the traffic is the same.
- Dueling DQN may show only a small gain with 3 servers. Report it as measured; optionally test 5-8 servers
  by changing `capacities` in `EnvConfig`.
