"""DQN / Dueling DQN agent. The ONLY difference between the two is the network class."""
import torch
torch.set_num_threads(1)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .networks import DuelingQNet, QNet
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, s_dim, a_dim, dueling=False, hidden=128, lr=1e-3, gamma=0.99,
                 batch_size=64, buffer_size=50_000, target_update=500,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=20_000,
                 learn_start=1_000, grad_clip=10.0, device="cpu", seed=0):
        self.s_dim, self.a_dim, self.dueling, self.hidden = s_dim, a_dim, dueling, hidden
        self.gamma, self.batch_size = gamma, batch_size
        self.target_update, self.learn_start, self.grad_clip = target_update, learn_start, grad_clip
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        Net = DuelingQNet if dueling else QNet
        self.q = Net(s_dim, a_dim, hidden).to(self.device)
        self.q_target = Net(s_dim, a_dim, hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, s_dim, seed)
        self.total_steps = 0

    # ---- exploration ------------------------------------------------------
    @property
    def epsilon(self):
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.a_dim))
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(x).argmax(dim=1).item())

    # ---- learning ---------------------------------------------------------
    def remember(self, s, a, r, s2, done):
        self.buffer.add(s, a, r, s2, done)
        self.total_steps += 1

    def learn(self):
        """One gradient step. Returns the loss, or None while the buffer is warming up."""
        if len(self.buffer) < max(self.batch_size, self.learn_start):
            return None
        s, a, r, s2, d = (torch.as_tensor(x, device=self.device) for x in self.buffer.sample(self.batch_size))
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * (1.0 - d) * self.q_target(s2).max(dim=1).values
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        if self.total_steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # ---- persistence ------------------------------------------------------
    def save(self, path, extra=None):
        torch.save({"state_dict": self.q.state_dict(), "s_dim": self.s_dim, "a_dim": self.a_dim,
                    "dueling": self.dueling, "hidden": self.hidden, "extra": extra or {}}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ck["s_dim"], ck["a_dim"], dueling=ck["dueling"], hidden=ck["hidden"],
                    buffer_size=1, device=device)
        agent.q.load_state_dict(ck["state_dict"])
        agent.q.eval()
        return agent
