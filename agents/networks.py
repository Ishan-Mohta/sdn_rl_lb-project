"""Q-networks: plain DQN head vs Dueling head. Same depth/width for a fair comparison."""
import torch
import torch.nn as nn


class QNet(nn.Module):
    """Plain DQN: state -> Q(s, a) for every action."""

    def __init__(self, s_dim, a_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, a_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingQNet(nn.Module):
    """Dueling DQN: shared features -> V(s) stream and A(s,a) stream.
    Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)      (Wang et al., 2016, Eq. 9)
    """

    def __init__(self, s_dim, a_dim, hidden=128):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(s_dim, hidden), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.adv = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, a_dim))

    def forward(self, x):
        f = self.feat(x)
        v, a = self.value(f), self.adv(f)
        return v + a - a.mean(dim=1, keepdim=True)
