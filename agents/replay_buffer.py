import numpy as np


class ReplayBuffer:
    """Fixed-size circular buffer of (s, a, r, s', done) stored in numpy arrays."""

    def __init__(self, capacity, s_dim, seed=0):
        self.cap, self.size, self.ptr = capacity, 0, 0
        self.s = np.zeros((capacity, s_dim), np.float32)
        self.s2 = np.zeros((capacity, s_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(done)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.size
