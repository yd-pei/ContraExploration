from collections import deque
import zlib
import numpy as np


class NStep:
    def __init__(self, n=3, gamma=0.99):
        self.n, self.gamma, self.queue = n, gamma, deque()

    def add(self, obs, action, reward, next_obs, terminated, truncated):
        self.queue.append((obs, action, reward, next_obs, terminated))
        out = []
        if len(self.queue) >= self.n:
            out.append(self._pop())
        if terminated or truncated:
            while self.queue:
                out.append(self._pop())
        return out

    def _pop(self):
        seq = list(self.queue)[:self.n]
        reward = sum(self.gamma ** i * t[2] for i, t in enumerate(seq))
        obs, action = seq[0][:2]
        next_obs, terminal = seq[-1][3:]
        self.queue.popleft()
        return obs, action, reward, next_obs, 0. if terminal else self.gamma ** len(seq)


class PrioritizedReplay:
    """Proportional PER with O(log N) sum tree and compressed uint8 stacks.

    Sampling is stratified with replacement. Global minimum probability normalises
    importance weights. No emulator, file handles, or GPU tensors enter replay.
    """
    def __init__(self, capacity=20000, alpha=0.5, seed=0):
        self.capacity, self.alpha = capacity, alpha
        self.base = 1 << (capacity - 1).bit_length()
        self.tree = np.zeros(2 * self.base, dtype=np.float64)
        self.minimum = np.full(2 * self.base, np.inf)
        self.data = [None] * capacity
        self.pos = self.size = 0
        self.max_priority = 1.
        self.rng = np.random.default_rng(seed)

    def _update(self, i, priority):
        j = self.base + i
        self.tree[j] = self.minimum[j] = priority ** self.alpha
        while j > 1:
            j //= 2
            self.tree[j] = self.tree[j * 2] + self.tree[j * 2 + 1]
            self.minimum[j] = min(self.minimum[j * 2], self.minimum[j * 2 + 1])

    def add(self, transition):
        o, a, r, no, d = transition
        self.data[self.pos] = (zlib.compress(o.tobytes(), 1), a, r, zlib.compress(no.tobytes(), 1), d)
        self._update(self.pos, self.max_priority)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch, beta):
        values = (np.arange(batch) + self.rng.random(batch)) * self.tree[1] / batch
        indices = []
        for value in values:
            j = 1
            while j < self.base:
                j *= 2
                if value >= self.tree[j]:
                    value -= self.tree[j]
                    j += 1
            indices.append(j - self.base)
        probabilities = self.tree[np.array(indices) + self.base] / self.tree[1]
        weights = (probabilities / (self.minimum[1] / self.tree[1])) ** (-beta)
        rows = [self.data[i] for i in indices]
        def decode(blob):
            return np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(4, 84, 84)
        return (np.stack([decode(t[0]) for t in rows]), np.array([t[1] for t in rows]),
                np.array([t[2] for t in rows], dtype=np.float32), np.stack([decode(t[3]) for t in rows]),
                np.array([t[4] for t in rows], dtype=np.float32), weights.astype(np.float32), indices)

    def update(self, indices, priorities):
        # Duplicate samples share the largest new priority, independent of order.
        unique = {}
        for i, p in zip(indices, priorities):
            unique[i] = max(unique.get(i, 0.), float(p) + 1e-6)
        for i, p in unique.items():
            if not np.isfinite(p):
                raise ValueError('Non-finite replay priority')
            self.max_priority = max(self.max_priority, p)
            self._update(i, p)
