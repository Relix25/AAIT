from collections import deque

import numpy as np


class ReplayBuffer:
    """Uniform replay with compact uint8 pixels and preallocated arrays."""

    def __init__(self, capacity: int, shape=(64, 64, 3)):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.obs = np.empty((capacity, *shape), dtype=np.uint8)
        self.next_obs = np.empty((capacity, *shape), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, discount):
        i = self.pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.discounts[i] = discount
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx], self.discounts[idx]


class NStepAccumulator:
    """Emit each transition once with its actual n-step horizon."""

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = n, gamma
        self.queue = deque()

    def _pop(self):
        reward, discount = 0.0, 1.0
        horizon = min(self.n, len(self.queue))
        for k in range(horizon):
            item = self.queue[k]
            reward += discount * item[2]
            discount *= self.gamma * item[5]
        first, last = self.queue[0], self.queue[horizon - 1]
        self.queue.popleft()
        return first[0], first[1], reward, last[3], discount

    def add(self, obs, action, reward, next_obs, episode_done, bootstrap_mask):
        self.queue.append((obs, action, reward, next_obs, episode_done, bootstrap_mask))
        emitted = []
        if len(self.queue) >= self.n:
            emitted.append(self._pop())
        if episode_done:
            while self.queue:
                emitted.append(self._pop())
        return emitted
