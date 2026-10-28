"""Мера новизны и архив (Lehman, Stanley, 2011).

Новизна особи x — средняя дистанция от её поведения до k ближайших соседей среди текущей
популяции и архива:

    ρ(x) = (1/k) · Σ_{i=1..k} dist(x, μ_i)

Особь попадает в архив, если ρ(x) > ρ_min.
"""
from __future__ import annotations

import numpy as np


class NoveltyArchive:
    def __init__(self, k: int = 15, threshold: float = 6.0, dynamic: bool = True,
                 max_add: int = 4, timeout: int = 10, floor: float = 0.25,
                 max_size: int | None = None):
        self.k = k
        self.threshold = threshold
        self.dynamic = dynamic
        self.max_add = max_add
        self.timeout = timeout
        self.floor = floor
        self.max_size = max_size
        self.items: list[np.ndarray] = []
        self._since_add = 0

    def __len__(self):
        return len(self.items)

    def novelty(self, pop_behaviors: np.ndarray) -> np.ndarray:
        ref = pop_behaviors if not self.items else np.vstack([pop_behaviors, np.array(self.items)])
        d = np.sqrt(((pop_behaviors[:, None, :] - ref[None, :, :]) ** 2).sum(-1))   # (n, n+m)
        n = len(pop_behaviors)
        d[np.arange(n), np.arange(n)] = np.inf                                       # без самой себя
        k = min(self.k, d.shape[1] - 1)
        part = np.partition(d, k - 1, axis=1)[:, :k]
        return part.mean(axis=1)

    def update(self, pop_behaviors: np.ndarray, scores: np.ndarray, eligible: np.ndarray | None = None) -> int:
        mask = scores > self.threshold
        if eligible is not None:
            mask &= eligible
        if not self.items and not mask.any():
            mask = np.zeros_like(mask)
            mask[int(np.argmax(scores))] = True
        added = int(mask.sum())
        self.items.extend(pop_behaviors[mask])
        if self.max_size is not None and len(self.items) > self.max_size:
            self.items = self.items[-self.max_size:]
        if self.dynamic:
            if added > self.max_add:
                self.threshold *= 1.2
            if added == 0:
                self._since_add += 1
                if self._since_add >= self.timeout:
                    self.threshold = max(self.floor, self.threshold * 0.95)
                    self._since_add = 0
            else:
                self._since_add = 0
        return added
