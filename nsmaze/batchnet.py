"""Векторизованное вычисление сразу всех сетей популяции.

neat-python считает каждую сеть отдельно, в цикле на чистом Python: 250 роботов × 400 шагов =
100 000 вызовов за поколение — это основная часть времени. Здесь все сети популяции
«склеиваются» в одну большую разреженную сеть и считаются по уровням (глубине узла) общими операциями numpy.
"""
from __future__ import annotations

import numpy as np


def _is_supported(net) -> bool:
    for _node, act, agg, *_ in net.node_evals:
        if getattr(act, "__name__", "") != "sigmoid_activation":
            return False
        if getattr(agg, "__name__", "") not in ("sum_aggregation", "sum"):
            return False
    return True


class BatchNetwork:
    def __init__(self, nets):
        self.nets = list(nets)
        self.n = len(self.nets)
        self.vectorized = all(_is_supported(nt) for nt in self.nets)
        if not self.vectorized:
            return
        n_in = len(self.nets[0].input_nodes)
        n_out = len(self.nets[0].output_nodes)
        in_idx = np.empty((self.n, n_in), dtype=np.int64)
        out_idx = np.empty((self.n, n_out), dtype=np.int64)
        levels: dict[int, dict[str, list]] = {}
        total = 0
        for r, net in enumerate(self.nets):
            gid: dict[int, int] = {}
            depth: dict[int, int] = {}

            def g(key):
                nonlocal total
                if key not in gid:
                    gid[key] = total
                    total += 1
                return gid[key]

            for j, k in enumerate(net.input_nodes):
                in_idx[r, j] = g(k)
                depth[k] = 0
            for j, k in enumerate(net.output_nodes):
                out_idx[r, j] = g(k)
            for node, _act, _agg, bias, response, links in net.node_evals:
                d = 1 + max((depth.get(i, 0) for i, _ in links), default=0)
                depth[node] = d
                lv = levels.setdefault(d, {"node": [], "bias": [], "resp": [], "src": [], "dst": [], "w": []})
                pos = len(lv["node"])
                lv["node"].append(g(node))
                lv["bias"].append(bias)
                lv["resp"].append(response)
                for i, w in links:
                    lv["src"].append(g(i))
                    lv["dst"].append(pos)
                    lv["w"].append(w)
        self.total = total
        self.in_idx = in_idx
        self.out_idx = out_idx
        self.levels = []
        for d in sorted(levels):
            lv = levels[d]
            self.levels.append((np.array(lv["node"], np.int64), np.array(lv["bias"], float),
                                np.array(lv["resp"], float), np.array(lv["src"], np.int64),
                                np.array(lv["dst"], np.int64), np.array(lv["w"], float)))

    def __len__(self):
        return self.n

    def activate(self, X: np.ndarray) -> np.ndarray:
        if not self.vectorized:
            return np.array([nt.activate(x) for nt, x in zip(self.nets, X)], float)
        V = np.zeros(self.total)
        V[self.in_idx] = X
        for node, bias, resp, src, dst, w in self.levels:
            s = np.bincount(dst, weights=w * V[src], minlength=len(node)) if len(src) else np.zeros(len(node))
            z = np.clip(5.0 * (bias + resp * s), -60.0, 60.0)
            V[node] = 1.0 / (1.0 + np.exp(-z))
        return V[self.out_idx]
