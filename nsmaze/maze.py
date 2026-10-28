"""Лабиринты: загрузка/сохранение, генерация лабиринтов разной сложности и оценка их сложности.

Формат файла лабиринта из книги Я. Омельяненко «Hands-On Neuroevolution with Python» :

    <число стен>
    <x старта> <y старта>
    <начальный курс робота, градусы>
    <x выхода> <y выхода>
    <x1> <y1> <x2> <y2>      # по одной стене-отрезку на строку
    ...

Пустые строки игнорируются.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAZES_DIR = Path(__file__).resolve().parent.parent / "mazes"


@dataclass
class Maze:
    walls: np.ndarray
    start: tuple[float, float]
    heading: float
    exit: tuple[float, float]
    name: str = "maze"
    exit_radius: float = 5.0     # робот «нашёл выход», если ближе этого расстояния
    info: dict = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = np.concatenate([self.walls[:, 0], self.walls[:, 2]])
        ys = np.concatenate([self.walls[:, 1], self.walls[:, 3]])
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    @property
    def diagonal(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return float(np.hypot(x1 - x0, y1 - y0))

    @classmethod
    def load(cls, path: str | Path) -> "Maze":
        path = Path(path)
        if not path.exists():
            cand = MAZES_DIR / f"{path.name}.txt" if path.suffix == "" else MAZES_DIR / path.name
            path = cand
        lines = [ln.split("#")[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        lines = [ln for ln in lines if ln]
        n = int(lines[0])
        sx, sy = map(float, lines[1].split())
        heading = float(lines[2])
        ex, ey = map(float, lines[3].split())
        walls = np.array([list(map(float, ln.split())) for ln in lines[4:4 + n]], dtype=float)
        if len(walls) != n:
            raise ValueError(f"{path}: ожидалось {n} стен, найдено {len(walls)}")
        return cls(walls=walls, start=(sx, sy), heading=heading, exit=(ex, ey), name=path.stem)

    def save(self, path: str | Path) -> None:
        out = [str(len(self.walls)), f"{self.start[0]:g} {self.start[1]:g}", f"{self.heading:g}",
               f"{self.exit[0]:g} {self.exit[1]:g}"]
        out += [" ".join(f"{v:g}" for v in w) for w in self.walls]
        Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")

    def free_space_grid(self, cell: float = 2.0, radius: float = 8.0):
        """Сетка проходимости: True там, где центр робота радиуса `radius` не задевает стены."""
        x0, y0, x1, y1 = self.bbox
        xs = np.arange(x0 + cell / 2, x1, cell)
        ys = np.arange(y0 + cell / 2, y1, cell)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        d = point_segment_distance(pts, self.walls).min(axis=1)
        return (d >= radius).reshape(gy.shape), xs, ys

    def shortest_path_length(self, cell: float = 2.0, radius: float = 8.0) -> float:
        free, xs, ys = self.free_space_grid(cell, radius)
        def to_idx(p):
            return int(np.argmin(np.abs(ys - p[1]))), int(np.argmin(np.abs(xs - p[0])))
        s, g = to_idx(self.start), to_idx(self.exit)
        h, w = free.shape
        dist = np.full(free.shape, np.inf)
        dist[s] = 0.0
        # Дейкстра на равномерной сетке с весами 1 и √2
        q = deque([s])
        steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5)]
        while q:
            r, c = q.popleft()
            for dr, dc, wgt in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and dist[r, c] + wgt < dist[rr, cc] - 1e-9:
                    dist[rr, cc] = dist[r, c] + wgt
                    q.append((rr, cc))
        gyy, gxx = np.meshgrid(ys, xs, indexing="ij")
        near = np.hypot(gxx - self.exit[0], gyy - self.exit[1]) <= max(self.exit_radius, cell)
        best = dist[near].min() if near.any() else dist[g]
        return float(best * cell)

    def difficulty(self) -> dict:
        """Показатели сложности лабиринта.

        * path — длина кратчайшего пути до выхода;
        * euclid — расстояние до выхода по прямой;
        * deception — во сколько раз путь длиннее прямой (чем больше, тем обманчивее
          «целевая» функция «будь ближе к выходу»).
        """
        path = self.shortest_path_length()
        euclid = float(np.hypot(self.exit[0] - self.start[0], self.exit[1] - self.start[1]))
        return {"walls": int(len(self.walls)), "path": round(path, 1), "euclid": round(euclid, 1),
                "deception": round(path / euclid, 2) if euclid > 0 else float("inf")}


def point_segment_distance(points: np.ndarray, walls: np.ndarray) -> np.ndarray:
    """Расстояния от точек (N, 2) до отрезков (W, 4) → матрица (N, W)."""
    a = walls[None, :, 0:2]
    b = walls[None, :, 2:4]
    p = points[:, None, :]
    ab = b - a
    denom = (ab ** 2).sum(-1)
    denom = np.where(denom == 0, 1e-12, denom)
    t = np.clip(((p - a) * ab).sum(-1) / denom, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.sqrt(((p - proj) ** 2).sum(-1))

def generate_grid_maze(rows: int, cols: int, cell: float = 40.0, seed: int = 0,
                       braid: float = 0.0, name: str | None = None,
                       exit_mode: str = "deceptive") -> Maze:
    """Генерирует лабиринт
    Параметры:
      * cell — размер клетки (ширина коридора), робот имеет радиус 8;
      * braid — доля стен, удаляемых случайно после генерации (0 — одна дорога к выходу,
        больше — появляются петли и альтернативные пути, лабиринт становится проще);
      * exit_mode — "deceptive": выход в клетке, наиболее далёкой по пути при малом расстоянии
        по прямой (максимизируем отношение путь/прямая); "far": выход в клетке, самой далёкой по пути.
    Старт — в левой нижней клетке.
    """
    rng = random.Random(seed)
    h = [[True] * cols for _ in range(rows)]
    v = [[True] * cols for _ in range(rows)]
    seen = [[False] * cols for _ in range(rows)]
    stack = [(rows - 1, 0)]
    seen[rows - 1][0] = True
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < rows and 0 <= c + dc < cols and not seen[r + dr][c + dc]]
        if not nbrs:
            stack.pop()
            continue
        nr, nc = rng.choice(nbrs)
        if nr == r + 1: h[r][c] = False
        elif nr == r - 1: h[nr][nc] = False
        elif nc == c + 1: v[r][c] = False
        else: v[nr][nc] = False
        seen[nr][nc] = True
        stack.append((nr, nc))
    inner = [("h", r, c) for r in range(rows - 1) for c in range(cols) if h[r][c]] + \
            [("v", r, c) for r in range(rows) for c in range(cols - 1) if v[r][c]]
    for kind, r, c in rng.sample(inner, int(len(inner) * braid)):
        (h if kind == "h" else v)[r][c] = False

    def neighbours(r, c):
        if r > 0 and not h[r - 1][c]: yield r - 1, c
        if r < rows - 1 and not h[r][c]: yield r + 1, c
        if c > 0 and not v[r][c - 1]: yield r, c - 1
        if c < cols - 1 and not v[r][c]: yield r, c + 1
    start_cell = (rows - 1, 0)
    dist = {start_cell: 0}
    q = deque([start_cell])
    while q:
        cur = q.popleft()
        for nb in neighbours(*cur):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    if exit_mode == "far":
        exit_cell = max(dist, key=lambda k: dist[k])
    else:
        def score(k):
            eu = np.hypot(k[0] - start_cell[0], k[1] - start_cell[1])
            return dist[k] / max(eu, 1.0) + 0.01 * dist[k]
        exit_cell = max((k for k in dist if k != start_cell), key=score)

    m = 5.0
    W, H = cols * cell, rows * cell
    segs = [(m, m, m + W, m), (m + W, m, m + W, m + H), (m + W, m + H, m, m + H), (m, m + H, m, m)]
    for r in range(rows - 1):           # горизонтальные стены
        c = 0
        while c < cols:
            if h[r][c]:
                c0 = c
                while c < cols and h[r][c]: c += 1
                y = m + (r + 1) * cell
                segs.append((m + c0 * cell, y, m + c * cell, y))
            else:
                c += 1
    for c in range(cols - 1):           # вертикальные стены
        r = 0
        while r < rows:
            if v[r][c]:
                r0 = r
                while r < rows and v[r][c]: r += 1
                x = m + (c + 1) * cell
                segs.append((x, m + r0 * cell, x, m + r * cell))
            else:
                r += 1

    def centre(rc):
        return (m + (rc[1] + 0.5) * cell, m + (rc[0] + 0.5) * cell)
    maze = Maze(walls=np.array(segs, dtype=float), start=centre(start_cell), heading=0.0,
                exit=centre(exit_cell), name=name or f"grid{rows}x{cols}_s{seed}")
    maze.info = {"rows": rows, "cols": cols, "cell": cell, "braid": braid, "seed": seed,
                 "path_cells": dist[exit_cell]}
    return maze


def open_maze(size: float = 200.0) -> Maze:
    """Простейший лабиринт без препятствий — контрольная (лёгкая) задача."""
    m = 5.0
    walls = np.array([(m, m, m + size, m), (m + size, m, m + size, m + size),
                      (m + size, m + size, m, m + size), (m, m + size, m, m)], dtype=float)
    return Maze(walls=walls, start=(m + 30, m + size - 30), heading=0.0,
                exit=(m + size - 30, m + 30), name="open")
