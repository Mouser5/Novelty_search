"""
Модель робота повторяет эксперимент Лемана и Стэнли (2011) в реализации Я. Омельяненко:
  * 6 дальномеров под углами −90°, −45°, 0°, 45°, 90°, 180° к курсу, дальность 100;
  * 4 «радара» цели — секторы по 90° (впереди, слева, сзади, справа), срабатывает тот,
    в котором находится выход;
  * 2 выхода сети: изменение угловой скорости и изменение линейной скорости
    (выход сети − 0,5), обе ограничены ±3;
  * радиус робота 8; при столкновении со стеной робот не сдвигается;
  * 400 шагов на одну оценку; выход найден, если робот ближе 5 к точке выхода.

Входы сети нормированы: дальномеры делятся на дальность (0…1), радары — 0 или 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .batchnet import BatchNetwork
from .maze import Maze, point_segment_distance

RANGEFINDER_ANGLES = np.radians([-90.0, -45.0, 0.0, 45.0, 90.0, -180.0])
N_INPUTS = len(RANGEFINDER_ANGLES) + 4
N_OUTPUTS = 2


@dataclass
class RobotParams:
    radius: float = 8.0
    range: float = 100.0
    max_speed: float = 3.0
    max_angular: float = 3.0
    steps: int = 400


@dataclass
class SimResult:
    final_pos: np.ndarray        # (n, 2) — точка остановки
    solved: np.ndarray           # (n,) bool
    solved_step: np.ndarray      # (n,) номер шага, на котором найден выход, или −1
    samples: np.ndarray          # (n, s, 2) — позиции через каждые sample_every шагов (траектория)
    min_dist: np.ndarray         # (n,) минимальное расстояние до выхода за всё время
    paths: np.ndarray | None     # (n_rec, steps+1, 2) — полные траектории для выбранных роботов


def _walls(maze: Maze):
    c = getattr(maze, "_wall_cache", None)
    if c is None:
        w = maze.walls
        c = (w[:, 0], w[:, 1], w[:, 2] - w[:, 0], w[:, 3] - w[:, 1])
        maze._wall_cache = c
    return c


def _min_wall_dist2(points, maze: Maze):
    wx, wy, sx, sy = _walls(maze)
    ss = getattr(maze, "_wall_len2", None)
    if ss is None:
        ss = np.maximum(sx * sx + sy * sy, 1e-12)
        maze._wall_len2 = ss
    px = points[:, 0:1] - wx
    py = points[:, 1:2] - wy
    t = np.clip((px * sx + py * sy) / ss, 0.0, 1.0)
    dx = px - t * sx
    dy = py - t * sy
    return (dx * dx + dy * dy).min(axis=1)


def _sensors(pos, heading_deg, maze: Maze, rp: RobotParams):

    n = len(pos)
    wx, wy, sx, sy = _walls(maze)
    theta = np.radians(heading_deg)[:, None] + RANGEFINDER_ANGLES[None, :]
    c = np.cos(theta)[:, :, None]
    s = np.sin(theta)[:, :, None]
    qpx = (wx[None, :] - pos[:, 0:1])[:, None, :]
    qpy = (wy[None, :] - pos[:, 1:2])[:, None, :]
    denom = c * sy - s * sx
    denom[denom == 0] = 1e-12
    inv = 1.0 / denom
    t = (qpx * sy - qpy * sx) * inv
    u = (qpx * s - qpy * c) * inv
    miss = (u < 0) | (u > 1) | (t < 0)
    t[miss] = rp.range
    ranges = np.minimum(t.min(axis=2), rp.range) / rp.range

    gx = maze.exit[0] - pos[:, 0]
    gy = maze.exit[1] - pos[:, 1]
    rel = (np.degrees(np.arctan2(gy, gx)) - heading_deg) % 360.0
    radar = np.zeros((n, 4))
    radar[:, 0] = (rel >= 315) | (rel < 45)        # впереди
    radar[:, 1] = (rel >= 45) & (rel < 135)        # слева (при оси y вниз)
    radar[:, 2] = (rel >= 135) & (rel < 225)       # сзади
    radar[:, 3] = (rel >= 225) & (rel < 315)       # справа
    return np.concatenate([ranges, radar], axis=1)


def simulate(maze: Maze, nets, rp: RobotParams | None = None, sample_every: int = 50,
             record: list[int] | None = None) -> SimResult:
    rp = rp or RobotParams()
    n = len(nets)
    if isinstance(nets, BatchNetwork):
        batch = nets if nets.vectorized else None
        nets = nets.nets
    else:
        batch = None
    pos = np.tile(np.asarray(maze.start, float), (n, 1))
    heading = np.full(n, float(maze.heading))
    speed = np.zeros(n)
    ang = np.zeros(n)
    solved = np.zeros(n, bool)
    solved_step = np.full(n, -1)
    ex = np.asarray(maze.exit, float)
    min_dist = np.hypot(*(pos - ex).T)
    samples = []
    rec = list(record or [])
    paths = np.zeros((len(rec), rp.steps + 1, 2)) if rec else None
    if rec:
        paths[:, 0] = pos[rec]
    idle = np.array([0.5, 0.5])

    for step in range(rp.steps):
        active = np.flatnonzero(~solved)
        if len(active) == 0:
            if rec:
                paths[:, step + 1:] = pos[rec][:, None]
            break
        inp = _sensors(pos[active], heading[active], maze, rp)
        if batch is not None:
            full = np.zeros((n, inp.shape[1]))
            full[active] = inp
            out = batch.activate(full)[active]
        else:
            out = np.empty((len(active), 2))
            for j, i in enumerate(active):
                o = nets[i].activate(inp[j])
                out[j, 0], out[j, 1] = o[0], o[1]
        ang[active] = np.clip(ang[active] + (out[:, 0] - 0.5), -rp.max_angular, rp.max_angular)
        speed[active] = np.clip(speed[active] + (out[:, 1] - 0.5), -rp.max_speed, rp.max_speed)
        rad = np.radians(heading[active])
        new_pos = pos[active] + np.stack([np.cos(rad), np.sin(rad)], 1) * speed[active, None]
        heading[active] = (heading[active] + ang[active]) % 360.0
        hit = _min_wall_dist2(new_pos, maze) < rp.radius ** 2
        moved = active[~hit]
        pos[moved] = new_pos[~hit]

        d = np.hypot(*(pos[active] - ex).T)
        min_dist[active] = np.minimum(min_dist[active], d)
        just = active[d < maze.exit_radius]
        solved[just] = True
        solved_step[just] = step + 1
        if rec:
            paths[:, step + 1] = pos[rec]
        if (step + 1) % sample_every == 0:
            samples.append(pos.copy())

    while len(samples) < rp.steps // sample_every:      # если все решили досрочно
        samples.append(pos.copy())
    return SimResult(final_pos=pos.copy(), solved=solved, solved_step=solved_step,
                     samples=np.stack(samples, 1) if samples else pos[:, None].copy(),
                     min_dist=min_dist, paths=paths)


def behaviors(res: SimResult, kind: str = "endpoint") -> np.ndarray:
    """Описание поведения для меры новизны.

    * "endpoint" — точка остановки (x, y), как в статье авторов;
    * "trajectory" — позиции через каждые sample_every шагов (вектор длины 2·s).
    """
    if kind == "endpoint":
        return res.final_pos.copy()
    if kind == "trajectory":
        return res.samples.reshape(len(res.samples), -1)
    raise ValueError(f"неизвестное описание поведения: {kind}")
