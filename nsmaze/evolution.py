"""Эволюция

  * "novelty"   — NEAT + поиск новизны (Lehman, Stanley, 2011) — исследуемая модель;
  * "fitness"   — NEAT с целевой функцией f = D − d (D — диагональ лабиринта, d — расстояние
                   от точки остановки до выхода), классический «поиск к цели»;
  * "random"    — NEAT со случайной приспособленностью (контроль: отбор без информации);
  * "mcns"      — поиск новизны с минимальным критерием (Lehman, Stanley, 2010): особи, не
                   выполнившие критерий, получают нулевую оценку и не попадают в архив;
  * "mapelites" — MAP-Elites (Mouret, Clune, 2015) на тех же геномах NEAT: сетка по точке
                   остановки, в каждой клетке хранится лучшая по целевой функции особь.
"""
from __future__ import annotations

import copy
import itertools
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import neat

from .batchnet import BatchNetwork
from .maze import Maze
from .novelty import NoveltyArchive
from .simulator import RobotParams, behaviors, simulate

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "neat_maze.ini"
METHODS = ("novelty", "fitness", "random", "mcns", "mapelites")


@dataclass
class Params:
    method: str = "novelty"
    pop_size: int = 250
    max_evals: int = 100_000

    # поиск новизны
    k: int = 15
    threshold: float = 6.0
    dynamic_threshold: bool = True
    behavior: str = "endpoint"
    sample_every: int = 50
    # MCNS
    mc_kind: str = "progress"
    mc_value: float = 0.01
    # MAP-Elites
    me_bins: int = 20
    # NEAT (None — значение из ini-файла)
    node_add_prob: float | None = None
    conn_add_prob: float | None = None
    weight_mutate_rate: float | None = None
    weight_mutate_power: float | None = None
    compatibility_threshold: float | None = None

    steps: int = 400
    coverage_cell: float = 10.0
    config_path: str = str(DEFAULT_CONFIG)
    verbose: bool = False
    progress_every: int = 25


@dataclass
class RunResult:
    maze: str
    seed: int
    params: dict
    solved: bool
    evals: int                       # оценок до решения (или всего, если не решено)
    generations: int
    archive_size: int
    final_threshold: float
    coverage: float                  # доля проходимых клеток лабиринта, где побывали точки остановки
    best_dist: float                 # лучшее (минимальное) расстояние до выхода в точке остановки
    solver_nodes: int
    solver_conns: int
    time_s: float
    history: list = field(default_factory=list)
    endpoints: np.ndarray | None = None      # (E, 2) точки остановки всех оценённых роботов
    solver_path: np.ndarray | None = None    # траектория решившего (или лучшего) робота
    solver_genome: bytes | None = None

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("history", "endpoints", "solver_path", "solver_genome", "params")}
        d.update({f"p_{k}": v for k, v in self.params.items() if k not in ("config_path", "verbose", "progress_every")})
        return d

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "RunResult":
        with open(path, "rb") as f:
            return pickle.load(f)


class _Stop(Exception):
    pass


def _progress(p: "Params", seed: int, st: dict, t0: float, extra: str = "") -> None:
    if not p.progress_every or st["gen"] % p.progress_every != 0:
        return
    el = time.time() - t0
    left = el / max(st["evals"], 1) * max(p.max_evals - st["evals"], 0)
    print(f"  [{p.method} seed={seed}] поколение {st['gen']:4d}, оценок {st['evals']:6d}/{p.max_evals}, "
          f"ближе всего к выходу {st['best_d']:5.1f}{extra}; прошло {el / 60:4.1f} мин, "
          f"до лимита ≤ {left / 60:4.1f} мин", flush=True)


# ---------------------------------------------------------------------- вспомогательное
class Coverage:
    """Покрытие лабиринта: доля проходимых клеток сетки, в которые попала хотя бы одна точка остановки."""

    def __init__(self, maze: Maze, cell: float = 10.0, radius: float = 8.0):
        x0, y0, x1, y1 = maze.bbox
        self.x0, self.y0, self.cell = x0, y0, cell
        self.nx = int(np.ceil((x1 - x0) / cell))
        self.ny = int(np.ceil((y1 - y0) / cell))
        free, xs, ys = maze.free_space_grid(cell=2.0, radius=radius)
        gx, gy = np.meshgrid(xs, ys)
        ix = np.clip(((gx[free] - x0) // cell).astype(int), 0, self.nx - 1)
        iy = np.clip(((gy[free] - y0) // cell).astype(int), 0, self.ny - 1)
        self.free = np.zeros((self.ny, self.nx), bool)
        self.free[iy, ix] = True
        self.visited = np.zeros_like(self.free)

    def add(self, pts: np.ndarray) -> None:
        ix = np.clip(((pts[:, 0] - self.x0) // self.cell).astype(int), 0, self.nx - 1)
        iy = np.clip(((pts[:, 1] - self.y0) // self.cell).astype(int), 0, self.ny - 1)
        self.visited[iy, ix] = True

    @property
    def value(self) -> float:
        return float((self.visited & self.free).sum() / max(self.free.sum(), 1))


def load_config(p: Params) -> neat.Config:
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, p.config_path)
    config.pop_size = p.pop_size
    gc = config.genome_config
    for name in ("node_add_prob", "conn_add_prob", "weight_mutate_rate", "weight_mutate_power"):
        val = getattr(p, name)
        if val is not None:
            setattr(gc, name, val)
    if p.compatibility_threshold is not None:
        config.species_set_config.compatibility_threshold = p.compatibility_threshold
    return config


def _net_size(genome) -> tuple[int, int]:
    conns = sum(1 for c in genome.connections.values() if c.enabled)
    return len(genome.nodes), conns


def _goal_fitness(maze: Maze, final_pos: np.ndarray) -> np.ndarray:
    d = np.hypot(final_pos[:, 0] - maze.exit[0], final_pos[:, 1] - maze.exit[1])
    return maze.diagonal - d


def run(maze: Maze, p: Params, seed: int = 0) -> RunResult:
    if p.method not in METHODS:
        raise ValueError(f"method должен быть одним из {METHODS}")
    random.seed(seed)
    np.random.seed(seed)
    if p.method == "mapelites":
        return _run_map_elites(maze, p, seed)

    config = load_config(p)
    rp = RobotParams(steps=p.steps)
    archive = NoveltyArchive(k=p.k, threshold=p.threshold, dynamic=p.dynamic_threshold)
    cov = Coverage(maze, p.coverage_cell)
    start_dist = float(np.hypot(maze.exit[0] - maze.start[0], maze.exit[1] - maze.start[1]))
    st = {"evals": 0, "gen": 0, "solved": False, "solver": None, "best": None, "best_d": np.inf,
          "history": [], "endpoints": []}
    t0 = time.time()

    def eval_genomes(genomes, config):
        nets = BatchNetwork([neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomes])
        res = simulate(maze, nets, rp, sample_every=p.sample_every)
        final = res.final_pos
        st["endpoints"].append(final.astype(np.float32))
        cov.add(final)
        d_exit = np.hypot(final[:, 0] - maze.exit[0], final[:, 1] - maze.exit[1])

        if p.method in ("novelty", "mcns"):
            b = behaviors(res, p.behavior)
            nov = archive.novelty(b)
            eligible = None
            if p.method == "mcns":
                if p.mc_kind == "progress":
                    eligible = (start_dist - d_exit) / start_dist >= p.mc_value
                else:
                    eligible = np.hypot(final[:, 0] - maze.start[0], final[:, 1] - maze.start[1]) \
                               >= p.mc_value * maze.diagonal
                if eligible.any():
                    nov = np.where(eligible, nov, 0.0)
                else:
                    eligible = None
            archive.update(b, nov, eligible)
            scores = nov
        elif p.method == "fitness":
            scores = _goal_fitness(maze, final)
        else:  # random
            scores = np.random.rand(len(genomes))

        for (gid, g), s in zip(genomes, scores):
            g.fitness = float(s)

        ib = int(np.argmin(d_exit))
        if d_exit[ib] < st["best_d"]:
            st["best_d"], st["best"] = float(d_exit[ib]), copy.deepcopy(genomes[ib][1])
        sizes = np.array([_net_size(g) for _, g in genomes])
        st["history"].append({"gen": st["gen"], "evals": st["evals"] + len(genomes),
                              "best_dist": float(d_exit.min()), "mean_dist": float(d_exit.mean()),
                              "archive": len(archive), "threshold": archive.threshold,
                              "coverage": cov.value, "mean_nodes": float(sizes[:, 0].mean()),
                              "mean_conns": float(sizes[:, 1].mean())})
        if res.solved.any():
            first = int(np.flatnonzero(res.solved)[0])
            st["evals"] += first + 1
            st["solved"], st["solver"] = True, copy.deepcopy(genomes[first][1])
            raise _Stop
        st["evals"] += len(genomes)
        st["gen"] += 1
        _progress(p, seed, st, t0, f", архив {len(archive)}, покрытие {cov.value:.2f}")
        if st["evals"] >= p.max_evals:
            raise _Stop

    pop = neat.Population(config)
    n_gen = int(np.ceil(p.max_evals / p.pop_size)) + 1
    try:
        pop.run(eval_genomes, n_gen)
    except _Stop:
        pass
    return _finish(maze, p, seed, st, archive, cov, config, rp, t0)


def _finish(maze, p, seed, st, archive, cov, config, rp, t0) -> RunResult:
    g = st["solver"] or st["best"]
    nodes, conns = _net_size(g) if g is not None else (0, 0)
    path = None
    if g is not None:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        path = simulate(maze, [net], rp, record=[0]).paths[0]
    return RunResult(maze=maze.name, seed=seed, params=asdict(p), solved=st["solved"],
                     evals=int(st["evals"]), generations=int(st["gen"]) + (1 if st["solved"] else 0),
                     archive_size=len(archive) if archive is not None else 0,
                     final_threshold=float(archive.threshold) if archive is not None else 0.0,
                     coverage=cov.value, best_dist=0.0 if st["solved"] else float(st["best_d"]),
                     solver_nodes=nodes, solver_conns=conns, time_s=round(time.time() - t0, 1),
                     history=st["history"],
                     endpoints=np.concatenate(st["endpoints"]) if st["endpoints"] else None,
                     solver_path=path, solver_genome=pickle.dumps(g) if g is not None else None)


def _run_map_elites(maze: Maze, p: Params, seed: int) -> RunResult:
    config = load_config(p)
    rp = RobotParams(steps=p.steps)
    cov = Coverage(maze, p.coverage_cell)
    pop = neat.Population(config)
    gc = config.genome_config
    keys = itertools.count(10 ** 7)
    x0, y0, x1, y1 = maze.bbox
    bins = p.me_bins
    elites: dict[tuple[int, int], tuple[float, object]] = {}
    st = {"evals": 0, "gen": 0, "solved": False, "solver": None, "best": None, "best_d": np.inf,
          "history": [], "endpoints": []}
    t0 = time.time()
    batch = list(pop.population.values())

    while True:
        nets = BatchNetwork([neat.nn.FeedForwardNetwork.create(g, config) for g in batch])
        res = simulate(maze, nets, rp, sample_every=p.sample_every)
        final = res.final_pos
        st["endpoints"].append(final.astype(np.float32))
        cov.add(final)
        fit = _goal_fitness(maze, final)
        d_exit = maze.diagonal - fit
        ix = np.clip(((final[:, 0] - x0) / (x1 - x0) * bins).astype(int), 0, bins - 1)
        iy = np.clip(((final[:, 1] - y0) / (y1 - y0) * bins).astype(int), 0, bins - 1)
        for g, f, cx, cy in zip(batch, fit, ix, iy):
            g.fitness = float(f)
            cell = (int(cx), int(cy))
            if cell not in elites or f > elites[cell][0]:
                elites[cell] = (float(f), g)
        ib = int(np.argmin(d_exit))
        if d_exit[ib] < st["best_d"]:
            st["best_d"], st["best"] = float(d_exit[ib]), copy.deepcopy(batch[ib])
        st["history"].append({"gen": st["gen"], "evals": st["evals"] + len(batch),
                              "best_dist": float(d_exit.min()), "mean_dist": float(d_exit.mean()),
                              "archive": len(elites), "threshold": 0.0, "coverage": cov.value,
                              "mean_nodes": float(np.mean([len(g.nodes) for g in batch])),
                              "mean_conns": float(np.mean([_net_size(g)[1] for g in batch]))})
        if res.solved.any():
            first = int(np.flatnonzero(res.solved)[0])
            st["evals"] += first + 1
            st["solved"], st["solver"] = True, copy.deepcopy(batch[first])
            break
        st["evals"] += len(batch)
        st["gen"] += 1
        _progress(p, seed, st, t0, f", клеток {len(elites)}, покрытие {cov.value:.2f}")
        if st["evals"] >= p.max_evals:
            break
        parents = [v[1] for v in elites.values()]
        batch = []
        for _ in range(p.pop_size):
            child = copy.deepcopy(random.choice(parents))
            child.key = next(keys)
            child.fitness = None
            child.mutate(gc)
            batch.append(child)

    class _A:
        threshold = 0.0
        def __len__(self): return len(elites)
    return _finish(maze, p, seed, st, _A(), cov, config, rp, t0)


# ---------------------------------------------------------------------- серия запусков
def _worker(args):
    maze_spec, p, seed = args
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    maze = maze_spec if isinstance(maze_spec, Maze) else Maze.load(maze_spec)
    return run(maze, p, seed)


def run_many(maze_spec, p: Params, seeds, workers: int | None = None, out_dir: str | Path | None = None,
             tag: str = ""):
    from multiprocessing import get_context
    workers = workers or max(1, (os.cpu_count() or 2) - 0)
    jobs = [(maze_spec, p, s) for s in seeds]
    results = []
    workers = min(workers, len(jobs))
    name = maze_spec.name if isinstance(maze_spec, Maze) else maze_spec
    print(f"[{p.method}] {name}: {len(jobs)} запусков на {workers} процессах, лимит {p.max_evals} оценок на запуск. "
          f"Запуск, не нашедший выход, идёт до лимита; ход печатается каждые {p.progress_every} поколений.",
          flush=True)
    if workers == 1:
        it = map(_worker, jobs)
    else:
        pool = get_context("spawn").Pool(workers)
        it = pool.imap_unordered(_worker, jobs)
    for r in it:
        results.append(r)
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            r.save(Path(out_dir) / f"{tag}seed{r.seed}.pkl")
        if True:
            status = f"решено за {r.evals} оценок" if r.solved else f"не решено, лучшее расстояние {r.best_dist:.1f}"
            print(f"[{p.method}] {r.maze} seed={r.seed}: {status}; архив {r.archive_size}; "
                  f"покрытие {r.coverage:.2f}; {r.time_s} с  (готово {len(results)}/{len(jobs)})", flush=True)
    if workers != 1:
        pool.close(); pool.join()
    results.sort(key=lambda r: r.seed)
    return results
