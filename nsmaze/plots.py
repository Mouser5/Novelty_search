from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .maze import Maze

def _save(fig, out):
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close(fig)


METHOD_NAMES = {"novelty": "Поиск новизны", "fitness": "Поиск к цели", "random": "Случайный отбор",
                "mcns": "Новизна + мин. критерий", "mapelites": "MAP-Elites"}
METHOD_COLORS = {"novelty": "#A8432A", "fitness": "#2E5E8C", "random": "#8C8496",
                 "mcns": "#E8946F", "mapelites": "#3C8D5A"}


def draw_maze(ax, maze: Maze, show_points: bool = True, lw: float = 2.5):
    for x1, y1, x2, y2 in maze.walls:
        ax.plot([x1, x2], [y1, y2], color="#1E1C2A", lw=lw, solid_capstyle="round", zorder=3)
    if show_points:
        ax.scatter(*maze.start, s=120, color="#2E5E8C", zorder=5, edgecolor="white", lw=1.5, label="старт")
        ax.scatter(*maze.exit, s=180, color="#A8432A", marker="*", zorder=5, edgecolor="white", lw=1, label="выход")
    x0, y0, x1, y1 = maze.bbox
    pad = 4
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y1 + pad, y0 - pad)          # ось y вниз, как в картах авторов: старт снизу
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def heatmap(ax, maze: Maze, points: np.ndarray, cell: float = 4.0, cmap: str = "magma_r", title: str | None = None,
            log: bool = True):
    """Тепловая карта """
    x0, y0, x1, y1 = maze.bbox
    bx = np.arange(x0, x1 + cell, cell)
    by = np.arange(y0, y1 + cell, cell)
    H, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[bx, by])
    H = np.ma.masked_where(H.T == 0, H.T)
    im = ax.pcolormesh(bx, by, H, cmap=cmap, norm=LogNorm(vmin=1, vmax=max(H.max(), 2)) if log else None,
                       shading="flat", zorder=1)
    draw_maze(ax, maze)
    if title:
        ax.set_title(title, fontsize=12)
    return im


def path(ax, maze: Maze, xy: np.ndarray, color: str = "#A8432A", label: str | None = None):
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2, zorder=4, label=label)


def figure_run(result, maze: Maze, out: str | Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.1, 1.1, 1.3]})
    name = METHOD_NAMES.get(result.params["method"], result.params["method"])
    im = heatmap(axes[0], maze, result.endpoints, title=f"Точки остановки всех роботов ({len(result.endpoints)})")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="число роботов")
    draw_maze(axes[1], maze)
    if result.solver_path is not None:
        path(axes[1], maze, result.solver_path, label="решение" if result.solved else "лучший робот")
    axes[1].set_title("Траектория " + ("решившего робота" if result.solved else "лучшего робота"), fontsize=12)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9, frameon=False)
    h = result.history
    ev = [r["evals"] for r in h]
    best_gen = np.array([r["best_dist"] for r in h])
    axes[2].plot(ev, best_gen, color="#8C8496", lw=1.2, label="ближайший к выходу в поколении")
    axes[2].plot(ev, np.minimum.accumulate(best_gen), color="#1E1C2A", lw=2, label="ближайший за всё время")
    axes[2].set_xlabel("оценок"); axes[2].set_ylabel("расстояние до выхода")
    ax2 = axes[2].twinx()
    ax2.plot(ev, [r["coverage"] for r in h], color="#A8432A", label="покрытие лабиринта")
    ax2.set_ylabel("покрытие", color="#A8432A"); ax2.set_ylim(0, 1)
    axes[2].set_title("Динамика поиска", fontsize=12)
    lines = axes[2].get_lines() + ax2.get_lines()
    axes[2].legend(lines, [l.get_label() for l in lines], fontsize=9, loc="upper right")
    status = f"решено за {result.evals} оценок" if result.solved else "не решено"
    fig.suptitle(f"{name} · лабиринт «{maze.name}» · seed {result.seed} · {status}", fontsize=14)
    fig.tight_layout()
    _save(fig, out)


def figure_heatmaps_compare(results_by_method: dict, maze: Maze, out: str | Path, max_points: int = 60_000):
    n = len(results_by_method)
    fig, axes = plt.subplots(1, n, figsize=(max(4.6 * n, 8.0), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (method, results) in zip(axes, results_by_method.items()):
        pts = np.concatenate([r.endpoints for r in results])
        if len(pts) > max_points:
            pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
        heatmap(ax, maze, pts)
        solved = sum(r.solved for r in results)
        cov = np.mean([r.coverage for r in results])
        ax.set_title(f"{METHOD_NAMES.get(method, method)}\nрешено {solved}/{len(results)}, покрытие {cov:.2f}",
                     fontsize=12)
    fig.suptitle(f"Где останавливались роботы · лабиринт «{maze.name}»", fontsize=14)
    fig.tight_layout()
    _save(fig, out)


def figure_methods_bars(summary_rows: list[dict], out: str | Path):
    mazes = list(dict.fromkeys(r["maze"] for r in summary_rows))
    methods = list(dict.fromkeys(r["p_method"] for r in summary_rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    w = min(0.8 / len(methods), 0.25)
    for j, m in enumerate(methods):
        rate, med = [], []
        for mz in mazes:
            rows = [r for r in summary_rows if r["maze"] == mz and r["p_method"] == m]
            rate.append(np.mean([r["solved"] for r in rows]) if rows else 0)
            ev = [r["evals"] for r in rows if r["solved"]]
            med.append(np.median(ev) if ev else np.nan)
        x = np.arange(len(mazes)) + j * w - 0.4 + w / 2
        axes[0].bar(x, rate, w, color=METHOD_COLORS.get(m), label=METHOD_NAMES.get(m, m))
        axes[1].bar(x, med, w, color=METHOD_COLORS.get(m), label=METHOD_NAMES.get(m, m))
    for ax, t in zip(axes, ["Доля решённых запусков", "Медиана оценок до решения (только решённые)"]):
        ax.set_xticks(np.arange(len(mazes))); ax.set_xticklabels(mazes); ax.set_title(t)
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out)


def figure_sweep_1d(rows: list[dict], param: str, out: str | Path, maze_name: str = ""):
    vals = sorted(set(r[f"p_{param}"] for r in rows), key=lambda v: (isinstance(v, str), v))
    def agg(fn):
        return [fn([r for r in rows if r[f"p_{param}"] == v]) for v in vals]
    rate = agg(lambda rs: np.mean([r["solved"] for r in rs]))
    med = agg(lambda rs: np.median([r["evals"] for r in rs if r["solved"]]) if any(r["solved"] for r in rs) else np.nan)
    q1 = agg(lambda rs: np.percentile([r["evals"] for r in rs if r["solved"]], 25) if any(r["solved"] for r in rs) else np.nan)
    q3 = agg(lambda rs: np.percentile([r["evals"] for r in rs if r["solved"]], 75) if any(r["solved"] for r in rs) else np.nan)
    cov = agg(lambda rs: np.mean([r["coverage"] for r in rs]))
    x = np.arange(len(vals))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(x, rate, "o-", color="#A8432A"); axes[0].set_ylim(0, 1.05); axes[0].set_title("Доля решённых запусков")
    axes[1].plot(x, med, "o-", color="#1E1C2A")
    axes[1].fill_between(x, q1, q3, color="#1E1C2A", alpha=.15)
    axes[1].set_title("Оценок до решения (медиана, межквартильный размах)")
    axes[2].plot(x, cov, "o-", color="#2E5E8C"); axes[2].set_ylim(0, 1.05); axes[2].set_title("Покрытие лабиринта")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([str(v) for v in vals]); ax.set_xlabel(param); ax.grid(alpha=.3)
    fig.suptitle(f"Характеристическая карта: влияние «{param}»" + (f" · лабиринт «{maze_name}»" if maze_name else ""),
                 fontsize=14)
    fig.tight_layout()
    _save(fig, out)


def figure_sweep_2d(rows: list[dict], p1: str, p2: str, out: str | Path, maze_name: str = ""):
    v1 = sorted(set(r[f"p_{p1}"] for r in rows))
    v2 = sorted(set(r[f"p_{p2}"] for r in rows))
    rate = np.full((len(v2), len(v1)), np.nan)
    med = np.full_like(rate, np.nan)
    for i, b in enumerate(v2):
        for j, a in enumerate(v1):
            rs = [r for r in rows if r[f"p_{p1}"] == a and r[f"p_{p2}"] == b]
            if rs:
                rate[i, j] = np.mean([r["solved"] for r in rs])
                ev = [r["evals"] for r in rs if r["solved"]]
                med[i, j] = np.median(ev) if ev else np.nan
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, M, t, cm in ((axes[0], rate, "Доля решённых запусков", "YlOrRd"),
                         (axes[1], med, "Медиана оценок до решения", "Blues")):
        im = ax.imshow(M, cmap=cm, origin="lower", aspect="auto")
        ax.set_xticks(range(len(v1))); ax.set_xticklabels(v1); ax.set_xlabel(p1)
        ax.set_yticks(range(len(v2))); ax.set_yticklabels(v2); ax.set_ylabel(p2)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}" if M is rate else f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9)
        ax.set_title(t); fig.colorbar(im, ax=ax)
    fig.suptitle(f"Характеристическая карта «{p1}» × «{p2}»" + (f" · «{maze_name}»" if maze_name else ""), fontsize=14)
    fig.tight_layout()
    _save(fig, out)


def figure_mazes(mazes: list[Maze], out: str | Path):
    n = len(mazes)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, mazes):
        draw_maze(ax, m)
        d = m.difficulty()
        ax.set_title(f"{m.name}\nпуть {d['path']:.0f}, по прямой {d['euclid']:.0f}, обманчивость {d['deception']}",
                     fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout(h_pad=2.5)
    _save(fig, out)
